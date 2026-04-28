import torch.nn as nn
import torch 
from torch.distributions import Normal, Bernoulli, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from utils import creat_sequential_model_1D



# -------------------------- Neural Network definition --------------------------


class Encoder(nn.Module):
    def __init__(self, input_shape, output_size, config):
        super().__init__()
        
        self.config = config
        height, width, chanels = input_shape
        activation = getattr(nn, self.config.activation)

        convolutionnal_layers = nn.Sequential(
            nn.Conv2d(chanels,             self.config.depth*1, self.config.kernel_size, self.config.stride, self.config.padding), activation(),
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernel_size, self.config.stride, self.config.padding), activation(),
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernel_size, self.config.stride, self.config.padding), activation(),
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernel_size, self.config.stride, self.config.padding), activation(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_output_size = convolutionnal_layers(torch.zeros(1, chanels, height, width)).shape[-1]

        self.convolutionnal_net = nn.Sequential(
            convolutionnal_layers,
            nn.Linear(conv_output_size, output_size),
            activation(),
        )

    def forward(self, x):
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0 if x.max() > 1.0 else x.float()
        return self.convolutionnal_net(x)


class Decoder(nn.Module):
    def __init__(self, input_size, output_shape, config):
        super().__init__()

        self.config = config
        self.height, self.width, self.channels = output_shape
        activation = getattr(nn, self.config.activation)
        output_padding = self.config.stride - 1

        self.network = nn.Sequential(
            nn.Linear(input_size, self.config.depth*8*4*4),
            activation(),
            nn.Unflatten(1, (self.config.depth*8, 4, 4)),
            nn.ConvTranspose2d(self.config.depth*8, self.config.depth*4, self.config.kernel_size, self.config.stride, self.config.padding, output_padding=output_padding), activation(),
            nn.ConvTranspose2d(self.config.depth*4, self.config.depth*2, self.config.kernel_size, self.config.stride, self.config.padding, output_padding=output_padding), activation(),
            nn.ConvTranspose2d(self.config.depth*2, self.config.depth*1, self.config.kernel_size, self.config.stride, self.config.padding, output_padding=output_padding), activation(),
            nn.ConvTranspose2d(self.config.depth*1, self.channels,       self.config.kernel_size, self.config.stride, self.config.padding, output_padding=output_padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.network(x)
        return x.permute(0, 2, 3, 1)


class RecurrentModel(nn.Module):
    def __init__(self, recurrent_size, latent_size, action_size, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()
        
        self.linear = nn.Linear(latent_size + action_size, self.config.hidden_size)
        self.gru    = nn.GRUCell(self.config.hidden_size, recurrent_size)

    def forward(self, previous_recurrent_stat, latent_state, action):
        x = torch.cat((latent_state, action), dim=-1)
        x = self.activation(self.linear(x))
        return self.gru(x, previous_recurrent_stat)


class Prior(nn.Module):
    def __init__(self, input_size, latent_size, config):
        super().__init__()
        self.config = config

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size] * self.config.nb_layers, 2 * latent_size, self.config.activation)

    def forward(self, recurrent_state):
        out = self.network(recurrent_state)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = Independent(Normal(mean, std), 1)
        sample = dist.rsample()
        return dist, sample
    

class Posterior(nn.Module):
    def __init__(self, input_size, latent_size, config):
        super().__init__()
        self.config = config

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size] * self.config.nb_layers, 2 * latent_size, self.config.activation)

    def forward(self, recurrent_state_plus_encoded_obs):
        out = self.network(recurrent_state_plus_encoded_obs)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = Independent(Normal(mean, std), 1)
        sample = dist.rsample()
        return dist, sample


class RewardModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size]*self.config.nb_layers, 2, self.config.activation)

    def forward(self, x):
        mean, log_std = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(log_std).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = creat_sequential_model_1D(inputSize, [self.config.hidden_size]*self.config.nb_layers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))

 
class Actor(nn.Module):
    def __init__(self, input_size, action_size, config):
        super().__init__()
        self.config = config
        self.action_size = int(action_size)
        self.mean_scale = getattr(config, "mean_scale", 5.0)
        self.init_std = getattr(config, "init_std", 5.0)
        self.min_std = getattr(config, "min_std", 1e-4)

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size]*self.config.nb_layers, 2 * self.action_size, self.config.activation)

    def forward(self, x, training=False, deterministic=False):
        out = self.network(x)
        mean, std = out.chunk(2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std  = torch.nn.functional.softplus(std + self.init_std) + self.min_std

        base = Independent(Normal(mean, std), 1)          # event_dim = action_size
        dist = TransformedDistribution(base, TanhTransform(cache_size=1))
        action = torch.tanh(mean) if deterministic else dist.rsample()

        if training:
            logprob = dist.log_prob(action)
            entropy = base.entropy()
            return action, logprob, entropy
        return action


class Critic(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config 

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size]*self.config.nb_layers, 1, self.config.activation)

    def forward(self, x):
        mean = self.network(x).squeeze(-1)
        return Normal(mean, 1.0)
