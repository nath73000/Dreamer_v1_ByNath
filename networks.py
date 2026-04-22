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
        activation = getattr(nn, self.config.activation)()

        self.convolutionnal_net = nn.Sequential(
            nn.Conv2d(chanels,             self.config.depth*1, self.config.kernel_size, self.config.stride, self.config.padding), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self,config.kernel_size, self.config.srtide, self.config.padding), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self,config.kernel_size, self.config.srtide, self.config.padding), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self,config.kernel_size, self.config.srtide, self.config.padding), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth * 8 * (height//(self.config.stride**4)) * (width//(self.config.stride**4)), output_size), activation
        )

    def forward(self, x):
        return self.convolutionnal_net(x)


class Decoder(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()

        self.config = config
        self.height, self.width, self.channels = outputShape
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernelSize,     self.config.stride, self.config.padding), activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, self.config.kernelSize,     self.config.stride, self.config.padding), activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, self.config.kernelSize + 1, self.config.stride, self.config.padding), activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels,       self.config.kernelSize + 1, self.config.stride, self.config.padding))

    def forward(self, x):
        return self.network(x)


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
    def __init__(self, input_size, lattent_size, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size] * self.config.nb_layers, 2 * lattent_size, self.activation)

    def forward(self, recurrent_state):
        out = self.network(recurrent_state)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        return dist, sample
    

class Posterior(nn.Module):
    def __init__(self, input_size, lattent_size, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size] * self.config.nb_layers, 2 * lattent_size, self.activation)

    def forward(self, recurrent_state_plus_encoded_obs):
        out = self.network(recurrent_state_plus_encoded_obs)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = torch.distributions.Normal(mean, std)
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
        self.action_size = action_size

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size]*self.config.nb_layers, 2 * action_size, self.config.activation)

    def forward(self, x):
        out = self.network(x)
        mean, std = out.chunk(2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std  = torch.nn.functional.softplus(std + self.init_std) + self.min_std

        base = Independent(Normal(mean, std), 1)          # event_dim = action_size
        dist = TransformedDistribution(base, TanhTransform(cache_size=1))
        return dist


class Critic(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config 

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size]*self.config.nb_layers, 1, self.config.activation)

    def forward(self, x):
        mean = self.network(x).squeeze(-1)
        return Normal(mean, 1.0)