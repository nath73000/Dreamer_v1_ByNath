import torch.nn as nn
import torch 
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
        # recurrent_state : (B, recurrent_size)
        out = self.network(recurrent_state)             # (B, 2*latent_size)
        mean, std = torch.chunk(out, 2, dim=-1)         # chacun (B, latent_size)
        std = torch.nn.functional.softplus(std) + 0.1   # std > 0, évite collapse
        dist = torch.distributions.Normal(mean, std)    # distribution gaussienne
        sample = dist.rsample()                         # (B, latent_size)
        return dist, sample
    

class Posterior(nn.Module):
    def __init__(self, input_size, lattent_size, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)

        self.network = creat_sequential_model_1D(input_size, [self.config.hidden_size] * self.config.nb_layers, 2 * lattent_size, self.activation)

    def forward(self, recurrent_state_plus_encoded_obs):
        # recurrent_state : (B, recurrent_size)
        out = self.network(recurrent_state_plus_encoded_obs)             # (B, 2*latent_size)
        mean, std = torch.chunk(out, 2, dim=-1)         # chacun (B, latent_size)
        std = torch.nn.functional.softplus(std) + 0.1   # std > 0, évite collapse
        dist = torch.distributions.Normal(mean, std)    # distribution gaussienne
        sample = dist.rsample()                         # (B, latent_size)
        return dist, sample


class ActionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, X):
        return self.network(X)


class ValueModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, X):
        return self.network(X)