import torch.nn as nn


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
    def __init__(self, input_size, config):
        super().__init__()

        self.config = config
        self.activation = getattr(nn, self.config.activation)()
        
        self.linear = nn.Linear()
        self.gru    = nn.GRUCell()

    def forward(x):
        
        return x




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