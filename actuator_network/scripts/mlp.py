import torch.nn as nn
import numpy as np
import torch


class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size, init_scale=2): # shape: hidden layer
        super(MLP, self).__init__()
        self.activation_fn = activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [init_scale]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(init_scale)

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(init_scale)

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, x):
        return self.architecture(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


# if __name__ == "__main__":
#     mlp = MLP()