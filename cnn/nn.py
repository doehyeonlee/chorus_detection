import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

    @staticmethod
    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        weight = nn.Parameter(torch.randn(*shape) * 0.1)
        return weight

    @staticmethod
    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        bias = nn.Parameter(torch.full(shape, 0.1))
        return bias
