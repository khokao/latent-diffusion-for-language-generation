import torch.nn as nn


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.)
