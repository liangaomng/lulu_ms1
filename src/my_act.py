import torch
from torch import nn
import torch.nn.functional as F


def srelu(x):
    return F.relu(1 - x) * F.relu(x)


def phi(x):  #是否要乘1/2
    return 1/2*(F.relu(x)**2 - 3 * F.relu(x - 1)**2 + 3 * F.relu(x - 2)**2  - F.relu(x - 3)**2)


def ricker(x):
    a = 0.3
    return 1/(15*a) * torch.pi ** (1/4) * (1-(x/a)**2) * torch.exp(-0.5*(x/a)**2)



