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

class LearnPsi(nn.Module):
    def __init__(self):
        super(LearnPsi, self).__init__()
        # register parameter
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习的参数
        self.beta = nn.Parameter(torch.tensor(2.0))  # 可学习的参数
        self.gamma = nn.Parameter(torch.tensor(3.0))  # 可学习的参数
        self.coff = nn.Parameter(torch.tensor([1.0, -3.0, 3.0, -1.0]))  # 可学习的参数

    def forward(self, x):
        return (
            self.coff[0] * F.relu(x - 0) ** 2
            + self.coff[1] * F.relu(x - self.alpha) ** 2
            + self.coff[2] * F.relu(x - self.beta) ** 2
            + self.coff[3] * F.relu(x - self.gamma) ** 2
        )



