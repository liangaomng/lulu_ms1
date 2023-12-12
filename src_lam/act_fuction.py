import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class LearnPsi(nn.Module):
    def __init__(self):
        super(LearnPsi, self).__init__()
        # register parameter
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习的参数
        self.beta = nn.Parameter(torch.tensor(2.0))  # 可学习的参数
        self.gamma = nn.Parameter(torch.tensor(3.0))  # 可学习的参数
        self.coff = nn.Parameter(torch.tensor([1.0, -3.0, 3.0, -1.0]))  # 可学习的参数
        self.omega1=nn.Parameter(torch.tensor(1.0))
        self.omega2=nn.Parameter(torch.tensor(2.0))
        self.omega3=nn.Parameter(torch.tensor(4.0))
        self.omega4=nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        return (
            self.coff[0] * F.relu(self.omega1*x - 0) ** 2
            + self.coff[1] * F.relu(self.omega2*x - self.alpha) ** 2
            + self.coff[2] * F.relu(self.omega3*x - self.beta) ** 2
            + self.coff[3] * F.relu(self.omega4*x - self.gamma) ** 2
        )
learn_psi= LearnPsi()
class phi(nn.Module):  #是否要乘1/2
    def __init__(self):
        super(phi, self).__init__()
        # register parameter
    def forward(self,x):

        return 1/2*(F.relu(x)**2 - 3 * F.relu(x - 1)**2 + 3 * F.relu(x - 2)**2  - F.relu(x - 3)**2)
phi=phi()

class SincPSi(nn.Module):
    def __init__(self):
        super(SincPSi, self).__init__()
        # 可学习的参数
        self.omega = nn.Parameter(torch.tensor(20.0))  # 调整 sinc 函数频率的参数
        self.offset=nn.Parameter(torch.tensor(1e-6))
    def forward(self, x):
        # sinc(omega * x) 的实现
        scaled_x = self.omega * x
        out=torch.where(torch.abs(x)<1e-20, torch.tensor(1.0, device=x.device), torch.sin(scaled_x) / (scaled_x+self.offset))
        return out
sinc_psi=SincPSi()
class St_act_in_4_subnet_space():
    def __init__(self):
        pass
    @classmethod
    def initializer_dict_torch(cls,identifier):
        return {
            "Glorot-normal": torch.nn.init.xavier_normal_,
            "Glorot-uniform": torch.nn.init.xavier_uniform_,
            "He-normal": torch.nn.init.kaiming_normal_,
            "He-uniform": torch.nn.init.kaiming_uniform_,
            "xavier_uniform": torch.nn.init.xavier_normal_,
        }[identifier]
    @classmethod
    def act_dict_torch(cls,identifier):
        return {
                "relu": torch.nn.ReLU(),
                "sinc_psi": sinc_psi,
                "learn_psi": learn_psi,
                "phi": phi,
        }[identifier]
    @classmethod
    def activations_get(cls,act_info:np.ndarray)->list:


        '''
           return torch.nn.function,every layer every activation function
           every subnet should have different activation function
           5 个子网络，3层，说明是5*3的矩阵，每个元素是一个字符串，代表激活函数的名称
        '''
        print(act_info.shape)
        subs,layers=act_info.shape
        act_torch = nn.ModuleList()
        for i in range(subs):#every sub-net
            row = nn.ModuleList()
            for j in range(layers):
                activation_func = St_act_in_4_subnet_space.act_dict_torch(act_info[i][j])
                row.append(activation_func)
            act_torch.append(row)
        return act_torch

    @classmethod
    def weight_ini_method(self,init_info:np.ndarray)->list:
        '''

        Args:
            init_info:4*1,4个子网络

        Returns: [],4个torch.nn.init,method of weight initialization

        '''
        net_method=[]
        for i in range(init_info.shape[0]):
            net_method.append(St_act_in_4_subnet_space.initializer_dict_torch(init_info[i][0]))

        return net_method

