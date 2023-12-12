import torch
from torch import nn
import torch.nn.functional as F

from .my_act import *

def initializer_dict_torch(identifier):
    return {
        "Glorot-normal": torch.nn.init.xavier_normal_,
        "Glorot-uniform": torch.nn.init.xavier_uniform_,
        "He-normal": torch.nn.init.kaiming_normal_,
        "He-uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }[identifier]

learn_psi= LearnPsi() #对象实例

def activation_dict_torch(identifier):
    return {
            "relu": torch.nn.functional.relu,
            "sRELU": srelu,
            "phi": phi,
            "ricker": ricker,
            "sin": torch.sin,
            "cos": torch.cos,
            "learn_psi": learn_psi,
        }[identifier]


def linear(x):
    return x

def activations_get(identifier):
    """Returns function.
    Args:
        identifier: Function or string.
    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        return activation_dict_torch[identifier]

    if callable(identifier): #callable() 函数用于检查一个对象是否是可调用的 这里是检查是否是一个函数
        return identifier
    raise TypeError(
        "Could not interpret activation function identifier: {}".format(identifier)
    )


#神经网络的权重初始化不能取constant 这样每一层相当于是一个神经元在演化了 没有什么区别

class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes,activation,kernel_initializer):
        super().__init__()

        if isinstance(activation, list):
            self.activation = list(map(activations_get, activation))
        else:
            self.activation = activation_dict_torch(activation)

        initializer = initializer_dict_torch(kernel_initializer)
        initializer_zero = initializer_dict_torch("zeros")
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, x):

        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        return x


class MscaleLayer(nn.Module):

    def __init__(self, in_features, out_features, omega, kernel_initializer):

        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.omega = omega #第一层或者每一层的系数向量
        initializer = initializer_dict_torch(kernel_initializer)
        initializer_zero = initializer_dict_torch("zeros")
        initializer(self.linear.weight)
        initializer_zero(self.linear.bias)

    def forward(self, input):

        output = self.omega * self.linear(input)

        return output


class MscaleNN(nn.Module):

    def __init__(self, layer_sizes, activation, kernel_initializer, first_omega, hidden_omega):

        super().__init__()

        if isinstance(activation, list):
            self.activation = list(map(activations_get, activation))
        else:

            self.activation = activation_dict_torch(activation)

        self.linears = torch.nn.ModuleList()
        self.linears.append(
                MscaleLayer(
                    layer_sizes[0], layer_sizes[1], omega=first_omega, kernel_initializer=kernel_initializer
                )
            )

        for i in range(2, len(layer_sizes)-1):
            self.linears.append(
                MscaleLayer(
                    layer_sizes[i - 1], layer_sizes[i], omega=hidden_omega, kernel_initializer=kernel_initializer
                )
            )


        final_linear = nn.Linear(layer_sizes[-2], layer_sizes[-1]) #最后一层用一般的线性层 不用乘尺度因子 scale factor
        self.linears.append(
               final_linear
            )
        initializer = initializer_dict_torch(kernel_initializer)
        initializer_zero = initializer_dict_torch("zeros")
        initializer(self.linears[-1].weight)
        initializer_zero(self.linears[-1].bias)


    def forward(self, x):

        x = (self.activation[0](self.linears[0](x))
             if isinstance(self.activation, list)
            else self.activation(self.linears[0](x))
            )

        for j, linear in enumerate(self.linears[1:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)

        return x


class MscaleNN2(nn.Module):

    def __init__(self, sub_layer_sizes, activation, kernel_initializer, sub_omegas):

        super().__init__()
        print("test")
        print(activation)
        if isinstance(activation, list):

            self.activation = list(map(activations_get, activation))
        else:

            self.activation = activation_dict_torch(activation)

        self.subnets=nn.ModuleList()
        #get the sub_omegas‘ itemmiju
        for index,_ in enumerate(sub_omegas):
            fnn = FNN(layer_sizes =sub_layer_sizes,
                      activation=activation,
                      kernel_initializer=kernel_initializer)
            self.subnets.append(fnn)

        self.sub_omegas = sub_omegas
        print("__net___", self.subnets)
        exit()

    def forward(self, x):

        #y1 y2 y3 y4 is numbers of subscale neural networks
        y = []
        for index, subnet in enumerate(self.subnets):
           y.append(self.sub_omegas[index]*subnet(x))

        out = torch.sum(torch.stack(y), dim=0)

        return out

