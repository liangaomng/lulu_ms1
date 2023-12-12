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

        if isinstance(activation, list):
            self.activation = list(map(activations_get, activation))
        else:
            self.activation = activation_dict_torch(activation)
           
        self.subnet_1 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_2 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_3 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_4 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_5 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_6 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.sub_omegas = sub_omegas

            
    def forward(self, x):

        y1 = self.subnet_1(self.sub_omegas[0]* x)

        y2 = self.subnet_2(self.sub_omegas[1]*x)

        y3 = self.subnet_3(self.sub_omegas[2]*x)

        y4 = self.subnet_4(self.sub_omegas[3]*x)

        y5 = self.subnet_5(self.sub_omegas[4]*x)
 
        y6 = self.subnet_6(self.sub_omegas[5]*x)

  
        out = y1 +  y2 +   y3 +  y4 +   y5 + y6

        return out

class MscaleNN2_multi_alphai(nn.Module):

    def __init__(self, sub_layer_sizes, activation, kernel_initializer, sub_omegas):

        super().__init__()

        if isinstance(activation, list):
            self.activation = list(map(activations_get, activation))
        else:
            self.activation = activation_dict_torch(activation)
           
        self.subnet_1 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_2 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_3 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_4 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_5 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_6 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.sub_omegas = sub_omegas

            
    def forward(self, x):

        y1 = self.subnet_1(self.sub_omegas[0]* x)

        y2 = self.subnet_2(self.sub_omegas[1]*x)

        y3 = self.subnet_3(self.sub_omegas[2]*x)

        y4 = self.subnet_4(self.sub_omegas[3]*x)

        y5 = self.subnet_5(self.sub_omegas[4]*x)
 
        y6 = self.subnet_6(self.sub_omegas[5]*x)

        
        out =  self.sub_omegas[0] * y1 +  self.sub_omegas[1] * y2 +  self.sub_omegas[2] * y3 +  self.sub_omegas[3] * y4 +  self.sub_omegas[4] * y5 +  self.sub_omegas[5] * y6
        # out =  1/self.sub_omegas[0] * y1 +  1/self.sub_omegas[1] * y2 +  1/self.sub_omegas[2] * y3 +  1/self.sub_omegas[3] * y4 +  1/self.sub_omegas[4] * y5 +  1/self.sub_omegas[5] * y6
        # out = 1/64 * out
        out = 6/(self.sub_omegas[0]+self.sub_omegas[1]+self.sub_omegas[2]+self.sub_omegas[3]+self.sub_omegas[4]+self.sub_omegas[5]) * out
        # out = 1/32 * out

        return out

class MscaleNN2_multi_alphai_ci(nn.Module):

    def __init__(self, sub_layer_sizes, activation, kernel_initializer, sub_omegas):

        super().__init__()

        if isinstance(activation, list):
            self.activation = list(map(activations_get, activation))
        else:
            self.activation = activation_dict_torch(activation)
           
        self.subnet_1 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_2 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_3 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_4 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_5 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.subnet_6 = FNN(layer_sizes =sub_layer_sizes,activation=activation,kernel_initializer=kernel_initializer)

        self.sub_omegas = sub_omegas

        # self.c_s = torch.nn.parameter.Parameter(1/64*torch.ones(6))
        # self.c_s = torch.nn.parameter.Parameter(1/torch.sqrt(150)*torch.ones(6))
        self.c_s = torch.nn.parameter.Parameter(6/(self.sub_omegas[0]+self.sub_omegas[1]+self.sub_omegas[2]+self.sub_omegas[3]+self.sub_omegas[4]+self.sub_omegas[5])*torch.ones(6))
            
    def forward(self, x):

        y1 = self.subnet_1(self.sub_omegas[0]* x)

        y2 = self.subnet_2(self.sub_omegas[1]*x)

        y3 = self.subnet_3(self.sub_omegas[2]*x)

        y4 = self.subnet_4(self.sub_omegas[3]*x)

        y5 = self.subnet_5(self.sub_omegas[4]*x)
 
        y6 = self.subnet_6(self.sub_omegas[5]*x)

        out = self.c_s[0] * self.sub_omegas[0] * y1 + self.c_s[1] * self.sub_omegas[1] * y2 + self.c_s[2] * self.sub_omegas[2] * y3 + self.c_s[3] * self.sub_omegas[3] * y4 + self.c_s[4] * self.sub_omegas[4] * y5 + self.c_s[5] * self.sub_omegas[5] * y6
        
        

        return out