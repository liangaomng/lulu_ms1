
import deepxde.geometry as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)
#https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from torch.autograd import grad
class  PDE_base():
    def __init__(self):
        pass
    def Get_Data(self):
        pass
    def torch_u(self,x,t):
        pass
    def pde(self,net,data):
        pass
    def train(self,net=None):
        pass
class PDE_HeatData(PDE_base):
    def __init__(self):
        self.a = 0.4
        self.L = 1
        self.n = 1

    def torch_u(self, x, t):
        # 确保 x 和 t 是 torch.Tensor 类型
        out= torch.exp(-(self.n ** 2 * np.pi ** 2 * self.a * t) / (self.L ** 2)) * torch.sin(
            self.n * np.pi * x / self.L)
        out=out.unsqueeze(1)
        return out
    def pde(self,net,data):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - a * dy_xx
    def pde_loss(self,net,data):
        #data:[batch,2]
        # 确保 data 的相关列设置了 requires_grad=True  对于data：第0维度是x，t是1维度
        u=net(data)  # 计算网络输出
        grad_outputs = torch.ones_like(u)  # 创建一个与u形状相同且元素为1的张量

        # 计算一阶导数
        du_data=grad(u,data,grad_outputs,create_graph=True)[0]
        du_dt=du_data[:,1]
        du_dx=du_data[:,0].unsqueeze(1)
        # 计算二阶导数
        ddu_ddata=grad(du_dx,data,grad_outputs, create_graph=True)[0]
        ddu_ddx=ddu_ddata[:,0]

        # 计算 PDE 残差
        pde_loss =  du_dt - self.a * ddu_ddx
        #mse
        pde_loss=torch.mean(torch.square(pde_loss))
        return pde_loss

    def bc_loss(self,net,data):
        #data:[batch,2]
        # self.data.train_x_all=None
        # self.data.train_x_bc=None
        # self.data.train_x = None
        # self.data.train_x,_,_=self.data.train_next_batch()
        # self.data.train_x_all=self.data.train_points()
        inputs=torch.from_numpy(data).float()
        #targets=self.torch_u(x=inputs[:,0],t=inputs[:,1])  # 创建一个与u形状相同且元素为1的张量(,1)
        outputs=net(inputs) # 计算网络输出

        #标记bc序列
        bcs_start = np.cumsum([0] + self.data.num_bcs)
        bcs_start = list(map(int, bcs_start))
        losses= []
        for i, bc in enumerate(self.data.bcs): #ic and bc
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.train_x有序
            error = bc.error(inputs,inputs,outputs, beg, end)
            #求mse
            error_scalar = torch.mean(torch.square(error))
            losses.append(error_scalar)
        # 将losses列表转换为Tensor
        losses_tensor = torch.stack(losses)
        bc_mse = torch.mean(losses_tensor)
        print("bcmse",bc_mse)

        return bc_mse

    def data_loss(self,net,data):
        #data:[batch,2]
        u = net()  # 计算网络输出
        # 计算 MSE 损失
        data_loss = nn.mse(u,self.torch_u())
        return data_loss

    def Get_Data(self)->dde.data.TimePDE: #u(x,t)
        geom = dde.geometry.Interval(0, self.L)
        timedomain = dde.geometry.TimeDomain(0, 2)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(
            geomtime,
            lambda x: np.sin(self.n * np.pi * x[:, 0:1] / self.L),
            lambda _, on_initial: on_initial,
        )
        self.data = dde.data.TimePDE(
            geomtime,
            self.pde,
            [bc, ic],
            num_domain=1,#sqrt(3600)=60
            num_boundary=1,
            num_initial=1,
            num_test=1,
            train_distribution="pseudo",
        )

        return self.data

    def heat_eq_exact_solution(self,x, t):
        """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        return np.exp(-(self.n ** 2 * np.pi ** 2 * self.a * t) / (self.L ** 2)) * np.sin(self.n * np.pi * x / self.L)

    def gen_exact_solution(self):
        """Generates exact solution for the heat equation for the given values of x and t."""
        # Number of points in each dimension:
        x_dim, t_dim = (256, 201)

        # Bounds of 'x' and 't':
        x_min, t_min = (0, 0.0)
        x_max, t_max = (self.L, 1.0)

        # Create tensors:
        t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
        x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
        usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(t_dim):
                usol[i][j] = self.heat_eq_exact_solution(x[i], t[j])

        return usol

    def plot(self):

        u=self.gen_exact_solution()
        print(u.shape)
        plt.imshow(u)

        plt.xlabel("t")
        plt.ylabel("x")

        plt.title("Exact $u(x,t)$")
        plt.show()

    def train(self,net=None):
        pass

        # optimizer= torch.optim.Adam(net.parameters(), lr=0.001)
        # criterion = torch.nn.MSELoss()
        # for epoch in range(0, 1000, 1):
        #
        #     epoch_loss = 0.0
        #     data=torch.from_numpy(vars(self.data)["train_x_all"])
        #     data.requires_grad_(True)
        #
        #     # 确保 data 的相关列设置了 requires_grad=Tru
        #     pde_loss=self.pde(net,data)
        #     print("pde_lss",pde_loss.shape)
        #     #pde_loss
        #     pde_loss= criterion(pde_loss,torch.zeros_like(pde_loss))
        #
        #     #data_loss
        #     pred = net(data)
        #     mse=criterion(pred,self.torch_u(data[:,0],data[:,1]))
        #     #boundary_loss
        #     train_boundary=torch.from_numpy(vars(self.data)["train_x_bc"])
        #
        #     boundary_loss=criterion(net(train_boundary),
        #                             self.torch_u(train_boundary[:,0],train_boundary[:,1]))
        #
        #
        #
        #
        #     optimizer.zero_grad()
        #     loss =  pde_loss+mse
        #     loss.backward()
        #
        #     optimizer.step()
        #     epoch_loss += loss.item()
        #
        #     print('epoch: {}, train loss: {:.9f}'.format(epoch, pde_loss))
        # return net


if __name__ == "__main__":
    heat=PDE_HeatData()
    data=heat.Get_Data()
    if (type(data)==dde.data.TimePDE):
        print(vars(data))
    data.train_points()











