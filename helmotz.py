import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special
from torch.utils.data import Dataset

def get_mgrid(sidelen, dim=2):
    # 给定一个方向的网格点数量，生成给定维度dim的[-1,1]^dim上的等距网格坐标tensor，形如(1,sidelen,sidelen,dim)
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords
def lin2img(tensor, image_resolution=None):
    # 给定一个形如(batch_size,num_samples,channels)的点云函数，将其转换为网格函数表示(batchsize,channels,height=sqrt(num_samples),width =sqrt(num_samples) )
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]
    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    # x形如[...,2]为计算点位置,mu为均值位置，sigma为标准差，d为函数的维度.
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()
    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()
class SingleHelmholtzSource(Dataset):
    def __init__(self, sidelength, velocity='uniform', source_coords=[0., 0.]):
        super().__init__()
        torch.manual_seed(0)
        # sidelength: 单方向网格点数量
        # velocity: 波速(正问题中已知)
        # wavenumber:波数
        # N_src_samples:在点源
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.wavenumber = 20.
        self.N_src_samples = 100
        self.sigma = 1e-4
        self.source = torch.Tensor([1.0, 1.0]).view(-1, 2)  # (1,2) [[1.0,  1.0]]
        self.source_coords = torch.tensor(source_coords).view(-1, 2)  # (1,2) [[-0.3500,  0.0000]]
        square_meshgrid = lin2img(self.mgrid[None, ...]).numpy()  # 求解区域的网格化坐标(1,2,sidelength,sidelength)
        x = square_meshgrid[0, 0, ...]  # 网格化坐标的X (sidelength,sidelength)
        y = square_meshgrid[0, 1, ...]  # 网格化坐标的Y (side_length,side_length)

        # 下面计算解析解
        source_np = self.source.numpy()  # (1,2)
        hx = hy = 2 / self.sidelength
        field = np.zeros((sidelength, sidelength)).astype(np.complex64)
        for i in range(source_np.shape[0]):
            x0 = self.source_coords[i, 0].numpy()
            y0 = self.source_coords[i, 1].numpy()
            # 源项强度为复数
            s = source_np[i, 0] + 1j * source_np[i, 1]
            hankel = scipy.special.hankel2(0, self.wavenumber * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6)
            field += 0.25j * hankel * s * hx * hy
        # 方程解析解实部与虚部:(sidelength*side_length,2)
        field_r = torch.from_numpy(np.real(field).reshape(-1, 1))
        field_i = torch.from_numpy(np.imag(field).reshape(-1, 1))
        self.field = torch.cat((field_r, field_i), dim=1)

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        # 计算慢度平方，输入为点云坐标[...,2]，输出为对应的慢度平方(复数) [...,2]，这里取为均质材料，即为常数1
        squared_slowness = torch.ones_like(coords)
        squared_slowness[..., 1] = 0.
        return squared_slowness

    def __getitem__(self, idx):
        # dataloader调用的数据迭代产生器，输出包含两个dict
        # 第一个为均匀采样自(-1,1)^2的点云坐标，并将其中一部分点(N_src)替换为源项附近的点，形为(sidelength^2,2)
        # 第二个包含点云坐标形式的 
        # 1.源项函数:boundary_values; 2.解析解:gt; 3.慢度平方:squared_slowness;
        # 4.慢度平方网格表示:squared_slowness_grid 形为(1,self.sidelength,self.sidelength,1);5.波数:wavenumbeer;

        # 采样配点坐标
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1.,
                                                               1.)  # 从-1,1采样uniform distribution， shape[self.sidelength*self.sidelength,2]
        source_coords_r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        source_coords_theta = 2 * np.pi * torch.rand(self.N_src_samples, 1)
        source_coords_x = source_coords_r * torch.cos(source_coords_theta) + self.source_coords[0, 0]
        source_coords_y = source_coords_r * torch.sin(source_coords_theta) + self.source_coords[0, 1]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)

        # 将采样的coords的一部分替换为source周围的点
        coords[-self.N_src_samples:, :] = source_coords

        # 用Gaussian逼近dirac源项
        boundary_values = self.source * gaussian(coords, mu=self.source_coords, sigma=self.sigma)[:, None]
        boundary_values[boundary_values < 1e-5] = 0.

        # 计算采样点的square_slowness
        squared_slowness = self.get_squared_slowness(coords)
        squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, 0, None]
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'gt': self.field,
                                    'squared_slowness': squared_slowness,
                                    'squared_slowness_grid': squared_slowness_grid,
                                    'wavenumber': self.wavenumber}
