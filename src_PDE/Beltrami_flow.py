"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import torch
import scipy.io
from scipy.interpolate import griddata


class Beltrami_flow():

    def __init__(self):
        self.a = 1
        self.d = 1
        self.Re = 1

    def pde(self,x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )
        continuity = u_vel_x + v_vel_y + w_vel_z

        return [momentum_x, momentum_y, momentum_z, continuity]


    def u_func(self,x):
        return (
            -a
            * (
                np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def v_func(self,x):
        return (
            -a
            * (
                np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def w_func(self,x):
        return (
            -a
            * (
                np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def p_func(self,x):
        return (
            -0.5
            * a ** 2
            * (
                np.exp(2 * a * x[:, 0:1])
                + np.exp(2 * a * x[:, 1:2])
                + np.exp(2 * a * x[:, 2:3])
                + 2
                * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
                * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
                + 2
                * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
                * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
                + 2
                * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
                * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
            )
            * np.exp(-2 * d ** 2 * x[:, 3:4])
        )

    def Get_Data(self):

        spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
        temporal_domain = dde.geometry.TimeDomain(0, 10)
        spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

        boundary_condition_u = dde.icbc.DirichletBC(
            spatio_temporal_domain, self.u_func, lambda _, on_boundary: on_boundary, component=0
        )
        boundary_condition_v = dde.icbc.DirichletBC(
            spatio_temporal_domain,  self.v_func, lambda _, on_boundary: on_boundary, component=1
        )
        boundary_condition_w = dde.icbc.DirichletBC(
            spatio_temporal_domain,  self.w_func, lambda _, on_boundary: on_boundary, component=2
        )

        initial_condition_u = dde.icbc.IC(
            spatio_temporal_domain,  self.u_func, lambda _, on_initial: on_initial, component=0
        )
        initial_condition_v = dde.icbc.IC(
            spatio_temporal_domain,  self.v_func, lambda _, on_initial: on_initial, component=1
        )
        initial_condition_w = dde.icbc.IC(
            spatio_temporal_domain,  self.w_func, lambda _, on_initial: on_initial, component=2
        )

        data = dde.data.TimePDE(
            spatio_temporal_domain,
            self.pde,
            [
                boundary_condition_u,
                boundary_condition_v,
                boundary_condition_w,
                initial_condition_u,
                initial_condition_v,
                initial_condition_w,
            ],
            num_domain=50000,
            num_boundary=5000,
            num_initial=5000,
            num_test=5000,
        )

        return data


    def Save_4mat(self):

        self.data=self.Get_Data()

        x, y, z = np.meshgrid(
            np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
        )

        X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

        t_0 = np.zeros(1000).reshape(1000, 1)
        t_1 = np.ones(1000).reshape(1000, 1)

        # 提取 u, v, w, p
        physical_quantities = vars(data)["train_x_all"]
        x, y, z, t = physical_quantities[:, 0], physical_quantities[:, 1], physical_quantities[:, 2], physical_quantities[:, 3]
        u=u_func(physical_quantities)
        w=w_func(physical_quantities)
        v=v_func(physical_quantities)
        p=p_func(physical_quantities)
        # 创建一个 (10, 10, 10) 的网格坐标
        xi, yi, zi = np.meshgrid(np.linspace(min(x), max(x), 10),
                                 np.linspace(min(y), max(y), 10),
                                 np.linspace(min(z), max(z), 10))

        # 将一维数据映射到网格上
        points = np.column_stack((x, y, z))  # 一维坐标数据
        u_mapped = griddata(points, u, (xi, yi, zi), method='nearest')
        v_mapped = griddata(points, v, (xi, yi, zi), method='nearest')
        w_mapped = griddata(points, w, (xi, yi, zi), method='nearest')
        p_mapped = griddata(points, p, (xi, yi, zi), method='nearest')

        # 移除最后的维度
        u_sliced = u_mapped[:, :, :, 0]
        v_sliced = v_mapped[:, :, :, 0]
        w_sliced = w_mapped[:, :, :, 0]
        p_sliced = p_mapped[:, :, :, 0]
        print(u_sliced)
        # Prepare data for saving
        data_to_save = {
            "x": xi,
            "y": yi,
            "z": zi,
            "t": t,
            "u": u_sliced,
            "w": v_sliced,
            "v": w_sliced,
            "p": p_sliced
        }
        # Save data to .mat file
        mat_file_path = 'physical_quantities_data.mat'
        scipy.io.savemat(mat_file_path, data_to_save)


b_flow=Beltrami_flow()

data=b_flow.Get_Data()






