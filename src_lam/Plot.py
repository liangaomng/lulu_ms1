import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Analyzer import Analyzer4scale
import numpy as np

# 根据行列数自动创建子图
class Plot_Adaptive:
    def __init__(self):
        self.fig = None
        self.axes = None
    def _create_subplot_grid1(self,nrow, ncol):
        self.fig = plt.figure(figsize=(1.6*ncol * 3, 1.4*nrow * 3))
        gs = GridSpec(nrow, ncol, figure=self.fig,hspace=0.4,wspace=0.3)
        self.axes = []

        # 添加跨列的子图
        for r in range(nrow - 1):
            ax = self.fig.add_subplot(gs[r, :])
            self.axes.append(ax)

        # 添加最后一行的子图
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[-1, c])
            self.axes.append(ax)

    def _create_subplot_grid2(self,nrow, ncol):
        self.fig = plt.figure(figsize=(1.6 * ncol * 3, 1.4 * nrow * 3))
        gs = GridSpec(nrow, ncol, figure=self.fig, hspace=0.4, wspace=0.3)
        self.axes = []

        for c in range(ncol):
            ax = self.fig.add_subplot(gs[0, c])
            self.axes.append(ax)

        # 添加第2行的图
        for r in [1]:
            ax = self.fig.add_subplot(gs[r, :])
            self.axes.append(ax)

        # 添加最后一行的子图
        for c in range(ncol):
            ax = self.fig.add_subplot(gs[-1, c])
            self.axes.append(ax)
    def plot_1d(self,nrow,ncol,**kwagrs):
        # 绘制一些示例数据
        c_map=["Green","Blue","Purple"]
        # 从kwagrs中获取参数
        analyzer=kwagrs["analyzer"]
        if self.fig is None:
            self._create_subplot_grid1(nrow,ncol)
        Record=kwagrs["contribution_record"]

        for i, ax in enumerate(self.axes):

            if i==0: #   第一张图
                # 在第一个子图上绘制预测值的散点图
                ax.cla()
                x_test=kwagrs["x_test"]
                pred=kwagrs["pred"]
                y_true=kwagrs["y_true"]
                avg_test_loss=kwagrs["avg_test_loss"]
                epoch=kwagrs["epoch"]
                ax.scatter(x_test, pred, label="Pred", color="red")
                # 在第一个子图上绘制真实值的散点图
                ax.scatter(x_test, y_true, label="True", color="blue")
                # 设置第一个子图的图例、坐标轴标签和标题
                ax.legend(loc="best", fontsize=16)
                ax.set_xlabel('x', fontsize=16)
                ax.set_ylabel('y', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=16, width=2, colors='black')
                ax.set_title("Test_MSE={:.6f}_Epoch{}".format(avg_test_loss, epoch))
                ax.legend()
            if i==1: #   第二张图
                # 在第最后子图上绘制损失曲线
                ax.cla()
                loss_record_df=kwagrs["loss_record_df"]
                ax.plot(loss_record_df["epoch"], loss_record_df["train_loss"], label="Train Loss", color="blue")
                ax.plot(loss_record_df["epoch"], loss_record_df["valid_loss"], label="Valid Loss", color="red")
                ax.plot(loss_record_df["epoch"], loss_record_df["test_loss"], label="Test Loss", color="green")

                # 设置第二个子图的图例、坐标轴标签和标题
                ax.set_yscale('log')  # 将y轴设置为对数尺度
                ax.legend(loc="best", fontsize=16)
                ax.set_xlabel('Epoch', fontsize=16)
                ax.set_ylabel('Loss', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=16, width=2, colors='black')
                ax.set_title("Loss_Epoch{}".format(epoch))
                # # 画三条虚线
                for j,value in enumerate(Record):
                    ax.axvline(x=value, color=c_map[j], linestyle='--')
            if i==2: #   第三行图开始画贡献度
                if (epoch == Record[0]):
                    analyzer.plot_contributions(ax=self.axes[i],fig=self.fig,cmap=c_map[0])
            if i==3:
                if (epoch == Record[1]):
                    analyzer.plot_contributions(ax=self.axes[i],fig=self.fig,cmap=c_map[1])
            if i==4:
                if (epoch == Record[2]):
                    # for j in epoch_axv:
                    analyzer.plot_contributions(ax=self.axes[i],fig=self.fig,cmap=c_map[2])


        return self.fig,self.axes
    def plot_2d(self, nrow, ncol, **kwagrs):
        # 绘制一些示例数据
        c_map = ["Green", "Blue", "Purple"]
        # 从kwagrs中获取参数
        analyzer = kwagrs["analyzer"]
        #画图计算的solver
        solver=kwagrs["solver"]
        model=kwagrs["model"]
        if self.fig is None:
            self._create_subplot_grid2(nrow, ncol)
        Record = kwagrs["contribution_record"]

        for i, ax in enumerate(self.axes):

            if i == 2:  # 第一张图
                # 在第一个子图上绘制预测值的散点图
                avg_test_loss = kwagrs["avg_test_loss"]
                epoch = kwagrs["epoch"]
                # 在第一个子图上绘制真实值的散点图
                U_pred=solver.plot_2dfrom_model(ax=self.axes[0],
                                               model=model,
                                               title="Pred",
                                               cmap="bwr")

                # 在第一个子图上绘制真实值的散点图
                U_true=solver.plot_2dfrom_model(ax=self.axes[1],
                                               model=None,
                                               title="True",
                                               cmap="bwr")
                self.axes[2].imshow((np.abs(U_pred-U_true)),
                                    cmap="bwr",vmin=-1,vmax=1,origin="lower")
                # 隐藏 x 和 y 轴的刻度和标签
                self.axes[2].set_xticks([])
                self.axes[2].set_yticks([])
                # 设置第一个子图的图例、坐标轴标签和标题
                self.axes[2].set_title("Test_MSE={:.6f}_Epoch{}".format(avg_test_loss, epoch))


            if i == 3:  # 第二张图
                # 在第最后子图上绘制损失曲线
                ax.cla()
                loss_record_df = kwagrs["loss_record_df"]
                ax.plot(loss_record_df["epoch"], loss_record_df["train_loss"], label="Train Loss", color="blue")
                ax.plot(loss_record_df["epoch"], loss_record_df["valid_loss"], label="Valid Loss", color="red")
                ax.plot(loss_record_df["epoch"], loss_record_df["test_loss"], label="Test Loss", color="green")

                # 设置第二个子图的图例、坐标轴标签和标题
                ax.set_yscale('log')  # 将y轴设置为对数尺度
                ax.legend(loc="best", fontsize=16)
                ax.set_xlabel('Epoch', fontsize=16)
                ax.set_ylabel('Loss', fontsize=16)
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.tick_params(labelsize=16, width=2, colors='black')
                ax.set_title("Loss_Epoch{}".format(epoch))
                # # 画三条虚线
                for j, value in enumerate(Record):
                    ax.axvline(x=value, color=c_map[j], linestyle='--')
            if i == 4:  # 第三行图开始画贡献度
                if (epoch == Record[0]):
                    analyzer.plot_contributions(ax=self.axes[i], fig=self.fig, cmap=c_map[0])
            if i == 5:
                if (epoch == Record[1]):
                    analyzer.plot_contributions(ax=self.axes[i], fig=self.fig, cmap=c_map[1])
            if i == 6:
                if (epoch == Record[2]):
                    # for j in epoch_axv:
                    analyzer.plot_contributions(ax=self.axes[i], fig=self.fig, cmap=c_map[2])

        return self.fig, self.axes

    # 使用示例
if __name__ == '__main__':
    Plot_Adaptive1= Plot_Adaptive()
    fig, axes = Plot_Adaptive1.create_subplot_grid2(3, 3)  # 举例：3 行，4 列
    plt.show()
