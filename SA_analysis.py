import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from captum.attr import LayerConductance
from src_lam.Agent import Expr_Agent
import argparse

class ModelAnalyzer:
    def __init__(self, model_path, read_set_path):
        self.model_path = model_path
        self.read_set_path = read_set_path
        self.model,self.scale_coeffs = self._load_model_scale_set()
        self.contributions= None
    def _load_model_scale_set(self):
        # 配置解析器和参数
        parser = argparse.ArgumentParser(description="Pytorch")
        args = parser.parse_args()
        # 创建模型实例
        expr = Expr_Agent(args=args,
                          Read_set_path=self.read_set_path,
                          Loss_Save_Path=self.read_set_path)
        # 加载模型状态
        state_dict = torch.load(self.model_path)
        expr.model.load_state_dict(state_dict)
        expr.model.eval()
        print("实验的sacles",expr.args.Scale_Coeff)
        return expr.model, expr.args.Scale_Coeff

    def _analyze_scales(self, input_tensor= torch.tensor([1]),
                       baseline=-1,
                       n_steps=1000,
                       target=0):
        scales_contribution = []
        for i, scale in enumerate(self.model.Multi_scale):
            layer_conductance = LayerConductance(self.model, scale)
            cond = layer_conductance.attribute(input_tensor, baselines=baseline, n_steps=n_steps, target=target)
            scales_contribution.append(cond.item())
        return scales_contribution

    def plot_contributions(self):

        self.contributions=self._analyze_scales()
        # 归一化 scale_coeffs 以便用于颜色映射
        norm = Normalize(vmin=min(self.scale_coeffs),
                         vmax=max(self.scale_coeffs))
        normed_coeffs = norm(self.scale_coeffs)
        fig, ax = plt.subplots()
        cmap = plt.cm.Greens
        for i, (contrib, coeff_norm) in enumerate(zip(self.contributions, normed_coeffs)):
            plt.bar(i, contrib, color=cmap(coeff_norm), label=f'Scale {i}: Coeff {self.scale_coeffs[i]}')
            plt.text(i, contrib, f'{self.scale_coeffs[i]}', ha='center', va='bottom')

        # 添加颜色条
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Scale-Coefficients')

        # 设置图表标题和轴标签
        plt.title('Contribution per Scale')
        plt.xlabel('Scale Index')
        plt.ylabel('Contribution')
        plt.grid(True)
        plt.legend(loc='best')
        # 返回图形对象
        return fig


if __name__== "__main__":
    # 使用示例
    model_path = "/Users/liangaoming/Desktop/neural_study/lulu_ms/Result/Expr2_1/mscalenn2.pth"
    read_set_path = "Expr2d/Expr_1.xlsx"

    analyzer = ModelAnalyzer(model_path, read_set_path)

    fig=analyzer.plot_contributions()
    plt.show()
