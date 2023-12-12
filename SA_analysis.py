import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from src_lam.Agent import Expr_Agent
import argparse
import numpy as np
from captum.attr import LayerConductance
# 读取数据
parser = argparse.ArgumentParser(description="Pytorch")
args = parser.parse_args()
read_set_path = "Expr1d/Expr_1.xlsx"
expr = Expr_Agent(args=args,
                  Read_set_path=read_set_path,
                  Loss_Save_Path=read_set_path)
import matplotlib.pyplot as plt

state_dict =torch.load("/Users/liangaoming/Desktop/neural_study/lulu_ms/Result/Expr1_/mscalenn2.pth")

expr.model.load_state_dict(state_dict)
print(state_dict.keys())
#
expr.model.eval()
# 这里的键名应与模型的状态字典中的键名匹配
# 定位到您关注的层
model = expr.model
input_tensor= torch.tensor([[1]])

# 存储每个 scale 的贡献
scales_contribution = []

for i, scale in enumerate(model.Multi_scale):
    layer_conductance = LayerConductance(model, scale)
    cond = layer_conductance.attribute(input_tensor, target=0)

    # 我们获取每个神经元平均贡献的绝对值，确保它是一个标量
    scales_contribution.append(cond.abs().item())

# 横轴 - 'scale' 的索引
scales = np.arange(len(scales_contribution))

# 创建柱状图
plt.bar(scales, scales_contribution, color='skyblue')

# 添加标题和轴标签
plt.title('Contribution per Scale')
plt.xlabel('Scale Index')
plt.ylabel('Contribution')

# 显示横轴标签
plt.xticks(scales)

# 显示图形
plt.show()