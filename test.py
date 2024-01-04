import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

# 1. 生成模拟数据
torch.manual_seed(1)
n_samples = 100
X = torch.randn(n_samples, 1)
true_coeff = torch.tensor([3.0])
intercept = torch.tensor([1.0])
noise = torch.randn(n_samples) * 0.5
y = intercept + torch.matmul(X, true_coeff) + noise


# 2. 定义神经网络模型
class RegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


neural_net = RegressionModel(1)


# 3. 定义Pyro模型和指南
def model(X, y=None):
    # 将神经网络的参数作为先验
    priors = {}
    for name, param in neural_net.named_parameters():
        priors[name] = dist.Normal(torch.zeros_like(param), torch.ones_like(param)).to_event(1)
    lifted_module = pyro.random_module("module", neural_net, priors)

    # 采样一个神经网络实例
    lifted_reg_model = lifted_module()

    with pyro.plate("data", X.size(0)):
        # 模型输出
        prediction_mean = lifted_reg_model(X).squeeze(-1)

        # 仅当提供了 'y' 时，才包含观测数据的采样
        if y is not None:
            pyro.sample("obs", dist.Normal(prediction_mean, 0.1), obs=y)
        else:
            return prediction_mean


guide = pyro.infer.autoguide.AutoDiagonalNormal(model)

# 4. 使用SVI进行模型训练
optimizer = Adam({"lr": 0.001})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

losses = []
num_iterations = 10000
for step in range(num_iterations):
    loss = svi.step(X, y)
    losses.append(loss)
    if step % 500 == 0:
        print(f"Step {step} : loss = {loss / len(X)}")

# 5. 使用训练好的模型进行预测
# 使用训练好的模型进行预测
predictive = pyro.infer.Predictive(model, guide=guide, num_samples=1000)
samples = predictive(X)
w_samples = samples["module$$$linear.weight"]
b_samples = samples["module$$$linear.bias"]

# 确保 w 是一维张量
w_mean = w_samples.mean(0).squeeze().unsqueeze(-1)
b_mean = b_samples.mean(0).squeeze()

preds = torch.matmul(X, w_mean) + b_mean

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.scatter(X.numpy(), y.numpy())
plt.plot(X.numpy(), preds.detach().numpy(), color='red')
plt.title("Model Prediction vs Actual Data")
plt.xlabel("X")
plt.ylabel("Y")
# 6
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(w_samples.numpy().flatten(), bins=50, color='orange', alpha=0.7)
plt.title("Posterior Distribution of Weight")
plt.xlabel("Weight")
plt.ylabel("Frequency")

# 偏差的后验分布
plt.subplot(1, 2, 2)
plt.hist(b_samples.numpy().flatten(), bins=50, color='green', alpha=0.7)
plt.title("Posterior Distribution of Bias")
plt.xlabel("Bias")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

means = preds.detach().numpy()
stds = samples["obs"].std(0).detach().numpy()



