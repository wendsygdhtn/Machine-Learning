import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 0. 准备数据 (从 Task 1 的 make_moons 取极小 batch)
# ==========================================
from sklearn.datasets import make_moons
X_np, y_np = make_moons(n_samples=3, noise=0.1, random_state=42)
X_np = X_np.astype(np.float32)          # X二维数组形状 (3, 2)
y_np = y_np.astype(np.float32).reshape(-1, 1)  # Y二维数组形状 (3, 1)
#每个样本有2个特征
batch_size = X_np.shape[0]  # 3
in_dim = 2
hidden_dim = 4              # 设定隐藏层神经元为 4
out_dim = 1

# 固定随机种子，保证 Numpy 和 PyTorch 初始化一致
np.random.seed(42)
torch.manual_seed(42)

# 初始化参数 (使用标准的 He 初始化，方便 PyTorch 复制)
W1_np = np.random.randn(in_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
b1_np = np.zeros((1, hidden_dim), dtype=np.float32)
W2_np = np.random.randn(hidden_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
b2_np = np.zeros((1, out_dim), dtype=np.float32)

print("=== 参数形状标注 ===")
print(f"X: {X_np.shape}, y: {y_np.shape}")
print(f"W1: {W1_np.shape}, b1: {b1_np.shape}")
print(f"W2: {W2_np.shape}, b2: {b2_np.shape}\\n")

# ==========================================
# 1 & 2. 使用 NumPy 完成前向计算
# ==========================================
print("=== Numpy 前向传播 ===")
# 第一层：Linear
Z1 = np.dot(X_np, W1_np) + b1_np            # 形状: (3, 4)
# 激活函数：ReLU
A1 = np.maximum(0, Z1)                      # 形状: (3, 4)
# 第二层：Linear
Z2 = np.dot(A1, W2_np) + b2_np              # 形状: (3, 1)
# 激活函数：Sigmoid
Y_hat = 1.0 / (1.0 + np.exp(-Z2))           # 形状: (3, 1)

# 二元交叉熵损失 (Binary Cross Entropy)
# 加 1e-8 防止 log(0) 报错
loss_np = -np.mean(y_np * np.log(Y_hat + 1e-8) + (1 - y_np) * np.log(1 - Y_hat + 1e-8))
print(f"NumPy Loss: {loss_np:.6f}")

# ==========================================
# 3. 使用 NumPy 按照链式法则计算梯度
# ==========================================
print("\\n=== Numpy 反向传播（手动推导） ===")
# 1. Loss 对 Z2 的梯度 (BCE+Sigmoid 的神奇化简)
dZ2 = (Y_hat - y_np) / batch_size           # 形状: (3, 1)

# 2. Loss 对 W2, b2 的梯度
dW2 = np.dot(A1.T, dZ2)                     # 形状: (4, 3) @ (3, 1) -> (4, 1)
db2 = np.sum(dZ2, axis=0, keepdims=True)    # 形状: (1, 1)

# 3. Loss 对 A1 的梯度
dA1 = np.dot(dZ2, W2_np.T)                  # 形状: (3, 1) @ (1, 4) -> (3, 4)

# 4. Loss 对 Z1 的梯度 (ReLU 的导数：大于0为1，小于等于0为0)
dZ1 = dA1 * (Z1 > 0).astype(np.float32)     # 形状: (3, 4)

# 5. Loss 对 W1, b1 的梯度
dW1 = np.dot(X_np.T, dZ1)                   # 形状: (2, 3) @ (3, 4) -> (2, 4)
db1 = np.sum(dZ1, axis=0, keepdims=True)    # 形状: (1, 4)


# ==========================================
# 4. 使用 PyTorch 计算前向和反向
# ==========================================
print("\\n=== PyTorch 前向与自动求导 ===")
# 将 Numpy 参数装入 PyTorch 的 Tensor，并开启 requires_grad
# 将 Numpy 参数装入 PyTorch 的 Tensor，并开启 requires_grad
X_pt = torch.tensor(X_np, dtype=torch.float32, requires_grad=False)
y_pt = torch.tensor(y_np, dtype=torch.float32, requires_grad=False)
W1_pt = torch.tensor(W1_np, dtype=torch.float32, requires_grad=True)  # <--- 加 dtype
b1_pt = torch.tensor(b1_np, dtype=torch.float32, requires_grad=True)  # <--- 加 dtype
W2_pt = torch.tensor(W2_np, dtype=torch.float32, requires_grad=True)  # <--- 加 dtype
b2_pt = torch.tensor(b2_np, dtype=torch.float32, requires_grad=True)  # <--- 加 dtype

# 前向计算
Z1_pt = torch.matmul(X_pt, W1_pt) + b1_pt
A1_pt = torch.relu(Z1_pt)
Z2_pt = torch.matmul(A1_pt, W2_pt) + b2_pt
Y_hat_pt = torch.sigmoid(Z2_pt)

loss_pt = nn.BCELoss()(Y_hat_pt, y_pt)
print(f"PyTorch Loss: {loss_pt.item():.6f}")

# 反向传播
loss_pt.backward()


# ==========================================
# 5. 逐项比较与误差验证
# ==========================================
print("\\n=== 梯度对比结果 ===")
def compare(name, np_grad, pt_grad):
    pt_grad_np = pt_grad.detach().numpy()
    max_err = np.max(np.abs(np_grad - pt_grad_np))
    print(f"[{name}] 形状: {np_grad.shape}")
    print(f"  Numpy 结果:\\n{np_grad}")
    print(f"  Torch 结果:\\n{pt_grad_np}")
    print(f"  最大绝对误差: {max_err:.2e}")
    print(f"  结论: {'一致 ✅' if max_err < 1e-6 else '不一致 ❌'}\\n")

compare("dW1", dW1, W1_pt.grad)
compare("db1", db1, b1_pt.grad)
compare("dW2", dW2, W2_pt.grad)
compare("db2", db2, b2_pt.grad)
