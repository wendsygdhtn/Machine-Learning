# 实践 2：神经网络的梯度
**从 Task 1 的 make_moons 数据中取出一个样本或一个很小的 batch，构造下面的两层二分类网络（参数任意）：X → Linear → ReLU → Linear → Sigmoid.需要完成一次前向传播和反向传播，不要求更新参数或训练模型：**

**1.自己确定隐藏层神经元数量，并初始化 W1,b1,W2,b2.**

根据输出，我设置隐藏层神经元数量为 4，输入特征为 2。下面是初始化代码：

```python
#每个样本有2个特征
batch_size = X_np.shape[0]  # 3
in_dim = 2
hidden_dim = 4              # 设定隐藏层神经元为 4
out_dim = 1
# 初始化参数 (使用标准的 He 初始化，方便 PyTorch 复制)
W1_np = np.random.randn(in_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
b1_np = np.zeros((1, hidden_dim), dtype=np.float32)
W2_np = np.random.randn(hidden_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
b2_np = np.zeros((1, out_dim), dtype=np.float32)
```

**2.使用 NumPy 完成线性层、ReLU、Sigmoid 和二元交叉熵损失的前向计算；**

前向计算代码：
```python
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
```
其中进行的计算是：Z1=X@W1+B1 , A1=ReLU(Z1) , Z2=A1@W2+b2

**3.画出从输入到 loss 的计算图，按照链式法则从后向前计算每个参数的梯度；**



**4.使用相同的输入和参数在 PyTorch 中重新完成一次前向传播，调用 autograd 得到梯度；**



**5.将 NumPy 手算梯度与 PyTorch 自动计算的梯度逐项比较，输出它们的形状、数值和最大绝对误差，并判断结果是否一致。**



**(请在代码注释或笔记中标出 X,W1,b1,W2,b2,各层中间结果及其梯度的形状。为了降低难度，可以把隐藏层设置得很小，并先只使用一个样本。验证成功后，再尝试换成一个小 batch，观察哪些公式和形状发生了变化。)**







# 趁热打铁 
**1.前向传播时为什么要保存每一层的输入或输出？反向传播会在哪里用到它们？这和显存有什么关系？**

**2.链式法则怎样把损失函数的梯度传回第一层权重？请结合自己的计算图说明。**

**3.二元交叉熵与 Sigmoid 配合后，loss 对 logit 的梯度是什么？这个结果为什么方便反向传播？**

**4.NumPy 梯度与 PyTorch 梯度出现差异时，你会怎样判断问题出在公式、矩阵转置、广播还是损失的求平均方式？**

**5.完成两次验证后，你现在怎样理解 loss.backward()？它替我们保存和完成了哪些工作？**


