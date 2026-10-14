#task1代码
"""
整体步骤：
1.导入库
2.固定随机种子
3.生成并可视化原始数据
4.进行训练/验证/测试三集的划分
5.搭建MLP神经网络
6.构建训练，验证函数
7.构建预测和绘制函数，画loss/accuracy曲线和二维决策边界
8.构建主程序，训练/验证/测试模型，保存和加载模型
9.输出混淆矩阵，并分析部分错误样本
10.改变因素进行对照试验
"""
#1.导入库
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#2.固定随机种子（42）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
#3.生成并可视化原始数据
#3.1生成数据
X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)#X是特征矩阵，有1000个特征向量（1000*2），y是标签矩阵（1000*1）
X = X.astype(np.float32)   #排查出的错误：特征转 float32
y = y.astype(np.int64)     
#3.2可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')#c:用y值映射颜色，cmap：值色映射表
plt.show()
#4用train_test_split函数划分训练/验证/测试集合，比例为7：1.5：1.5
#4.1分出训练集
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
#4.2分出验证/测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
"""
职责：
训练集：训练模型参数
验证集：调超参数、早停、模型选择，间接影响模型但不参与梯度更新
测试集：最终评估，只能用一次，避免数据泄漏
"""
#5.搭建MLP
ACTS = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
#定义类，用relu作为激活函数A
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, n_layers=1, out_dim=2, act='relu'):#参数：输入维度2，单层宽32，隐藏层1，输出维度2，激活函数relu
        super().__init__()
        A, layers, d = ACTS[act], [], in_dim        #一会将数据放入layer列表
        for _ in range(n_layers):                    # 堆叠1个隐藏层
            layers += [nn.Linear(d, hidden), A()]   #线性变换接激活函数
            d = hidden                           #更新输入维度
        layers.append(nn.Linear(d, out_dim))         # 输出层，输出2维logits
        self.net = nn.Sequential(*layers)           #将layer打包为nn.sequential，其会自动向前传播
    def forward(self, x):                            #定义数据流
        return self.net(x)
#6.构建训练/验证模型函数
#6.1训练
def run_experiment(cfg, X_train, y_train, X_val, y_val, verbose=False):
    torch.manual_seed(SEED)               # 每个实验都从同一初始化出发，保证公平
    model = MLP(hidden=cfg['hidden'], n_layers=cfg['n_layers'], act=cfg['act']).to(DEVICE)     #按传入的cfg参数创建MLP
    n_params = sum(p.numel() for p in model.parameters())
    opt = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[cfg['optimizer']](
        model.parameters(), lr=cfg['lr'])
    loss_fn = nn.CrossEntropyLoss()
 
    Xtr = torch.from_numpy(X_train).to(DEVICE)
    ytr = torch.from_numpy(y_train).to(DEVICE)
    Xv  = torch.from_numpy(X_val).to(DEVICE)
    yv  = torch.from_numpy(y_val).to(DEVICE)
 
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}   #保存每轮训练数据，用于绘制loss/accuracy曲线
    best_val, best_state = float('inf'), None
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
 
    t0, n = time.time(), len(Xtr)
    for ep in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_loss, correct = 0.0, 0                                #进入训练，记录错误和正确数
        for i in range(0, n, cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            xb, yb = Xtr[idx], ytr[idx]                          #按batch_size取数据。
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_loss, tr_acc = ep_loss / n, correct / n        #计算正确/错误率
#6.2验证
        model.eval()
        with torch.no_grad():                       # 验证集：只前向、不更新参数防泄漏
            vlogits = model(Xv)                         #记录模型给出的预测分数
            v_loss = loss_fn(vlogits, yv).item()
            v_acc  = (vlogits.argmax(1) == yv).float().mean().item()
        hist['train_loss'].append(tr_loss); hist['train_acc'].append(tr_acc)
        hist['val_loss'].append(v_loss);   hist['val_acc'].append(v_acc)
 
        if v_loss < best_val:                       # 按验证 loss 保存最优权重模型
            best_val, best_state = v_loss, copy.deepcopy(model.state_dict())
        if verbose and (ep + 1) % 50 == 0:
            print(f'  ep{ep+1:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}'
                  f' | val loss {v_loss:.4f} acc {v_acc:.4f}')
 
    info = {'time': time.time() - t0, 'n_params': n_params,                 #恢复最佳模型，统计信息并返回
            'peak_mem_MB': (torch.cuda.max_memory_allocated()/1024**2)
                           if DEVICE.type == 'cuda' else float('nan')}
    model.load_state_dict(best_state)
    return model, hist, info
#7.构建预测和绘制函数
#7.1预测类别函数
@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X.astype(np.float32)).to(DEVICE)).argmax(1).cpu().numpy()
#7.2预测类别概率函数
@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    p = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE))
    return torch.softmax(p, dim=1)[:, 1].cpu().numpy()      #输出类别为1的概率
#7.3分别绘制loss/accuracy曲线
def plot_curves(hist, title, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist['train_loss'], label='train'); ax[0].plot(hist['val_loss'], label='val')
    ax[0].set_title('Loss');  ax[0].set_xlabel('epoch'); ax[0].legend()
    ax[1].plot(hist['train_acc'], label='train'); ax[1].plot(hist['val_acc'], label='val')
    ax[1].set_title('Accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend()
    plt.suptitle(title)
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.show()
#7.4绘制决策边界，并标出预测错误的样本
def plot_boundary(model, Xs, ys, title, fname, wrong_idx=None, resolution=300):
    x0, x1 = Xs[:, 0].min()-.5, Xs[:, 0].max()+.5
    y0, y1 = Xs[:, 1].min()-.5, Xs[:, 1].max()+.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    zz = predict_proba(model, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=20, cmap='coolwarm', alpha=0.85)
    plt.colorbar(label='P(class=1)')
    plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap='coolwarm', s=15,
                edgecolors='k', linewidths=0.3)
    if wrong_idx is not None and len(wrong_idx):    # 用黑圈标出判断错误的样本
        plt.scatter(Xs[wrong_idx, 0], Xs[wrong_idx, 1], s=120,
                    facecolors='none', edgecolors='black', linewidths=1.8,
                    label='misclassified')
        plt.legend()
    plt.title(title)
    plt.savefig(fname, dpi=150, bbox_inches='tight'); 
    plt.show()
#8.构建主程序，训练/验证/测试模型
#8.1定义基线配置
if __name__ == '__main__':
    BASE = dict(hidden=32, n_layers=1, act='relu', lr=0.01,         
                optimizer='adam', batch_size=64, epochs=200)
 
#8.2开始训练/验证，绘制曲线
    model, hist, info = run_experiment(BASE, X_train, y_train, X_val, y_val, verbose=True)
    print(f"[baseline] params={info['n_params']}  time={info['time']:.2f}s  "
          f"peak_mem={info['peak_mem_MB']:.1f}MB")
 
    plot_curves(hist, 'Baseline: h=32, ReLU, Adam(lr=0.01), bs=64', 'fig2_curves.png')
    plot_boundary(model, X_train, y_train,
                  'Baseline decision boundary (train set)', 'fig3_boundary.png')
 
#8.3保存和加载模型
    torch.save({'state_dict': model.state_dict(), 'cfg': BASE}, 'mlp_moons.pt')
    ckpt = torch.load('mlp_moons.pt', map_location=DEVICE, weights_only=True)
    loaded = MLP(hidden=ckpt['cfg']['hidden'], n_layers=ckpt['cfg']['n_layers'],
                 act=ckpt['cfg']['act']).to(DEVICE)
    loaded.load_state_dict(ckpt['state_dict'])
    print(f"加载后测试集 acc = {(predict(loaded, X_test) == y_test).mean():.4f}")  # 此处应与保存前一致
 
#8.4测试
    y_pred = predict(model, X_test)
    test_acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {test_acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')               #混淆矩阵的绘制
    plt.title('Confusion Matrix (test)')
    plt.savefig('fig4_confusion.png', dpi=150, bbox_inches='tight'); plt.show()
 
#9.错误样本分析
    wrong = np.where(y_pred != y_test)[0]
    proba = predict_proba(model, X_test)
    print(f'错误样本数: {len(wrong)} / {len(y_test)}')
    for i in wrong:
        print(f"  idx={i:3d}  X=({X_test[i,0]:+.2f},{X_test[i,1]:+.2f})  "
              f"true={y_test[i]}  pred={y_pred[i]}  P(class1)={proba[i]:.3f}")
    plot_boundary(model, X_test, y_test, 'Test set with misclassified points',
                  'fig5_errors.png', wrong_idx=wrong)
#10.每次改变单个变量，进行对照试验
    EXPERIMENTS = {
        'baseline h32/relu/adam/lr.01/bs64': BASE,
        'E1a hidden=4':                      {**BASE, 'hidden': 4},
        'E1b hidden=128':                    {**BASE, 'hidden': 128},
        'E2a act=tanh':                      {**BASE, 'act': 'tanh'},
        'E2b act=sigmoid':                   {**BASE, 'act': 'sigmoid'},
        'E3a lr=0.5':                        {**BASE, 'lr': 0.5},
        'E3b lr=0.001':                      {**BASE, 'lr': 0.001},
        'E4 optimizer=SGD':                  {**BASE, 'optimizer': 'sgd'},
        'E5a batch_size=8':                  {**BASE, 'batch_size': 8},
        'E5b batch_size=1024':               {**BASE, 'batch_size': 1024},
    }
    results = []
    for name, cfg in EXPERIMENTS.items():
        if name.startswith('baseline'):
            m, h, i2, acc = model, hist, info, test_acc   # 复用基线结果
        else:
            m, h, i2 = run_experiment(cfg, X_train, y_train, X_val, y_val)
            acc = (predict(m, X_test) == y_test).mean()
            plot_curves(h, name, f'curves_{name.replace(" ", "_").replace("/", "-")}.png')
            plot_boundary(m, X_train, y_train, f'{name} (boundary)', f'bd_{name[:12]}.png')
        results.append([name, i2['n_params'], round(i2['time'], 2),
                        f"{h['val_acc'][-1]:.4f}", f"{acc:.4f}", i2['peak_mem_MB']])
 
    print(f"\n{'experiment':<35}{'params':>8}{'time(s)':>9}{'val_acc':>10}{'test_acc':>10}{'mem(MB)':>10}")
    for r in results:
        print(f"{r[0]:<35}{r[1]:>8}{r[2]:>9}{r[3]:>10}{r[4]:>10}{r[5]:>10.1f}")