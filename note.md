task1:代码，运行指令和输出

1.

代码  
print("Hello,machine learning!")
运行指令  
PS C:\WINDOWS\System32> cd D:/python练习
输出  
PS D:\python练习> python hello_ml.py

Hello,machine learning!

2.

代码  
scores = {
    "张三": 88,
    "李四": 95,
    "王五": 72,
    "赵六": 60,
    "钱七": 45,
    "孙八": 100,
    "周九": 83,
    "吴十": 59,
}
def max_scores(scores_dict):
    scores=scores_dict.values()
    print("最高分为%d"%max(scores))

def average_scores(scores_dict):
    scores=scores_dict.values()
    length=len(scores)
    total=sum(scores)
    ave=total/length
    print("平均分为%f"%ave)
max_scores(scores)
average_scores(scores)













task2:








task3:

1.不同项目往往依赖不同版本的库，甚至需要不同的 Python 版本。如果所有项目共用一个环境，安装或升级某个包很容易引发版本冲突，导致其他项目突然无法运行。虚拟环境让每个项目拥有独立隔离的依赖集合，互不干扰，还能通过导出依赖清单方便复现，删除环境也不会污染系统全局配置。

2.CPU擅长复杂的逻辑判断、串行任务和通用控制流程，适合处理操作系统调度、数据预处理等低延迟的顺序计算。而GPU但擅长大规模同质化的并行计算，对海量数据执行相同操作。
训练神经网络用GPU是因为训练的本质是海量的矩阵，张量乘法和卷积运算，同样的运算要在数百万个参数和样本上反复执行，天然高度并行。GPU 可以一次并行计算数千个这样的乘加操作，速度通常是 CPU 的几十到上百倍。

3.CUDA 是 NVIDIA 的 GPU 并行计算平台，让程序能用 GPU 做通用计算。关系是层层依赖：显卡驱动连接操作系统与GPU硬件，且决定能支持的最高 CUDA 版本；PyTorch 编译时绑定某个 CUDA 版本并自带运行时库，只需驱动够新即可，无需单独装完整 CUDA Toolkit。





