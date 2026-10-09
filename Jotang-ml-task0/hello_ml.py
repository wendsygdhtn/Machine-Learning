#所有测试代码
print("Hello,machine learning!")

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

import numpy as np
A=np.array([[1,2],[3,4],[5,6]])
B=np.array([[5,6],[7,8]])
print(A @ B)

