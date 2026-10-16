# 本文件用来记录所有试验代码和结果

01.
```python
from PIL import Image
import numpy as np
import torch

img = Image.open("highway.png") 
print(img.size)   # (宽, 高)
print(img.mode)   # 通常是 'RGB'

#arr(x,y)=(R(x,y),G(x,y),B(x,y));arr(x,y,颜色)=(0~255（该颜色强度）)
arr = np.array(img)   #arr是二维数组，每个元素对应一个小方格，每个元素包含该格红/绿/蓝三色强度的三个数值     
print(arr.shape)  #arr.shape=(高/行数，宽/列数，颜色通道数（3）)
print(arr.dtype)   #每个数值占1字节，为0~255
print(arr.max(), arr.min())  #整张图最暗和最亮的通道值
print(arr[0, 0])   # 左上角像素的 RGBA

tensor = torch.from_numpy(arr)          # 转换Tensor，此时形状仍是 (H, W, 4)
print(tensor.shape, tensor.dtype)
print(tensor.max(), tensor.min())
print(tensor[0, 0])                     

# 变成CHW并归一化
tensor_chw = tensor.permute(2, 0, 1).float() / 255.0  #C换到H，W前面
print(tensor_chw.shape)

```

运行结果：
(1512, 852)
RGBA
(852, 1512, 4)
uint8
255 18
[145 168 193 255]
torch.Size([852, 1512, 4]) torch.uint8
tensor(255, dtype=torch.uint8) tensor(18, dtype=torch.uint8)
tensor([145, 168, 193, 255], dtype=torch.uint8)
torch.Size([4, 852, 1512])
