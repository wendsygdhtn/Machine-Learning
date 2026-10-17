# 本文件用来记录所有试验代码



from PIL import Image
import numpy as np
import torch 
import matplotlib
matplotlib.use('TkAgg')

#01
img = Image.open("highway.png") 
img=img.convert("RGB")   #RGBA转RGB
"""
print(img.size)   # (宽, 高)
print(img.mode)   # 通常是 'RGB'

#arr(x,y)=(R(x,y),G(x,y),B(x,y));arr(x,y,颜色)=(0~255（该颜色强度）)
arr = np.array(img)   #arr是二维数组，每个元素对应一个小方格，每个元素包含该格红/绿/蓝三色强度的三个数值     
print(arr.shape)  #arr.shape=(高/行数，宽/列数，颜色通道数（3）)
print(arr.dtype)   #每个数值占1字节，为0~255
print(arr.max(), arr.min())  #整张图最暗和最亮的通道值
print(arr[0, 0])   # 左上角像素的 RGBA

tensor = torch.from_numpy(arr)          # 转换Tensor，此时形状仍是 (H, W, 3)
print(tensor.shape, tensor.dtype)
print(tensor.max(), tensor.min())
print(tensor[0, 0])                     

# 变成CHW并归一化
tensor_chw = tensor.permute(2, 0, 1).float() / 255.0  #C换到H，W前面
print(tensor_chw.shape)

"""
#02
#处理灰度图的卷积函数
def convolve2d(gray_img, kernel):
    
    gray_arr = np.array(gray_img)   

    W,H=gray_img.size[0],gray_img.size[1]
    kH,kW=kernel.shape

    Hout,Wout=H-kH+1,W-kW+1
    cv = np.zeros((Hout,Wout), dtype=np.float64)#输出灰度矩阵
    for i in range (Hout):
        for j in range (Wout):
            patch = gray_arr[i:i+kH, j:j+kW]   #左闭右开
            cv[i,j] = (kernel * patch).sum()
    #print(cv)
    return cv
import matplotlib.pyplot as plt
"""
#以下为测试
gray_img = img.convert('L')   #转灰度图
kernel_1 = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=float)

result=convolve2d(gray_img,kernel_1)
print(result)



plt.imshow(result, cmap='gray')
plt.axis('off')   # 不显示坐标轴
plt.show()
"""

#处理彩色图的卷积函数
def convolve2dRGB(img, kernel):
    
    arr = np.array(img)   

    W,H=img.size[0],img.size[1]
    kH,kW=kernel.shape

    Hout,Wout=H-kH+1,W-kW+1
    cv = np.zeros((Hout,Wout,3), dtype=np.float64)#输出三维矩阵，每个元素表示某行某列某颜色通道的值
    for i in range (Hout):
        for j in range (Wout):          
            for k in range (3):
                patch = arr[i:i+kH, j:j+kW,k]   #左闭右开
                cv[i,j,k] = (kernel * patch).sum()
    #print(cv)
    return cv


kernel_2=np.array([[0,-1,0],
                 [-1,5,-1],
                 [0,-1,0]],dtype=float)
kernel_3=np.ones((3,3))/9
kernel_gaussian = np.array([
    [ 1,  4,  6,  4, 1],
    [ 4, 16, 24, 16, 4],
    [ 6, 24, 36, 24, 6],
    [ 4, 16, 24, 16, 4],
    [ 1,  4,  6,  4, 1]
], dtype=float) / 256
result_2=convolve2dRGB(img,kernel_gaussian)
result_uint8 = np.clip(result_2, 0, 255).astype(np.uint8)
plt.imshow(result_uint8)
plt.axis('off')
plt.show()

#边缘图
#函数
def convolve2d_array(image, kernel):
    H, W = image.shape[:2]
    kH, kW = kernel.shape
    Hout, Wout = H - kH + 1, W - kW + 1
    
    output = np.zeros((Hout, Wout), dtype=np.float64)
    for i in range(Hout):
        for j in range(Wout):
            patch = image[i:i+kH, j:j+kW].astype(np.float64)
            output[i, j] = (kernel * patch).sum()
    return output

img = Image.open("highway.png").convert("RGB") 
gray_arr = np.array(img.convert("L"))

#定义sobel_x,sobel_y
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)

#计算梯度
grad_x = convolve2d_array(gray_arr, sobel_x)
grad_y = convolve2d_array(gray_arr, sobel_y)

#合成边缘图
edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
edge_norm = (edge_magnitude / (edge_magnitude.max() + 1e-8)) * 255
edge_show = edge_norm.astype(np.uint8)

#原图
plt.figure(figsize=(8, 8))
plt.imshow(gray_arr, cmap='gray')
plt.title("原图")
plt.axis('off')
plt.show() 
#Sobel-X
plt.figure(figsize=(8, 8))
plt.imshow(np.abs(grad_x), cmap='gray')
plt.title("Sobel-X")
plt.axis('off')
plt.show()  
#Sobel-Y
plt.figure(figsize=(8, 8))
plt.imshow(np.abs(grad_y), cmap='gray')
plt.title("Sobel-Y")
plt.axis('off')
plt.show()  
# 合成边缘图
plt.figure(figsize=(8, 8))
plt.imshow(edge_show, cmap='gray')
plt.title("合成边缘图")
plt.axis('off')
plt.show()  
