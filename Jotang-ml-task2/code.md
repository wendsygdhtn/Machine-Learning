# 本文件用来记录所有试验代码和结果

01.
```python
from PIL import Image

img = Image.open("highway.png")  
print(img.size)   # 图片的(宽, 高)
print(img.mode)   
```

运行结果：

(1518, 852)  
RGBA
