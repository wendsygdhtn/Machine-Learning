import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#1. 定义模型结构 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

#2. 推理函数
def predict_single_image(img_path, model_path="best_cat_dog_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：未找到模型文件 '{model_path}'！")
        print("👉 请确保模型文件与 predict.py 在同一目录下，或前往网盘/GitHub 下载。")
        return

    # 定义与训练严格一致的预处理
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # 加载模型
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    try:
        # 读取并预处理图片
        img_pil = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 错误：找不到你输入的图片路径 '{img_path}'，请检查路径是否正确。")
        return
        
    img_tensor = val_tf(img_pil).unsqueeze(0).to(device) # 增加 batch 维度 [1, 3, 224, 224]
    
    # 推理预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0] 
        
    # ImageFolder 按字母序：cat=0, dog=1
    dog_prob = probs[1].item()
    cat_prob = probs[0].item()
    
    if dog_prob > 0.5:
        pred_class = "狗 (Dog)"
        confidence = dog_prob
    else:
        pred_class = "猫 (Cat)"
        confidence = cat_prob

    print(f"\\n--- 预测结果 ---")
    print(f"输入图片: {img_path}")
    print(f"预测类别: {pred_class}")
    print(f"置信度  : {confidence:.4f} (猫:{cat_prob:.4f}, 狗:{dog_prob:.4f})")

    # 可视化：展示经过预处理（裁剪、归一化还原后）的图片
    img_show = img_tensor[0].cpu().numpy().transpose((1, 2, 0))
    img_show = np.array(std) * img_show + np.array(mean)
    img_show = np.clip(img_show, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img_show)
    plt.title(f"Pred: {pred_class} ({confidence*100:.2f}%)", fontsize=14, color='blue')
    plt.axis('off')
    plt.show()

#3. 主程序
if __name__ == "__main__":
    # 使用时修改这里为你想要测试的任意本地图片路径，我这里先用我的一张图片的图片路径填充了
    test_img_path = r"D:\\kaggle_data\\dogs-vs-cats-redux-kernels-edition\\test\\test\\5.jpg" 
    
    # 调用推理函数
    predict_single_image(test_img_path, model_path="best_cat_dog_model.pth")