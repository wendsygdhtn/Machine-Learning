import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

#1.数据路径

src_dir = Path(r"D:\\kaggle_data\\dogs-vs-cats-redux-kernels-edition\\train\\train")
val_dir = src_dir.parent / "val"

#2.预处理
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

#3.Dataset/DataLoader
train_ds = datasets.ImageFolder(str(src_dir), transform=train_tf)
val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

#4.模型
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}     # 记录训练过程的字典
    
    best_val_acc = 0.0 # 记录最高准确率

#5.训练循环
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss, train_acc = running_loss / total, correct / total

        # 验证环节
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        wrong_samples = [] 

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                wrong_mask = (predicted != labels)
                wrong_idx = torch.where(wrong_mask)[0]
                
                for idx in wrong_idx:
                    img_tensor = images[idx].cpu()
                    true_label = labels[idx].item()
                    pred_label = predicted[idx].item()
                    confidence = probs[idx, pred_label].item()
                    wrong_samples.append((img_tensor, true_label, pred_label, confidence))
                    
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        #保存表现最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cat_dog_model.pth")
            print(f"验证集准确率提升至 {val_acc:.4f}，已保存当前最佳模型！")
        

    print(f"训练结束，最高验证集准确率为: {best_val_acc:.4f}，模型已保存为best_cat_dog_model.pth")

#6.绘制曲线
    plt.figure(figsize=(12, 4))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_acc'], label='Train Acc', marker='o')
    plt.plot(range(1, epochs+1), history['val_acc'], label='Val Acc', marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

#7.可视化预测错误的样本
    print(f"验证集中共有 {len(wrong_samples)} 张图片预测错误。")
    
    random.seed(42)
    show_samples = random.sample(wrong_samples, min(8, len(wrong_samples)))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    class_names = ['Cat', 'Dog']
    
    for i, (img_tensor, true_label, pred_label, conf) in enumerate(show_samples):
        img_np = img_tensor.numpy().transpose((1, 2, 0))
        img_np = np.array(std) * img_np + np.array(mean)
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        title = f"True: {class_names[true_label]}\\nPred: {class_names[pred_label]} ({conf:.2f})"
        axes[i].set_title(title, color='red', fontsize=10)

    plt.suptitle("Misclassified Validation Samples", fontsize=14)
    plt.tight_layout()
    plt.show()
