模型结构图：

```mermaid
graph TD
    %% 定义节点样式（类似你之前图的颜色）
    classDef input fill:#c6f6d5,stroke:#48bb78,stroke-width:2px;
    classDef block fill:#ebf8ff,stroke:#4299e1,stroke-width:2px;
    classDef flatten fill:#feebc8,stroke:#ed8936,stroke-width:2px;
    classDef classifier fill:#fefcbf,stroke:#ecc94b,stroke-width:2px;
    classDef output fill:#fed7d7,stroke:#f56565,stroke-width:2px;

    %% 定义节点和内容
    In["Input Image<br/>[B, 3, 224, 224]"]:::input
    
    B1["Block 1<br/>Conv2d(3->16, 3x3)<br/>ReLU<br/>MaxPool2d(2x2)<br/><br/>Tensor: [B, 16, 112, 112]"]:::block
    
    B2["Block 2<br/>Conv2d(16->32, 3x3)<br/>ReLU<br/>MaxPool2d(2x2)<br/><br/>Tensor: [B, 32, 56, 56]"]:::block
    
    Fl["Flatten<br/>32 * 56 * 56 = 100352<br/><br/>Tensor: [B, 100352]"]:::flatten
    
    Cl["Classifier<br/>Linear(100352 -> 128)<br/>ReLU<br/>Dropout(0.5)<br/><br/>Tensor: [B, 128]"]:::classifier
    
    Out["Output<br/>Linear(128 -> 2)<br/><br/>Tensor: [B, 2]"]:::output

    %% 定义连接线
    In --> B1
    B1 --> B2
    B2 --> Fl
    Fl --> Cl
    Cl --> Out
