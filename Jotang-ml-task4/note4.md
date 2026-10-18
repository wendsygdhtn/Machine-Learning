# task4 任务集




**1.阅读论文中的 Figure 1，分别找出 Encoder 和 Decoder。参考原图，用自己的方式重新画一张 简化/拓展 的 Transformer 结构图。**

```mermaid
flowchart TB
    subgraph ENC["Encoder（N=6 层堆叠）"]
        direction TB
        E_IN["输入 Embedding<br/>+ Positional Encoding"]
        E_ATT["Multi-Head<br/>Self-Attention"]
        E_ADD1["Add & LayerNorm"]
        E_FFN["Position-wise<br/>Feed-Forward"]
        E_ADD2["Add & LayerNorm"]
        E_OUT["Encoder 输出<br/>（memory）"]

        E_IN --> E_ATT --> E_ADD1 --> E_FFN --> E_ADD2 --> E_OUT
        E_IN -. 残差 .-> E_ADD1
        E_ADD1 -. 残差 .-> E_ADD2
    end

    subgraph DEC["Decoder（N=6 层堆叠）"]
        direction TB
        D_IN["输出 Embedding<br/>+ Positional Encoding<br/>（右移一位）"]
        D_MASK["Masked Multi-Head<br/>Self-Attention"]
        D_ADD1["Add & LayerNorm"]
        D_CROSS["Multi-Head<br/>Encoder-Decoder Attention"]
        D_ADD2["Add & LayerNorm"]
        D_FFN["Position-wise<br/>Feed-Forward"]
        D_ADD3["Add & LayerNorm"]
        D_LIN["Linear"]
        D_SOFT["Softmax"]
        D_OUT["输出概率<br/>（下一个 token）"]

        D_IN --> D_MASK --> D_ADD1 --> D_CROSS --> D_ADD2 --> D_FFN --> D_ADD3 --> D_LIN --> D_SOFT --> D_OUT
        D_IN -. 残差 .-> D_ADD1
        D_ADD1 -. 残差 .-> D_ADD2
        D_ADD2 -. 残差 .-> D_ADD3
    end

    E_OUT -->|"K, V（编码器输出）"| D_CROSS
    D_ADD1 -->|"Q（解码器当前层）"| D_CROSS
```

**2.重点阅读论文第 3 节，了解 Scaled Dot-Product Attention、Multi-Head Attention、Position-wise Feed-Forward Network 和 Positional Encoding。**

**3.整理一份阅读笔记，记录你使用了哪些资料或工具、遇到了哪些读不懂的地方、后来怎样理解，以及目前仍然存在哪些疑问。**

**4.这篇论文最初处理的是什么任务？它为什么尝试不用 RNN 和 CNN？**

**5.Self-Attention 中的“Self”是什么意思？它和普通 Attention 有什么区别？**

**6.Q、K、V 分别来自哪里？它们为什么要经过不同的线性变换？**

**7.Attention 的结果为什么要经过 softmax？得到的每一行权重表示什么？**

**8.Multi-Head Attention 为什么要分成多个头？多个头只是把同一件事重复多次吗？**

**9.Transformer 没有循环结构，它怎样知道词语的先后顺序？**

**10.Decoder 中为什么需要 Masked Attention？如果能提前看到后面的答案，会发生什么？**

**11.残差连接、Layer Normalization 和 Feed-Forward Network 分别出现在结构图的哪里？它们大致有什么作用？**
