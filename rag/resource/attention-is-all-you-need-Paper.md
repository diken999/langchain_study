# Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. 发表于 NIPS 2017。本文档根据原论文整理，便于 RAG / Markdown 标题切分测试。

## Abstract

主流序列转导模型基于复杂的循环或卷积神经网络，并包含编码器与解码器；表现最好的模型还会通过注意力机制连接编码器与解码器。本文提出一种全新的简单网络架构——Transformer，完全基于注意力机制，彻底摒弃循环与卷积。在两项机器翻译任务上的实验表明，该模型质量更优、更易并行，且训练时间显著更短。模型在 WMT 2014 英德翻译任务上达到 28.4 BLEU，比此前最佳结果（含集成）高出超过 2 BLEU；在 WMT 2014 英法翻译任务上，单模型达到新的 SOTA BLEU 分数 41.0，仅在 8 块 GPU 上训练 3.5 天，远低于文献中最佳模型的训练成本。

## 一、Introduction

循环神经网络，尤其是 LSTM 与门控循环网络，已是序列建模与转导问题（如语言模型与机器翻译）的主流方法。循环模型通常沿输入输出序列的符号位置分解计算：将位置对齐到计算时间步，生成隐藏状态序列 ht，它是前一隐藏状态 ht−1 与位置 t 输入的函数。这种固有的序列性阻碍了训练样本内部的并行化；序列越长，受内存限制，跨样本 batching 越困难。

注意力机制已成为序列建模与转导模型的重要组成部分，可在不考虑输入/输出序列中距离的情况下建模依赖关系。但除少数情况外，注意力通常仍与循环网络结合使用。

本文提出 Transformer：摒弃循环，完全依赖注意力机制刻画输入与输出之间的全局依赖。Transformer 可显著提升并行度，并在 8 块 P100 GPU 上仅训练约 12 小时即可达到翻译质量新 SOTA。

### 1.1 核心贡献

Transformer 是首个完全依赖自注意力计算输入与输出表示、且不使用序列对齐 RNN 或卷积的转导模型。

## 二、Background

降低序列计算量也是 Extended Neural GPU、ByteNet、ConvS2S 等工作的基础；它们以 CNN 为基本模块，对所有输入输出位置并行计算隐藏表示。在这些模型中，关联任意两个位置信号所需的操作数随距离增长：ConvS2S 为线性，ByteNet 为对数，因此更难学习远距离依赖。在 Transformer 中，该操作数降为常数，但会因对注意力加权位置求平均而降低有效分辨率；作者用 Multi-Head Attention 加以缓解。

### 2.1 Self-Attention

Self-attention（有时也称 intra-attention）是一种将同一序列中不同位置关联起来以计算序列表示的注意力机制。它已成功用于阅读理解、摘要、文本蕴含、以及任务无关的句表示等任务。

### 2.2 与其他模型的关系

End-to-end memory networks 基于循环注意力而非序列对齐循环，在简单语言问答与语言建模上表现良好。据作者所知，Transformer 是首个完全依赖自注意力、不使用 RNN 或卷积来计算表示的转导模型。

## 三、Model Architecture

多数竞争性神经序列转导模型采用 encoder-decoder 结构。编码器将输入符号序列 (x1,...,xn) 映射为连续表示序列 z = (z1,...,zn)；解码器再基于 z 逐步生成输出符号序列 (y1,...,ym)。每一步模型是自回归的：生成下一符号时会把已生成符号作为额外输入。

Transformer 沿用这一整体架构，编码器与解码器均使用堆叠的自注意力与逐位置全连接层。

### 3.1 Encoder and Decoder Stacks

#### 3.1.1 Encoder

编码器由 N = 6 个相同层堆叠而成。每层有两个子层：第一个是 multi-head self-attention；第二个是逐位置全连接前馈网络。每个子层周围使用残差连接，后接层归一化，即输出为 LayerNorm(x + Sublayer(x))。为便于残差连接，模型中所有子层以及 embedding 层的输出维度均为 dmodel = 512。

##### 残差与归一化

每个子层输出为 LayerNorm(x + Sublayer(x))，其中 Sublayer(x) 为该子层实现的函数。

###### dmodel 设定

所有子层与 embedding 层输出维度 dmodel = 512。

#### 3.1.2 Decoder

解码器同样由 N = 6 个相同层堆叠。除编码器中的两个子层外，解码器还插入第三个子层：对编码器栈输出做 multi-head attention。同样使用残差连接与层归一化。解码器栈中的自注意力子层被修改，以阻止当前位置关注后续位置。该 masking 与输出 embedding 右移一位相结合，保证位置 i 的预测只能依赖位置小于 i 的已知输出。

##### Decoder Masking

Masking 确保自回归性质：位置 i 不能看到未来位置。

###### 自回归约束

预测位置 i 时，只能依赖 positions < i 的已知输出。

### 3.2 Attention

注意力函数可描述为：将一个 query 与一组 key-value 对映射到输出；query、key、value、输出均为向量。输出是 value 的加权和，权重由 query 与对应 key 的兼容性函数计算。

#### 3.2.1 Scaled Dot-Product Attention

作者称其注意力为 Scaled Dot-Product Attention。输入为维度 dk 的 queries 与 keys，以及维度 dv 的 values。计算 query 与所有 keys 的点积，各除以 √dk，再经 softmax 得到 value 的权重。

实践中，将一组 queries 打包为矩阵 Q，keys 与 values 打包为 K、V，输出为：

Attention(Q, K, V) = softmax(QK^T / √dk) V

##### 与 Additive Attention 对比

两种常用注意力是 additive attention 与 dot-product（乘性）attention。Dot-product attention 与本文算法相同，只是多了缩放因子 1/√dk。Additive attention 用带单隐层的前馈网络计算兼容性。二者理论复杂度相近，但点积注意力更快、更省空间，可用高度优化的矩阵乘法实现。

###### 为何需要 Scaling

当 dk 较大时，点积幅度变大，会把 softmax 推到梯度极小的区域。假设 q、k 各分量独立、均值 0、方差 1，则点积 q·k 均值 0、方差为 dk。因此用 1/√dk 缩放点积。

#### 3.2.2 Multi-Head Attention

与其用 dmodel 维的 keys/values/queries 做单次注意力，作者发现将 queries、keys、values 用不同的可学习线性投影投影 h 次到 dk、dk、dv 维更有益。在每个投影版本上并行做注意力，得到 dv 维输出，再拼接并再次投影得到最终值。

MultiHead(Q, K, V) = Concat(head1, ..., headh) W^O

其中 headi = Attention(Q W^Q_i, K W^K_i, V W^V_i)

##### 超参数设定

本文使用 h = 8 个并行注意力头；对每个头 dk = dv = dmodel/h = 64。由于每头维度降低，总计算量与全维度单头注意力相近。

###### Multi-Head 的好处

Multi-head attention 使模型能共同关注不同位置、不同表示子空间的信息；单头时平均会抑制这种能力。

#### 3.2.3 Applications of Attention in our Model

Transformer 以三种方式使用 multi-head attention：

1. Encoder-decoder attention：queries 来自上一解码器层，keys 与 values 来自编码器输出；解码器每个位置可关注输入序列所有位置。
2. Encoder self-attention：keys、values、queries 均来自编码器上一层输出；编码器每个位置可关注上一层所有位置。
3. Decoder self-attention：解码器每个位置可关注到当前位置为止的所有解码器位置；通过在 scaled dot-product attention 中把非法连接对应的 softmax 输入设为 −∞ 来 masking，以保持自回归性质。

### 3.3 Position-wise Feed-Forward Networks

除注意力子层外，编码器与解码器每层还包含逐位置全连接前馈网络，对每个位置单独且相同地应用，由两次线性变换与中间 ReLU 组成：

FFN(x) = max(0, xW1 + b1) W2 + b2

不同位置线性变换相同，但各层参数不同。也可看作核大小为 1 的两次卷积。输入输出维度 dmodel = 512，内层维度 dff = 2048。

### 3.4 Embeddings and Softmax

与其他序列转导模型类似，使用可学习 embedding 将输入/输出 token 转为 dmodel 维向量；并用可学习线性变换与 softmax 将解码器输出转为下一 token 概率。模型在两个 embedding 层与 pre-softmax 线性变换之间共享同一权重矩阵；在 embedding 层将这些权重乘以 √dmodel。

### 3.5 Positional Encoding

由于模型无循环、无卷积，为利用序列顺序，必须注入 token 的相对或绝对位置信息。作者在编码器与解码器栈底部将 positional encodings 加到输入 embeddings 上。positional encodings 与 embeddings 同维 dmodel，以便相加。

#### 3.5.1 正弦位置编码

本文使用不同频率的正弦与余弦函数：

PE(pos, 2i) = sin(pos / 10000^(2i/dmodel))

PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))

其中 pos 为位置，i 为维度。每个维度对应一条正弦曲线，波长从 2π 到 10000·2π 成几何级数。选择该函数是因为假设模型易于学习相对位置注意力：对任意固定偏移 k，PE_{pos+k} 可表示为 PE_pos 的线性函数。

##### 与可学习位置编码对比

作者也实验了可学习 positional embeddings，结果与正弦版本几乎相同。选择正弦版本是因为可能外推到训练时未见过的更长序列。

###### 位置编码维度

positional encodings 维度与 embeddings 相同，均为 dmodel，两者相加后送入编码器/解码器栈。

## 四、Why Self-Attention

本节比较自注意力层与常用于序列映射的循环层、卷积层，从三方面衡量：每层总计算复杂度、可并行计算量（最少顺序操作数）、网络中长程依赖的最大路径长度。

### 4.1 复杂度对比

如表所示（n 为序列长度，d 为表示维度，k 为卷积核大小，r 为受限自注意力邻域大小）：

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| Self-Attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r · n · d) | O(1) | O(n/r) |

### 4.2 主要优势

自注意力层用常数次顺序操作连接所有位置，而循环层需要 O(n) 次顺序操作。当序列长度 n 小于表示维度 d 时（机器翻译中常见），自注意力层比循环层更快。对很长序列，可将自注意力限制在大小为 r 的邻域，此时最大路径长度变为 O(n/r)。

### 4.3 可解释性

自注意力还可带来更可解释的模型：不同注意力头会学到不同任务，许多头表现出与句子句法、语义结构相关的行为。

## 五、Training

本节描述模型的训练设定。

### 5.1 Training Data and Batching

在标准 WMT 2014 英德数据集上训练，约 450 万句对；使用 byte-pair encoding，共享源-目标词表约 37000 tokens。英法任务使用更大的 WMT 2014 英法数据，约 3600 万句，词表约 32000 word-piece。按近似序列长度组 batch；每个训练 batch 约含 25000 个源 tokens 与 25000 个目标 tokens。

### 5.2 Hardware and Schedule

在一台配备 8 块 NVIDIA P100 GPU 的机器上训练。base 模型每步约 0.4 秒，共训练 100,000 步（约 12 小时）。big 模型每步约 1.0 秒，训练 300,000 步（约 3.5 天）。

### 5.3 Optimizer

使用 Adam 优化器，β1 = 0.9，β2 = 0.98，ε = 10^−9。学习率按公式变化：

lrate = dmodel^(−0.5) · min(step_num^(−0.5), step_num · warmup_steps^(−1.5))

即前 warmup_steps 步线性升温，之后按步数平方根的倒数衰减。warmup_steps = 4000。

### 5.4 Regularization

训练中使用三类正则化：

#### Residual Dropout

对每个子层输出（在加到子层输入并归一化之前）施加 dropout；也对 embeddings 与 positional encodings 之和施加 dropout。base 模型 Pdrop = 0.1。

#### Label Smoothing

训练时使用 label smoothing，ε_ls = 0.1。这会损害 perplexity（模型更不确定），但提升准确率与 BLEU。

##### Dropout 位置小结

子层输出与 embedding+位置编码之和都会 dropout。

###### Base 模型 Dropout 率

base 模型使用 Pdrop = 0.1。

## 六、Results

### 6.1 Machine Translation

在 WMT 2014 英德翻译任务上，Transformer (big) 比此前最佳模型（含集成）高出超过 2.0 BLEU，达到新 SOTA 28.4 BLEU；在 8 块 P100 上训练 3.5 天。即使 base 模型也超过此前所有已发表模型与集成，且训练成本仅为竞争模型的一小部分。

在 WMT 2014 英法翻译任务上，big 模型达到 BLEU 41.0，超过此前所有已发表单模型，训练成本不到此前 SOTA 的 1/4。英法 big 模型使用 Pdrop = 0.1（而非 0.3）。

#### 推理设定

base 模型对最后 5 个 checkpoint（每 10 分钟写一次）取平均；big 模型对最后 20 个 checkpoint 取平均。使用 beam search：beam size = 4，长度惩罚 α = 0.6。推理时最大输出长度为输入长度 + 50，可尽早终止。

##### 主要 BLEU 结果

- Transformer (base)：EN-DE 27.3，EN-FR 38.1
- Transformer (big)：EN-DE 28.4，EN-FR 41.0

###### 训练成本（FLOPs，约）

- Transformer (base) EN-DE：3.3 × 10^18
- Transformer (big) EN-DE：2.3 × 10^19

### 6.2 Model Variations

为评估各组件重要性，作者对 base 模型做多种变体，在英德开发集 newstest2013 上测量性能。

#### 注意力头数

在计算量不变时改变注意力头数与 key/value 维度：单头比最佳设置差 0.9 BLEU；头数过多质量也会下降。减小注意力 key 维度 dk 会损害质量，说明兼容性判定并不容易，可能需要比点积更复杂的兼容性函数。

##### 模型规模与 Dropout

更大模型通常更好；dropout 对避免过拟合很有帮助。

###### 位置编码变体

用可学习 positional embeddings 替换正弦位置编码后，结果与 base 模型几乎相同。

## 七、Conclusion

本文提出 Transformer：首个完全基于注意力的序列转导模型，用 multi-headed self-attention 取代 encoder-decoder 中常用的循环层。在翻译任务上，Transformer 可比基于循环或卷积的架构训练得显著更快；在 WMT 2014 英德与英法任务上均达到新 SOTA，英德任务上最佳模型甚至超过此前所有已报道的集成模型。

作者计划将 Transformer 扩展到文本以外的输入输出模态，并研究局部/受限注意力以高效处理图像、音频、视频等大输入输出；使生成更少序列依赖也是研究目标之一。训练与评估代码见：https://github.com/tensorflow/tensor2tensor
