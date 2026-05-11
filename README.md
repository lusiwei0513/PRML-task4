# Transformer Ablation Homework

本项目基于经典论文 **Attention Is All You Need** 中的 Transformer 架构，使用一个小规模的序列反转任务对 Transformer 进行复现实验与消融分析。

实验重点包括：

1. 复现 Encoder-Decoder Transformer 的核心结构；
2. 比较不同位置编码方式对顺序建模能力的影响；
3. 分析 Residual Connection 与 LayerNorm 对模型训练稳定性和性能的影响；
4. 通过 loss 曲线、准确率指标和 attention heatmap 对实验结果进行可视化分析。

---

## 1. 实验任务

本项目采用 synthetic reverse task 作为实验任务。

给定一个输入序列：

```text
[6, 27, 10, 22, 15]
```

模型需要输出其反转序列：

```text
[15, 22, 10, 27, 6]
```

该任务虽然规模较小，但对位置关系非常敏感，适合观察 Transformer 是否能够学习 token 顺序信息以及输入输出位置之间的对应关系。

---

## 2. 实验内容

### 2.1 位置编码实验

比较以下四种位置编码方式：

| 实验名称 | 位置编码方式 | 说明 |
|---|---|---|
| `pe_none` | 无位置编码 | 不向模型提供任何位置信息 |
| `pe_simple` | 简单绝对位置编码 | 使用归一化位置编号并映射到 embedding 维度 |
| `pe_learned` | 可学习位置编码 | 为每个位置设置可训练向量 |
| `pe_sinusoidal` | 正弦/余弦位置编码 | 原论文采用的位置编码方式 |

该实验用于分析位置编码在 Transformer 顺序建模中的必要性。

### 2.2 残差结构实验

在固定使用 sinusoidal positional encoding 的基础上，比较以下结构：

| 实验名称 | 结构形式 | 说明 |
|---|---|---|
| `res_standard` / `pe_sinusoidal` | `LayerNorm(x + Sublayer(x))` | 原论文标准 Add & Norm 结构 |
| `res_no_residual` | `LayerNorm(Sublayer(x))` | 去掉残差连接 |
| `res_no_layernorm` | `x + Sublayer(x)` | 去掉 LayerNorm |
| `res_no_residual_no_layernorm` | `Sublayer(x)` | 同时去掉残差连接和 LayerNorm |

该实验用于分析 residual connection 与 LayerNorm 对模型训练稳定性和泛化能力的影响。

---

## 3. 项目结构

```text
transformer-ablation-homework/
├── run_ablation.py
├── README.md
└── my_experiment/
    ├── __init__.py
    ├── constants.py
    ├── utils.py
    ├── data.py
    ├── masks.py
    ├── positional_encoding.py
    ├── attention.py
    ├── layers.py
    ├── model.py
    ├── scheduler.py
    ├── engine.py
    ├── experiments.py
    ├── visualization.py
    ├── main.py
    └── results/
```

主要文件说明：

| 文件 | 功能 |
|---|---|
| `constants.py` | 定义特殊 token：`PAD`、`BOS`、`EOS` |
| `utils.py` | 设置随机种子，保证实验可复现 |
| `data.py` | 构造 reverse task 数据集和 DataLoader |
| `masks.py` | 构造 source padding mask 和 decoder causal mask |
| `positional_encoding.py` | 实现四种位置编码方式 |
| `attention.py` | 实现 Multi-Head Attention |
| `layers.py` | 实现 Feed Forward、Sublayer Connection、EncoderLayer、DecoderLayer |
| `model.py` | 实现完整 Seq2Seq Transformer |
| `scheduler.py` | 实现 Transformer warmup learning rate schedule |
| `engine.py` | 实现训练、验证和 greedy decoding |
| `experiments.py` | 管理不同消融实验配置 |
| `visualization.py` | 绘制 loss 曲线和 attention heatmap |
| `main.py` | 参数解析和实验主流程 |
| `run_ablation.py` | 项目启动脚本 |

---

## 4. 环境依赖

建议使用 Python 3.10 或 Python 3.11。

安装依赖：

```bash
pip install torch numpy pandas matplotlib tqdm
```

本项目没有使用大型外部数据集，实验数据由代码自动生成。

---

## 5. 运行方式

在项目根目录下运行：

```bash
python run_ablation.py --mode all
```

如果 `run_ablation.py` 放在 `my_experiment/` 文件夹内部，则运行：

```bash
python my_experiment/run_ablation.py --mode all
```

---

## 6. 常用命令

### 运行 baseline

```bash
python run_ablation.py --mode baseline
```

### 运行位置编码实验

```bash
python run_ablation.py --mode pe
```

### 运行残差结构实验

```bash
python run_ablation.py --mode residual
```

### 运行全部实验

```bash
python run_ablation.py --mode all
```

### CPU 快速测试

如果设备性能有限，可以使用更小规模参数先跑通流程：

```bash
python run_ablation.py --mode all --train_size 3000 --valid_size 500 --test_size 500 --epochs 6 --num_layers 2 --d_model 64 --d_ff 256 --batch_size 64
```

---

## 7. 主要参数说明

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--mode` | `all` | 选择运行 baseline、位置编码实验、残差实验或全部实验 |
| `--train_size` | `12000` | 训练集样本数 |
| `--valid_size` | `1000` | 验证集样本数 |
| `--test_size` | `1000` | 测试集样本数 |
| `--train_min_len` | `5` | 训练序列最小长度 |
| `--train_max_len` | `20` | 训练序列最大长度 |
| `--ood_min_len` | `21` | OOD 测试序列最小长度 |
| `--ood_max_len` | `40` | OOD 测试序列最大长度 |
| `--d_model` | `128` | Transformer hidden dimension |
| `--num_heads` | `4` | Multi-head attention 的 head 数量 |
| `--num_layers` | `4` | Encoder 和 Decoder 的层数 |
| `--d_ff` | `512` | Feed Forward 中间层维度 |
| `--epochs` | `12` | 训练轮数 |
| `--batch_size` | `128` | batch size |

---

## 8. 输出结果

实验结果默认保存在：

```text
my_experiment/results/
```

主要输出文件包括：

| 文件 | 说明 |
|---|---|
| `summary.csv` | 所有实验组的最终测试结果 |
| `*_history.csv` | 每个实验组每个 epoch 的训练过程记录 |
| `valid_loss_curves.png` | 不同实验组的验证集 loss 曲线 |
| `baseline_attention_heatmap.png` | baseline 模型的 decoder-source attention heatmap |

每个 `*_history.csv` 文件中包含以下指标：

| 指标 | 含义 |
|---|---|
| `train_loss` | 训练集平均交叉熵损失 |
| `valid_loss` | 验证集平均交叉熵损失 |
| `valid_teacher_token_acc` | teacher forcing 条件下的逐 token 准确率 |
| `valid_greedy_token_acc` | greedy decoding 条件下的逐 token 准确率 |
| `valid_sequence_acc` | 完整序列预测准确率 |
| `grad_norm` | 梯度范数 |
| `lr` | 当前学习率 |

---

## 9. 实验结论概述

实验结果表明：

1. 不加位置编码时，Transformer 难以完成序列反转任务，说明 self-attention 本身不包含显式顺序信息；
2. 简单绝对位置编码可以提供一定顺序信息，但效果有限；
3. 可学习位置编码在训练长度范围内表现较好，但对长度外推能力存在一定限制；
4. 原论文 sinusoidal positional encoding 在本实验中收敛最快，最终性能最好；
5. 去掉 residual connection 后，模型难以有效学习序列映射；
6. 去掉 LayerNorm 后，训练过程容易出现数值不稳定；
7. 标准 Add & Norm 结构对 Transformer 的稳定训练和最终性能具有关键作用。

---

## 10. 参考论文

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).  
**Attention Is All You Need.**  
Advances in Neural Information Processing Systems.
