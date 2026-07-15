# Transformer 推理优化学习指南

> **适用读者**：已初步了解 Transformer 框架（Self-Attention、Multi-Head Attention、FFN、LayerNorm、Position Encoding 等基本组件），希望系统学习推理优化技术。
>
> **基于项目**：本文档围绕 DNDX 多模态推理调优赛的实际代码（Qwen3.5-2B 模型 + HuggingFace Transformers），从原理到实践逐层深入。

---

## 目录

1. [前置回顾：Transformer 推理的全景图](#1-前置回顾transformer-推理的全景图)
2. [KV Cache：推理优化的第一性原理](#2-kv-cache推理优化的第一性原理)
3. [Prefill vs Decode：推理的两个阶段](#3-prefill-vs-decode推理的两个阶段)
4. [注意力机制优化：从标准 Attention 到 Flash Attention](#4-注意力机制优化从标准-attention-到-flash-attention)
5. [内存优化：精度、量化与设备映射](#5-内存优化精度量化与设备映射)
6. [TTFT 优化：首 Token 延迟的攻防战](#6-ttft-优化首-token-延迟的攻防战)
7. [吞吐量优化：让 GPU 持续忙碌](#7-吞吐量优化让-gpu-持续忙碌)
8. [高级话题：CUDA Graph、Kernel Fusion 与自定义算子](#8-高级话题cuda-graphkernel-fusion-与自定义算子)
9. [本项目实战：从 882ms 到 350ms 的优化路线图](#9-本项目实战从-882ms-到-350ms-的优化路线图)
10. [延伸学习资源](#10-延伸学习资源)

---

## 1. 前置回顾：Transformer 推理的全景图

### 1.1 你已经知道的东西

你了解的标准 Transformer 架构：

```
输入 Token 序列
    │
    ├── Embedding（词嵌入 + 位置编码）
    │
    ├── [Transformer Block] × N
    │   ├── Multi-Head Self-Attention
    │   │   ├── Linear → Q, K, V
    │   │   ├── Q @ K^T / √d  →  Attention Scores
    │   │   ├── Softmax → Attention Weights
    │   │   └── Weights @ V → Output
    │   ├── Add & LayerNorm (残差连接)
    │   ├── FFN (两层 MLP + 激活函数)
    │   └── Add & LayerNorm
    │
    └── LM Head → 词表概率分布 → 采样/贪心 → 下一个 Token
```

### 1.2 从"训练"切换到"推理"，世界变了

训练时和推理时有本质区别：

| 维度 | 训练 | 推理 |
|:---|:---|:---|
| **输入方式** | 一次给整个序列 | 逐 token 生成 |
| **计算模式** | 所有位置并行算 | 每次只算一个新 token |
| **显存瓶颈** | 需要存梯度、优化器状态 | 主要存模型权重 + KV Cache |
| **延迟要求** | 不敏感（离线） | 极致敏感（用户等待） |
| **Attention 复杂度** | O(n²) 对 batch 内所有位置的并行矩阵乘 | O(n) 每次只与历史 KV 交互 |

**关键洞察**：推理时，每生成一个新 token，模型要"看一遍"之前所有的 token。如果每次都重新计算所有历史的 K 和 V，那生成第 256 个 token 时就要重复前面 255 次的 Attention 计算——这显然不可接受。于是有了 KV Cache。

---

## 2. KV Cache：推理优化的第一性原理

### 2.1 为什么需要 KV Cache？

回顾 Self-Attention 的计算：

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

在**生成第 t 个 token** 时：

- **不用 KV Cache**：
  ```
  Q_t（仅 1 个新 token 的 Query）
  K_[1:t]（重新计算全部历史 token 的 Key）  ← 浪费！
  V_[1:t]（重新计算全部历史 token 的 Value） ← 浪费！
  ```

- **用 KV Cache**：
  ```
  Q_t（仅 1 个新 token 的 Query）
  K_[1:t-1]（从缓存读取）+ K_t（只算新的）  ← 高效！
  V_[1:t-1]（从缓存读取）+ V_t（只算新的）  ← 高效！
  ```

**直观比喻**：就像你看书，每翻一页不需要把前面所有页重新读一遍——你把之前的内容"记在脑子里"（缓存起来），只需理解当前新的一页。

### 2.2 KV Cache 的显存计算

这是每个做推理优化的人必须会算的公式：

```
KV Cache 大小(字节) = 2 × batch_size × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

对本项目的 Qwen3.5-2B 模型（使用 GQA，仅 2 个 KV head）：

```
num_layers     = 24           # 配置文件中的 num_hidden_layers
num_kv_heads   = 2            # 仅 2 个 KV head（GQA 的关键）
head_dim       = 256          # 配置文件的 head_dim
dtype_bytes    = 2            # bfloat16 = 2 字节

单 token 的 KV Cache = 2 × 1 × 24 × 2 × 256 × 1 × 2 = 49,152 字节 ≈ 48 KB

生成 256 token 时：48 KB × 256 = 12 MB
batch_size=8 时：12 MB × 8 = 96 MB
```

看起来不大？但序列长了就恐怖了：

```
seq_len=4096：48 KB × 4096 × 8 = ~1.5 GB
seq_len=32768：48 KB × 32768 × 8 = ~12 GB
```

**关键认知**：KV Cache 是推理时显存的隐性消耗大户，而显存 = 能跑多大的 batch = 吞吐量。

### 2.3 GQA（Grouped Query Attention）：减少 KV Cache 的关键技术

Qwen3.5-2B 的配置中有个重要细节：

```json
"num_attention_heads": 8,      // 8 个 Query head
"num_key_value_heads": 2,      // 但只有 2 个 KV head
```

这就是 **GQA（分组查询注意力）**：
- 8 个 Q head 分成 2 组，每组 4 个 Q head 共享 1 对 K、V head
- KV Cache 从 8 对缩减到 2 对 = **节省 75% 的 KV Cache 显存**

```
MHA (标准多头注意力):    Q₁Q₂  Q₃Q₄  Q₅Q₆  Q₇Q₈
                         K₁V₁  K₂V₂  K₃V₃  K₄V₄  K₅V₅  K₆V₆  K₇V₇  K₈V₈
                         → 8 组 KV，显存需求大

GQA (分组查询注意力):    Q₁…Q₄ 属于组1          Q₅…Q₈ 属于组2
                         K₁V₁（组1共享）          K₂V₂（组2共享）
                         → 2 组 KV，显存减少 75%
```

### 2.4 Qwen3.5-2B 的混合注意力：Linear Attention + Full Attention

这是更进阶的设计。查看配置文件中的 `layer_types`：

```
第 1-3 层:   linear_attention
第 4 层:     full_attention       ← 每 4 层插入 1 个全局注意力
第 5-7 层:   linear_attention
第 8 层:     full_attention
...以此类推，共 24 层，其中 6 层 full_attention + 18 层 linear_attention
```

| 类型 | 计算复杂度 | 每 token KV Cache | 适用场景 |
|:---|:---|:---|:---|
| **Full Attention** | O(n²) | 较大（完整的 K/V） | 需要全局依赖的任务 |
| **Linear Attention** | O(n) | 极小（固定大小的隐状态） | 长序列、高吞吐 |

**Linear Attention 的核心思想**：不存储完整的 K、V 矩阵，而是用一个固定大小的"压缩状态"来近似 Attention 结果。这样无论序列多长，每步的计算量和显存都是常数。

> 这个设计让 Qwen3.5-2B 支持高达 262K token 的上下文窗口，同时保持推理速度。

---

## 3. Prefill vs Decode：推理的两个阶段

### 3.1 一张图理解两个阶段

```
时间轴 →

┌────────────── Prefill 阶段 ──────────────┐  ┌────── Decode 阶段 ──────────┐
│                                            │  │                             │
│  一次性处理所有输入 token（Prompt + 图片）   │  │  逐 token 生成输出            │
│                                            │  │                             │
│  输入: "请回答图中是什么动物？"              │  │  "一" → "只" → "猫" → <eos>  │
│  + 图片的视觉 token (~500 tokens)           │  │                             │
│                                            │  │                             │
│  计算特点：                                 │  │  计算特点：                  │
│  • 计算密集（Compute-bound）               │  │  • 内存密集（Memory-bound）  │
│  • 大量并行矩阵乘法                         │  │  • 每次只做小的矩阵-向量乘法   │
│  • GPU 利用率高                            │  │  • GPU 利用率低（~20-30%）   │
│  • 瓶颈在算力                              │  │  • 瓶颈在显存带宽             │
│                                            │  │                             │
│  TTFT 的一部分                             │  │  吞吐量的主要决定因素          │
└────────────────────────────────────────────┘  └─────────────────────────────┘
```

### 3.2 为什么 Decode 阶段 GPU 利用率低？

Prefill 阶段：做的是 **矩阵 × 矩阵**（GEMM）—— GPU 的天堂。

```
Q: [seq_len, d_model]  @  K^T: [d_model, seq_len]  →  [seq_len, seq_len]
   ↑ 大矩阵乘法，GPU 满载运行
```

Decode 阶段：做的是 **矩阵 × 向量**（GEMV）—— GPU 的地狱。

```
Q: [1, d_model]  @  K^T: [d_model, seq_len]  →  [1, seq_len]
   ↑ 只算 1 个 token，99% 的 GPU 算力闲置，时间全花在"从显存读 KV Cache"上
```

**核心矛盾**：Decode 阶段是 Memory-bound，优化方向不是"算得更快"而是"减少显存读写"。

### 3.3 对本项目的启示

TTFT 受 Prefill 阶段主导 → 优化 Prefill（Flash Attention、算子融合）
吞吐量受 Decode 阶段主导 → 优化显存带宽（KV Cache 布局、量化、更大 batch）

---

## 4. 注意力机制优化：从标准 Attention 到 Flash Attention

### 4.1 标准 Attention 的显存墙

标准 Attention 的实现步骤：

```python
# 伪代码：标准 Attention
Q, K, V = linear_projection(x)         # 形状: [batch, heads, seq, head_dim]

S = Q @ K.transpose(-2, -1)            # Step 1: 计算 Attention Score
                                        # [batch, heads, seq, seq] ← 这个矩阵很大！
S = S / sqrt(head_dim)                 # Step 2: 缩放
P = softmax(S, dim=-1)                 # Step 3: Softmax
                                        # [batch, heads, seq, seq] ← 又一个大矩阵在显存里
O = P @ V                              # Step 4: 加权求和

# 问题：S 和 P 都是 [seq, seq] 大小的矩阵
# seq=4096 时 → 4096² = 16M 个元素 × 4 字节(float32) × 32 head = 2 GB 临时显存！
```

这个 `[seq, seq]` 矩阵要完整地写入显存、读出、再写入，而显存带宽（HBM bandwidth）是 GPU 最稀缺的资源。

### 4.2 Flash Attention：分块计算的革命

Flash Attention 的核心思想极其优雅：**不把整个 Attention 矩阵写进显存**。

```
标准 Attention:
  GPU 显存（HBM）
  ┌──────────────────────────────────┐
  │ Q, K, V  →  [计算 S]  →  [S 矩阵全部存 HBM]  →  [读回 S]  →  [Softmax]  →  [写回 P]  →  ... 
  │                    ↑ 大量 HBM 读写，速度瓶颈 ↑
  └──────────────────────────────────┘

Flash Attention:
  GPU 显存（HBM）           GPU 芯片内（SRAM）
  ┌─────────────────┐      ┌──────────────────────────┐
  │ Q, K, V         │  →   │ 把 Q, K, V 分小块加载     │
  │                  │  ←   │ 在 SRAM 内完成全部计算     │
  │ O (最终结果)     │      │ S → Softmax → O 全部在片内 │
  └─────────────────┘      └──────────────────────────┘
                              ↑ SRAM 带宽是 HBM 的 10-15 倍
```

**Flash Attention 的三个关键技术**：

1. **分块（Tiling）**：把 Q, K, V 切成小块，一次只加载一个块到 SRAM
2. **重计算（Recomputation）**：不存中间的 Attention 矩阵，需要时重新算
3. **Online Softmax**：用数学技巧让 Softmax 也能分块算（在线归一化）

**效果**：显存占用从 O(n²) 降到 O(n)，速度提升 2-4 倍，数学上等价（精确，不是近似）。

### 4.3 Flash Attention 2/3 的演进

| 版本 | 主要改进 |
|:---|:---|
| **Flash Attention 1** | 分块 + 重计算，减少 HBM 读写 |
| **Flash Attention 2** | 更好的并行策略，forward 提速 2× |
| **Flash Attention 3** | 利用 Hopper 架构的新特性（TMA, FP8），再提速 ~1.5-2× |

### 4.4 如何使用？（两步就能在本项目中启用）

```python
# evaluation_wrapper.py 中加两行：

# 方法 1：在模型加载后调用
self._model = AutoModelForImageTextToText.from_pretrained(...).eval()

# 启用 Flash Attention 2（需要 torch >= 2.0, flash_attn 已安装）
import flash_attn
self._model = self._model.to(dtype=torch.bfloat16)  # Flash Attn 需要 bf16/fp16

# 方法 2：更简单——用 torch.compile + SDPA backend
# PyTorch 2.0+ 自带 scaled_dot_product_attention，会自动选择最优实现
```

实际上，PyTorch 2.0 的 `torch.nn.functional.scaled_dot_product_attention` 已经自动集成了 Flash Attention 作为 backend。只要满足条件（bf16/fp16 + CUDA），它会自动启用。

---

## 5. 内存优化：精度、量化与设备映射

### 5.1 精度格式的选择

```
float32 (FP32): 1 位符号 + 8 位指数 + 23 位尾数 = 4 字节
  ├── 范围：±3.4 × 10³⁸，精度约 7 位有效数字
  └── 训练时用，推理太浪费

float16 (FP16): 1 位符号 + 5 位指数 + 10 位尾数 = 2 字节
  ├── 范围：±65504，精度约 3-4 位有效数字
  └── 范围可能溢出（梯度 > 65504 就炸了）

bfloat16 (BF16): 1 位符号 + 8 位指数 + 7 位尾数 = 2 字节
  ├── 范围：和 FP32 一样大！精度略低但范围不丢
  └── 推理的最优选择（本项目用的就是 bfloat16）
```

本项目的配置中：

```json
"dtype": "bfloat16"
```

```python
# evaluation_wrapper.py 第 101 行
dtype=torch.bfloat16,
```

**为什么 bfloat16 是推理的甜点？**
- 范围与 FP32 相同（8 位指数），不会溢出
- 精度损失在推理中几乎不可察觉
- 节省 50% 显存和带宽

### 5.2 量化（Quantization）：进一步压缩

| 方案 | 每参数位数 | 2B 模型大小 | 精度影响 |
|:---|:---|:---|:---|
| FP32 | 32 bit | ~8 GB | 基准 |
| BF16/FP16 | 16 bit | ~4 GB | 几乎无损 |
| INT8 | 8 bit | ~2 GB | 极小下降 |
| INT4 (GPTQ/AWQ) | 4 bit | ~1 GB | 略有下降 |
| FP8 (H100 专用) | 8 bit | ~2 GB | 极小下降 |

**量化时机选择**：

```
Post-Training Quantization (PTQ):
  训练完成 → 直接量化 → 推理
  优点：简单，不需要训练数据
  典型工具：GPTQ, AWQ, bitsandbytes

Quantization-Aware Training (QAT):
  训练时模拟量化 → 微调恢复精度 → 推理
  优点：精度损失更小
  代价：需要额外训练
```

### 5.3 设备映射（Device Map）

本项目中的 `device_map="auto"` 做了什么？

```python
# evaluation_wrapper.py 第 102 行
device_map=self.device,  # "auto" 会让 accelerate 自动分配层
```

```
单 GPU 场景:
  ┌──────────────────────────┐
  │        GPU (12GB)         │
  │  ├── Embedding (~200MB)   │
  │  ├── Layer 0-23 (~4GB)   │
  │  ├── LM Head (~200MB)    │
  │  └── KV Cache (~动态)     │
  └──────────────────────────┘

多 GPU 场景 (device_map="auto"):
  ┌────────────┐  ┌────────────┐
  │   GPU 0    │  │   GPU 1    │
  │  Layer 0-11│  │  Layer 12-23│
  │  Embedding │  │  LM Head   │
  └────────────┘  └────────────┘

CPU offload 场景 (device_map="auto" + 显存不够):
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │   GPU 0    │  │   GPU 1    │  │    CPU     │
  │  Layer 0-15│  │  Layer 16-20│  │  Layer 21-23│ ← 放不下就到 CPU
  └────────────┘  └────────────┘  └────────────┘
```

---

## 6. TTFT 优化：首 Token 延迟的攻防战

### 6.1 TTFT 的完整时间分解

回到项目文档中的分析（`generate_with_transformers_详解.md`）：

```
882ms TTFT 是怎么分配的？

├── 🔴 Thread 创建/销毁 + Streamer 通信开销     ~380ms (43%)
│   └── 每次推理都 new 一个 Thread + TextIteratorStreamer！
│
├── 🟡 Prefill 计算（标准 Attention）           ~400ms (45%)
│   └── 处理 ~500 个输入 token + 图片特征
│
├── 🟢 图片编码 + Tokenization                  ~50ms  (6%)
│
├── 🟢 GPU 数据传输                              ~20ms  (2%)
│
└── 🟡 第 1 个 token decode                     ~30ms  (3%)
```

### 6.2 优化 1：消除 Thread 开销（最简单、最有效）

**问题**：当前代码每次推理都创建 `threading.Thread` + `TextIteratorStreamer`。

```python
# 当前实现（每次推理一把新的 Thread）
worker = threading.Thread(target=_run_generate, daemon=True)
worker.start()
for chunk in streamer:  # 主线程阻塞等待
    ...
worker.join()
```

**根本原因**：Thread 不是为了 Streamer 而生的——是为了**精确计时 TTFT**！Streamer 在 generate 过程中回调，主线程能精确记录第一个 chunk 到达的时间。

**优化方案**：用 `StoppingCriteria` 替代 Thread+Streamer。

```python
# 优化后（已经在 evaluation_wrapper.py 中实现）
first_token_time = {"value": None}

class FirstTokenTimer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        if first_token_time["value"] is None:
            first_token_time["value"] = time.perf_counter()
        return False  # 不终止生成

# 直接在 generate() 中同步调用
generation_kwargs = {
    ...
    "stopping_criteria": [FirstTokenTimer()],
}

output_ids = self._model.generate(**generation_kwargs)
self._sync_device()
```

**收益**：省掉 ~300-400ms 的 Thread 调度、创建销毁、GIL 竞争开销。

### 6.3 优化 2：Flash Attention 加速 Prefill

Prefill 阶段处理所有输入 token，正是 Flash Attention 最能发挥作用的场景。将标准 Attention 换成 Flash Attention，Prefill 可以提速 2-3 倍。

**收益**：Prefill 从 ~400ms → ~150-200ms。

### 6.4 优化 3：静态 Padding vs 动态 Padding + 连续批处理

当前每次只处理 1 个样本（batch_size=1），prefill 时 GPU 利用率不高。但这不是本轮的重点，先跳过。

---

## 7. 吞吐量优化：让 GPU 持续忙碌

### 7.1 吞吐量的计算公式

```
吞吐量 = 有效解码 token 数 / (总时间 - TTFT)

       = (token_count - 1) / (elapsed_seconds - ttft_seconds)
```

见 `benchmark_public.py` 的 `compute_throughput` 函数（第 145 行）。

### 7.2 提升吞吐量的主要手段

#### 7.2.1 Continuous Batching（连续批处理）

传统推理：一次处理一个请求，batch 固定。

```
请求1: ████████████████████████████████
请求2:                                 ██████████████████████████
请求3:                                                         ██████████████████
       ←──────────────── 串行，GPU 大量空闲 ────────────────→
```

Continuous Batching：动态组合请求，随时增减。

```
Batch:  [请求1, 请求2, 请求3]  [请求1, 请求3]  [请求2, 请求3]
         ↑ 趁大家都在 prefill/decode，GPU 跑满
```

**关键**：请求1 短，先结束；请求 2 和 3 还在继续，但它们俩可以一起继续 decode。

#### 7.2.2 KV Cache 显存管理

KV Cache 碎片化是 Continuous Batching 的头号敌人：

```
显存视图（碎片化）:
[A的KV][空闲][B的KV][C的KV][空闲][D的KV]...
       ↑ 虽然有空闲，但放不下 E 的大 KV Cache → 浪费

显存视图（连续管理，类似 vLLM 的 PagedAttention）:
[Block0: A0+B0][Block1: A1+C0][Block2: B1+D0][Block3: A2+E0]...
 ↑ 像操作系统分页一样管理 KV Cache，零碎片
```

#### 7.2.3 增大有效 Batch Size

```python
# 用 vLLM 或 TGI 替代原生 transformers
# 它们内置了 Continuous Batching + PagedAttention
# 吞吐量提升通常 10-30 倍（是的，没写错）
```

---

## 8. 高级话题：CUDA Graph、Kernel Fusion 与自定义算子

### 8.1 CUDA Graph

**问题**：每次 kernel launch 都有 CPU→GPU 调度开销。Decode 阶段每次只生成一个新 token，这个开销占比很大。

**解决方案**：CUDA Graph 把一系列 GPU 操作"录制"下来，之后像放录像一样一次提交整段操作。

```
不用 CUDA Graph:
  CPU: launch_kernel1 → 等待 → launch_kernel2 → 等待 → launch_kernel3 → ...
  GPU:        run1            run2            run3

用 CUDA Graph:
  CPU: 创建 Graph: [kernel1 → kernel2 → kernel3]  ← 只做一次
  CPU: replay_graph  ← 每次只需一次 CPU→GPU 通信
  GPU: run1 → run2 → run3  ← 无缝衔接，零调度延迟
```

**适用场景**：batch_size 固定、序列长度变化小的情况（Decode 阶段完全满足）。

### 8.2 Kernel Fusion（算子融合）

多个小操作合成一个大操作，减少显存读写：

```
融合前:
  x → LayerNorm → 写 HBM → 读 HBM → Linear → 写 HBM → 读 HBM → GELU → ...
      ↑ kernel1              ↑ kernel2              ↑ kernel3
      每个 kernel 都要读/写 HBM

融合后 (如 FlashAttention 或 fused MLP):
  x → [LayerNorm + Linear + GELU 融合] → 写 HBM
      ↑ 一个 kernel，中间结果都在寄存器/SRAM
```

**常见的融合机会**：
- `LayerNorm + Linear`（尤其 QKV 投影）
- `Linear + Activation + Linear`（FFN 的 mlp 块）
- `Softmax + Dropout + Attention output`

### 8.3 自定义算子

比赛中允许用自定义 CUDA kernel 替换 PyTorch 实现。常见目标：

```
可替换的算子:
  ├── matmul（尤其小矩阵的 GEMV，可用 Triton 写专门优化版本）
  ├── attention（各种 Flash Attention 变体）
  ├── rms_norm（Qwen 用 RMSNorm 而非 LayerNorm）
  ├── silu / gelu 激活函数
  ├── rope（旋转位置编码）
  └── sampling（top-p, top-k, temperature）
```

**Triton**：Python 写的 GPU kernel 语言，比写 CUDA C 简单很多，性能接近手写 CUDA。

---

## 9. 本项目实战：从 882ms 到 350ms 的优化路线图

### 9.1 快速见效（改几行代码，立竿见影）

| 优先级 | 优化项 | 预期收益 | 代码改动量 |
|:---:|:---|:---:|:---:|
| **P0** | 消除 Thread + Streamer（用 StoppingCriteria） | TTFT -380ms | 中等（已实现） |
| **P0** | 启用 Flash Attention 2 | TTFT -150ms | 极简（2 行） |
| **P1** | PyTorch `torch.compile` | TTFT -10~20% | 极简（1 行） |
| **P1** | 图片预处理缓存（resize 固定大小） | TTFT -20ms | 简单 |

### 9.2 中等投入（改写部分推理逻辑）

| 优先级 | 优化项 | 预期收益 | 说明 |
|:---:|:---|:---:|:---|
| **P2** | 用 `torch.inference_mode()` 替代 `torch.no_grad()` | 微小加速 | 比 no_grad 更快，禁止所有 autograd 操作 |
| **P2** | CUDA Graph 封装 Decode 阶段 | 吞吐量 +20~40% | 录制 graph 后 replay |
| **P2** | 预热 + 固定 batch_size，消除首次 kernel launch 开销 | TTFT -50ms | 稳定性能 |

### 9.3 高投入（替换推理后端）

| 优先级 | 优化项 | 预期收益 | 说明 |
|:---:|:---|:---:|:---|
| **P3** | 替换为 vLLM 后端 | 吞吐量 +10-30× | 内置 PagedAttention + Continuous Batching |
| **P3** | 用 ONNX + TensorRT-LLM | TTFT -40%，吞吐量 +3-5× | 需要模型导出，工作量大 |

### 9.4 本项目当前已实现的优化

对比 `evaluation_wrapper.py`（最新版）和 `evaluation_wrapper_annotated.py`（原始注释版）：

| 优化 | 说明 |
|:---|:---|
| ✅ 同步推理 | 移除了 Thread + Streamer，改用 `StoppingCriteria` 计时 |
| ✅ 设备同步 | `_sync_device()` 确保 GPU 计算完成后才计时 |
| ✅ CUDAGraph 识别 | 接口干净，可直接在其上封装 CUDAGraph |
| ✅ bfloat16 精度 | 节省 50% 显存 |

---

## 10. 延伸学习资源

### 10.1 必读论文（按学习顺序）

| 论文 | 核心贡献 | 难度 |
|:---|:---|:---:|
| **Attention Is All You Need** (2017) | 原始 Transformer | ⭐ 基础 |
| **Flash Attention** (2022) | IO-aware 的精确注意力 | ⭐⭐ 进阶 |
| **Flash Attention 2** (2023) | 更好的并行策略 | ⭐⭐ 进阶 |
| **GQA: Training Generalized Multi-Query** (2023) | 分组查询注意力 | ⭐ 基础 |
| **PagedAttention (vLLM)** (2023) | 操作系统式 KV Cache 管理 | ⭐⭐⭐ 深入 |
| **Llama 3 / Qwen 3 技术报告** | 最新架构设计（MLA, MoE, etc.） | ⭐⭐ 进阶 |

### 10.2 关键开源项目

| 项目 | 用途 |
|:---|:---|
| [vLLM](https://github.com/vllm-project/vllm) | 高吞吐推理引擎，PagedAttention 原实现 |
| [SGLang](https://github.com/sgl-project/sglang) | 下一代推理引擎，RadixAttention |
| [Flash Attention](https://github.com/Dao-AILab/flash-attention) | 官方 Flash Attention 实现 |
| [Triton](https://github.com/triton-lang/triton) | Python GPU Kernel 语言 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU/边缘推理（量化技术参考） |

### 10.3 本项目相关文件速查

```
D:\AI+TEST\dndx_participant-v1.1\
├── evaluation_wrapper.py              ← 核心：你需要优化这个文件
├── evaluation_wrapper_annotated.py    ← 逐行注释版，学习用
├── benchmark_public.py                ← 自测脚本，可看到完整评测流程
├── run.py                             ← benchmark_public.py 的注释版
├── docs\generate_with_transformers_详解.md  ← 零基础入门文档
├── Qwen3.5-2B\config.json             ← 模型配置，包含所有架构参数
└── README.md                          ← 比赛规则

D:\AI+TEST\transformer_task\
├── analysis_answer.py                 ← 答案偏差分析（包含 MSE、KL 散度）
├── test.py                            ← TSV 数据集读取和分析
└── trans_image.py                     ← Base64 图片解码
```

---

## 附录 A：Qwen3.5-2B 架构参数速查

| 参数 | 值 | 含义 |
|:---|:---|:---|
| `num_hidden_layers` | 24 | Transformer 层数 |
| `hidden_size` | 2048 | 隐藏层维度 |
| `num_attention_heads` | 8 | Query head 数量 |
| `num_key_value_heads` | 2 | KV head 数量（GQA） |
| `head_dim` | 256 | 每个 head 的维度 |
| `intermediate_size` | 6144 | FFN 中间层维度 |
| `vocab_size` | 248320 | 词表大小 |
| `max_position_embeddings` | 262144 | 最大上下文长度 |
| `full_attention_interval` | 4 | 每 4 层插入 1 个全局注意力 |
| `linear_conv_kernel_dim` | 4 | Linear Attention 的卷积核大小 |
| `rope_theta` | 10M | RoPE 的基础频率 |
| Vision `depth` | 24 | 视觉编码器层数 |
| Vision `hidden_size` | 1024 | 视觉编码器隐层维度 |
| Vision `patch_size` | 16 | 图片分块大小（16×16 像素） |

## 附录 B：关键概念速查表

| 术语 | 一句话解释 |
|:---|:---|
| **TTFT** | Time To First Token，从接收到请求到第一个输出 token 的时间 |
| **TPOT** | Time Per Output Token，每个输出 token 的平均生成时间 |
| **KV Cache** | 缓存历史的 Key 和 Value 矩阵，避免重复计算 |
| **Prefill** | 一次性处理所有输入 token 的阶段（计算密集） |
| **Decode** | 逐 token 生成的阶段（内存密集） |
| **GQA** | Grouped Query Attention，多个 Q head 共享一对 KV head |
| **MQA** | Multi-Query Attention，所有 Q head 共享 1 对 KV head（GQA 的特例） |
| **Flash Attention** | 分块 + 重计算，避免将 Attention 矩阵写入 HBM |
| **Continuous Batching** | 动态增删 batch 中的请求，GPU 不闲置 |
| **PagedAttention** | 用分页机制管理 KV Cache，消除碎片 |
| **torch.compile** | PyTorch 2.0 的 JIT 编译器，自动做算子融合和优化 |
| **CUDA Graph** | 录制 GPU 操作序列，消除单次 kernel launch 的调度开销 |
| **GEMM** | General Matrix Multiply，矩阵乘法的统称 |
| **GEMV** | General Matrix-Vector Multiply，矩阵-向量乘法（Decode 阶段的主要运算） |
| **HBM** | High Bandwidth Memory，GPU 上的大容量显存（带宽 ~2 TB/s） |
| **SRAM** | GPU 芯片内的高速缓存（带宽 ~19 TB/s，但容量仅 ~40MB） |

---

> **学习建议**：建议按目录顺序阅读，每读完一章就对照本项目的实际代码加深理解。KV Cache（第 2 章）和 Prefill/Decode（第 3 章）是理解所有后续优化的基础，务必吃透。Flash Attention（第 4 章）是单次推理加速的最有效手段，建议优先实践。
