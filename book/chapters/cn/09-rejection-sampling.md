<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "直接对齐"
prev-url: "08-direct-alignment"
page-title: Rejection Sampling
search-title: "第9章：Rejection Sampling"
next-chapter: "什么是 Preferences"
next-url: "10-preferences"
lecture-video: "https://youtu.be/4gIwiSPmQkU"
lecture-label: "Lecture 2: IFT, Reward Modeling, Rejection Sampling (Chap. 4, 5, & 9)"
---

# Rejection Sampling

Rejection Sampling（RS）是 preference fine-tuning 中使用最广泛却记录最少的方法之一。
许多知名的 RLHF 论文将其作为 training pipeline 的核心组件，但目前既没有规范的实现，也没有对其为何效果如此出色的充分解释。
RS 可应用于 training pipeline 的多个环节——instruction fine-tuning 之后、基于 RL 的优化之后，甚至 RLVR 之后——使其成为一种用途广泛却难以定位的工具。
加之其文档记录不足的特性，这正是它出现在核心优化方法末尾的原因。

Rejection sampling 的运作方式是：生成一批新的候选 completions，依据训练好的 reward model 对其进行过滤，然后仅在排名靠前的 completions 上对原始模型进行 fine-tuning（损失函数与 instruction tuning 相同）。

该名称源自计算统计学 [@gilks1992adaptive]，其目标是从复杂分布中采样，但没有直接的方法可以做到这一点。
为此，人们从一个更易建模的简单分布中采样，并使用启发式方法检验样本是否可接受。
对于 language models，目标分布是对 prompt 的高质量 completions，过滤器是 reward model，采样分布则是当前模型。

WebGPT [@nakano2021webgpt]、Anthropic 的 Helpful and Harmless agent [@bai2022training]、OpenAI 关于 process reward models 的经典论文 [@lightman2023let]、Llama 2 Chat models [@touvron2023llama] 以及其他开创性工作都使用了这一基线方法；近期的工作对其进行了更正式的阐述（例如，RAFT [@dong2023raft] 将其应用于多模态的对齐，以及 Statistical Rejection Sampling Optimization（RSO）[@liu2023statistical]，该工作系统地概述了 rejection sampling 与其他 preference learning 目标的关系）。

*在本章中，我们用 $x$ 表示 prompts，用 $y$ 表示 completions。这一符号约定在 language model 文献中十分常见，相关方法作用于完整的 prompt-completion 对，而非单个 tokens。*

## 训练过程分步详解

Rejection sampling 总体上遵循以下几个阶段。

0. **Prompt 与 reward model 的选择：** 首先，需要相对于训练的其他阶段，选定用于训练的 prompts。最简单的方法是复用第一阶段 SFT/IFT 中的所有 prompts，但这可能导致一定程度的过拟合。在进行 rejection sampling 之前，还必须已经训练好一个 reward model（详见第 5 章）。
1. **从起始 checkpoint 生成 completions：** 接下来，需要用待优化模型对所选 prompts 生成 completions。这一步涉及调整多种设置，如 sampling temperature、top-p、最大序列长度、每个 prompt 的 completions 数量等。
2. **用 reward model 选取最优 completions：** 所有 completions 由 reward model 排序。此阶段还可能包含去重操作，以确保每个 prompt 只保留一个 completion，但许多此类设计选择最终取决于实证消融研究。
3. **在最优 completions 上进行 SFT：** Rejection sampling 的最后一步是以起始 checkpoint 为基础，在所选 completions 上进行 instruction fine-tuning。

Rejection sampling 过程的可视化概览如下图 @fig:rs-overview 所示。

![Rejection sampling 概览。](images/rejection-sampling.png){#fig:rs-overview}

关于使用哪些 prompts、如何选择 reward model、如何安排 rejection sampling 的顺序等具体细节，在文献中尚无充分记录。
本章提供方法概述，将进一步的实验留给读者自行探索。

### 1. 生成 Completions

为每个 prompt 生成一组多个候选 completions，我们将 $M$ 个 prompts 定义为一个向量：

$$X = [x_1, x_2, ..., x_M]$$ {#eq:rs_prompt_vector}

这些 prompts 可以来自多种来源，但最常见的是来自 instruction training set。

对于每个 prompt $x_i$，我们生成 $N$ 个 completions。可以用矩阵表示如下：

$$Y = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,N} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,N}
\end{bmatrix}$$ {#eq:rs_completion_matrix}

其中 $y_{i,j}$ 表示第 $i$ 个 prompt 的第 $j$ 个 completion。
第 $i$ 行对应单个 prompt $x_i$ 及其 $N$ 个候选 completions；第 $j$ 列对应所有 prompts 的第 $j$ 个采样 completion。

### 2. 对 Completions 评分

现在，将所有这些 prompt-completion 对输入 reward model，得到一个 rewards 矩阵。
将 rewards 表示为矩阵 $R$：

$$R = \begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,N} \\
r_{2,1} & r_{2,2} & \cdots & r_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
r_{M,1} & r_{M,2} & \cdots & r_{M,N}
\end{bmatrix}$$ {#eq:rs_reward_matrix}

每个 reward $r_{i,j}$ 通过将 completion $y_{i,j}$ 及其对应 prompt $x_i$ 输入 reward model $\mathcal{R}$ 来计算：

$$r_{i,j} = \mathcal{R}(y_{i,j} \mid x_i)$$ {#eq:rs_reward_computation}

选取用于训练的最优 completions 有多种方法。

为了将基于 reward 矩阵选取最优 completions 的过程形式化，可以定义一个作用于 reward 矩阵 $R$ 的选择函数 $S$。

#### 每个 Prompt 的最优结果

第一种候选选择函数取每个 prompt 的最大 reward。

$$S(R) = [\arg\max_{j} r_{1,j}, \arg\max_{j} r_{2,j}, ..., \arg\max_{j} r_{M,j}]$$ {#eq:rs_selection_per_prompt}

函数 $S$ 返回一个索引向量，其中每个索引对应 $R$ 中每一行 reward 最大的列。
然后用这些索引选取对应的 completions：

$$Y_{chosen} = [y_{1,S(R)_1}, y_{2,S(R)_2}, ..., y_{M,S(R)_M}]$$ {#eq:rs_chosen_completions}


#### 全局最优 Pairs
另一种方法是从全部集合中选取排名前 K 的 prompt-completion pairs。
首先将 reward 矩阵 R 展平为一个向量：

$$R_{flat} = [r_{1,1}, r_{1,2}, ..., r_{1,N}, r_{2,1}, r_{2,2}, ..., r_{2,N}, ..., r_{M,1}, r_{M,2}, ..., r_{M,N}]$$ {#eq:rs_flattened_rewards}

$R_{flat}$ 向量的长度为 $M \times N$，其中 $M$ 是 prompts 数量，$N$ 是每个 prompt 的 completions 数量。

然后定义选择函数 $S_K$，选取 $R_{flat}$ 中 K 个最大值的索引：

$$S_K(R_{flat}) = \text{argsort}(R_{flat})[-K:]$$ {#eq:rs_topk_selection}

其中 $\text{argsort}$ 返回按升序排列的索引，取最后 K 个索引即得到 K 个最大值。

为获得所选 completions，需将这些展平的索引映射回原始 completion 矩阵 $Y$。
对于从零开始的展平索引 $k$，可通过 $i = \lfloor k / N \rfloor + 1$ 和 $j = (k \bmod N) + 1$ 恢复对应的 prompt-completion pair $(i,j)$。

#### 选择示例
考虑如下情况：有五个 prompts 和四个 completions。
我们将展示两种基于 reward 选择 completions 的方法。

$$R = \begin{bmatrix}
0.7 & 0.3 & 0.5 & 0.2 \\
0.4 & 0.8 & 0.6 & 0.5 \\
0.9 & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & 0.8 & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_matrix}

首先是**每个 prompt 的最优结果**。直观地，可以如下高亮 reward 矩阵：

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \textbf{0.6}
\end{bmatrix}$$ {#eq:rs_example_per_prompt}

使用 argmax 方法，为每个 prompt 选取最优 completion：

$$S(R) = [\arg\max_{j} r_{i,j} \text{ for } i \in [1,5]]$$ {#eq:rs_example_selection_formula}

$$S(R) = [1, 2, 1, 3, 4]$$ {#eq:rs_example_selection_result}

这意味着我们将选取：

- 对于 prompt 1：completion 1（reward 0.7）
- 对于 prompt 2：completion 2（reward 0.8）
- 对于 prompt 3：completion 1（reward 0.9）
- 对于 prompt 4：completion 3（reward 0.8）
- 对于 prompt 5：completion 4（reward 0.6）

现在是**全局最优**。
高亮全局前五名的 completion pairs。

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & \textbf{0.7} \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_top_overall}


首先展平 reward 矩阵：

$$R_{flat} = [0.7, 0.3, 0.5, 0.2, 0.4, 0.8, 0.6, 0.5, 0.9, 0.3, 0.4, 0.7, 0.2, 0.5, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6]$$ {#eq:rs_example_flattened}

然后选取五个最大值的索引：
$$S_5(R_{flat}) = [8, 5, 14, 0, 11]$$ {#eq:rs_example_topk_result}

将这些索引映射回原始矩阵：

- 索引 8 → prompt 3，completion 1（reward 0.9）
- 索引 5 → prompt 2，completion 2（reward 0.8）
- 索引 14 → prompt 4，completion 3（reward 0.8）
- 索引 0 → prompt 1，completion 1（reward 0.7）
- 索引 11 → prompt 3，completion 4（reward 0.7）

#### 实现示例

以下代码片段展示了选择方法的可能实现方式。

```python
import numpy as np

x = np.random.randint(10, size=10)
print(f"{x=}")
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
print(f"{x_sorted=}")

# first way to recover the original array
i_rev = np.zeros(10, dtype=int)
i_rev[sorted_indices] = np.arange(10)
np.allclose(x, x_sorted[i_rev])

# second way to recover the original array
np.allclose(x, x_sorted[np.argsort(sorted_indices)])
```

### 3. Fine-tuning

选出 completions 后，在当前版本的模型上执行标准的 instruction fine-tuning。
更多详情可参见 [instruction tuning 章节](https://rlhfbook.com/c/04-instruction-tuning)。

## 实现细节

执行此训练的核心超参数非常直观：

- **Sampling 参数**：Rejection sampling 直接依赖于模型生成的 completions。常见的设置包括温度大于零，例如在 0.7 到 1.0 之间，以及对 top-p 或 top-k sampling 等参数进行相应调整。
- **每个 Prompt 的 Completions 数量**：成功的 rejection sampling 实现通常为每个 prompt 生成 10 到 30 个或更多的 completions。使用过少的 completions 会使训练产生偏差和/或噪声。
- **Instruction tuning 细节**：目前尚未公开 rejection sampling 阶段 instruction tuning 的明确训练细节。很可能使用的设置与模型初始 instruction tuning 阶段略有不同。
- **异构模型生成**：部分 rejection sampling 实现中，生成内容来自多个模型，而非仅限于待训练的当前模型。关于如何执行此操作的最佳实践尚未建立。
- **Reward model 训练**：所使用的 reward model 将对最终结果产生重大影响。有关 reward model 训练的更多资源，请参见[相关章节](https://rlhfbook.com/c/05-reward-models)。

在进行批量 reward model inference 时，可以按长度对 tokenized completions 进行排序，使批次长度相近。
这样可以减少在 padding tokens 上运行 inference 的需要，以较小的实现复杂度换取更高的吞吐量。

## 相关方法：Best-of-N Sampling

Best-of-N（BoN）是 rejection sampling 的近亲方法，同样遵循生成-评分流程，但**不**对模型在所选 completions 上进行 fine-tuning。
相反，BoN 在 inference 时计算对某个静态 prompt（或一组 prompts）的最优 completion，相关技术常用于聊天模型的"Pro"版本中，这些版本会花费额外计算资源来回答用户的查询。

Best-of-N sampling 通常作为 RLHF 训练方法的基线进行比较。
重要的是要记住，BoN *不会*修改底层模型，而是一种 sampling 技术。
因此，在某些情况下，将 BoN sampling 与 PPO 等在线训练方法进行比较仍然是有意义的。
例如，相对于任何其他 policy，仍然可以测量运行 BoN sampling 时的 KL distance。

这里我们将展示，当对单个 prompt 使用简单 BoN sampling 时，上述两种选择标准是等价的。

设 R 为单个 prompt 的 reward 向量，包含 N 个 completions：

$$R = [r_1, r_2, ..., r_N]$$ {#eq:rewards_vector}

其中 $r_j$ 表示第 j 个 completion 的 reward。

使用 argmax 方法，为该 prompt 选取最优 completion：

$$S(R) = \arg\max_{j \in [1,N]} r_j$$ {#eq:selection_function}

使用 $K=1$ 的 top-K 方法可归约为同一方法，这是常见做法。
