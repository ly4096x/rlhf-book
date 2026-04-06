<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "指令微调"
prev-url: "04-instruction-tuning"
page-title: Reward Models
search-title: "第5章：Reward Models"
next-chapter: "强化学习"
next-url: "06-policy-gradients"
lecture-video: "https://youtu.be/4gIwiSPmQkU"
lecture-label: "第2讲：IFT、Reward Modeling 与拒绝采样（第4、5、9章）"
---

# Reward Modeling

Reward models 是现代 RLHF 方法的核心，负责学习复杂的人类偏好。
正是它们使我们的模型能够从难以明确指定的信号中进行学习。
它们将数据中的复杂特征压缩成一种可用于下游训练的表示——这是一种再次展现现代深度学习强大能力的"魔法"。
这些模型充当核心优化的代理目标，将在后续章节中加以研究。
如 @fig:rm-role-in-rlhf 所示，reward model 扮演着类似于标准 RL 环境的角色，为 agent 提供学习信号，但与固定环境不同的是，我们可以从人类偏好中学习得到它。

Reward models 在强化学习研究中被广泛用作环境奖励的代理 [@sutton2018reinforcement]。
Reward models 以其现代形式被提出，作为研究值对齐问题的工具 [@leike2018scalable]。
这些模型通常接受某种形式的输入并输出单一的 scalar reward 值。
该奖励可以有多种形式——在传统 RL 问题中，它试图近似问题的精确环境奖励，但我们将在 RLHF 中看到，reward models 实际上输出的是某个输入"质量较高"的概率（即在成对偏好关系中被选中的答案）。
针对 RLHF 的 reward modeling 实践与逆强化学习密切相关，后者的问题是在给定行为轨迹的情况下近似 agent 的奖励函数 [@ng2000algorithms]，以及深度强化学习的其他领域。
高层次的问题表述相同，但实现方式和关注领域完全不同，因此它们通常被视为完全独立的研究领域。

最常见的 reward model，通常称为 Bradley-Terry reward model，也是本章的主要关注点，它预测一段文本接近训练比较中"偏好"文本的概率。
本节后续部分还将对其与 Outcome Reward Models（ORM）、Process Reward Models（PRM）以及其他类型的 reward models 进行比较。

*在本章中，我们用 $x$ 表示提示（prompt），用 $y$ 表示补全（completion）。这种符号在语言模型文献中很常见，其中方法作用于完整的 prompt-completion 对，而非单个 token。*

![RLHF 中的 reward model 扮演着标准 RL 中返回奖励的环境组件角色。关键区别在于，在 RLHF 中，我们可以从人类偏好中控制并学习这个奖励函数，而不是由环境固定给定。](images/rlhf-overview.png){#fig:rm-role-in-rlhf}

## 训练 Bradley-Terry Reward Model

Reward model 的标准实现源自 Bradley-Terry 偏好模型 [@BradleyTerry]。
训练标准 RLHF reward model 有两种流行的表达形式——它们在数学上是等价的。
首先，Bradley-Terry 偏好模型定义了在对两个项目 $i$ 和 $j$ 进行成对比较时，评判者偏好 $i$ 胜于 $j$ 的概率：

$$P(i > j) = \frac{p_i}{p_i + p_j}.$$ {#eq:bradterry}

Bradley-Terry 模型假设每个项目具有一个潜在强度 $p_i > 0$，且观测到的偏好是这些底层强度的带噪反映。
通常对 Bradley-Terry 模型使用无界分数进行重参数化，其中 $p_i = e^{r_i}$，由此得到以下形式：

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).$$ {#eq:bradterry_unbounded}

只有分数之间的差值才重要：对所有 $r_i$ 加上相同的常数不会改变 $P(i > j)$。
这些形式并非自然法则，而是对人类偏好的有用近似，在 RLHF 中通常效果良好。

为了训练 reward model，我们必须构造一个满足上述关系的损失函数。
在实践中，这通过将语言模型转换为输出 scalar 分数的模型来实现，通常借助一个小型线性头，从模型的最终隐藏状态生成单一奖励值。
给定一个 prompt $x$ 和两个采样的 completion $y_1$ 与 $y_2$，我们用 reward model $r_\theta$ 对两者进行打分，并将条件分数写为 $r_\theta(y_i \mid x)$。

reward model 将 $y_1$ 优于 $y_2$ 的概率表示为：

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.$$ {#eq:bradterryrm}

我们将偏好的 completion 记为 $y_c$（chosen），将被拒绝的 completion 记为 $y_r$。

由此得到的损失函数鼓励 reward model 为人类偏好的 completion 分配比被拒绝的 completion 更高的分数，使用 sigmoid 函数将分数差转换为概率。
通过最大化上述函数的对数似然（或等价地最小化负对数似然），我们可以得到训练 reward model 的以下损失：

$$
\begin{aligned}
\theta^* = \arg\max_\theta P(y_c > y_r \mid x) &= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \arg\max_\theta \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\ 
&= \arg\max_\theta \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \arg\max_\theta \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \\
&= \arg\min_\theta - \log \left( \sigma \left(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\right) \right)
\end{aligned}
$$ {#eq:bradterryrm_deriv}

第一种形式是上面推导的 log-sigmoid 表达式，见 [@ouyang2022training] 等工作：
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

第二种是用 softplus 函数 $\log(1+e^x)$ 表达的数学等价形式，见 [@askell2021general] 等工作：
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

两者的等价性可通过令 $\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)$ 并利用 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$ 来验证，由此得出 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)$。
两种形式均出现在 RLHF 文献中。

![训练偏好 reward model 需要成对的 chosen 与 rejected completion。模型在序列结束（EOS）token 处为每个 completion 计算 scalar 分数，对比损失仅取决于两者之间的分数差。](images/pref_rm_training.png){#fig:pref_rm_training}

## 默认 Reward Model 架构

Reward models 最常见的实现方式是通过类似 Transformers 的 `AutoModelForSequenceClassification` 的抽象，该抽象在语言模型上附加一个小型线性头，并在训练或推理时为 prompt-completion 对生成 scalar reward 分数。
在推理时，模型以模型的单一 logit 形式输出*该文本片段被选中的相对可能性*。

也存在其他实现选项，例如直接从最终 embedding 提取线性层，但在开放工具中较为少见。

## 实现示例

实现 reward modeling 损失相当简单。
更多的实现挑战在于设置独立的数据加载器和推理流水线。
给定包含 tokenized、chosen 和 rejected prompt 及 completion 的正确数据加载器，损失实现如下：
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected 包含 prompt token x 以及各自的
# completion token（y_c 或 y_r），reward model 对其进行联合打分。
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

从更宏观的角度来看，这通常处于一个因果语言模型（从左到右逐个生成 token、在所有前序 token 的条件下预测每个 token 的模型）中，该模型附加了一个额外的头（使用上述损失进行学习），将最终隐藏状态转换为输入的分数。
代码接收标准的 transformer 输入——`input_ids`（tokenized 文本）和 `attention_mask`（标记真实 token 与填充 token）——并提取最后一个真实 token 处的隐藏状态（模型对输入的内部表示），然后通过线性层生成 scalar reward。
该模型的结构如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
