---
title: "LLM post-training 的 ML 基础"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "post-training 的 ML 基础"
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #84A98C; }
  :root {
    --colloquium-progress-fill: #84A98C;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
    letter-spacing: 0;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 课程先修：LLM post-training 的 ML 基础

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">从交叉熵到偏好与强化学习——第 1 讲之前的一份简短、可选的复习。</p>

<p class="colloquium-title-note" style="font-size: 0.6em; opacity: 0.7;">本幻灯片在 GLM 5.2 的协助下起草。</p>

---

## 本次复习涵盖的内容

**不要求具备任何强化学习基础。** 我们假设你熟悉入门级 ML 中的 loss、梯度和概率，并能熟练使用 PyTorch 张量。本讲是对自回归语言模型以及课程中用到的几个核心概念的复习：

- **语言模型**——log-prob 的自回归分解、LM head、softmax、交叉熵，以及张量形状的剖析
- **优化**——梯度下降、训练循环、预训练 → SFT
- **概率**——KL divergence（含熵）、sigmoid 与成对似然
- **强化学习框架**——MDP 设定与 RL 的目标

---

<!-- layout: section-break -->

## 语言模型

---

<!-- columns: 45/55 -->
## 什么是语言模型

语言模型以**自回归**的方式学习由 **token**（词 / 子词）组成的序列的联合概率——每一次预测都依赖于它之前的 token。

给定 $x = (x_1, \ldots, x_T)$，模型将序列概率分解为：

$$P_\theta(x) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, \ldots, x_{t-1}).$$

现代 LM（ChatGPT、Claude、Gemini）都是基于**自注意力**（self-attention）的 **decoder-only Transformer**——每个位置只关注自身及其之前的位置。

|||

![Next-token 预测：decoder-only 语言模型根据前缀，一次预测一个 token。](assets/pretraining_next_token_tikz.png)
<!-- cite-right: Vaswani2017AttentionIA -->

---

## LM head

Transformer 主干为每个 token 输出一个**隐状态**（hidden state）向量。**LM head** 是最后一个线性投影，将隐藏维度映射回**词表**（tokenizer 空间）。

- 隐状态 $\xrightarrow{\text{LM head}}$ **logits**（每个词表 token 一个分数）
- 对 logits 做 **softmax** $\rightarrow$ 得到下一个 token 的概率分布
- 从该分布中采样 / 取 argmax $\rightarrow$ 生成的 token

同一个主干可以接**不同的 head** 来完成不同的任务。本课程的第一个例子是 **reward model head**（第 5 章）——同样的 transformer，新的 head，新的目标。

---

## Softmax 与 log-probability

**Softmax** 将第 $t$ 步的 logits $z^{(t)}$ 转换为下一个 token 的分布——得到每个 token 的概率：

$$P_\theta(x_t \mid x_{<t}) = \mathrm{softmax}(z^{(t)})_{x_t} = \frac{e^{z^{(t)}_{x_t}}}{\sum_j e^{z^{(t)}_j}}.$$

整个序列的概率是这些项的**乘积**（链式法则），而在 **log 空间**中乘积变为求和：

$$\log P_\theta(x) = \log \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t}) = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}).$$

因此求和中的每一项，都是在真实 token 处读取的 logits 的 `log_softmax`——这恰好就是该位置上（负的）交叉熵 loss。

为什么用 log-prob？**数值稳定性**（许多小概率相乘会下溢），而且求和比乘积更容易求导。

---

## 一个 LM 训练样本的剖析

各讲会讨论 logits 的移位、目标 log-prob 的 gather 以及 masking。这里按张量形状的层面，完整走一遍整个流程：

```text
input_ids  ∈ ℕ^{B × L}          # INPUT TO MODEL -- token IDs, batch B, length L
   → model(input_ids).logits ∈ ℝ^{B × L × V}     # one vector per position, V = vocab
   → log_softmax + gather    ∈ ℝ^{B × (L-1)}     # log-prob of each observed token
   → (× completion_mask).sum ∈ ℝ^{B}             # one completion log-prob per sequence
```

接下来：同样的流程用 PyTorch 逐行来看。

---

## 一个 LM 训练样本的剖析

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
```

---

## 一个 LM 训练样本的结构

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
logits = logits[:, :-1, :]                  # drop the last position — its prediction has no next token to score
```

---

## 一个 LM 训练样本的结构

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
logits = logits[:, :-1, :]                  # drop the last position — its prediction has no next token to score
labels = input_ids[:, 1:]                   # the target at each position is just the actual next token (the shift)
```

---

## 一个 LM 训练样本的结构

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
logits = logits[:, :-1, :]                  # drop the last position — its prediction has no next token to score
labels = input_ids[:, 1:]                   # the target at each position is just the actual next token (the shift)
completion_mask = completion_mask[:, 1:]    # shift the mask the same way so it lines up with labels (1 = completion)
```

---

## 一个 LM 训练样本的结构

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
logits = logits[:, :-1, :]                  # drop the last position — its prediction has no next token to score
labels = input_ids[:, 1:]                   # the target at each position is just the actual next token (the shift)
completion_mask = completion_mask[:, 1:]    # shift the mask the same way so it lines up with labels (1 = completion)
log_probs = logits.log_softmax(dim=-1)      # normalize logits into log-probabilities over the vocab -> (B, L-1, V)
```

---

## 一个 LM 训练样本的结构

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
logits = logits[:, :-1, :]                  # drop the last position — its prediction has no next token to score
labels = input_ids[:, 1:]                   # the target at each position is just the actual next token (the shift)
completion_mask = completion_mask[:, 1:]    # shift the mask the same way so it lines up with labels (1 = completion)
log_probs = logits.log_softmax(dim=-1)      # normalize logits into log-probabilities over the vocab -> (B, L-1, V)
token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # read off the log-prob of each TRUE next token -> (B, L-1)
```

---

## 一个 LM 训练样本的结构

```python
logits = model(input_ids).logits            # (B, L, V): a score for every vocab token, at every position
logits = logits[:, :-1, :]                  # drop the last position — its prediction has no next token to score
labels = input_ids[:, 1:]                   # the target at each position is just the actual next token (the shift)
completion_mask = completion_mask[:, 1:]    # shift the mask the same way so it lines up with labels (1 = completion)
log_probs = logits.log_softmax(dim=-1)      # normalize logits into log-probabilities over the vocab -> (B, L-1, V)
token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # read off the log-prob of each TRUE next token -> (B, L-1)
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)           # zero out prompt/pad, then sum per sequence -> (B,)
```

**错位一个 token** 把自回归的这一步直观地写了出来：位置 $t$ 的 logits 预测的是 token $t+1$。把每个 token 的 log-prob 求和后调用 `loss.backward()`，autodiff 就会把这个和变成对应的逐 token 梯度之和。

---

## post-training 代码中常见的三种 mask

Mask 是 LLM 代码里隐蔽 bug 的藏身之处。三种不同的 mask 各司其职：

- **Causal mask**（内置于 decoder attention）——位置 $t$ 不能关注位置 $> t$ 的内容。正是它让模型具有自回归性；训练代码中你不需要碰它。
- **Attention / padding mask**——告诉 attention 忽略 padding token（用于把变长序列拼成一个 batch）。形状为 `(B, L)`。
- **Completion / loss mask**——只在 completion token 上取 `1`；求和之前先让 loss 乘上它。**Prompt token 会影响预测，但不产生 loss。**

> completion mask 出错的后果：要么在 prompt 上训练（把梯度浪费在无法改变的 token 上），要么在 padding 上训练（拟合的是噪声）。

---

## 解码方法小结

生成过程是在 $\pi_\theta$ 上**采样/搜索**——权重相同，选取每个 token 的方式却有很多种：

- **Greedy**——每一步取 $\arg\max_v P_\theta(v \mid x_{<t})$。结果确定但短视：局部最优的 token 可能毁掉整个序列。
- **Beam search**——按累积 log-prob 保留前 $b$ 个部分序列，扩展后再剪枝回 $b$ 个。适合低熵任务（如翻译），对开放式文本则显得平淡。
- **Temperature sampling**——$y_t \sim \mathrm{softmax}(z_t / T)$。$T$ 越高 → 越多样，$T$ 越低 → 越接近 greedy。
- **Truncated sampling**——只从 top-$k$ 个 token 中采样，或从累积概率质量 $\ge p$ 的最小集合中采样（top-$p$ / nucleus）。
- **Lookahead search（MCTS 等）**——在落定之前，先对模拟出的未来续写打分（通常借助 value/reward model）——相当于解码阶段的 inference-time reasoning。

关键区分：**分布** $\pi_\theta$ 与对它进行采样的**流程**。

---

## 训练 LM：Cross-entropy

在每个位置上，cross-entropy 把模型预测的分布与**真实**的下一个 token 标签 $q_t$ 作比较。该标签是实际 token $x_t$ 处的 *one-hot*，因此对词表的求和坍缩为单独一项：

$$H(q_t,\, P_\theta) = -\sum_{v \in \mathcal{V}} q_t(v)\,\log P_\theta(v \mid x_{<t}) = -\log P_\theta(x_t \mid x_{<t}).$$

所以“与真实 token 作比较”，其实就是 one-hot 挑出了一个 log-prob。对各位置求和、再对数据取平均，就得到 LM loss——等价于**负对数似然（NLL）**：

$$\mathcal{L}_{\text{LM}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right].$$

“预测下一个 token”**就是**监督学习——标签正是序列中的下一个 token。

---

<!-- layout: section-break -->

## 优化与 fine-tuning

---

## 梯度下降与训练循环

训练不断重复同一步：

$$\theta \leftarrow \theta - \eta \,\nabla_\theta \mathcal{L}(\theta).$$

- $\theta$ — 模型参数，$\eta$ — 学习率，$\nabla_\theta \mathcal{L}$ — loss 的梯度
- **AdamW** 及相关自适应优化器（adaptive、基于 momentum）是 LM 训练的标准选择，而非朴素的 SGD
- 训练以 optimizer **steps / tokens** 来衡量；对有限的 fine-tuning 数据集，**epochs** 是有用的单位。学习率调度与学习率本身同样重要

基本的 backprop 循环在 pretraining、mid-training 和 fine-tuning 中是共通的——不同的是 **masking、sampling、batching、数据构造和系统**，而不是优化器。

---

## Pretraining → mid-training → SFT

pretraining *末尾*的边界很模糊，但对 post-training 非常重要：

- **Pretraining** — 在海量原始网页/代码数据上做 next-token cross-entropy；**每个** token 都对 loss 有贡献。
- **Mid-training** — pretraining 最末尾的 annealing / 高质量数据阶段，在任何指令数据之前。数据质量在这里逐步提升。可以视为 post-training 的一部分，但通常不被单独讨论。
- **SFT（supervised fine-tuning）** — 仍然是 cross-entropy，但现在作用在**指令–回答**对上，并使用 **completion mask**：只有回答部分的 token 对 loss 有贡献，prompt 仅作为条件。模型学到的是*回答*，而不只是续写文本。

现代 post-training 流程从一个 **mid-trained** checkpoint 出发，因此这里正是本课程故事真正开始的地方。

---

<!-- layout: section-break -->

## 概率基础

---

## KL divergence（与 entropy）

**Kullback–Leibler divergence** 衡量一个分布与另一个分布的差异：

$$\mathcal{D}_{\text{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}.$$

需要记住三点：

- **不对称** — $\mathcal{D}_{\text{KL}}(p\|q) \neq \mathcal{D}_{\text{KL}}(q\|p)$；方向很重要
- **非负**，且仅当 $p = q$ 时为 $0$
- alignment 的**核心**对象：RLHF 的 KL 惩罚（第 3–4 讲）让 policy 靠近其 SFT 起点，而 DPO loss（第 6 讲）正是由 **KL 正则化的 reward 最大化**目标推导而来

相关：**Entropy** $H(p) = -\sum_i p_i \log p_i$ 是分布的不确定性（均匀 = 高，尖峰 = 低）。它出现在定义、推导以及对这些系统的理解中。计算方式非常相似。

你几乎会在每一讲 alignment 内容中见到 KL。

---

## KL divergence（与 entropy）

三个量，一幅信息论图景（用 $\ln$ 时单位为 *nats*，用 $\log_2$ 时为 *bits*）：

- **Entropy** $H(p) = -\sum_i p_i \log p_i$ — 来自 $p$ 的样本的平均意外程度；编码这些样本不可约减的代价。均匀 = 高，尖峰 = 低。
- **Cross-entropy** $H(p, q) = -\sum_i p_i \log q_i$ — 用为*错误*分布 $q$ 构造的编码来编码来自 $p$ 的样本的代价。
- **KL divergence** $\mathcal{D}_{\text{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}$ — 恰好就是使用 $q$ 而非 $p$ 所多付出的*额外*代价。

它们满足同一个恒等式：

$$H(p, q) = H(p) + \mathcal{D}_{\text{KL}}(p \,\|\, q).$$

所以最小化 cross-entropy **就是**最小化与真实分布之间的 KL——$H(p)$ 由数据决定，是固定的。在 LM 训练中，标签 $p$ 是 one-hot，因此 $H(p)=0$，**cross-entropy = KL = NLL**。而在 RLHF 中，KL 项是*有意*加入的，用来让 policy 靠近其 reference。

---

## KL divergence（与 entropy）

对序列而言，KL 背后的求和是不可计算的——但 KL 是一个**期望**，所以可以用 **Monte Carlo** 来估计：对从 policy 采样得到的样本，取 log-ratio 的平均。

$$\mathcal{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \mathbb{E}_{x \sim \pi_\theta}\!\left[\log \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)}\right] \approx \frac{1}{N}\sum_{i=1}^{N} \log \frac{\pi_\theta(x_i)}{\pi_{\text{ref}}(x_i)}.$$

- 被积项就是你本来就在生成的每条 rollout 上的**对数概率比**。
- 朴素平均（$\hat k_1 = \log\frac{\pi_\theta}{\pi_{\text{ref}}}$）是无偏的，但方差大，并且在有限样本上可能为负。
- RLHF 使用低方差、始终 $\ge 0$ 的 **$k_3$ estimator** $\hat k_3 = r - 1 - \log r$，其中 $r = \tfrac{\pi_{\text{ref}}}{\pi_\theta}$ [@schulman2020klapprox]。

这就是信息论的实际应用面：**从样本出发逼近目标分布**，用方差而非不可计算的求和来为估计付出代价。

---

## Sigmoid 与 pairwise likelihood

一个偏好（“回答 $y_w$ 优于 $y_l$”）是**二元**结果，因此分数之差会经过 **sigmoid** 被压缩成概率：

$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad P(y_w \succ y_l \mid x) = \sigma\!\left(r(x,y_w) - r(x,y_l)\right).$$

需要认出的三点：

- sigmoid 把**分数之差**转换为二元概率
- 只有**相对**的 reward 差才重要——把两个 reward 同时平移一个常数不会有任何影响
- $-\log \sigma(\cdot)$ 就是**二元 negative log-likelihood**

**第 2 讲会完整推导这个 Bradley–Terry 模型**来训练 reward model；**第 6 讲在 DPO 中复用了完全相同的 $\sigma(\Delta r)$ 形式**。

---

<!-- layout: section-break -->

## 强化学习框架

---

## 作为 MDP 的语言模型

强化学习研究智能体（agent）在**马尔可夫决策过程（Markov decision process）** $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ 中的行动：

| MDP 概念 | 语言模型 |
|-------------|---------------|
| 状态 $s_t$ | Prompt + 目前已生成的 token：$(x, y_{<t})$ |
| 动作 $a_t$ | 下一个 token $y_t$ |
| 转移 $P$ | 确定性：把 token 追加上去 |
| Policy $\pi_\theta(a_t \mid s_t)$ | LM 的 next-token 分布 |
| 奖励 $r$ | 通常是**终端奖励**：RM 分数或 verifier 输出 |
| 折扣 $\gamma$ | 通常为 $1.0$（不折扣） |

policy 就是 LM，按 token 做因子分解：

$$\pi_\theta(y \mid x) = \prod_t \pi_\theta(y_t \mid x, y_{<t}).$$

---

## RL 的目标，以及为什么用于语言模型

RL 的目标是选择策略参数 $\theta$，使策略自身下的**期望奖励**最大化：

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(x, y)].$$

人类偏好（或 verifier）是回复*末尾*的单个标量：

- **稀疏**——整个回复只有一个标签，而不是每个 token 一个
- **延迟**——一个 token 的好坏取决于整个 completion
- **没有 token 级目标**——而且普通梯度无法穿过离散的采样 token 反向传播

因此并不存在针对“偏好”的 cross-entropy 目标。Policy gradient 仍然可以优化评估者的期望分数——先把这个信号*学习*成 reward model，再去*优化*它。这就是 RLHF 的全部动机。

---

## 这些工具如何成为 post-training

本课件讲的都是基础机制——**概率、loss、梯度和采样**。当你用比原始 next-token prediction 更丰富的反馈去训练一个预训练模型时，就得到了 post-training：

- **示范**（“生成这样的回复”）→ supervised fine-tuning
- **偏好**（“让这个回复比那个更可能出现”）→ reward modeling 或偏好优化（DPO）
- **奖励**（“采样得分高的回复”）→ 强化学习（PPO、GRPO、RLVR）
- **约束**（“不要偏离起点太远”）→ 正则化，尤其是 KL 惩罚

数据不同，目标不同；但根本目的相同：

> **让模型在回复上的分布朝着我们想要的行为改变。**

这就是本课程其余部分的主题。

---

## 如果有些内容生疏了：

- **RL**——Sutton & Barto，*Reinforcement Learning: An Introduction*；David Silver 的 UCL 课程；UC Berkeley CS285
- **从零开始的语言模型**——Sebastian Raschka，*Build a Large Language Model (From Scratch)*
- **本书自带的参考**——附录 A（Definitions）涵盖了全书用到的 language modeling 概览、KL 和 RL 词汇

尽情钻研兔子洞、随意跳读、追逐让你兴奋的内容——这门课就是为此而设计的。
