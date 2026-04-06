<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "产品与特性"
prev-url: "17-product"
page-title: "附录 A：定义"
search-title: "附录 A：定义"
next-chapter: "风格与信息"
next-url: "appendix-b-style"
---

# 定义与背景

本章包含 RLHF 过程中频繁使用的所有定义、符号和操作，并对 language model 进行简要概述——language model 是本书的核心应用对象。

## Language Modeling 概述

现代 language model 的大多数都以自回归方式训练，学习 token（词、子词或字符）序列的联合概率分布。
自回归简单来说意味着每一步的预测都依赖于序列中前面的元素。
给定一个 token 序列 $x = (x_1, x_2, \ldots, x_T)$，模型将整个序列的概率分解为条件分布的乘积：

$$P_{\theta}(x) = \prod_{t=1}^{T} P_{\theta}(x_{t} \mid x_{1}, \ldots, x_{t-1}).$$ {#eq:llming}

为了拟合一个能准确预测上述概率的模型，目标通常是最大化当前模型对训练数据的似然。
为此，我们可以最小化负对数似然（NLL）损失：

$$\mathcal{L}_{\text{LM}}(\theta)=-\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T}\log P_{\theta}\left(x_t \mid x_{<t}\right)\right]. $$ {#eq:nll}

在实践中，针对每个下一个 token 的预测使用交叉熵损失，通过将序列中的真实 token 与模型预测结果进行比较来计算。

Language model 有多种架构，在知识、速度以及其他性能特征方面各有取舍。
现代 LM，包括 ChatGPT、Claude、Gemini 等，最常使用**仅解码器 Transformer**[@Vaswani2017AttentionIA]。
Transformer 的核心创新在于大量使用**自注意力**[@Bahdanau2014NeuralMT]机制，使模型能够直接关注上下文中的概念并学习复杂的映射关系。
在本书中，特别是在第 5 章介绍 reward model 时，我们将讨论添加新的输出头或修改 transformer 的 language modeling（LM）head。
LM head 是一个最终的线性投影层，将模型的内部 embedding 空间映射到 tokenizer 空间（即词表）。
我们将在本书中看到，language model 的不同"head"可以用于将模型 fine-tune 到不同目的——在 RLHF 中，这最常见于训练 reward model，相关内容在第 5 章中重点介绍。

## ML 定义

- **Kullback-Leibler（KL）散度（$\mathcal{D}_{\text{KL}}(P || Q)$）**，也称为 KL divergence，是衡量两个概率分布之间差异的指标。
对于定义在同一概率空间 $\mathcal{X}$ 上的离散概率分布 $P$ 和 $Q$，从 $Q$ 到 $P$ 的 KL 距离定义为：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:def_kl}


## NLP 定义

- **Prompt（$x$）**：给予 language model 以生成回复或补全内容的输入文本。

- **Completion（$y$）**：language model 响应 prompt 所生成的输出文本。Completion 通常记为 $y\mid x$。Reward 和其他值通常计算为 $r(y\mid x)$ 或 $P(y\mid x)$。

- **Chosen Completion（$y_c$）**：在多个选项中被选择或偏好的 completion，通常记为 $y_{chosen}$。

- **Rejected Completion（$y_r$）**：在成对比较设置中不被偏好的 completion。

- **偏好关系（$\succ$）**：表示一个 completion 优于另一个的符号，例如 $y_{chosen} \succ y_{rejected}$。例如，reward model 预测偏好关系的概率 $P(y_c \succ y_r \mid x)$。

- **Policy（$\pi$）**：对可能的 completion 的概率分布，由 $\theta$ 参数化：$\pi_\theta(y\mid x)$。

## RL 定义

- **Reward（$r$）**：表示某个动作或状态可取程度的标量值，通常记为 $r$。

- **Action（$a$）**：智能体在环境中做出的决策或行动，通常表示为 $a \in A$，其中 $A$ 是可能动作的集合。

- **State（$s$）**：环境的当前配置或情况，通常记为 $s \in S$，其中 $S$ 是状态空间。

- **Trajectory（$\tau$）**：trajectory $\tau$ 是智能体经历的状态、动作和 reward 的序列：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$。

- **Trajectory 分布（$(\tau\mid\pi)$）**：policy $\pi$ 下 trajectory 的概率为 $P(\tau\mid\pi) = p(s_0)\prod_{t=0}^T \pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)$，其中 $p(s_0)$ 是先验状态分布，$p(s_{t+1}\mid s_t,a_t)$ 是转移概率。

- **Policy（$\pi$）**，在 RLHF 中也称为 **policy model**：在 RL 中，policy 是智能体在给定状态下决定采取何种动作的策略或规则：$\pi(a\mid s)$。

- **折扣因子（$\gamma$）**：标量 $0 \le \gamma < 1$，对未来 reward 在回报中进行指数衰减加权，在即时性与长期收益之间进行权衡，并保证无限时域求和的收敛性。有时不使用折扣，这等价于 $\gamma=1$。

- **Value Function（$V$）**：估计从给定状态开始的期望累积 reward 的函数：$V(s) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s]$。

- **Q-Function（$Q$）**：估计在给定状态下采取特定动作后的期望累积 reward 的函数：$Q(s,a) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a]$。

- **Advantage Function（$A$）**：advantage function $A(s,a)$ 量化了在状态 $s$ 中采取动作 $a$ 相对于平均动作的相对收益。它定义为 $A(s,a) = Q(s,a) - V(s)$。Advantage function（以及 value function）可以依赖于特定的 policy，$A^\pi(s,a)$。

- **Policy 条件值（$[]^{\pi(\cdot)}$）**：在 RL 的推导和实现中，理论与实践的关键组成部分是收集以特定 policy 为条件的数据或值。在本书中，我们将在更简单的 value function 记号（$V,A,Q,G$）与其特定 policy 条件值（$V^\pi,A^\pi,Q^\pi$）之间切换。在期望值计算中同样关键的是从以特定 policy 为条件的数据 $d$ 中采样，即 $d_\pi$（例如，在估计 $\mathbb{E}_{s\sim d_\pi,\,a\sim\pi(\cdot\mid s)}[A^\pi(s,a)]$ 时，$s \sim d_\pi$ 且 $a \sim \pi(\cdot\mid s)$）。

- **Reward 优化的期望**：RL 中的主要目标，涉及最大化期望累积 reward：

  $$\max_{\theta} \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}[\sum_{t=0}^{\infty} \gamma^t r_t]$$ {#eq:expect_reward_opt}

  其中 $\rho_\pi$ 是 policy $\pi$ 下的状态分布，$\gamma$ 是折扣因子。

- **有限时域 Reward（$J(\pi_\theta)$）**：由 $\theta$ 参数化的 policy $\pi_\theta$ 的期望有限时域折扣回报定义为：

  $$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$ {#eq:finite_horizon_return}
  
  其中 $\tau \sim \pi_\theta$ 表示按照 policy $\pi_\theta$ 采样的 trajectory，$T$ 是有限时域。

- **On-policy**：在 RLHF 中，特别是在 RL 与 Direct Alignment Algorithm 之间的争论中，关于 **on-policy** 数据的讨论很常见。在 RL 文献中，on-policy 意味着数据完全由智能体的当前形式生成，但在一般的偏好调优文献中，on-policy 被扩展为来自该版本模型的生成内容——例如，在运行任何偏好 fine-tuning 之前的 instruction-tuned checkpoint。在这种语境下，off-policy 可以是 post-training 中使用的任何其他 language model 生成的数据。

## 仅适用于 RLHF 的定义

- **Reference Model（$\pi_{\text{ref}}$）**：这是 RLHF 中保存的一组参数，其输出用于对优化过程进行正则化。

## 扩展词汇表

- **合成数据（Synthetic Data）**：这是指 AI 模型的任何训练数据，其来源是另一个 AI 系统的输出。这可以是任何形式，从模型对开放式 prompt 生成的文本，到模型对现有内容的改写。
- **蒸馏（Distillation）**：Distillation 是训练 AI 模型的一类通用做法，其中一个模型在更强大模型的输出上进行训练。这是一种已知能够生成强大、较小模型的合成数据类型。大多数模型通过许可证（对于开放权重模型）或服务条款（对于仅通过 API 访问的模型）明确说明了关于 distillation 的规则。distillation 这一术语现在已经与 ML 文献中的特定技术定义相互交织。
- **（师生）知识蒸馏（Knowledge Distillation）**：从特定教师到学生模型的知识蒸馏是上述 distillation 的一种具体类型，也是该术语的起源。它是一种特定的深度学习方法，其中神经网络的损失被修改为从教师模型在多个潜在 token/logit 上的对数概率中学习，而不是直接从选定的输出中学习 [@hinton2015distilling]。使用知识蒸馏训练的现代模型系列的一个例子是 Gemma 2 [@team2024gemma] 或 Gemma 3。对于 language modeling 设置，next-token 损失函数可以按如下方式修改 [@agarwal2024policy]，其中学生模型 $P_\theta$ 从教师分布 $P_\phi$ 中学习：

$$\mathcal{L}_{\text{KD}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} P_{\phi}(x_t \mid x_{<t}) \log P_{\theta}(x_t \mid x_{<t})\right]. $$ {#eq:knowledge_distillation}

- **上下文学习（In-context Learning，ICL）**：这里的"上下文"指 language model 上下文窗口内的任何信息。通常，这是添加到 prompt 中的信息。上下文学习最简单的形式是在 prompt 之前添加类似形式的示例。高级版本可以学习为特定用例选择哪些信息。
- **思维链（Chain-of-Thought，CoT）**：思维链是 language model 的一种特定行为，引导模型以逐步分解问题的方式进行推理。其最初版本是通过 prompt "Let's think step-by-step" 实现的 [@wei2022chain]。
