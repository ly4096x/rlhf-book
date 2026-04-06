<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "奖励模型"
prev-url: "05-reward-models"
page-title: 强化学习
search-title: "第6章：强化学习"
next-chapter: "推理"
next-url: "07-reasoning"
---

# 强化学习（即 policy gradient 算法）

在 RLHF 过程中，强化学习算法根据 reward model 的反馈，逐步更新模型的权重。
policy——即被训练的模型——对训练集中的 prompt 生成补全内容，然后 reward model 对其评分，强化学习优化器根据这些信息执行梯度步骤（概览参见 @fig:rlhf-overview）。
本章介绍各种算法的数学原理与权衡，这些算法用于从 reward model 对 on-policy 数据所给出的信号中学习。
这些算法会运行多个 epoch，通常是在更大规模的 prompt 集合上运行数千乃至数百万个 batch，每个 batch 之间执行梯度更新。

让 RLHF 在语言模型中得以普及的，是 policy-gradient 强化学习算法。
这些算法，如 Proximal Policy Optimization（PPO）、Group Relative Policy Optimization（GRPO）和 REINFORCE，使用最近生成的样本来更新模型（而非像 Deep Q-Networks、DQN 这类算法那样将评分存储在 replay buffer 中，DQN 被用于 AlphaGo 等知名项目）。
本节将介绍 policy gradient 算法的基础知识，以及它们在现代 RLHF 框架中的应用方式。

从机器学习层面来看，本节是 RLHF 过程中复杂度最高的部分。
不过，与大多数现代 AI 模型一样，决定其成功的最大因素是作为输入提供给该过程的数据。

![RLHF 训练循环概览。数据集中的 prompt 被传入调优后的 policy，由其生成补全内容。reward model 对该补全内容评分，而冻结的初始模型（通常是 RL 之前的 instruction-tuned 模型）则对相同文本计算对数概率，以计算防止过度偏移的 KL penalty。综合后的 reward 信号随后驱动对 policy 参数的强化学习更新。](images/rlhf-overview.png){#fig:rlhf-overview}

当 RLHF 随 ChatGPT 进入公众视野时，业界普遍知晓 OpenAI 使用了 PPO 的一个变体，许多早期工作也以此为基础。
随着时间推移，多个研究项目展示了 REINFORCE 风格算法的潜力 [@ahmadian2024back] [@wang2024helpsteer2p]，其优点在于相比 PPO 更为简洁——无需 reward model（节省内存，进而减少所需 GPU 数量），且 value 估计更简单（无需 Generalized Advantage Estimation，GAE，这是一种用于在 policy gradient 算法中减少方差的 advantage 计算方法）。
更多算法相继涌现，其中 Group Relative Policy Optimization 在推理任务中尤为流行，但总体而言，这些算法大多可以调整以适应特定任务。
本章涵盖核心 policy gradient 框架，以及上述三种算法，因为它们在建立规范化 RLHF 文献中发挥了核心作用。

最简情形下，RLHF 的 RL 阶段需要两个模型：一个 policy（被训练的模型）和一个对其输出评分的 reward model（如前一章所述）。
RL 之前的 policy 副本充当参考模型，用于计算 KL penalty（该模型被冻结，即不会被自动微分引擎的梯度所更新）。
本章介绍的最复杂算法——PPO，增加了第四个模型——一个学习得到的 value function，用于估计动作中每个 token 的优劣，该模型也是一个在训练过程中被更新的大型语言模型。
本章中的算法主要在以下两方面有所不同：一是如何估计称为 *advantage* 的量——衡量模型当前动作（补全内容）相对于平均水平的优劣；二是如何约束 policy 更新以保证优化在数值上的稳定性。
该 RLHF 过程的可视化概览（不含 value 模型）如 @fig:rlhf-overview 所示。

符号定义请参见问题设置章节。

*本章使用强化学习文献中的 $(s, a)$ 符号，其中 $s$ 表示状态，$a$ 表示动作。在语言模型语境中，你会经常看到 $(x, y)$ 的写法，其中 $x$ 是 prompt，$y$ 是补全内容。$(s, a)$ 的表述更为通用——这些算法是为每个时间步都需要采取动作的序列决策问题而设计的。然而，许多 RLHF 实现将整个补全内容视为单一动作，这使得 $(x, y)$ 符号同样适用。*

***RL 速查表：** 本章所有核心 RL 损失函数的单页参考资料可在 [rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet) 获取。*

## Policy Gradient 算法

本章的核心在于理解如下形式的方程。
该方程计算的是我们正在训练的语言模型 $\pi_\theta$ 的梯度 $\Delta \theta$：

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$ {#eq:policy_gradient_intuition}

该方程由两个关键部分组成：
1. $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — 参数空间中使动作 $a_t$ 更可能发生的方向。
2. $\Psi_t$ — 该动作的优劣程度？一个对结果评分的标量。

将这两部分结合——即两个量相乘——便得到 policy gradient 更新。
一些简单的性质如下：$\Psi_t > 0$ 时更新参数使 $a_t$ 更可能发生，$\Psi_t < 0$ 时则使其更不可能发生。
policy gradient 计算的是哪些参数对某一动作有所贡献，以及我们是否应该让该动作在未来更或更少发生。
本章的其余部分将深入探讨实现这一目标的不同方式，以及使其适用于 LLM 的具体技巧。

现在，让我们进一步将其形式化。
强化学习算法旨在最大化跨状态 $s \in \mathcal{S}$ 和动作 $a \in \mathcal{A}$ 的 trajectory 上的未来折扣奖励（更多符号说明请参见附录 A，定义）。
智能体的目标，通常称为 *return*，是在给定时刻 $t$ 的未来折扣奖励之和（其中 $\gamma\in [0,1]$ 是优先考虑近期奖励的折扣因子）：

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}.$$ {#eq:return_definition}

return 的定义也可以递归估计为：
$$G_{t} = \gamma{G_{t+1}} + R_{t+1}.$$ {#eq:recursive_return}

该 return 是学习 value function $V(s)$ 的基础，$V(s)$ 表示给定当前状态下对未来 return 的估计：

$$V(s) = \mathbb{E}\big[G_t | S_t = s \big].$$ {#eq:value_function}

所有 policy gradient 算法都通过优化 policy $\pi_\theta(a\mid s)$ 来最大化期望 return；该目标可以使用由 policy 导出的 value function $V^{\pi_\theta}(s)$ 来表达。

其中 $d^{\pi_\theta}(s)$ 是由 policy $\pi_\theta(a \mid s)$ 导出的状态访问分布，我们最大化的目标可以写为：
$$
J(\theta)
\;=\;
\sum_{s} d^{\pi_\theta}(s) V^{\pi_\theta}(s),
$$ {#eq:policy_objective}

在有限 MDP 中，这是对所有状态的求和，但在实践中我们从不精确计算它。
相反，我们通过从当前 policy 中采样 rollout 来从数据中估计它。
在 RLHF 中，这通常意味着从数据集中采样 prompt $x_i$，并生成补全内容 $y_i \sim \pi_\theta(\cdot\mid x_i)$，然后计算经验均值，例如：

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} R(x_i, y_i),
$$ {#eq:empirical_batch_estimate}

或者，在具有逐步奖励的 MDP 视角下：

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \sum_{t=0}^{T_i} \gamma^t r_{i,t}.
$$ {#eq:empirical_mdp_estimate}

在实践中，语言模型的 RLHF 设置 $\gamma = 1$（不折扣），因为优化单元是整体补全内容，而非单个 token——这一选择将在本章后面的 MDP 与 Bandit 部分进一步讨论。

policy gradient 算法的核心是计算关于当前 policy 有限时间期望 return 的梯度。
有了该期望 return $J$，参数更新可按如下方式计算，其中 $\alpha$ 为学习率：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$ {#eq:policy_update}

核心实现细节在于如何计算上述梯度。

### 推导 Policy Gradient

另一种表述我们希望最大化的 RL 目标的方式如下：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right],
$$ {#eq:policy_objective_expectation}

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条 trajectory，$R(\tau) = \sum_{t=0}^\infty r_t$ 是该 trajectory 的总奖励。或者，我们可以将期望写成对所有可能 trajectory 的积分：
$$
J(\theta) = \int_\tau p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_objective_integral}
注意，我们可以将 trajectory 概率表达如下，其中 $\pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$ 是从某一状态和动作转移到一组下一状态的转移概率：
$$
p_\theta (\tau) = p(s_0) \prod_{t=0}^\infty \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t),
$$ {#eq:trajectory_probability}

若我们对目标函数（@eq:policy_objective_expectation）关于 policy 参数 $\theta$ 求梯度：
$$
\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_gradient_integral}

注意，我们可以利用 [log-derivative trick](https://andrewcharlesjones.github.io/journal/log-derivative.html) 将积分的梯度改写为期望的形式：
$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau) &= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} &\text{（由链式法则）} \\
\implies \nabla_\theta p_\theta(\tau) &= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) &\text{（整理得）}
\end{aligned}
$$ {#eq:log_chain_rule}

利用该 log-derivative trick：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau \\
&= \int_\tau p_\theta (\tau) \nabla_\theta \log p_\theta (\tau) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p_\theta (\tau) R(\tau) \right]
\end{aligned}
$$ {#eq:policy_gradient_expectation}

最后一步使用了在 trajectory 分布 $p_\theta(\tau)$ 下期望的定义：对任意函数 $f$，$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau)\,p_\theta(\tau)\,d\tau$（离散情形下为求和）。
将其写成期望的形式非常有用，因为我们可以用 Monte Carlo rollout 来近似，例如对 trajectory $\tau_i \sim \pi_\theta$，用 $\frac{1}{B}\sum_{i=1}^{B} f(\tau_i)$ 来近似。

回到推导过程，展开 trajectory 的对数概率：

$$
\log p_\theta (\tau) = \log p(s_0) + \sum_{t=0}^\infty \log \pi_\theta(a_t|s_t) + \sum_{t=0}^\infty \log p(s_{t+1}|s_t, a_t)
$$ {#eq:trajectory_log_prob}

现在，若对上式求梯度，可得：

- $\nabla_\theta \log p(s_0) = 0$（初始状态不依赖于 $\theta$）
- $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$（环境的转移动态不依赖于 $\theta$）
- 只有 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 保留下来

因此，trajectory 对数概率的梯度化简为：
$$
\nabla_\theta \log p_\theta (\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)
$$ {#eq:trajectory_log_grad}

在此稍作说明，推导至此已触及实现上的关键一步。
我们已经可以看出，trajectory 分布的梯度可以化简为语言模型 policy 概率的梯度之和（即被训练模型给出的 token 概率）。
在实践中，这导出了 policy gradient 方程的一种常见形式：loss 中是对数概率之和，然后通过自动微分计算梯度。
你会反复见到如下简短代码片段：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

本章后续将多次用到这一形式。现在回到正式的 policy gradient 数学推导。

将 @eq:trajectory_log_grad 代回 @eq:policy_gradient_expectation，得：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]
$$ {#eq:policy_gradient_returns}

通常人们会使用更一般化的 policy gradient 公式：
$$
g = \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t) \Psi_t \right]
$$ {#eq:general_gradient}

其中 $\Psi_t$ 可以是以下各项之一（reward 通常还可以用 $\gamma$ 折扣），这一分类框架沿用自 Schulman et al. 2015 [@schulman2015high]：

1. $R(\tau) = \sum_{t=0}^{\infty} r_t$：trajectory 的总 reward。
2. $\sum_{t'=t}^{\infty} r_{t'}$：执行动作 $a_t$ 之后的 reward，也称为 return $G$。
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$：上一公式的带 baseline 版本。
4. $Q^{\pi}(s_t, a_t)$：状态-动作 value function。
5. $A^{\pi}(s_t, a_t)$：advantage function，若能精确计算，理论上方差最低。
6. $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$：Temporal Difference (TD) 残差。

*baseline* 是一个用于降低 policy 更新方差的值（详见下文）。

对于语言模型而言，其中一些概念并不那么直观。
例如，对于确定性 policy $\pi$，状态 value 为 $V^{\pi}(s_t) = Q^{\pi}(s_t, \pi(s_t))$（最优 value function 则满足 $V^*(s_t)=\max_{a_t} Q^*(s_t,a_t)$）；对于随机性 policy，对应的恒等式为 $V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot\mid s_t)}[Q^{\pi}(s_t,a_t)]$。
Bellman 方程将 Q 与 V 联系起来：一般情况下 $Q^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t]$，但对于状态转移确定性的语言模型，这可以化简为 $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$。
advantage function 衡量动作 $a_t$ 相对于平均水平的优势程度：

$$A(s_t,a_t) = Q(s_t,a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$ {#eq:advantage_trick}

最后这一形式恰好就是 temporal difference (TD) 残差（即上面第 6 项）——RL 中的一个基本量，衡量 value function 预测值与实际发生值之间的差距，驱动 value function 向更准确的估计更新。在实践中，会使用学习到的 value function $\hat{V}$ 通过该 TD 误差来估计 advantage。

### Vanilla Policy Gradient

Vanilla policy gradient 的实现通过对 policy 参数求导来优化上述 $J(\theta)$ 表达式。
一个针对总 return 的简单版本如下：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]$$ {#eq:vanilla_policy_gradient}

vanilla policy gradient 算法的一个常见问题是梯度更新的高方差，可以通过多种方式加以缓解。
高方差的来源在于：梯度更新是通过估计 return $G$ 得到的，而所用的 rollout 数量通常较少，且容易受到噪声影响（例如，从语言模型以 temperature $>0$ 进行采样时的随机性）。
在 reward 稀疏的场景下，return 估计的方差更大，因为更多样本的值为 0 或 1，而非紧密聚集在一起。
为缓解这一问题，人们使用各种技术来归一化 value 估计，称为 *baseline*。
baseline 以多种方式实现这一目标，实际上是相对于后续动作对状态 value 进行归一化（例如 advantage 就是 Q 值与 value 之差）。
最简单的 baseline 是对一批 reward 取平均，或使用移动平均。
即使是这些与动作无关的 baseline，也能在不改变期望梯度的前提下降低方差，因为对任意仅依赖状态的 $b(s)$，有 $\mathbb{E}_{a \sim \pi(a|s)}[b(s) \nabla_\theta \log \pi_\theta(a|s)] = 0$，从而显著改善学习信号。

本章讨论的许多 policy gradient 算法都建立在 advantage 形式的 policy gradient 之上：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$$ {#eq:advantage_policy_gradient}


### REINFORCE

REINFORCE 算法的名称很可能是一个反向首字母缩写词，但它所代表的算法组成部分与现代强化学习算法密切相关。
该算法定义于开创性论文 *Simple statistical gradient-following algorithms for connectionist reinforcement learning* [@williams1992simple]：

> 该名称是以下短语的首字母缩写："REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility。"
这三个部分共同构成了*奖励增量*（即 policy gradient 步骤）的实现方式。
更新规则包含三个组成部分：

1. 非负因子：这是学习率（步长），必须为正数，例如下文中的 $\alpha$。
2. 偏移 Reinforcement：这是一个基线 $b$ 或其他用于提升稳定性的奖励归一化因子。
3. 特征资格（Characteristic Eligibility）：这决定了学习如何按 token 进行归因。它可以是每个参数的一般值 $e$，但在现代方程中通常为 policy 的对数概率。

因此，其形式相当熟悉：

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

使用更现代的符号表示以及广义回报 $G$，REINFORCE 算子表达为：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\Big[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,(G_t - b(s_t))
\Big],
$$ {#eq:REINFORCE_with_baseline}

其中，$G_t - b(s_t)$ 是 policy 在当前状态下的 *advantage*，因此我们可以用 advantage $A$ 重新表述 policy gradient，并在后续章节中沿用这一形式：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\Big[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,A_t
\Big],
$$ {#eq:REINFORCE_with_advantage}

REINFORCE 是 vanilla policy gradient 的一种具体实现，采用蒙特卡洛梯度估计器。

![用于语言模型的基础 REINFORCE 架构。shaped reward 将 reward model 分数与来自参考模型的 KL penalty 相结合。本章将在此基础上进一步展开。](images/reinforce_tikz.png){#fig:reinforce-arch}

### REINFORCE Leave One Out（RLOO）

REINFORCE Leave One Out 与标准 REINFORCE 的核心实现差异在于：RLOO 使用批次中*其他*样本的平均奖励来计算基线，而非对批次内所有奖励取平均 [@huang2024putting]、[@ahmadian2024back]、[@kool2019buy]。
通过将当前样本的奖励排除在其自身基线之外，RLOO 基线与被评估的动作相互独立，从而保证梯度估计器是严格无偏的。

关键在于，这一方法仅在每个状态（prompt）生成多条 trajectory（completion）时才有效，这在使用 RL 对语言模型进行 fine-tuning 的多个领域中是通行做法。

具体而言，对于 REINFORCE Leave-One-Out（RLOO）基线，给定 $K$ 条采样 trajectory（以 prompt 为条件所采取的动作）$a_1, \dots, a_K$，针对给定 prompt $s$，我们逐 prompt 定义基线如下：

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}

由此得到 advantage：

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}

等价地，这也可以表示为：

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}

这是一种简单、低方差的*逐 prompt* advantage 估计，与 Group Relative Policy Optimization（GRPO）中使用的组相对 advantage 密切相关（GRPO 将在 Proximal Policy Optimization，即 PPO 之后讨论）。
在实践中，GRPO 风格的训练主要在以下方面有所不同：KL 正则化的施加方式（作为显式损失项，还是折叠入奖励中）以及是否使用 PPO 风格的 ratio clipping。
具体来说，标准 GRPO 实现在损失层面施加 KL penalty，而 RLOO 或传统 policy gradient 的推导则是将 KL penalty 应用于奖励本身。
随着从 RLHF 向推理以及使用可验证奖励的强化学习（RLVR）的转变，KL penalty 的使用整体上有所减少，许多推理领域对 RLHF 代码的改造版本甚至将其完全关闭。
尽管如此，RLOO 的 advantage 仍可与 PPO 的 clipping 相结合，由此可见这些算法之间的高度相似性。

RLOO 及其他不使用 value network 的算法——即不使用额外的模型副本（critic）来逐 token 预测标量值 $V(s_t)$——在计算损失时，对每个 token 分配相同的序列级 advantage（或奖励）。
而使用学习型 value network 的算法（例如 PPO）则对每个 token 分别分配不同的值，并从 EOS token 处实现的最终奖励进行折扣。
在引入 KL distance penalty 的情况下，RLOO 将 completion 中各 token 的 KL 累加，并将该标量折叠入序列奖励，从而将所得 advantage 广播至所有 token。
PPO 则在计算 $A_t$ 之前，从逐 token 奖励中减去逐 token KL，实现 token 级别的信用分配。
GRPO 通常保留序列级 advantage，但会在损失中额外添加一个逐 token 项，而非从奖励中减去。
这些细节与权衡将在本章后续部分进一步讨论。

![REINFORCE Leave-One-Out（RLOO）架构。每个 prompt 生成多条 completion，通过留一法基线进行 advantage 估计，无需学习 value function。](images/rloo_tikz.png){#fig:rloo-arch}

<!-- A nice formulation of LM RL loss functions is found here https://arxiv.org/pdf/2502.01600 -->

### Proximal Policy Optimization（PPO）

Proximal Policy Optimization（PPO）[@schulman2017proximal] 是 Deep RL 诸多成功背后的基础算法之一（例如 OpenAI 的 Five，它掌握了 DOTA 2 [@berner2019dota]，以及大量相关研究）。
PPO 所最大化的目标函数——关于 advantage 和 policy 概率——如下所示：

$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).$$ {#eq:PPO_EQN}

其中，$\pi_\theta(a|s)$ 是当前被优化的 policy，$\pi_{\theta_{\text{old}}}(a|s)$ 是用于收集训练数据的 policy（即上一轮迭代的 policy）。
这两个 policy 之间的比率源于*重要性采样*，它使我们能够重用在旧 policy 下收集的数据来估计新 policy 的梯度。

回顾 policy gradient 的 advantage 表述（@eq:advantage_policy_gradient），有：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right].$$ {#eq:advantage_policy_gradient_recall}

该期望是在 $\pi_\theta$ 采样的 trajectory 上取的，但实际上我们希望对从固定 policy $\pi_{\theta_{\text{old}}}$ 收集的一批数据进行多次梯度更新。
为了修正分布不匹配的问题，我们乘以重要性权重 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$，以反映当前 policy 与数据收集 policy 相比，各样本出现可能性的变化比例。
若不加约束，当比率偏离 1 较远时，优化这一重要性加权目标可能导致具有破坏性的大幅 policy 更新。
PPO 通过将比率 clipping 至范围 $[1-\varepsilon, 1+\varepsilon]$ 来解决这一问题，从而确保 policy 在单次更新中的变化幅度不会过大。

为完整起见，PPO 通常被写成关于时间步的*期望* clipped surrogate 目标函数：

$$
J(\theta)
=
\mathbb{E}_{t}\left[
\min\left(\rho_t(\theta)A_t,\ \text{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)A_t\right)
\right],
\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$ {#eq:PPO_EQN_EXPECTED}

该目标函数通常通过添加负号转化为损失函数，使优化器朝着使其尽可能小的方向求解。

对于语言模型，目标函数（或损失）是逐 token 计算的，这在直觉上可以从计算整个序列自回归预测概率的方式来理解——即通过概率的乘积。
在此基础上，常见的实现方式采用*对数概率*，以简化在现代语言模型框架中的计算。
实践中，通常计算 token 对数概率之差并取指数，以还原 policy ratio $\rho_t$。

$$ J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).  $$  {#eq:PPO_EQN_EXPANDED}

这是 PPO 的逐 token 版本，同样适用于其他 policy gradient 方法，将在本章实现部分进一步探讨。
其中，按动作中 token 数量取平均的项 $\frac{1}{|a|}$ 源于常见的实现惯例，并非损失的正式推导所要求（参见 [@liu2025understanding]）。
下面我们将解释在不同的 advantage 值和 policy ratio 下，该损失函数所触发的不同情形。
在实现层面，PPO 的内部计算涉及两个主要项：1）带有已学习 advantage 的标准 policy gradient，以及 2）基于最大步长限制的 clipped policy gradient。

为了理解不同情形的产生方式，我们可以将 policy ratio 定义为：

$$\rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$ {#eq:PPO_POL_RATIO}

Policy ratio 是 PPO 及相关算法的核心。
它由计算策略梯度推导而来，并以一种非常直观的方式控制参数更新。
对于任意一批数据，在该批次的第一次梯度步骤中，policy ratio 从 1 开始，因为此时 $\pi_{\theta}$ 与 $\pi_{\theta_{\text{old}}}$ 相同。随后，在下一次梯度步骤中，若该步骤提升了某些具有正 advantage 的 token 的概率，则 policy ratio 将大于 1；反之则小于 1。一个常见做法是在更新 $\pi_{\theta_{\text{old}}}$ 之前，对每批数据执行 1 到 4 次梯度步骤。

#### 理解 PPO 目标函数

总体而言，PPO 目标函数可以通过目标值相对于 policy ratio 的曲线图中的两条线来可视化，如 @fig:ppo-obj 所示。
PPO 目标函数通过改变被采样动作的概率来最大化。
在数值上，目标函数通过巧妙运用最小值操作，同时处理正 advantage 和负 advantage 的情形，使得更新幅度最多偏离 policy ratio 为 1 的位置一个 epsilon 距离。

在 trust region 内部，PPO 的行为与其他 policy gradient 算法基本相同。
这是有意为之的！Trust region 是一个用于限制 PPO 及其同类算法最大更新步长的概念，以保证更新的稳定性。PPO 算法的核心，即 clip 和 min/max 函数，正是用来定义这一区域的。目标函数在该区域之外变为平坦。

"trust region" 的概念源自数值优化文献 [@nocedal2006numerical]，但在深度 RL 领域，它因算法 Trust Region Policy Optimization（TRPO）而广为人知，TRPO 被公认为 PPO 的前身 [@schulman2015trust]。
Trust region 是完整 policy gradient 步骤得以应用的区域，因为在此区域内更新不会被 PPO 目标函数的 max/min 操作所 "clipped"。

![PPO 目标函数在假设 advantage 下不同区域的可视化。"trust region" 描述的是 policy ratio $\rho$ 处于 $1\pm\varepsilon$ 范围内的区域。](images/ppo-viz-4x.png){#fig:ppo-obj}

Policy ratio 与 advantage 的组合可能出现几种不同的配置。我们将这些情形分为两组：正 advantage 和负 advantage。

**正 Advantage（$A_t > 0$）**

这意味着根据 value function，所采取的动作是有益的，我们希望在未来提高采取该动作的概率。现在，让我们来看 policy ratio $\rho(\theta)$ 的不同情形：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：新策略下该动作的概率低于旧策略
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标值**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生了什么**：正常的 policy gradient 更新——提高该动作的概率

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：新策略与旧策略下该动作的概率几乎相同
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标值**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生了什么**：正常的 policy gradient 更新——提高该动作的概率

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：新策略下该动作的概率高于旧策略
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标值**：$(1 + \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 + \varepsilon) A_t = 0$
    - **发生了什么**：不更新——该动作在新策略下已经更加可能被采取

总结来说，当 advantage 为正（$A_t>0$）时，我们希望提升该动作的概率。因此：

- 仅当 $\pi_{\text{new}}(a) \leq (1+\varepsilon) \pi_{\text{old}}(a)$ 时，我们才执行梯度步骤。直观上，由于 advantage 为正，我们希望提升该动作的概率，但不应提升过多以至于其概率显著增大。
- 关键在于，当 $\pi_{\text{new}}(a) > (1+\varepsilon) \pi_{\text{old}}(a)$ 时，我们不执行任何更新，clipped 目标函数的梯度为 $0$。直观上，该动作在新策略下已经得到了更多的体现，因此我们不希望过度强化它。

**负 Advantage（$A_t < 0$）**

这意味着根据 value function，所采取的动作是有害的，我们希望在未来降低采取该动作的概率。现在，让我们来看 policy ratio $\rho(\theta)$ 的不同情形：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：新策略下该动作的概率低于旧策略
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标值**：$(1 - \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 - \varepsilon) A_t = 0$
    - **发生了什么**：不更新——该动作在新策略下已经更不可能被采取

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：新策略与旧策略下该动作的概率几乎相同
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标值**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生了什么**：正常的 policy gradient 更新——降低该动作的概率

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：新策略下该动作的概率高于旧策略
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标值**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生了什么**：正常的 policy gradient 更新——降低该动作的概率

总结来说，当 advantage 为负（$A_t < 0$）时，我们希望降低该动作的概率。因此：

- 仅当 $\pi_{\text{new}}(a) \geq (1-\varepsilon) \pi_{\text{old}}(a)$ 时，我们才执行梯度步骤。直观上，由于 advantage 为负，我们希望降低该动作的概率，且降低幅度与 advantage 成比例。
- 关键在于，当 $\pi_{\text{new}}(a) < (1-\varepsilon) \pi_{\text{old}}(a)$ 时，我们不执行任何更新，clipped 目标函数的梯度为 $0$。直观上，该动作在新策略下已经更不可能被采取，因此我们不希望过度压制它。

必须牢记，在 trust region 内，PPO 与标准形式的 policy gradient 大体相同。


#### Value Functions 与 PPO

PPO 中的 value function 是模型的一个额外副本，用于预测每个 token 的价值。
在传统 RL 中，token（或状态）的价值是预测从该时刻起（通常带有折扣）的未来回报。
PPO 中的这个价值被用作已学习的 baseline，代表了在 REINFORCE 中使用的简单 Monte Carlo 版本的演进（REINFORCE 不需要已学习的 value network）。
这突显了 PPO 是如何在多个方面对 REINFORCE 和 vanilla policy gradient 进行演进的，涵盖优化形式、baseline 等。
在实践中，对于 PPO 和其他用于语言模型的算法，这是在扣除 KL 惩罚后预测每个 token 的回报（如前所述，per-token 损失传统上包含来自 reward 的 KL）。

有几种不同的方法（或目标）用于学习 value functions。
Generalized Advantage Estimation（GAE）被认为是现代系统中最先进的标准实现，但它通过计算多步骤的 value 预测误差而带来了更高的复杂性——详见本章后面关于 GAE 的章节。
Value function 也可以用来自 rollout 的 Monte Carlo 估计来学习，这些 rollout 也用于更新策略。
PPO 有两个损失——一个用于学习 value function，另一个用于利用该 value function 来更新策略。

![Value function 训练使用 on-policy rollout 来计算目标。模型在每个 token 处预测 $V_t$，通过均方误差（MSE）与目标回报 $\hat{V}_t$ 进行训练。Advantage $A_t = \hat{V}_t - V_t$ 随后对 policy gradient 更新进行加权。](images/value_fn_training.png){#fig:value_fn_training}

下面展示了一个 value network 损失的简单示例实现。
```python
# Basic PPO critic targets & loss (no GAE)
#
# B: Batch Size
# L: Completion Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards; EOS row includes outcome
#   done_mask: (B, L) 1.0 at terminal token (EOS or truncation if penalized), else 0.0
#   completion_mask: (B, L) 1.0 on response tokens to supervise (ignore the prompt)
#   values: (B, L) current critic predictions V_theta(s_t)
#       because a value network is a running update
#   old_values: (B, L) critic predictions at rollout time V_{theta_old}(s_t)
#   gamma: discount factor, float (often 1.0 for LM RLHF)
#   epsilon_v: float value clip range (e.g., 0.2), similar to PPO Loss Update itself, optional
#
# Returns:
#   value_loss: scalar; advantages: (B, L) detached (for policy loss)

B, L = rewards.shape

# 1) Monte Carlo returns per token (reset at terminals)
# Apply discounting, if enabled
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns  # y_t = G_t (post-KL)

# 2) PPO-style value clipping (optional)
v_pred = values
v_old  = old_values
v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)

vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss_tok  = torch.max(vf_unclipped, vf_clipped)

# 3) Mask to response tokens and aggregate
denom = completion_mask.sum(dim=1).clamp_min(1)
value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()

# 4) Advantages for policy loss (no GAE): A_t = G_t - V(s_t)
advantages = (targets - v_pred).detach()

# The value loss is applied later, often with the PG loss, e.g.
# total_loss = policy_loss + vf_coef * value_loss
```

### 群体相对策略优化（GRPO）

群体相对策略优化（Group Relative Policy Optimization，GRPO）在 DeepSeekMath [@shao2024deepseekmath] 中被引入，并在其他 DeepSeek 相关工作中得到应用，例如 DeepSeek-V3 [@deepseekai2025deepseekv3technicalreport] 和 DeepSeek-R1 [@guo2025deepseek]。
GRPO 可以视为一种受 PPO 启发的算法，具有非常相似的代理损失，但它避免了使用原始 policy 语言模型的另一个副本（或另一个用于初始化的检查点）来学习 value function。
这带来了两个被认为有益的特点：

1. 避免了从语言模型骨干网络学习 value function 的挑战，而目前研究尚未确立该领域的最佳实践。
2. 由于不需要在内存中保存额外的模型权重，从而节省了内存（从需要当前 policy、reference policy 和 value function 三份副本，减少为仅需前两份）。

GRPO 通过简化 value 估计来实现这一点，对 episode 中的每个 token 分配相同的值（即在针对某一 prompt 的 completion 中，每个 token 被分配相同的值，而不是标准 value function 中的折扣奖励），方法是对 advantage 或 baseline 进行估计。
该估计通过从同一初始状态/prompt（$s$）收集多个 completion（$a_i$）和奖励（$r_i$）来完成，即蒙特卡洛估计。

从形式化的角度来看，GRPO 目标与上述 PPO 目标非常相似。
对于 GRPO，目标函数（或损失）是在针对给定 prompt $s$ 的一组 completion $\{a_1, a_2, ..., a_G\}$ 上累积的。
以下展示 GRPO 目标：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right).$$ {#eq:GRPO}

注意，相对于 PPO，GRPO 的标准实现将 KL divergence 包含在损失中。
如上所述，我们可以将其展开为逐 token 的计算：

$$\begin{aligned}
J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \Bigg( &\min\!\left(\frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}A_{i,t},\; \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) \\
&- \beta \mathcal{D}_{\text{KL}}\!\left(\pi_\theta(\cdot|s_{i})\|\pi_{\text{ref}}(\cdot|s_{i})\right) \Bigg)
\end{aligned}$$ {#eq:GRPO_token}


其中，completion 索引 $i$ 对应的 advantage 计算为：

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.$$ {#eq:GRPO_ADV}

![GRPO 架构。Advantage 相对于群体均值和标准差进行归一化。KL 惩罚直接应用于损失，而非对奖励进行塑形。](images/grpo_tikz.png){#fig:grpo-arch}

直观而言，GRPO 更新是在一个 batch 内对同一问题的多个答案进行比较。
模型学习变得更像被标记为正确的答案，而更不像其他答案。
这是一种非常简单的 advantage 计算方式，即在给定状态下衡量某一特定动作相对于平均水平的优越程度。
相对于 PPO、REINFORCE 以及更广泛地使用 reward model 评分进行的 RLHF（相对于输出奖励），GRPO 通常以每个 prompt 更高数量的样本运行，因为 advantage 完全取决于一个 completion 相对于同一 prompt 下其他 completion 的相对价值。
在这里，当前 policy 对给定 prompt 生成多个响应，而群体范围内的 GRPO advantage 估计能获得有价值的上下文信息。
PPO 和原始 policy gradient 算法旨在准确估计每个 completion 的奖励（事实上，在某些情况下，更多的 completion 对改善 value 估计几乎没有帮助）。
GRPO 及其变体特别适合现代语言模型工具，因为对给定 prompt 生成多个 completion 非常自然（尤其是与机器人任务中从固定环境状态执行多个动作相比）。

GRPO 的 advantage 计算在其偏差上存在权衡。
对标准差的归一化会对 batch 中答案正确性变异性低的问题给予更高奖励。
对于几乎全部正确或全部错误的问题，标准差会更低，advantage 会更高。
Liu 等人 2025 [@liu2025understanding] 提出移除标准差项以消除这一偏差，但这会以降低那些只有少数正确答案的全错误问题的权重为代价——而这些情况可能被视为对模型有价值的学习信号。
那些高方差的 prompt 恰恰可能是最难的情形，只有极少数采样的 completion 找到了正确答案，从而提供了强烈的训练信号。

@eq:GRPO_ADV 是在结果监督（标准 reward model 或单一可验证奖励）下 GRPO 的实现，而在过程监督的情况下需要不同的实现。
在这种情况下，GRPO 将 advantage 计算为后续推理步骤归一化奖励的总和。

最后，GRPO 的 advantage 估计也可以在不使用 PPO clipping 的情况下应用于更原始的 policy gradient 版本（例如 REINFORCE），但这不是规范形式。
作为这些算法相互交织的一个示例，我们可以证明，GRPO 的一个变体——Dr. GRPO（GRPO Done Right）[@liu2025understanding]——中的 advantage 估计与 RLOO 估计（使用其他样本的平均奖励作为 baseline）在一个常数缩放因子上等价（由于归一化 advantage 的实现细节，这个常数因子通常不影响结果）。
Dr. GRPO 从 @eq:GRPO_ADV 中移除了标准差归一化项——注意这同时会*放大* advantage，相当于在答案分数存在方差的样本上提高 GRPO 的学习率。
这解决了对低奖励方差问题的偏差——即几乎所有答案都正确或错误的情况——但可能带来潜在的代价，即对于只有一个样本答对的问题，从中学习是重要的。
Dr. GRPO 中群体大小为 $G$ 的第 $i$ 个 completion 的 advantage 定义为：

$$ \tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum_{j=1}^G r_j $$ {#eq:DrGRPO_ADV}

在相同的符号表示下，我们可以回顾 RLOO advantage 估计：

$$ A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j $$ {#eq:RLOO_ADV_AGAIN}

因此，如果我们将 Dr. GRPO advantage 定义乘以 $\frac{G}{G-1}$，可以看到一个缩放等价关系：

$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1}^G r_j \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j - \frac{1}{G-1} r_i \\
&= r_i \left( \frac{G}{G-1} - \frac{1}{G-1} \right) - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

### 组序列策略优化（GSPO）

当对从先前策略中采集的一批数据进行多次梯度更新时，需要使用重要性采样（importance sampling）来修正数据采集策略与当前被优化策略之间的分布偏差。
标准的重要性采样恒等式允许我们利用来自另一个分布的样本来估计某一分布下的期望：

$$
\mathbb{E}_{p}[f(x)] = \mathbb{E}_{q}\left[f(x) \frac{p(x)}{q(x)}\right],
$$ {#eq:IS_identity}

其中 $p$ 是目标分布，$q$ 是采样分布，$\frac{p(x)}{q(x)}$ 是重要性权重。
在 policy gradient 方法中，$p = \pi_\theta$ 是我们想要优化的当前策略，$q = \pi_{\theta_{\text{old}}}$ 是生成训练数据的策略。
这使我们能够对在 $\pi_{\theta_{\text{old}}}$ 下采集的样本重新加权，以估计 $\pi_\theta$ 的梯度，从而在每批 rollout 中进行多次梯度更新。

这种分布偏差在两种常见场景中会出现：（1）对单批数据进行多次梯度更新时，$\pi_\theta$ 在每次更新后都会偏离 $\pi_{\theta_{\text{old}}}$；（2）在异步训练系统中，推理后端（例如 vLLM）和训练后端（例如 FSDP）可能因同步延迟而持有不同的模型权重（详见本章后续的异步性章节，该问题在以可验证奖励为目标的 RL 训练中尤为突出，在 RLHF 场景中同样适用）。

PPO 和 GRPO 在 token 层面应用重要性采样，并通过 clipping *代理目标函数*（surrogate objective）来稳定学习过程。
然而，这种方法存在一个微妙的失效模式：当某个 token 的重要性比率超出 clipping 范围 $[1-\varepsilon, 1+\varepsilon]$ 时，该 token 将获得零梯度。
对于罕见但重要的 token——例如模型最初赋予低概率的关键推理步骤——这种"token 丢弃"现象会阻碍模型学习更可靠地生成这些 token。

Group Sequence Policy Optimization（GSPO）[@zheng2025gspo] 在 GRPO 的基础上进行了扩展，将重要性比率的计算粒度从 token 层面提升至序列层面。
这一算法的实践动机，以及与之并列、同样修改了 policy gradient 算法中重要性采样计算方式的 CISPO（我们稍后将讨论），在于逐 token 的重要性采样比率在数值上往往不稳定。
其概念动机在于：当奖励在序列层面赋予时（如大多数 RLHF 和 RLVR 场景），重要性采样的修正也应与该粒度相匹配。

Token 级别的比率对于长序列和/或大型稀疏模型（例如现代混合专家模型，MoE）而言可能表现异常：单个比率较大的 token 可能主导整个策略更新，或者一个响应中的许多 token 各自独立被 clip，从而使单个响应的学习信号碎片化。
GSPO 通过为每个响应计算单一的重要性权重来解决这一问题。

回顾一下，完整响应的概率以自回归方式分解：

$$
\pi_\theta(a \mid s) = \prod_{t=1}^{|a|} \pi_\theta(a_t \mid s, a_{<t}).
$$ {#eq:response_factorization}

注意，为简便起见，我们通常将条件策略 $\pi_\theta(a_t \mid s, a_{<t})$ 简写为 $\pi_\theta(a_t \mid s)$，其中隐含地包含了补全中的前序动作（token）。
GSPO 使用几何平均数定义了一个长度归一化的序列级重要性比率（以避免长序列的数值问题）：

$$
\rho_i(\theta) = \left( \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)} \right)^{\frac{1}{|a_i|}} = \exp\left( \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})} \right).
$$ {#eq:GSPO_ratio}

GSPO 的目标函数与 GRPO 类似，但使用了该序列级比率：

$$
J_{\text{GSPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) A_i,\, \text{clip}(\rho_i(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) \right].
$$ {#eq:GSPO_objective}

由于比率经过了长度归一化，clipping 范围 $\varepsilon$ 在逐 token 平均的尺度上运作，使得有效约束在不同长度的响应之间具有可比性。
在实现上，序列级权重 $\rho_i$ 被均匀应用于响应 $a_i$ 中的所有 token，这在保持序列级重要性采样修正的同时简化了梯度计算。

advantage 的计算方式与 GRPO（@eq:GRPO_ADV）相同，使用组内相对均值和标准差归一化，也可根据其他 GRPO 衍生研究进行修改。
GSPO 可以概括为"使用序列级重要性比率的 GRPO"——重要性采样修正的粒度与奖励的粒度相匹配。

### 截断重要性采样策略优化（CISPO）

Clipped Importance Sampling Policy Optimization（CISPO）[@minimax2025minimax_m1] 采取了不同的方法：它不是 clipping 代理目标函数，而是直接 clipping 重要性权重本身，同时保留所有 token 的梯度信号。
该目标函数对截断后的重要性权重使用停止梯度（stop-gradient），回归到 REINFORCE 风格的表达形式，而非 PPO 风格的双侧 clipping：

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^K \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{\sum_{i=1}^K |a_i|} \sum_{i=1}^K \sum_{t=1}^{|a_i|} \text{sg}\left( \hat{\rho}_{i,t}(\theta) \right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s, a_{i,<t}) \right],
$$ {#eq:CISPO_objective}

其中 $\text{sg}(\cdot)$ 表示停止梯度（该权重被使用但不参与微分），截断的重要性比率为：

$$
\hat{\rho}_{i,t}(\theta) = \text{clip}\left( \rho_{i,t}(\theta),\, 1 - \varepsilon_{\text{low}},\, 1 + \varepsilon_{\text{high}} \right), \quad \rho_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})}.
$$ {#eq:CISPO_ratio}

与 PPO/GRPO 的关键区别微妙但重要：截断权重（而非目标函数）意味着每个 token 仍会收到与其 advantage 成比例的梯度信号——权重只是限制了该信号被重要性比率放大或抑制的程度。
这是一种偏差-方差权衡：截断权重引入了偏差，但控制了方差，并且关键在于避免完全丢弃 token 的梯度。

CISPO 和 GSPO 都由致力于在大规模 MoE 模型上推进 RL 极限的机构开发，而这类模型以数值问题著称。
相关论文强调，逐 token 的重要性采样比率不稳定，可能给梯度带来大量方差，从而阻碍学习。
这使得这些算法在大规模模型上可能产生显著影响，但在规模较小的学术实验中研究和受益相对较少。

CISPO 还允许非对称的 clipping 边界（$\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$），类似于本章后续讨论的 DAPO 的"clip-higher"修改，可通过允许对模型希望上调概率的 token 进行更大幅度的更新来鼓励探索。
相关工作还包括 Tapered Off-Policy REINFORCE（TOPR）[@leroux2025topr]，它同样直接截断重要性采样权重（类似 CISPO，而非像 PPO/GRPO 那样在目标函数内截断），但在序列层面运作（类似 GSPO），并根据奖励符号使用非对称截断——对正奖励不进行重要性采样修正，而对负奖励将比率截断至 $[0, 1]$——从而实现稳定的离策略（off-policy）学习。


## 实现

与最初开发这些算法的深度 RL 文献相比，为优化语言模型或其他大型 AI 模型而实现 RL 需要许多细微的实现细节。
本节将重点介绍区分流行算法实现的一些关键因素。

这一训练过程还涉及许多其他细节。
例如，在对语言模型进行 RLHF 时，一个关键步骤是生成文本，然后由 reward model 对其评分。
在正常情况下，模型应生成表示生成结束的序列结束（EOS）token，但通常的做法是对生成长度设置硬性上限，以高效利用基础设施。
RLHF 的一种失效模式是模型的回答被频繁截断，导致 reward model 的评分超出分布范围而产生不可预测的结果。
解决方案是*仅*对 `eos_token` 运行 reward model 评分，否则对生成过长的模型施加惩罚。

流行的开源 RLHF 工具在不同算法的实现细节上存在较大差异（参见 [@ivison2024unpacking] 中的表10）。
本节未涵盖的一些决策包括：

- **value network 初始化**：PPO 及其他类似算法所使用的内部学习型 value network 可以从相同架构的不同模型或随机选择的权重开始。这对性能影响显著。InstructGPT [@ouyang2022training] 建立的标准做法（并在 Tülu 3 的 RLVR 工作中沿用 [@lambert2024t]）是从 RLHF 中使用的 reward model 初始化 value network。其他方案包括使用上一个 RLHF 训练检查点（通常是 SFT 模型）并附加随机初始化的 value head，或完全重新初始化的语言模型（较不常见，因为 RLHF 需要更长时间才能收敛，但也是可行的）。
- **奖励归一化、奖励白化和/或 advantage 白化**：归一化将来自 RM（或环境）的所有值约束在 0 到 1 之间，有助于提高学习稳定性。[白化（whitening）](https://en.wikipedia.org/wiki/Whitening_transformation)更进一步，通过将奖励或 advantage 估计值变换为零均值和单位方差，对稳定性提供更强的保障。
- **不同的 KL divergence 估计器**：对于复杂的语言模型，精确计算模型间的 KL divergence 可能很复杂，因此使用多种近似方法代替精确计算 [@schulman2016klapprox]。
- **KL 控制器**：PPO 及相关算法的原始实现使用了动态控制器，以特定 KL 为目标并根据近期测量结果调整惩罚。大多数现代 RLHF 实现使用静态 KL 惩罚，但这一点也可能有所不同。

有关 RLHF 实现细节的更多信息，请参见 [@huang2024n]。
有关算法的进一步信息，请参见 [@weng2018PG]。

### Policy Gradient 基础

以下是一个使用 advantage 来估计梯度的简单 policy gradient 实现，为 PPO 和 GRPO 等高级算法做准备：
```python
pg_loss = -advantages * ratio
```
这里的 ratio 是新策略模型概率相对于参考模型的（逐 token）概率比率（通常由对数概率之差计算得出）。

为了理解这个公式，有必要了解一批更新中可能出现的不同情况。
请记住，我们希望随着模型在任务上的改进，损失能够*降低*。

情况1：advantage 为正，即该动作优于状态的期望值。我们希望强化这一点。在这种情况下，负号使得模型更倾向于采取这一动作。为此，它会增大对数比率。正的对数比率，即 token 的对数概率之和为正，意味着模型更可能生成这些 token。

情况2：advantage 为负，即该动作劣于状态的期望值。情况与之非常类似。此时，如果新模型具有更高的概率，损失将为正，因此模型将尝试调整策略参数，使该补全（completion）变得更不可能被生成。
### 损失聚合的权衡

在使用语言模型实现任何 policy gradient 算法时，都会遇到一个问题：如何将每个 token 的损失聚合为最终的标量损失？
给定样本 $i$ 在 token $t$ 处的每 token 损失 $\ell_{i,t}$，其中补全长度为 $|a_i|$，批次大小为 $B$，主要有三种策略：

**策略 1：按序列归一化**（标准 GRPO；部分 PPO 实现也采用此方式）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_per_sequence}

每条序列对批次损失的贡献相同，与序列长度无关。代码如下：

```python
# Strategy 1: Per-sequence normalization
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

**策略 2：按 token 归一化**（DAPO [@yu2025dapo]）

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$ {#eq:loss_per_token}

每个 token 对梯度的贡献相同；较长的序列对梯度的影响按比例更大。代码如下：

```python
# Strategy 2: Per-token normalization
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

**策略 3：固定长度归一化**（Dr. GRPO [@liu2025understanding]）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{L_{\max}} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_fixed_length}

以最大序列长度 $L_{\max}$ 进行归一化，在保持序列间每 token 尺度一致的同时，仍允许较长序列因包含更多有效 token 而贡献更多总梯度。代码如下：

```python
# Strategy 3: Fixed-length normalization 
fixed_len_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
            L_max).mean()
```

其中 $L_{\max}$ 通常是整个训练过程中的全局常量，用于指定生成 token 的最大数量。

注意，上述代码中的 `completion_mask` 是一个由 0 和 1 组成的矩阵，其中 prompt token 被掩码（置为 0），因为我们不希望模型从预测 prompt token 中学习。

#### 为什么这很重要？

直觉上，按序列归一化（策略 1）看起来最合理，因为我们关注的是*结果*，而非单个 token。
然而，这会基于序列长度引入微妙的偏差，可能导致模型过度思考，或根据偏差方向对那些本需要使用更多 token 的策略赋予较低权重。
考虑以下两条不同长度的序列，其每 token 损失为：

```python
seq_1_losses = [1, 1, 1, 1, 10]  # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

使用**策略 1**（按序列）：批次损失为 $(2.8 + 1.9)/2 = 2.35$，关键在于，短序列中每个 token 收到的梯度大于长序列中的 token。

使用**策略 2**（按 token）：批次损失为 $(14 + 19)/15 = 2.2$，所有 token 收到相同量级的梯度。

使用**策略 3**（固定长度，$L_{\max}=10$）：短序列贡献 $1.4$，长序列贡献 $1.9$，在按序列加权的同时平衡了每 token 的梯度。

如需查看更完整的示例以了解这些策略如何影响梯度，请参见下方脚本。

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """Compute sum of tensor with masked values. Use a constant to normalize."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)


advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0,],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)
```
```python
masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.0909, 0.0909, 0.0909, 0.0909, 0.0000, 0.0000, 0.0000],
# [0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]])
```

输出结果表明，采用策略 1（`masked_mean`）时，短序列的每个 token 梯度（0.25）大于长序列（0.14）。
策略 2 和策略 3 则使各序列间的每个 token 梯度趋于均等。
值得注意的是，若使用梯度累积——即在进行一次反向传播步骤前将多个 mini-batch 的梯度求和——上述结果可能会有显著变化：在这种情况下，短序列与长序列之间的权重平衡可能发生反转。

在实际操作中，最优策略取决于具体的训练配置。
在 RLHF 中，通常优先选择数值稳定性最佳或 loss 方差最小的方法。

#### 相关内容：MDP 框架与 Bandit 框架

loss 聚合方式的选择与我们如何定义 RL 问题的框架密切相关。
**MDP（token 级别）**视角将每个 token $a_t$ 视为一个动作，状态 $s_t$ 为当前的前缀序列。
在实际中，当我们通过学习得到的 value function $V(s_t)$（例如 GAE [@schulman2015high]）来计算 token 级别的 advantage function，并逐 token 施加 KL divergence 惩罚时，采用的正是这种框架。
带有学习 value network 的 PPO 是这一框架的典型代表 [@schulman2017proximal]。

相比之下，**bandit（序列级别）**视角将整个输出视为单一动作，对应一个标量 reward $R$。
在代码层面，这意味着计算序列级别的 advantage $A_{\text{seq}}$，并将其广播至所有 token。
RLOO 和 GRPO 风格的 advantage 通常用于这种 bandit 式设置 [@kool2019buy] [@ahmadian2024back] [@shao2024deepseekmath]。
DPO 和 A-LoL 等直接对齐方法也定义了序列级别的目标，尽管它们并非 policy gradient 估计器 [@baheti2023leftover]。

需要注意的是，许多 GRPO 实现采用 bandit 式 advantage，同时在 loss 中额外加入逐 token 的 KL 项；而许多 PPO/RLOO 实现则在计算 advantage 之前将 KL 折叠进 reward 中——两种做法在实践中均有使用。

下面的示例展示了两种方法的对比：

```python
# === Bandit-style（序列级别）===
# 每个序列对应一个标量 reward；advantage 广播至所有 token
reward = torch.tensor([3.0, 1.0])       # (B,) 例如 reward model 的评分
baseline = reward.mean()                 # 简单基线（RLOO 使用 leave-one-out）
advantage_seq = reward - baseline        # (B,)
advantages = advantage_seq[:, None].expand(-1, seq_len)  # (B, L)
# tensor([[ 1.,  1.,  1.,  1.],    <- 所有 token 的 advantage 相同
#         [-1., -1., -1., -1.]])

# === MDP-style（token 级别）===
# 逐 token reward + 学习得到的 V(s_t)；每个 token 有其自身的 advantage
# （也可使用逐 token 的 KL shaping、格式 reward 或其他 token 级别信号）
advantages = gae(per_token_rewards, values, done_mask, gamma=1.0, lam=0.95)
# tensor([[ 0.2,  0.5,  0.8,  1.5],    <- 随位置变化
#         [-0.3, -0.5, -0.8, -1.4]])
```

这一框架上的区别也解释了为何在几乎所有 RLHF 实现中，折扣因子 $\gamma$ 均被设置为 1.0。
在标准 RL 中，折扣（$\gamma < 1$）至关重要：它在多步骤 episode 中平衡了对短期与长期 reward 的优化，这对智能体学习有效的长期行为不可或缺。
但在 RLHF 设置中，即便采用 token 级别的 MDP 视角，优化的归纳偏置仍是整体输出的质量——reward 信号对整个响应进行评分，而非对单个 token 评分。
对较早出现的 token 进行折扣会任意降低其贡献权重，缺乏合理的理论依据。
随着 agentic RL 场景的日趋成熟——模型在其中执行真正的多步骤动作，如工具调用、代码执行和网页浏览——折扣因子可能再度变得相关，因为这类场景涉及真正具有不同长期后果的序贯决策。

### 异步 RL 系统

policy gradient 算法的默认实现方式是所谓的 **on-policy** 执行：智能体（语言模型）在采取动作（生成输出）后，先对动作进行评分，再更新模型。
policy gradient 的理论推导依赖于所有动作严格遵循 on-policy 原则，即模型始终与最新 trial/rollout 的结果保持同步。
然而在实践中，维持严格的 on-policy 执行会显著降低训练速度 [@noukhovitch2024asynchronous]——而且无论如何，完美同步在技术上都是不可能实现的。
因此，近期关于语言模型的大量实证结果在理论上都略微偏离了相应的数学证明。
实际发生的情况是：研究者根据什么真正有效来设计算法与系统。

![依据 Noukhovitch et al. 2024，同步与异步 RL 训练的生成-更新阶段对比。](images/async_v_synch_rl.png){#fig:async}

常用的解决方案是在独立的 GPU 节点上持续并行运行推理和训练，并使用专门为高效同步运行两者而设计的软件，如 @fig:async 底部所示。
在面向语言模型的主流开源 RL 工具中，通常的做法是使用 Ray 等分布式进程管理库，借助 vLLM 等高效推理引擎，在 policy gradient 学习循环与推理循环之间传递信息。
在这类系统中，负责执行 RL 更新步骤的 GPU 被称为"learner"，负责从语言模型采样的 GPU 被称为"actor"。
提升训练异步程度时面临的主要挑战在于保持训练稳定性和维持学习信号。

![一个分布式 RL 系统示例：系统通过两个队列向 learner GPU 和 actor GPU 传递数据，两者均可通过 Ray 等分布式计算库进行同步。Olmo Team 2025，CC-BY 许可。](images/distributed-rl.png){#fig:async_system}

这类系统的设计前提是：接近 on-policy 的数据足以支撑稳定的学习。
在这里，生成阶段与更新阶段可以轻松同步，避免训练系统任一部分出现计算资源空闲——即在 @fig:async_system 中将模型权重从 learner 传递至 actor 的过程。
对于推理模型而言，每个答案需要生成 10K 至 100K+ 个 token，其极长的推理特性使得 rollout 的生成成为远比其他环节更突出的瓶颈。
在同步性较强的 RL 基础设施上训练推理模型时，一个常见问题是：批次中某个 prompt 的回答可能需要消耗远多于其他 prompt 的生成时间（无论是更多的 token 还是更多的工具调用），导致大部分分配的计算资源在等待该回答完成期间处于闲置状态。
针对这一长度不匹配问题的第二种解决方案称为**序列级别打包（sequence-level packing）**：通过巧妙的 masking 将较短的样本叠加在同一批次中，使模型能够持续推进 rollout，并在批次内各样本间更好地分配长度归一化。
分布式 RL 基础设施的完整复杂性超出了本书的讨论范围，因为它还可能引发许多其他降低训练速度或导致不稳定的微妙问题。

随着推理模型的兴起，研究者进一步致力于将训练与推理循环转变为完全 off-policy 的模式：policy gradient 更新的训练批次由多个实例中最近完成的 rollout 填充 [@wu2025llamarl] [@fu2025areal]。
完全异步的训练也将使跨多个数据中心扩展 RL 训练规模变得更加容易，因为这样可以灵活增加 learner 节点（执行 policy gradient 步骤）与 actor（尝试求解问题）之间权重同步的时间间隔 [@primeintellectteam2025intellect2reasoningmodeltrained]。

相关方法正在探索完全 off-policy 的 policy gradient 算法 [@leroux2025topr]。

### 截断重要性采样（TIS）

截断重要性采样（Truncated Importance Sampling，TIS）是现代异步语言模型 RL 框架中用于稳定训练的重要工具。
重要性采样是一种修正方法，将从某一分布中采集的样本重新加权，以估计另一分布下的期望（如 @eq:IS_identity 中所介绍）。
截断重要性采样 [@ionides2008truncated] 使用 $\min(\rho, C)$（其中 $C$ 为某常数）对这些权重进行截断，以小幅偏差换取 policy gradient 中有界的方差。

这是一种应用于 policy gradient 的重要性采样修正，但与 PPO 和 CISPO 中的双边 clipping（将比率约束在 1 附近）不同，TIS 采用单侧上限：比率可以自由低于 1，但被截断至 $C$ 以防止极端的上加权。
在 PPO、GRPO、CISPO 及相关算法中，比率 $\rho_t^{\text{policy}} = \pi_\theta(a_t \mid s) / \pi_{\theta_{\text{old}}}(a_t \mid s)$ 用于修正同一 RL 批次中多次梯度步骤带来的策略漂移。
当我们转向以异步性为核心的真实 RL 框架时，还可能存在更大来源的数值差异（同样需要重要性采样进行数值修正）。
即便采样器与 learner 共享完全相同的参数 $\theta$，它们的有效 token 分布也可能存在差异，因为推理引擎（如 vLLM）与训练框架（如 FSDP）使用了不同的 kernel、精度和并行策略 [@yao2025offpolicy]。
因此，将同一策略在两个系统上的评估加以区分是有意义的，分别记为 $\pi_\theta^{\text{sampler}}$ 和 $\pi_\theta^{\text{learner}}$，并定义对应的比率及其截断形式：

$$
\rho_t^{\text{learner}} = \frac{\pi_\theta^{\text{learner}}(a_t \mid s, a_{<t})}{\pi_\theta^{\text{sampler}}(a_t \mid s, a_{<t})}, \qquad \tilde{\rho}_t^{\text{learner}} = \min(\rho_t^{\text{learner}},\; C).
$$ {#eq:tis_backend}

这两种修正相辅相成，但引入 policy gradient 实现中的原因各不相同——前者补偿 RL 批次训练过程中的策略漂移，后者则补偿实现层面引入的分布偏差——二者可以同时应用。
它们的组合方式取决于具体算法：

**带 TIS 的 REINFORCE**（单次梯度步骤）：不存在策略漂移（$\pi_\theta = \pi_{\theta_\text{old}}$），因此唯一的不匹配来自 learner 与 sampler 之间。
此时 $\pi_{\theta_\text{old}} = \pi_\text{gen}$，TIS 直接修正 learner–sampler 之间的差距：

$$
\nabla_\theta J \approx \mathbb{E}_{a \sim \pi_\theta^{\text{sampler}}} \left[ \tilde{\rho}_t^{\text{learner}} \cdot A_t \cdot \nabla_\theta \log \pi_\theta^{\text{learner}}(a_t \mid s, a_{<t}) \right].
$$ {#eq:reinforce_tis}
**带有 TIS 的 PPO/GRPO**（多次梯度步骤）：此时两个比率均处于激活状态。
在仔细的实现中，policy ratio 中的"旧 logprobs"在学习器（learner）上重新计算（GSPO 论文对此有所讨论），因此 policy ratio $\rho_t^{\text{policy}} = \pi_\theta^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{learner}}$ 捕捉的是纯粹的 policy 漂移，而 $\tilde{\rho}_t^{\text{learner}} = \min(\pi_{\theta_\text{old}}^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{sampler}},\; C)$ 则单独修正生成检查点处的后端不匹配：

$$
J_{\text{PPO+TIS}}(\theta) = \mathbb{E}\left[ \min\!\left( \rho_t^{\text{policy}}\, A_t,\; \text{clip}\!\left(\rho_t^{\text{policy}}, 1-\varepsilon, 1+\varepsilon\right) A_t \right) \cdot \tilde{\rho}_t^{\text{learner}} \right].
$$ {#eq:ppo_tis}

此处 $\pi_{\theta_\text{old}} \neq \pi_\text{gen}$：旧 logprobs 来自学习器，而非采样器。
若某框架跳过此次重新计算，直接将采样器的 logprobs 用作 $\pi_{\theta_\text{old}}$，则 policy ratio 已经捕捉了后端不匹配，无需单独进行 TIS 修正——但此时 clip 操作作用于一个更嘈杂的比率，即使在任何梯度步骤之前，该比率就已偏离 1.0。
这正是 Yao 等人 [-@yao2025offpolicy] 所指出的"你的框架悄悄带来了 off-policy RL"这一观察。

在实践中，LLM RL 系统将 TIS 作为 policy-gradient loss 上的逐 token 修正权重来应用：

```python
# Shape: (B*G, L)
C = 2.0  # TIS cap

logratio = learner_logprobs - sampler_logprobs
logratio = logratio.clamp(-10.0, 10.0)              # numerical safety
tis_weight = torch.exp(logratio).clamp(max=C)        # one-sided truncation

# Use as a fixed correction weight on the per-token PG loss
per_token_pg_loss = per_token_pg_loss * tis_weight.detach()
```

$[-10, 10]$ 的截断仅用于指数运算前的数值稳定性；真正的截断重要性采样步骤是在 $C$ 处进行的单侧截断。
在实践中，围绕这些 logprobs 的记录管理——存储生成时的采样器 logprobs、在旧检查点重新计算学习器 logprobs、以及在梯度步骤中追踪当前 logprobs——构成了分布式 RL 框架中脚手架代码的重要组成部分。
与 GSPO 不同，此修正是在 token 级别进行的，因为它处理的是 token 级别的数值不匹配，而非序列级别的 reward 粒度问题。
针对学习器–采样器比率的 TIS 已被主要开源 RL 框架所采用（VeRL、OpenRLHF、SkyRL、OAT，以及使用 $C = 2$ 的 Open Instruct），对于长推理轨迹（第 7 章）而言，其重要性日益凸显——在长推理轨迹中，每个 token 的微小差异会在数千个生成 token 上不断累积。


### 示例：Proximal Policy Optimization

PPO 有非常多的实现版本。
以下展示的是核心 *loss* 计算过程。
对于稳定性能至关重要的还有 *value* 的计算，其中存在多种选择（包括 *value model* loss 的多种选项）。

请注意，此处的参考 policy（或旧 logprobs）来自生成时所使用的版本，不一定是参考 policy 本身。
参考 policy 仅用于 KL 距离约束/惩罚。

```python
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute returns via backward pass (gamma typically 1.0 for LM RLHF)
# Mask rewards to avoid padding tokens (which may have KL penalties) leaking into returns
returns = torch.zeros_like(rewards)
running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(rewards.shape[1])):
    # Zero out padding: only accumulate rewards/returns for valid completion tokens
    running = (rewards[:, t] + self.gamma * running) * completion_mask[:, t]
    returns[:, t] = running

# Compute advantages: A_t = G_t - V(s_t)
advantages = returns - values.detach()  # Shape: (B*G, L)
# Note: We detach the value network here to not update the parameters of
# the value function when computing the policy-gradient loss

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Value function loss: predict returns
vf_loss = 0.5 * ((returns - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
    
    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

理解 PPO 的核心在于 policy gradient loss 是如何更新的。
重点关注以下三行：
```python
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```
`pg_losses1` 是普通的 advantage 加权 policy gradient loss。`pg_losses2` 应用相同公式，但将概率比率截断到 $[1-\varepsilon, 1+\varepsilon]$ 范围内，从而限制 policy 在单次更新中的变化幅度。

关键在于取两个 loss 的 `torch.max`。由于我们最小化的是 *负* loss（注意 advantages 前面的负号），取最大值选择的是更悲观的梯度——即产生更小 policy 更新的那个。当 advantage 为正时（好的动作），截断防止 policy 过于激进地提升该动作的概率。当 advantage 为负时（坏的动作），截断防止在反方向上的过度修正。

通过截断对数概率比率，PPO 限制了 policy 相对于生成训练数据时的版本所能漂移的距离，从而在无需显式信赖域计算的情况下稳定学习过程。

上述代码还展示了 PPO 在 policy 的同时学习一个 value function，这增加了实现的复杂性，但截断目标函数才是其核心机制。

#### 单次梯度步骤下 PPO/GRPO 的简化（无截断）

如果超参数"每个样本的梯度步骤数"等于 1，PPO（以及 GRPO）的实现可以大大简化。
该超参数的典型取值通常为 2–4 或更高。
在主要的 PPO 或 GRPO 方程（见 @eq:PPO_EQN）中，"参考" policy 是前一时刻的参数——即用于生成 completions 或动作的参数。
因此，如果只执行一次梯度步骤，则 $\pi_\theta = \pi_{\theta_{\text{old}}}$，更新规则简化为如下形式（符号 $[]_\nabla$ 表示停止梯度）：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right). $$ {#eq:ppo_1step}

这导致了 PPO 或 GRPO 的实现中，第二个 policy gradient 和 clipping 逻辑可以被省略，从而使优化器更接近标准的 policy gradient。


### 示例：Group Relative Policy Optimization

DeepSeekMath 论文描述了 GRPO 的一些实现细节，这些细节与 PPO 有所不同 [@shao2024deepseekmath]，尤其是与深度 RL 中 PPO 的标准应用相比，而非语言模型中的应用。
例如，RLHF 优化中的 KL penalty（回顾一下，KL penalty 也用于在无 reward model 的情况下，基于可验证奖励训练推理模型）是直接应用在 loss 更新中，而非 reward 函数中。
标准 RLHF 中 KL penalty 的应用方式为 $r=r_\theta - \beta \mathcal{D}_{\text{KL}}$，而 GRPO 的实现方式则为：

$$ L = L_{\text{policy gradient}} + \beta * \mathcal{D}_{\text{KL}} $$ {#eq:grpo_loss_kl}

不过，实现方式不止一种。
传统上，KL 距离是针对补全相对于提示词 $s$ 中每个 token 计算的。
在推理训练中，从一个提示词中采样多个补全，且一个 batch 中包含多个提示词，
因此 KL 距离的形状将为 [B, L, N]，其中 B 是 batch size，L 是序列长度，N 是每个提示词的补全数量。

将上述内容整合在一起，使用第一种 loss 累积方式，伪代码如下所示。

```python
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute grouped-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)    


# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
```

关于如何解读此代码的更多细节，请参阅上方的 PPO 章节。与 PPO 示例相比，核心差异如下：

- **优势计算**：GRPO 相对于组来归一化奖励（对同一提示词的各代取均值和标准差），而非使用学习到的价值函数作为 baseline。
- **无价值网络**：GRPO 完全移除了价值模型，消除了 `vf_loss` 及其相关的复杂性。
- **KL penalty 的位置**：GRPO 将 KL penalty 直接添加到 loss 中，而非从 reward 中减去（这是标准实现，但关于如何应用 KL 存在更多版本）。

#### RLOO 与 GRPO 的比较

RLOO 的优势更新与 GRPO 非常接近，突出了该算法在脱离 PPO 风格的 clipping 和 KL penalty 细节后，概念上的相似性。
具体而言，对于 RLOO，优势是相对于一个与 GRPO 极为相似的 baseline 来计算的——即某一补全的奖励相对于同一问题其他补全的奖励。
简而言之，RLOO 的优势估计如下（展开自 [TRL](https://github.com/huggingface/trl/blob/bfe20756082488350091352d1cdc19c172e42cd8/trl/trainer/rloo_trainer.py#L433) 的实现）：

```python
# rloo_k --> number of completions per prompt 
# rlhf_reward --> Initially a flat tensor of total rewards for all completions. Length B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) # 
# Now, Shape: (k, N), each column j contains the k rewards for prompt j.

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> Leave-one-out baseline rewards. Shape: (k, N)
#  baseline[i, j] is the avg reward of samples i' != i for prompt j.

advantages = rlhf_reward - baseline
# advantages --> Same Shape: (k, N)

advantages = advantages.flatten() # Same shape as original tensor
```

RLOO 其余的实现细节遵循 policy-gradient 实现中的其他权衡。

## 辅助主题

为了掌握 policy-gradient 算法的应用，还有无数其他需要考虑的因素。
本节将探讨成功部署 policy-gradient RL 算法时的一些长尾复杂性。

### 算法比较

本章中的每个算法都共享相同的核心梯度形式（@eq:policy_gradient_intuition），但在如何估计优势以及控制优化方面有所不同：

- **REINFORCE**：简单的 policy gradient 实现，包含对奖励的蒙特卡洛估计，并引入基于状态的 baseline 以降低方差。
- **RLOO**：每个提示词采用多个样本的 REINFORCE，每个样本的 baseline 为其他样本的平均奖励（leave-one-out），以降低梯度方差。
- **PPO**：增加了学习到的价值函数和 clipped policy ratio，以获得更准确、更稳定的梯度更新。
- **GRPO**：PPO 的简化变体，将每个提示词的多个补全分组，并在组内归一化奖励以计算优势，无需价值函数。
- **CISPO**：一种 REINFORCE 风格的算法，对重要性采样权重（而非 PPO/GRPO 中的目标函数）进行 clipping，并使用 stop-gradient 保证稳定性，使每个 token 都能接收到梯度信号。
- **GSPO**：与 GRPO 类似，但按补全长度对 policy ratio 进行归一化，以防止长度偏差。
- **DPO**：不是 RL 算法，而是一种通过完全绕过独立的 reward model 来解决相同偏好优化问题的方法，直接从偏好对进行优化（见第 8 章）。

上述所有 policy gradient 算法在推导上均为 on-policy，尽管大多数在实践中略微以 off-policy 方式应用。DPO 及第 8 章中的其他直接对齐算法默认为 off-policy。
所有算法均可与学习到的 reward model 或可验证奖励配对使用。
只有 PPO 需要学习到的价值函数。
REINFORCE 和 RLOO 没有重要性采样比率——其余算法各自引入了一个，以便对每批 rollout 执行多个梯度步骤，在粒度和 clipping 策略上有所不同，如下表所示。

| 方法 | IS 粒度 | Clipping 风格 | 优势 |
| :----- | :-----------: | :------------------: | :-------------------: |
| **REINFORCE** | None | None | Monte Carlo baseline |
| **RLOO** | None | None | Leave-one-out |
| **PPO** | Token | Objective (bilateral) | Learned value fn |
| **GRPO** | Token | Objective (bilateral) | Group-relative |
| **GSPO** | 序列 | 目标函数（双边） | 组相对 |
| **CISPO** | Token | 权重（stop-grad） | 组相对 |
表：policy gradient 算法对比。 {#tbl:pg_compare}

每种方法的核心损失函数 $\mathcal{L}(\theta)$ 如下：

$$\begin{aligned}
\textbf{REINFORCE:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\log \pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big) \\[6pt]
\textbf{RLOO:}\quad & -\frac{1}{K}\sum_{i=1}^{K}\sum_t \log \pi_\theta(a_{i,t}\mid s_{i,t})\left(R_i-\frac{1}{K-1}\sum_{j\neq i}R_j\right) \\[6pt]
\textbf{CISPO:}\quad & -\sum_{i,t} \mathrm{sg}(\hat{\rho}_{i,t})\, A_{i,t} \log \pi_\theta(a_{i,t}\mid s_{i,t}) \\
& \quad \hat{\rho}_{i,t} = \mathrm{clip}(\rho_{i,t},\, 1-\varepsilon,\, 1+\varepsilon) \\[6pt]
\textbf{PPO:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\min\!\big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\, A_t\big) \\
& \quad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} \\[6pt]
\textbf{GRPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\, A_i\big) \\
& \quad \rho_i = \frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_{\text{old}}}(a_i\mid s)},\quad A_i = \frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})} \\[6pt]
\textbf{GSPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\, A_i\big) \\
& \quad \rho_i = \left(\frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_{\text{old}}}(a_i\mid s)}\right)^{1/|a_i|} \\[6pt]
\textbf{DPO:}\quad & -\mathbb{E}_{(x,y^{w},y^{l})}\!\left[\log \sigma\!\big(\beta[\Delta\log \pi_\theta(x)-\Delta\log \pi_{\mathrm{ref}}(x)]\big)\right]
\end{aligned}$$


### 广义优势估计（GAE）

广义优势估计（Generalized Advantage Estimation，GAE）是一种用于计算 policy gradient 算法中优势值的替代方法 [@schulman2015high]，它能更好地平衡偏差与方差之间的权衡。
传统的单步优势估计可能引入过多偏差，而使用完整 trajectory 则可能导致高方差。
GAE 计算多步优势估计的指数加权平均值，其中超参数 $\lambda$ 控制偏差-方差权衡——范围从单步 TD（$\lambda=0$）到完整 trajectory 回报（$\lambda=1$）；$\lambda=0.95$ 是 LLM fine-tuning 的常用默认值。

优势估计可以有多种形式，但我们可以将 $n$ 步优势估计量（类似于本章开头的 TD 残差）定义如下：

$$
\hat{A}_t^{(n)} = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t), & n = 1 \\
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t), & n = 2 \\
\vdots \\
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t), & n = \infty
\end{cases}
$$ {#eq:K_STEP_ADV}

较短的 $n$ 具有较低的方差但较高的偏差，因为我们将更多的学习能力归因于每条 trajectory——这可能导致过拟合。
GAE 试图将这一公式推广为加权多步平均，而非固定的 $n$。
首先，我们需要定义预测价值的时序差分（TD）残差。

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$ {#eq:TD_RESIDUAL}

为了利用这一点，我们引入另一个变量 $\lambda$ 作为 GAE 混合参数。这折叠为我们希望估计的未来优势的指数衰减：

$$
\begin{array}{l}
\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda)(\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \cdots) \\
= (1-\lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \cdots) \\
= (1-\lambda)(\delta_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta_{t+1}^V(\lambda + \lambda^2 + \cdots) + \cdots) \\
= (1-\lambda)(\delta_t^V\frac{1}{1-\lambda} + \gamma\delta_{t+1}^V\frac{\lambda}{1-\lambda} + \cdots) \\
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V
\end{array}
$$ {#eq:GAE_DFN}

直觉上，这可以用一种简洁的方式对优势的多步估计进行平均。
下面展示一个示例实现：

```python
# GAE (token-level) for LM RLHF
#
# B: Batch Size
# L: Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards
#   values:  (B, L) current V_theta(s_t)
#   done_mask: (B, L) 1.0 at terminal token (EOS or penalized trunc), else 0.0
#   gamma: float (often 1.0), 
#   lam (short for lambda): float in [0,1]
#   (Padding beyond terminal should have rewards=0, values=0)
B, L = rewards.shape
advantages = torch.zeros_like(rewards)
next_v = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards[:, t] + gamma * not_done * next_v - values[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values[:, t]

targets = advantages + values      # y_t for value regression
advantages = advantages.detach()   # for policy loss
```

反向循环累积时序差分（TD）误差（$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$），这些误差衡量实际结果与价值函数预测相比有多好或多差，并以指数衰减 $(\gamma\lambda)^l$ 加权。
在终止 token 处，`not_done=0` 防止从未来状态进行自举并重置 GAE 累加器，因此每个回合的优势值是独立计算的（由于循环反向运行，终止 token 在回合边界处干净地停止指数加权累积——这使得实现对打包友好，能正确处理拼接到一起的多个序列）。
最终的 `targets` 作为在此 GAE 循环之外学习的独立价值函数的回归目标，而分离的 `advantages` 则为 policy gradient 加权——分离是为了防止 policy 更新通过价值网络反向传播。
在语言模型的 RLHF 中，$\gamma=1.0$ 很常见，因为回合是较短的 token 序列，其中更偏好无折扣的信用分配（且通常所有 token 都在一个回合中）。

*进一步阅读请参见 [@seita2017gae]。*

### 双重正则化

本章中我们已经看到两种类型的正则化。一种内置于 PPO 等算法中，以步长约束的形式存在；另一种是相对于优化起点的基于 KL divergence 的距离惩罚。

深度强化学习中许多流行的 policy gradient 算法，包括 PPO 及其前身，最初的提出都源于控制智能体学习过程的需要。
在 RLHF 中，正如第 15 章正则化和第 3 章训练概述中详细讨论的那样，存在一个内置的正则化项，即相对于正在 fine-tuning 的原始 policy 的距离惩罚。
从这个角度来看，像 PPO（内部具有步长正则化）和 REINFORCE（更简单，在某些超参数下 PPO 退化为 REINFORCE）这类算法之间的大部分差异，对于 fine-tuning 语言模型而言远不如从头训练智能体那样重要。

在 PPO 中，负责限制更新步长的目标函数被称为[代理目标函数](https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective)。
为了监控 PPO 正则化对 RLHF 更新的影响程度，可以查看许多流行实现中的裁剪比例（clip fraction）变量，即批次中概率比落在裁剪区间之外的样本百分比。
这是 PPO 正则化器可能生效频率的有用代理指标，但并非每个这样的样本梯度都为零：只有当裁剪分支被选中时代理函数才变得平坦，例如比率超过 $1+\varepsilon$ 的正优势样本或比率低于 $1-\varepsilon$ 的负优势样本。

在语言模型实践中，PPO 和 GRPO 等算法通常每批次只运行一个梯度步骤，这意味着 PPO 原生正则化从未被应用（因为只有当 policy 在一个批次内发生实质性变化时才会发生裁剪），而 KL 距离惩罚则占主导地位。
然而，这并非普遍规律。例如，DAPO 每批次使用 16 个梯度步骤 [@yu2025dapo]，而 Tülu 3 对 8B 和 70B 模型使用 4 次 PPO 更新迭代，但为了维持训练稳定性，对 405B 模型减少到 1 次 [@lambert2024t]。

### 延伸阅读

随着 RLHF 确立了其在现代后训练核心地位，其他 policy-gradient RL 算法以及 RL 算法通常都被提出用于改进训练过程，但它们在指导最佳实践方面并未占据核心地位。
延伸阅读的示例包括：

- **成对近端策略优化（P3O；Wu 等，2023）** [@wu2023pairwise] 直接在 PPO 风格的 policy 更新中使用成对数据，无需学习中间 reward model。
- **软自适应策略优化（SAPO）** [@gao2025sapo] 用平滑的温度控制门控替代了硬性的 PPO/GRPO 风格裁剪，旨在提供一个连续的信任域，在保留接近 on-policy 学习信号的同时降低 off-policy token 的权重。
- Off-policy policy-gradient 算法可以进一步实现异步训练，例如**对比策略梯度（CoPG）** [@flet2024contrastive]（直接对齐算法 IPO 和普通 policy gradient 的推广），该算法被 Cohere 用于其 Command A 模型 [@cohere2025command]。
- 其他 REINFORCE 算法的实现也专为语言模型设计，例如 **ReMax** [@li2023remax]，它实现了一种基线归一化，专门设计用于适应 reward model 推理中的不确定性来源。
- 一些基础模型，如 Apple Intelligence Foundation Models [@gunter2024apple] 或 Kimi k1.5 推理模型 [@team2025kimi]，使用了**镜像下降策略优化（MDPO）** [@tomar2020mirror] 的变体。相关基础研究仍在进一步发展 [@zhang2025improving]，但镜像下降是一种优化方法，而非直接的 policy gradient 算法。这里重要的是，它以与现有 RL 基础设施非常相似的方式被替换进去。
- **解耦裁剪与动态采样策略优化（DAPO）** 提出了对 GRPO 的 4 项修改，以更好地适应推理语言模型，其中需要长推理链，且新的、利用不足的 token 需要提高概率 [@yu2025dapo]。这些修改为：1，使用两个不同的裁剪超参数 $\varepsilon_\text{low}$ 和 $\varepsilon_\text{high}$，使得对数比正侧的裁剪可以采取更大的步骤以获得更好的探索；2，动态采样，移除批次中所有样本奖励 = 0 或奖励 = 1 的样本（无学习信号）；3，使用上文在"实现：GRPO"中讨论的按 token 的损失；4，对过长样本施加软惩罚，以避免从截断答案中学习。
