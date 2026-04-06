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
