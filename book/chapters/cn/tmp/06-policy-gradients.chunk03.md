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
