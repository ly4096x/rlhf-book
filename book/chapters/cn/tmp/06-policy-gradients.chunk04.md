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
