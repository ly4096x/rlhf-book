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
