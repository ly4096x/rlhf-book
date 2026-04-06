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
