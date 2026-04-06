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
