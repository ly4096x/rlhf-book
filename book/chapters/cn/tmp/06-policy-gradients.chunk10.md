
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
