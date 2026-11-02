---
title: "第 4 讲：RL 实现与实践"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 4 讲"
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
    letter-spacing: 0;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 4 讲：RL 实现与实践

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 6 章，第 2 部分</p>

---

<!-- rows: 50/50 -->
## 第 4 讲：RL 实现与实践

<!-- row-columns: 32/36/32 -->

```box
title: 概览
tone: muted
compact: true
content: |
  1. 引言
  2. 关键相关工作
  3. 训练概览
```

|||

```box
title: 核心训练流程
tone: accent
compact: true
content: |
  4. 指令微调
  5. Reward model
  6. **强化学习**
  7. 推理
  8. 直接对齐
  9. 拒绝采样
```

|||

```box
title: 数据与偏好
tone: muted
compact: true
content: |
  10. 什么是偏好
  11. 偏好数据
  12. 合成数据与 CAI
```

===

<!-- row-columns: 32/36/32 -->

```box
title: 实践考量
tone: muted
compact: true
content: |
  13. 工具使用
  14. 过度优化
  15. 正则化
  16. 评估
  17. 产品与角色
```

|||

```box
title: 附录
tone: surface
compact: true
content: |
  - A. 定义
  - B. 风格与信息
  - C. 实际问题
```

|||

```box
title: 课程主页
tone: surface
compact: true
content: |
  - [rlhfbook.com](https://rlhfbook.com)
  - [GitHub 仓库](https://github.com/natolambert/rlhf-book)
```

---

## 从数学到代码

第 3 讲讲的是**数学**：policy gradient 定理、REINFORCE、PPO、GRPO。

本讲：如何真正**实现、调试并运行** LLM 的 RL 训练。

最难的 bug 不是数学错误——而是隐蔽的实现错误：错误的掩码、过期的缓存、形状不匹配。

---

## 从数学到代码

第 3 讲讲的是**数学**：policy gradient 定理、REINFORCE、PPO、GRPO。

本讲：如何真正**实现、调试并运行** LLM 的 RL 训练。

最难的 bug 不是数学错误，而是悄无声息的实现错误：mask 用错、缓存过期、shape 不匹配。

*关于符号的提醒*：与第 6 章一样，我们使用强化学习文献中的 $(s, a)$，在 prompt-completion 记号更自然时使用 $(x, y)$。$(s, a)$ 的写法对应 token 级别的梯度计算；$(x, y)$ 对应序列级别的 reward。两种视角会贯穿全讲。

---

## 本讲内容

```box
title: 第 4 讲大纲
tone: accent
content: |
  1. **Policy gradient 代码** — log-prob、REINFORCE loss、RLOO baseline
  2. **PPO 实现** — GAE、clipped loss、value function、minibatch、调试
  3. **GRPO 及其变体** — 简化代码、与 PPO 的对比、GSPO 与 CISPO
  4. **Loss 聚合** — 按序列、按 token、固定长度归一化的权衡
  5. **训练基础设施** — on/off-policy、同步与异步系统
  6. **实践考量** — 数值稳定性、监控指标、开源代码库
```

---

<!-- layout: section-break -->

## 将 policy gradient 转化为代码

---

## 回顾：Policy gradient

目标函数及其梯度：

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

梯度的含义：对每个 token，先计算让它更可能出现的方向（$\nabla \log \pi$），再按它的好坏程度（$\Psi_t$）进行缩放。

---

## 回顾：Policy gradient 算法

所有方法最小化的都是同一族 **loss**（注意开头的负号），区别在于 $\Psi_t$ 以及如何限制更新幅度：

<div class="text-sm">

$$\begin{aligned}
\textbf{REINFORCE:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\log \pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big) \\[4pt]
\textbf{RLOO:}\quad & -\frac{1}{K}\sum_{i=1}^{K}\sum_t \log \pi_\theta(a_{i,t}\mid s_{i,t})\left(R_i-\frac{1}{K-1}\sum_{j\neq i}R_j\right) \\[4pt]
\textbf{PPO:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\min\!\big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\, A_t\big) \\[4pt]
\textbf{GRPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|a_i|}\sum_{t=1}^{|a_i|}\min\!\big(\rho_{i,t} \hat{A}_i,\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\, \hat{A}_i\big)
\end{aligned}$$

</div>

其中 $\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ 是 importance-sampling ratio。PPO：通过 GAE 得到 per-token advantage $A_t$。GRPO：ratio $\rho_{i,t}$ 是 per-token 的，但 advantage 是序列级的 $\hat{A}_i = \frac{R_i - \mu}{\sigma}$。



---

## 回顾：为什么 loss 是 log-prob 之和

来自第 3 讲，policy gradient 的推导给出：

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

在代码中，我们只需计算 **log-prob**，梯度交给 autodiff 处理：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()  # autodiff gives ∑ Ψ_t ∇ log π
```

本讲中的每个 loss 函数，都是这一模式的变体。

---

## 计算 log-probability（展开版）

最基本的构件：由 policy 得到的 per-token log-probability。下面逐步写出每一步：

```python
# Forward pass through the model
logits = model(input_ids).logits              # (B, L, vocab_size)

# Autoregressive shift: logit at position t predicts token at t+1
logits = logits[:, :-1, :]                    # (B, L-1, vocab_size)
labels = input_ids[:, 1:]                     # (B, L-1)
completion_mask = completion_mask[:, 1:]       # (B, L-1)

# Per-token log-probs
log_probs = logits.log_softmax(dim=-1)
token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)

# Per-sequence log-prob: sum over completion tokens
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
```

全部在 log 空间中计算，以避免许多小概率相乘造成的数值下溢。

---

## 计算 log-probability（紧凑版）

实践中，这通常会封装成一个辅助函数。来自本书的代码：

```python
def compute_log_probs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids,
                   attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return torch.gather(log_probs, dim=-1,
                        index=targets).squeeze(-1)
```

shift、gather 和 squeeze 与前面完全相同，只是写得更紧凑。它返回所有位置（prompt + completion）的 log-prob；mask 在之后再处理。在 rollout 阶段，对旧 policy 和 reference model 调用它（并缓存结果）。训练时，再对当前 policy 调用一次（每一步重新计算）。

---

## 代码中的 REINFORCE 损失

最简单的 policy gradient 损失：

```python
# rewards: (B,) — one reward per sequence
# seq_log_probs: (B,) — sum of log-probs over completion tokens

# Baseline: average reward in the batch
baseline = rewards.mean()
advantages = rewards - baseline

# REINFORCE loss (negative because we minimize)
loss = -(advantages * seq_log_probs).mean()
```

就是这样。advantage 对 log-probability 加权：advantage 为正 → 提高概率，为负 → 降低概率。（如何把逐 token 的损失归约为标量——这里用的是 `.mean()`——其影响比你预想的更大。我们会在损失聚合一节再回到这个问题。）

---

## 代码中的 RLOO

每个 prompt 生成 $K$ 个 completion，计算 leave-one-out baseline：

```python
# rlhf_reward: (B*K,) flat tensor of rewards
# Prompt-major layout: K sibling completions stay together
rlhf_reward = rlhf_reward.view(-1, rloo_k)  # (B, K)

# Leave-one-out baseline: avg of other K-1 rewards per prompt
baseline = (rlhf_reward.sum(dim=1, keepdim=True) - rlhf_reward) / (rloo_k - 1)

advantages = rlhf_reward - baseline        # (B, K)
advantages = advantages.reshape(-1)        # (B*K,)
```

其余部分与标准 policy gradient 相同——将 advantage 乘以 log-prob。 

注意：data loader 必须为每个 prompt 生成 $K$ 个 completion 并将它们分组放在一起。按 prompt 计算的 baseline 有助于降低方差，而标准 REINFORCE 则是在 batch 内所有状态上计算 baseline。

---

<!-- layout: section-break -->

## 近端策略优化（PPO）

---

## 回顾：近端策略优化（PPO）

裁剪代理损失（最小化）：

$$L^{\text{CLIP}}(\theta) = -\mathbb{E}_t\!\left[\min\!\Big(\rho_t A_t,\; \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\, A_t\Big)\right]$$

其中 $\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ 是重要性采样（importance sampling）比率。

**广义优势估计（GAE）**：

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE 通过将时序差分（TD）误差按指数衰减向后传播，得到逐 token 的 advantage。$\lambda = 0$ 是纯 TD（低方差、高偏差）；$\lambda = 1$ 是 Monte Carlo（高方差、低偏差）。

---

## 为什么要承担 PPO 的成本？

相比 REINFORCE/RLOO，PPO 增加了大量复杂性：

- **更低的方差**：通过 GAE 和学习得到的 value function 实现逐 token 的信用分配
- **Token 级信用**：每个 token 都有自己的 advantage 信号，而不只是序列级奖励
- **样本复用**：借助重要性采样，对同一个 batch 进行多次梯度更新

代价是：**内存中要放四个模型**（policy、value、reference、RM），value function 的初始化很脆弱，还需要调更多超参数。

在 2010 年代末和 2020 年代初，PPO 是发展最成熟、最为人所理解的 RL 算法，被广泛应用于各个 RL 领域。 
在语言模型之外的任务中，PPO 的性能远优于 REINFORCE。

---


## 代码中的 PPO 训练循环

```text
for each batch:
  1. Sample prompts from dataset
  2. Generate completions with current policy π_θ
  3. Score with reward model → per-sequence rewards
  4. Compute ref model log-probs → per-token KL penalty
  5. Shape rewards: r_t = r_t - β * KL_t (per token)
  6. Compute returns via backward pass (GAE or MC)
  7. Compute advantages: A_t = returns_t - V(s_t)
  8. For k = 1 to K epochs on this batch:
     - Compute ratio = π_θ(a_t|s_t) / π_old(a_t|s_t)
     - Policy loss: clipped surrogate objective
     - Value loss: MSE on returns
     - total_loss = policy_loss + vf_coef * value_loss
     - Backward + optimizer step
  9. Sync π_old ← π_θ
```

---

## 从标量奖励到逐 token 回报

reward model 在 completion 的末尾给出一个标量 $R(x, y)$。逐 token 奖励通过 KL 来塑形：

$$r_t = \begin{cases} R(x, y) - \beta \, \text{KL}_t & \text{if } t = T \text{ (final token)} \\ -\beta \, \text{KL}_t & \text{otherwise} \end{cases}$$

其中 $\text{KL}_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_\text{ref}(a_t \mid s_t)$。

这些逐 token 奖励输入 GAE，由 GAE 向后传播信用，从而分配逐 token 的 advantage。

---

## PPO 一步用到什么

训练开始之前，rollout 阶段必须计算并存储损失所需的一切：

| 张量 | 形状 | 用途 |
|---------------|-------|---------|
| Token ID | `(B, L)` | 全部 |
| Completion 掩码 | `(B, L)` | 全部 |
| 旧 log-prob $\log \pi_{\theta_\text{old}}$ | `(B, L)` | IS 比率 |
| 旧 value $V_{\phi_\text{old}}$ | `(B, L)` | PPO critic 裁剪 |
| Ref log-prob $\log \pi_\text{ref}$ | `(B, L)` | KL 惩罚 |
| 奖励 | `(B,)` 或 `(B, L)` | advantage 计算 |

---

## 缓存 vs. 重新计算

<!-- columns: 50/50 -->

**rollout 时存储**（只计算一次，之后冻结）：
- Token ID 与 completion mask
- 旧 log-probs $\log \pi_{\theta_\text{old}}$
- 旧 value 预测 $V_{\phi_\text{old}}$
- Ref log-probs $\log \pi_\text{ref}$
- 奖励（来自 RM）

|||

**每个训练步重新计算**：
- 新的 policy log-probs $\log \pi_\theta$
- 当前 value 预测 $V_\phi$

<div class="colloquium-spacer-md"></div>

IS ratio $\rho_t = \frac{\pi_\theta}{\pi_{\theta_\text{old}}}$ 同时用到一个新计算的量和一个缓存的量——这使得我们可以在同一批 rollout 上进行多个 epoch。

---

## 用 GAE 计算 advantage

<div class="text-sm">

```python
# rewards: (B,) terminal only (KL shaping omitted), values_old: (B, L) from rollout
B, L = completion_mask.shape
advantages = torch.zeros_like(values_old)
next_v = torch.zeros(B, device=values_old.device)
gae = torch.zeros(B, device=values_old.device)

last_idx = completion_mask.long().cumsum(-1).argmax(-1, keepdim=True)
done_mask = (torch.arange(L, device=values_old.device).unsqueeze(0) >= last_idx).float()
rewards_t = torch.zeros_like(values_old).scatter_(-1, index=last_idx, src=rewards)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards_t[:, t] + gamma * not_done * next_v - values_old[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values_old[:, t]

advantages = advantages * completion_mask
targets = (advantages + values_old).detach()
advantages = advantages.detach()
```

</div>

<span class="text-xs">*简化版：仅使用终止奖励。若使用 KL 塑形的奖励，则在此循环之前为每个 token 加上 $-\beta \cdot \text{KL}_t$。*</span>

---

## 代码中的 PPO policy loss

clipped 代理目标函数：

```python
# Compute probability ratio
ratio = torch.exp(new_per_token_logps - per_token_logps)  # (B, L), per_token_logps cached from rollout

# Clipped surrogate objective
eps = 0.2  # clip range
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
pg_loss = torch.max(pg_losses1, pg_losses2)
```

`torch.max` 选取更悲观（更保守）的梯度。由于我们最小化的是负的损失，这能防止单次更新过度冒进。

---

## 多个 epoch 与 minibatch

PPO（以及 GRPO）可以选择性地将每批 rollout 复用于多次梯度更新。只要 $\pi_\theta$ 偏离了 $\pi_{\theta_\text{old}}$，clipping 就会被触发——有两种机制导致这种偏离：


---

## 多个 epoch 与 minibatch

PPO（以及 GRPO）可以选择性地将每批 rollout 复用于多次梯度更新。只要 $\pi_\theta$ 偏离了 $\pi_{\theta_\text{old}}$，clipping 就会被触发——有两种机制导致这种偏离：

**Minibatching**：将 rollout 批次拆分为更小的 minibatch，使更大的总批量能够放进特定的 GPU 配置。在第一个 minibatch 上更新之后，$\pi_\theta$ 已经发生变化——因此*同一个* epoch 中后面的 minibatch 已经满足 $\rho_t \neq 1$。即使只有 $K = 1$ 个 epoch，clipping 也可能被触发。

---

## 多个 epoch 与 minibatch

PPO（以及 GRPO）可以选择性地将每批 rollout 复用于多次梯度更新。只要 $\pi_\theta$ 偏离了 $\pi_{\theta_\text{old}}$，clipping 就会被触发——有两种机制导致这种偏离：

**Minibatching**：将 rollout 批次拆分为更小的 minibatch，使更大的总批量能够放进特定的 GPU 配置。在第一个 minibatch 上更新之后，$\pi_\theta$ 已经发生变化——因此*同一个* epoch 中后面的 minibatch 已经满足 $\rho_t \neq 1$。即使只有 $K = 1$ 个 epoch，clipping 也可能被触发。

**多个 epoch**：对完整批次循环 $K$ 次，以便从给定的 rollout 中学到更多（代价可能较高）。每一遍看到的 $\pi_\theta$ 都更新过，使得 ratio 偏离得更远。典型取值 $K = 2$–$4$；超过约 $\sim$6 时，policy 就已经过于 off-policy。

当 $K = 1$ 且**不**使用 minibatching 时：$\pi_\theta = \pi_{\theta_\text{old}}$，ratio 恒为 1，clipping 永远不会触发——PPO 退化为带 GAE 的普通 policy gradient。

PPO 和 GRPO 都使用这种相同的样本复用结构。

---

## Value function 目标

critic $V_\phi$ 学习预测**回报（return）**——即从每个 token 起的总折扣奖励。GAE 通过将实际奖励与 critic 自身的预测相结合，给出了 advantage：

$$\hat{A}_t = \hat{G}_t - V_{\phi_\text{old}}(s_t)$$

$\hat{G}_t$ 比 $V_\phi$ 当前给出的预测更接近真实回报。我们把它作为回归目标来改进 critic：

$$\hat{G}_t = \hat{A}_t + V_{\phi_\text{old}}(s_t) \qquad \longrightarrow \qquad \min_\phi \left(V_\phi(s_t) - \hat{G}_t\right)^2$$

```python
targets = (advantages + values_old).detach()
```

`.detach()` 至关重要——目标在 rollout 时就已固定，不是我们要反向传播穿过的对象。

---

## 代码中的 PPO critic loss

value function 有**自己的 clipping**——思路与 policy 的 clip 相同，但很容易被忽略。它防止 critic 在一次更新中偏离 rollout 时的预测过远：

```python
# Current critic predictions
v_pred = value_net(completions)  # (B, L)

# old_values: critic predictions from rollout time (before any training updates, gradient detached)
# Clamp new predictions to stay within eps of the rollout values
v_clip = torch.clamp(v_pred, old_values - eps, old_values + eps)
vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss = torch.max(vf_unclipped, vf_clipped)
```

`torch.max` 会选取更差（更保守）的 loss——如果未 clip 的预测已经接近 target，clip 后的版本就不会产生干扰。

---

## PPO-RLHF：组合目标

组合 loss：policy + value（KL 通过 reward shaping 进入，而不是作为单独的 loss 项）：

```python
per_token_loss = pg_loss + vf_coef * vf_loss  # (B, L)

# Apply completion mask and aggregate
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         completion_mask.sum(dim=1)).mean()
```

`vf_coef`（通常为 0.5–1.0）用于平衡这两个目标。

---

## Advantage whitening

在 batch 内将 advantage 归一化为零均值、单位方差：

```python
valid_adv = advantages[completion_mask.bool()]
advantages = ((advantages - valid_adv.mean()) /
              (valid_adv.std() + 1e-8)) * completion_mask
```

**原因**：稳定不同 batch 之间的梯度幅度。如果不做 whitening，reward 整体偏高或偏低的 batch 可能产生过大的梯度。

---

## Value function 初始化

value function $V_\phi$ 需要从一开始就给出合理的估计：

- **从 RM 骨干网络初始化**（InstructGPT 的做法）：value 预测一开始就接近真实 reward
- **从 SFT 模型 + 随机初始化的 head 初始化**：成本更低，但训练早期不稳定
- **冷启动问题**：如果初始 value 估计很差，GAE advantage 噪声很大 → 训练早期可能非常混乱

Tülu 3 [@lambert2024t] 从 reward model 初始化。 

*细节*：许多面向 LLM 的 RL 设置会做“value function warmup”，即先在带有实测 reward 的数据上训练若干步来帮助 value function 初始化，使其在 policy 更新之前就已稳定。



---

## PPO 超参数

以下范围来自常见的 LLM RLHF 设置，仅供参考，并非通用默认值：

| 超参数 | 典型范围 | 说明 |
|---------------|:-------------:|-------|
| Clip $\varepsilon$ | 0.1–0.2 | 信任域宽度 |
| GAE $\lambda$ | 0.95 | advantage 的偏差-方差权衡 |
| Value 系数 | 0.5–1.0 | critic loss 的权重 |
| KL 系数 $\beta$ | 0.01–0.1 | 对 reference 约束的强度 |
| 每个 batch 的 epoch 数 $K$ | 2–6 | off-policy 预算 |
| 学习率 | $1 \times 10^{-6}$ 至 $5 \times 10^{-6}$ | 远低于 SFT |
| Batch size | 256–1024 个 prompt | 越大方差越低 |

---

## 为什么 batch size 在 RL 中更重要

在监督学习中，梯度噪声适中——临界 batch size 在几千量级。在 RL 中，梯度噪声尺度**高出几个数量级**，因为梯度来自 Monte Carlo rollout，而不是带标签的数据。

<!-- cite-right: mccandlish2018empirical -->

- **OpenAI Dota 2**：临界 batch size 达到**数百万**个 transition [@mccandlish2018empirical]
- **PPO 本身**并不具备 batch size 不变性——clipping 把 batch size 与有效步长耦合在一起，因此不能只靠调整学习率来补偿 [@hilton2021batch]
- **PPO plateau** 往往由噪声较大的 loss 估计造成，只有增加样本量才能消除 [@beukman2026preventing]

大 batch 能使梯度方差按 $1/N$ 的比例降低。在 RLHF 中，这是稳定训练成本最低的方法之一——比大多数超参数调优更有效。

---

## PPO 调试清单

训练期间需要检查：

- **Clip fraction**：被 clip 的 token 百分比——0% 表示 clipping 从未生效，>50% 表示 policy 变化过快
- **Advantage 统计量**：均值应接近 0（如果做了 whitening），标准差应保持稳定
- **Value loss**：应当下降——如果没有，说明 value function 没有在学习
- **KL divergence**：应逐渐增大，而不是爆炸式增长

---

## PPO 调试清单

训练期间需要检查：

- **Clip fraction**：被 clip 的 token 百分比——0% 表示 clipping 从未生效，>50% 表示 policy 变化过快
- **Advantage 统计量**：均值应接近 0（如果做了 whitening），标准差应保持稳定
- **Value loss**：应当下降——如果没有，说明 value function 没有在学习
- **KL divergence**：应逐渐增大，而不是爆炸式增长

常见的**隐性 bug**——训练能跑，但学到的是错误的东西：

- policy loss 中包含了 prompt token（抬高 loss，并把梯度浪费在无法改变的 token 上）
- reward 被放到了错误的位置（例如 `[:, -1]`，而不是最后一个生成的 token）
- 把过时的 log-prob 当作“当前”值使用（ratio 一直接近 1，clipping 从不生效）
- 标准差为零的 GRPO group（所有 completion 获得相同 reward → advantage 为 NaN）
- value 或 policy loss 在 completion mask 之外计算（在 padding 上训练）

---

## 内存中的四个模型

以 fp16 精度的 7B 模型为例：

| 模型 | 大小 | 用途 |
|-------|:----:|---------|
| Policy $\pi_\theta$ | ~14 GB | 训练对象 |
| Value function $V_\phi$ | ~14 GB | 学习得到的 critic |
| Reference policy $\pi_\text{ref}$ | ~14 GB | KL 锚点（冻结） |
| Reward model $r_\psi$ | ~14 GB | 打分（冻结） |

仅模型权重就需要 **~56 GB**，还不包括 optimizer states、activations 和 gradients。这就是 PPO 往往需要模型并行或 offloading 的原因。部分实现通过共享 backbone 来降低开销（例如在 policy 网络上加一个 value head）。

---

<!-- layout: section-break -->

## Group relative policy optimization（GRPO）及相关方法

---

## 回顾：Group relative policy optimization（GRPO）

<!-- cite-right: shao2024deepseekmath -->

对每个 prompt，采样 $G$ 个 completion，并计算组内归一化的 advantage：

$$\hat{A}_i = \frac{R_i - \mu_G}{\sigma_G}, \qquad \mu_G = \frac{1}{G}\sum_{j=1}^{G} R_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j - \mu_G)^2}$$

然后使用与 PPO 相同的 clipped loss（取最小化）——逐 token 的 ratio，但 advantage 是序列级的——再加上可选的 KL 惩罚项（详见推理讲）：

$$L^{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|a_i|}\sum_{t=1}^{|a_i|}\min\!\Big(\rho_{i,t} \hat{A}_i,\; \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\, \hat{A}_i\Big) + \beta \, \text{KL}(\pi_\theta \| \pi_\text{ref})$$

没有 value function，也没有 GAE——advantage 完全来自同一组内同 prompt 样本之间的比较。

---

## GRPO 代码实现

<!-- cite-right: shao2024deepseekmath -->

```python
# Generate G completions per prompt, then compute group advantages
mean_r = rewards.view(-1, G).mean(dim=1)
std_r = rewards.view(-1, G).std(dim=1)
mean_r = mean_r.repeat_interleave(G)
std_r = std_r.repeat_interleave(G)
advantages = ((rewards - mean_r) / (std_r + 1e-4)).unsqueeze(1)

# Importance sampling ratio
ratio = torch.exp(new_logps - old_logps)   # (B*G, L)

# Clipped surrogate (same as PPO)
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - eps, 1 + eps)
pg_loss = torch.max(pg_losses1, pg_losses2)

# KL penalty in loss (not in reward)
per_token_loss = pg_loss + beta * per_token_kl

loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()
```

---

## KL 惩罚的位置

<!-- rows: 30/70 -->

一个关键的实现细节——KL 惩罚放在哪里。目标相同（约束与 reference 的偏离），位置不同。GRPO 的做法避免了 KL 与 advantage 估计之间的相互影响。

===

<!-- row-columns: 50/50 -->

**PPO（KL 放在 reward 中）**：
```python
# per_token_kl is (B, L), rewards is (B, L)
# Scatter RM score to each sequence's
# last action token, not [:, -1]
rewards.scatter_(1, last_idx, rm_score)
rewards = rewards - beta * per_token_kl
advantages = gae(rewards, values, ...)
```

|||

**GRPO（KL 放在 loss 中）**：
```python
# Compute advantages from raw rewards
advantages = z_score(rewards)
# Add KL as separate loss term
loss = pg_loss + beta * per_token_kl
```

---

## GRPO 与 PPO 的实现对比

GRPO 相比 PPO 去掉了什么：

| 组件 | PPO | GRPO |
|-----------|:---:|:----:|
| Value network | 需要 | 无 |
| GAE 计算 | 需要 | 无 |
| Critic loss | 需要 | 无 |
| Value function 初始化 | 需要 | 无 |
| Advantage 计算 | 逐 token（GAE） | 逐序列（z-score） |
| KL 处理方式 | 折入 reward | 单独的 loss 项 |

代码显著减少，内存中也少一份模型副本（约 1 个）。

---

## RLOO 与 GRPO 的代码对比

<!-- columns: 50/50 -->

**RLOO advantage**：
```python
# Leave-one-out mean
rewards = rewards.view(-1, K)         # (N, K)
baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (K - 1)
advantages = rewards - baseline
advantages = advantages.reshape(-1)
```

|||

**GRPO advantage**：
```python
# Group z-score normalization
rewards = rewards.view(-1, G)       # (N, G)
mean_r = rewards.mean(dim=1, keepdim=True)
std_r = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean_r) \
             / (std_r + 1e-4)       # (N, G)
advantages = advantages.reshape(-1)
```

<div class="colloquium-spacer-md"></div>

结构相同，baseline 的计算方式不同。GRPO 增加了 std 归一化；RLOO 使用 leave-one-out 均值。

---

## 最新进展：GSPO 与 CISPO

<!-- columns: 50/50 -->

**GSPO** — 序列级 ratio：

$$\bar{\rho}_i = \exp\!\Big(\frac{1}{|a_i|}\sum_t \log \rho_{i,t}\Big)$$

将逐 token 的 ratio 合并为每条序列一个几何平均 ratio。每条序列裁剪一次，而不是每个 token 裁剪一次。

|||

**CISPO** — stop-gradient 裁剪：

$$\text{sg}\!\big[\text{clip}(\rho, 1\!-\!\varepsilon, 1\!+\!\varepsilon)\big] \cdot \hat{A} \cdot \log \pi_\theta$$

对裁剪后的 ratio 做 detach，使梯度只经过 $\log \pi_\theta$。ratio 相当于一个固定权重。

---

## GSPO 与 CISPO 的代码

<!-- columns: 50/50 -->

<div class="text-sm">

**GSPO** [@zheng2025gspo] — 序列级 ratio，GRPO 风格算法：
```python
log_ratio = (new_logps - old_logps) * mask
rho = torch.exp(log_ratio.sum(dim=1) / mask.sum(dim=1))  # (B*G,)
# Same clipped loss as GRPO, but per-sequence
rho_clipped = rho.clamp(1 - eps, 1 + eps)
loss = -torch.min(rho * advantages, rho_clipped * advantages).mean()
```

</div>

|||

<div class="text-sm">

**CISPO** [@minimax2025minimaxm1scalingtesttimecompute] — 对裁剪后的 ratio 做 stop-gradient，REINFORCE 风格算法：
```python
rho = torch.exp(new_logps - old_logps)
rho_clipped = torch.clamp(rho, 1 - eps, 1 + eps).detach()  # no grad through ratio
loss = -(rho_clipped * advantages.unsqueeze(1) * new_logps * mask).sum() / mask.sum()
```

</div>

---

<!-- layout: section-break -->

## 训练基础设施

---

## On-policy 与 off-policy

<!-- cite-right: noukhovitch2024asynchronous -->

**理想 / 理论（on-policy）**：生成 → 更新 → 生成 → 更新

每一批 completion 被打分后，只用于一个短的更新窗口（一个或几个 epoch），然后丢弃。

**现实（异步）**：生成与训练在不同的 GPU 组上重叠进行，以获得更高吞吐。用于生成的模型可能落后训练模型 1–N 步。

**权衡**：完美的 on-policy 很慢（生成或训练期间 GPU 空闲）。轻微的滞后通常没问题。

---

## 同步与异步训练

<!-- cite-right: noukhovitch2024asynchronous -->

<!-- img-align: center -->

![同步训练会使 GPU 空闲；将生成与训练分离有所帮助，但 on-policy 仍需等待。异步（off-policy）让两者重叠，实现满载利用。](assets/async_v_synch_rl.png)

---

## 异步 RL 训练

<!-- columns: 50/50 -->

现代 LLM 的 RL 将算力分为两组：

- **Actors**（推理 GPU）：使用 vLLM 等工具生成 completion
- **Learners**（训练 GPU）：计算 policy gradient 更新

由进程管理库（如 Ray）协调两者之间的数据流。模型权重定期从 learner 同步到 actor。

|||

![Distributed RL system with prompt and result queues between learner and actor GPUs.](assets/distributed-rl.png)

---

## 更多 RL 实现资源

- 我录制的一段[视频](https://www.youtube.com/watch?v=amrJDwMUFNs)，带大家阅读实现 GRPO、DAPO、Dr. GRPO 等论文的代码库。
- 约 24 分钟处，关于为 Olmo 3 扩展 RL 的[演讲](https://youtu.be/uaZ3yRdYg8A?si=iSGw56BFNQMWNjtr&t=1487)。
- Finbarr Timbers 关于如何让 RL 更快的[博客文章](https://finbarr.ca/making-rl-fast/)。

---

## 开源 RL 代码库

- **TRL** [@vonwerra2022trl] — Hugging Face 生态，支持 PPO/GRPO/DPO。最适合入门
- **Open Instruct** [@ivison2024unpacking] — Allen AI，支持多种算法。最适合研究与复现
- **veRL** — 在 RLVR 时代非常流行。
- **OpenRLHF** — 最初为 RLHF 而生，在 RLVR 等场景中同样流行。



---

<!-- layout: section-break -->

## 实践工程

---

## 关键数值问题

---

## 关键数值问题

- **对数空间运算**：始终使用 `log_softmax` + `gather`，绝不要先 `softmax` 再 `log` — softmax 会把很小的概率压成极小的浮点数，随后 `log` 会放大这种精度损失
- **屏蔽 padding token**：`completion_mask` 只能在 completion token 上为 1 — 要排除 prompt token、EOS 之后的 padding 以及 EOS token 本身（或统一地包含 EOS）。聚合之前先将损失乘以该 mask

---

## 关键数值问题

- **对数空间运算**：始终使用 `log_softmax` + `gather`，绝不要先 `softmax` 再 `log` — softmax 会把很小的概率压成极小的浮点数，随后 `log` 会放大这种精度损失
- **屏蔽 padding token**：`completion_mask` 只能在 completion token 上为 1 — 要排除 prompt token、EOS 之后的 padding 以及 EOS token 本身（或统一地包含 EOS）。聚合之前先将损失乘以该 mask
- **停止 token 的处理**：EOS token 需要一致的处理方式 — 计入损失或不计入都可以，但必须保持一致。处理不当是常见的隐性错误
- **序列长度处理**：变长 completion 需要谨慎地做归一化（下一节介绍）

---

## 关键数值问题

- **对数空间运算**：始终使用 `log_softmax` + `gather`，绝不要先 `softmax` 再 `log` — softmax 会把很小的概率压成极小的浮点数，随后 `log` 会放大这种精度损失
- **屏蔽 padding token**：`completion_mask` 只能在 completion token 上为 1 — 要排除 prompt token、EOS 之后的 padding 以及 EOS token 本身（或统一地包含 EOS）。聚合之前先将损失乘以该 mask
- **停止 token 的处理**：EOS token 需要一致的处理方式 — 计入损失或不计入都可以，但必须保持一致。处理不当是常见的隐性错误
- **序列长度处理**：变长 completion 需要谨慎地做归一化（下一节介绍）
- **Detach baseline**：`advantages.detach()` — 不要通过 advantage 的计算过程反向传播
- **除法保护**：凡是除以 `mask.sum()` 或序列长度的地方，都要使用 `.clamp_min(1)` 或 `+ eps`，避免空 completion（如立即输出 EOS）导致 NaN

---

## 超越损失公式

我们见过的每个算法计算的都是同一个核心梯度：$\nabla \log \pi \cdot A$。但如何把逐 token 的损失**聚合**成一个标量，对训练动态的影响超出你的预期。

下一节将介绍三种策略 — 按序列、按 token 和固定长度归一化 — 以及为什么这个选择在实践中很重要。

---

<!-- layout: section-break -->

## 损失聚合策略

---

## Bandit 式与 MDP 式 RLHF

<!-- columns: 50/50 -->

**Bandit 式**
- 每个 completion 对应一个 reward
- 序列级 $\Psi_t$ 广播到所有 token
- REINFORCE、RLOO、GRPO 默认采用

|||

**MDP 式**
- 每个 token 都被视为一个 action
- 逐 token 的 value 或 advantage
- 带 GAE 的 PPO 默认采用

<div class="colloquium-spacer-md"></div>

实践中大多数 RLHF 是混合式的：**序列级 reward**，但**token 级 log-prob 梯度**。PPO 式 RLHF 通常从序列级 reward model 分数出发，再通过 KL shaping 和 GAE 获得 token 级的 credit。

---

## 为什么这很重要

算法相同，聚合方式不同 → **训练动态不同**。

这通常通过训练过程中序列长度的变化来衡量，在 RLVR 中尤其如此。

---

## 按序列归一化

每个序列对 batch loss 的贡献**相等**，与长度无关：

$$L = \frac{1}{B}\sum_{i=1}^{B} \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \ell_{i,t}$$

```python
# Strategy 1: Per-sequence normalization
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         completion_mask.sum(dim=1)).mean()
```

GRPO 以及部分 PPO 实现的标准做法。

---

## 按序列归一化：效果

每个序列权重相等 → 单个 token 的梯度**与序列长度成反比**：

- 短序列（5 个 token）：每个 token 的梯度 $\propto 1/5 = 0.20$
- 长序列（10 个 token）：每个 token 的梯度 $\propto 1/10 = 0.10$

短序列的**单 token 梯度更大**，这可能使模型偏离冗长的回答。

---

## 按 token 归一化

整个 batch 中每个 token 的贡献**相等**：

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$

```python
# Strategy 2: Per-token normalization
loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
```

用于 DAPO [@yu2025dapo]。更长的序列按比例贡献更多梯度。

---

## 按 token 归一化：效果

所有 token 的梯度幅度相同。更长的序列因为 token 更多，对总梯度的贡献也更大。

可能偏向**冗长的 completion**——模型从更长的回答中获得更多梯度信号，这可能鼓励生成更长的内容。

---

## 固定长度归一化

除以常数 $L_\text{max}$（最大生成长度）进行归一化：

$$L = \frac{1}{B}\sum_{i=1}^{B} \frac{1}{L_\text{max}}\sum_{t=1}^{|a_i|} \ell_{i,t}$$

```python
# Strategy 3: Fixed-length normalization
loss = ((per_token_loss * completion_mask).sum(dim=1) /
         L_max).mean()
```

来自 Dr. GRPO [@liu2025understanding]。使单 token 的尺度一致，同时让更长的序列贡献更多总梯度（求和中有效 token 更多）。

---

## 聚合策略对比

```python
seq_1_losses = [1, 1, 1, 1, 10]           # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

| 策略 | Batch loss | 短序列梯度 | 长序列梯度 |
|----------|:----------:|:------------------:|:-----------------:|
| **按序列** | $(2.8 + 1.9)/2 = 2.35$ | 每个 token 0.20 | 每个 token 0.10 |
| **按 token** | $(14 + 19)/15 = 2.2$ | 每个 token 0.067 | 每个 token 0.067 |
| **固定长度**（$L=10$） | $(1.4 + 1.9)/2 = 1.65$ | 每个 token 0.10 | 每个 token 0.10 |

**按序列**使短序列的单 token 梯度更大；**按 token** 和**固定长度**则使其趋于一致。

---

## 训练中需要监控什么

以下范围仅作示意——阈值随模型规模、任务和算法而异：

| W&B 面板 | 健康 | 不健康 |
|-------------|---------|-----------|
| `reward/mean` | 稳步上升 | 尖峰、震荡、停滞 |
| `kl/mean` | 逐渐增大（0 → 2–5） | 爆炸（>10）或始终为 0 |
| `loss/policy` | 下降 | 发散或 NaN |
| `metrics/clip_frac` | 5–30% | 0% 或 >50% |
| `generation/length` | 稳定或小幅增加 | 单调增加（length hack） |
| Policy entropy | 缓慢下降 | 骤降至 0（mode collapse） |

此外还要监控：留出 benchmark 上的评测分数，并阅读样本输出以检查连贯性。

---

<!-- ## 扩展的 RL 训练实践

Reasoning 模型把 RL 推向极致，以下是变化之处：

- **移除 KL**：长时间训练时——reasoning 模型常常不加 KL 约束进行训练
- **放宽 clipping**（DAPO）：非对称的 $\varepsilon_\text{low} \neq \varepsilon_\text{high}$，以获得更好的探索
- **渐进式训练**：从较简单的问题开始，逐步提高难度
- **难度过滤**：当 20–80% 的 completion 能解出问题时效果最好（信号充足，又不会过于简单）
- **动态采样**：移除所有 completion 全对或全错的 prompt（没有学习信号）

--- -->

## 前瞻：正则化与过度优化

当 RL 训练出问题时，往往是因为 RL 抓住了虚假信号，而不是预期的 reward。 
这些主题值得单独开一讲：

- **KL 惩罚**：forward KL 与 reverse KL，数学与代码
- **Goodhart 定律**：当代理目标偏离真实质量时
- **Reward hacking**：谄媚（sycophancy）、冗长、过度拒答
- **隐式正则化**："SFT Memorizes, RL Generalizes"、"RL's Razor"

将在后续关于第 14 章和第 15 章的讲座中深入讲解。

---


<!-- layout: section-break -->

## 结论

---

## 本讲小结

从数学到可运行的代码——真正重要的实现细节：

1. **核心代码模式** — log-probs、masking，以及 REINFORCE/RLOO/PPO/GRPO 的损失函数
2. **Advantage 估计** — 按 prompt 的 baseline（RLOO、GRPO z-score）vs 按 token 的 credit 分配（GAE）
3. **损失聚合** — 按序列、按 token 与固定长度归一化会改变训练动态
4. **PPO 的复杂度预算** — value function 目标、critic clipping、双重正则化
5. **异步训练** — actor/learner 拆分、staleness 预算、分布式基础设施
6. **工程实践** — batch size、监控、调试、训练配方

---

<!-- rows: 50/50 -->
## 资源

<!-- row-columns: 50/50 -->

```box
title: 书籍与课程
tone: accent
compact: true
content: |
  - [rlhfbook.com](https://rlhfbook.com) — 全书
  - [RL Cheatsheet](https://rlhfbook.com/rl-cheatsheet) — 一页速查表
  - [GitHub repo](https://github.com/natolambert/rlhf-book)
```

|||

```box
title: 代码库
tone: surface
compact: true
content: |
  - [TRL](https://github.com/huggingface/trl) — Hugging Face
  - [Open Instruct](https://github.com/allenai/open-instruct) — Allen AI
  - [veRL](https://github.com/volcengine/verl) — 字节跳动
  - [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
```

===

<!-- row-columns: 50/50 -->

```box
title: 关键论文
tone: surface
compact: true
content: |
  - PPO [@schulman2017proximal]
  - REINFORCE baselines [@ahmadian2024back]
  - GRPO / DeepSeekMath [@shao2024deepseekmath]
  - GAE [@schulman2015high]
```

|||

```box
title: 延伸阅读
tone: surface
compact: true
content: |
  - Tülu 3 [@lambert2024t] — 完整的开源配方
  - DeepSeek R1 [@guo2025deepseek] — 推理 RL
  - Unpacking DPO & PPO [@ivison2024unpacking]
```

---

<!-- rows: 50/50 -->
## 接下来：DAAs 与 reasoning

<!-- row-columns: 32/36/32 -->

```box
title: 概览
tone: muted
compact: true
content: |
  1. 引言
  2. 关键相关工作
  3. 训练概览
```

|||

```box
title: 核心训练流程
tone: accent
compact: true
content: |
  4. 指令微调
  5. Reward model
  6. **强化学习**
  7. Reasoning
  8. 直接对齐
  9. 拒绝采样
```

|||

```box
title: 数据与偏好
tone: muted
compact: true
content: |
  10. 什么是偏好
  11. 偏好数据
  12. 合成数据与 CAI
```

===

<!-- row-columns: 32/36/32 -->

```box
title: 实践考量
tone: muted
compact: true
content: |
  13. 工具使用
  14. 过度优化
  15. 正则化
  16. 评估
  17. 产品与角色
```

|||

```box
title: 附录
tone: surface
compact: true
content: |
  - A. 定义
  - B. 风格与信息
  - C. 实践问题
```

|||

```box
title: 课程主页
tone: surface
compact: true
content: |
  - [rlhfbook.com](https://rlhfbook.com)
  - [GitHub 仓库](https://github.com/natolambert/rlhf-book)
```

---

## 课程大纲

1. **引言与训练概览** — 第 1–3 章
2. **指令微调、reward model、拒绝采样** — 第 4、5、9 章
3. **RL 理论** — 第 6 章（上）
4. **RL 实现与实践** — 第 6 章（下）
5. **Reasoning** — 第 7 章
6. **直接对齐算法** — 第 8 章
7. ...

---

<!-- rows: 85/15 -->
## 谢谢大家

提问 / 讨论

联系方式：nathan@natolambert.com

通讯：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===


```builtwith
repo: natolambert/colloquium
```
