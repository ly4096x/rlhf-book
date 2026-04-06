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
