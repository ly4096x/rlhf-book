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
