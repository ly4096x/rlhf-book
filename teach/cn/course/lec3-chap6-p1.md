---
title: "第 3 讲：强化学习"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 3 讲"
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

# 第 3 讲：强化学习的动机与数学

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">后训练课程。第 6 章，第 1 部分</p>

---

<!-- rows: 50/50 -->
## 第 3 讲：强化学习（以数学为主）

<!-- row-columns: 32/36/32 -->

```box
title: 总览
tone: muted
compact: true
content: |
  1. 引言
  2. 重要相关工作
  3. 训练概述
```

|||

```box
title: 核心训练流程
tone: accent
compact: true
content: |
  4. Instruction tuning
  5. Reward models
  6. **强化学习**
  7. 推理
  8. 直接对齐
  9. Rejection sampling
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
  17. 产品与 character
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
  - [GitHub repo](https://github.com/natolambert/rlhf-book)
```

---

<!-- rows: 30/70 -->
## 我们在流程中的位置

在指令微调和训练 reward model 之后，RL 步骤会针对学到的奖励信号更新 policy。

系统图视角。

===

![RLHF 训练流程——RL 步骤利用 reward model 信号优化 policy。](assets/rlhf-basic.png)

---

<!-- columns: 50/50 -->
## 我们在流程中的位置

在指令微调和训练 reward model 之后，RL 步骤会针对学到的奖励信号更新 policy。

RLHF 优化视角。

|||

![RLHF 训练流程——RL 步骤利用 reward model 信号优化 policy。](assets/rlhf-overview.png)

---

## 本讲内容

本讲涵盖语言模型 RL 的**数学与理论**。下一讲介绍具体实现。

```box
title: 第 3 讲大纲
tone: accent
content: |
  1. **动机** —— 为什么 RL 训练很重要
  2. **RL 基础回顾** —— MDP、回报、value function、policy gradient 目标
  3. **Policy gradient 推导** —— log-derivative trick，逐步推导
  4. **REINFORCE 与 RLOO** —— 最简单的 policy gradient 算法
  5. **PPO** —— trust region、clipping、GAE、value function
  6. **GRPO 与现代变体** —— GSPO、CISPO，以及走向简化的趋势
  7. **对比** —— 优势、劣势与问题
```

---

## RL 一个未被充分记录的好处

在第 2 讲中，我们介绍了 rejection sampling，之后还会介绍 direct alignment 算法（如 DPO）。它们更简单，但 RL 带来的好处难以衡量。

实现 RL 需要复杂得多的基础设施，但它提供的梯度更新“通常能大幅帮助模型”。这很难量化，但：
- RL 可以“修复”模型的粗糙之处——让输出更鲁棒、格式更规范，并与 vLLM 等推理服务框架更兼容
- RL 作用于较窄的 prompt 分布，往往不会“压垮”通用能力——可以有针对性地改进特定行为，而不损害其他能力

总体而言，语言模型上的 RL 损失函数鲁棒、可扩展、有效且灵活，由此开启了大量新的实验领域。让我们走上这条道路的最初方法，正是 RLHF 相关工作。


---

<!-- layout: section-break -->

## 为什么 RL 很重要

---

<!-- columns: 50/50 -->
## RL 扩展时代

<!-- cite-right: openai2024o1 -->

RL 如今是训练最强模型的**关键环节**。

推理模型（o1、DeepSeek R1 等）正是用这些算法在可验证奖励上训练的。

- 通过 RL 扩展训练时算力
- 通过延长推理扩展测试时算力
- RL 从“锦上添花”变成了核心能力的驱动力

|||

![OpenAI o1：扩大 RL 训练算力可提升推理性能。](assets/o1-train-time.png)

---

<!-- columns: 50/50 -->
## RLVR：相同的算法，可验证的奖励

<!-- cite-right: lambert2024t -->

本讲的 RL 方法同时驱动 **RLHF 和 RLVR**：

- **RLHF**：reward model 对主观质量打分
- **RLVR**：验证函数检查正确性（数学、代码）

相同的 policy gradient 算法，不同的奖励来源。我们会在后续的讲座中深入介绍推理训练的技巧和一系列有趣的模型。

|||

![RLVR 用验证函数代替 reward model，但 RL 算法是相同的。](assets/rlvr-system.png)

---

<!-- layout: section-break -->

## Policy gradient：核心直觉

---

<!-- columns: 50/50 -->
## 回顾：经典 RL 与 RLHF

<!-- cite-right: christiano2017, ouyang2022training -->

<div class="text-sm">

**经典 RL**：MDP 中的 agent $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$
- 奖励是环境在每一步给出的已知函数
- 优化累计回报：$J(\pi) = \mathbb{E}_{\tau \sim p_\pi}\!\left[\sum_{t} \gamma^t r(s_t, a_t)\right]$

**RLHF**：prompt 来自数据集，没有环境
- 奖励是从人类偏好中**学习**得到的
- **回复级别**的奖励，并用 KL 惩罚进行正则化
- $J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$

</div>


|||

![](assets/rlhf.png)


---

<!-- columns: 50/50 -->
## 本讲的符号约定

<div class="text-sm">

本讲采用强化学习文献中的 $(s, a)$ 符号，其中 $s$ 表示状态，$a$ 表示动作。在语言模型的语境下，你更常看到的是 $(x, y)$，其中 $x$ 是 prompt，$y$ 是 completion。

$(s, a)$ 的框架更为通用——这些算法最初是为序贯决策问题设计的，每个时间步都要执行一个动作。不过，许多 RLHF 实现把整段 completion 视为单个动作，因此 $(x, y)$ 符号同样成立。

**符号**：$r_t$ 表示每步奖励，$R(\tau)$ 表示轨迹回报，$\rho_t$ 表示重要性采样比率。我们把 $\rho$ 专门留给比率，以免与奖励混淆。

</div>


|||

![](assets/rlhf.png)


---

## 语言模型作为 episodic MDP

| MDP 概念 | 语言模型 |
|-------------|---------------|
| 状态 $s_t$ | prompt + 目前已生成的 token：$(x, y_{<t})$ |
| 动作 $a_t$ | 下一个 token $y_t$ |
| 状态转移 | 确定性：把 token 追加到序列末尾 |
| 策略 $\pi_\theta(a_t \mid s_t)$ | 语言模型的下一个 token 分布 |
| Episode | 一个 prompt → completion |
| 终止奖励 | RM 打分或 verifier 输出 |
| $\gamma$ | 通常为 1.0（不折扣） |

这个 MDP 有两种视角：**token 级**（每个 token 是一个独立动作，用于带 GAE 的 PPO）与**序列级**（整段 completion 是一个动作，用于 REINFORCE/GRPO）。

---

## 我们在优化什么？

选择策略参数 $\theta$，使当前策略下的奖励**平均**最大。

设 $p_\theta(\tau)$ 为当前策略诱导出的轨迹分布。

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$$

实际中，我们用采样得到的 rollout 来估计这个期望：

$$J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} R(\tau_i), \qquad \tau_i \sim p_\theta$$

我们构造一个梯度/导数，用它来进行优化。

---


## policy gradient 如何工作

让带来更好结果的动作更有可能被选中。

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

<div class="text-sm">

可以把它理解为同时回答两个问题：

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — **是哪个动作？**
- $\Psi_t$ — **它有多好？** 一个对结果打分的标量。

</div>

---

## policy gradient 如何工作

让带来更好结果的动作更有可能被选中。

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

<div class="text-sm">

可以把它理解为同时回答两个问题：

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — **是哪个动作？** 每个参数对在状态 $s_t$ 下采取动作 $a_t$ 的概率有多大影响。它把结果与造成这一结果的“旋钮”联系起来。
- $\Psi_t$ — **它有多好？** 一个对结果打分的标量。正值表示好，负值表示差，大小表示程度。

> 一种简化的理解（例如 batch size 为 1 时的直觉）：梯度是一个向量，每个参数对应一个分量。分量为正表示“增大这个参数会让该动作更有可能出现”，为负则相反。实际中，更新会对许多这样的向量取平均——最终留下的是整个 batch 的净投票结果。

相乘之后：$\Psi_t > 0$ 会更新参数，使 $a_t$ 更有可能出现；$\Psi_t < 0$ 则使其更不可能出现。

本节余下的内容都在讨论如何选择一个合适的 $\Psi_t$——不同的选择（总回报、advantage、TD residual）在方差与偏差之间各有取舍，但都适用于同一个更新公式。

</div>

---

## policy gradient 公式

先预览我们最终会得到什么：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

核心思想是：我们*在环境中对多次 trial 采样*，并**估计**梯度。

其中 $\Psi_t$ 是**学习信号**，告诉优化器该动作有多好。$\Psi_t$ 的选择决定了算法的方差、偏差和计算开销。


---

## RL 的关键量

本讲中反复出现三个量：

**Value function** $V(s)$：从状态 $s$ 出发的期望未来 return

$$V(s) = \mathbb{E}\big[G_t \mid S_t = s\big] \quad \text{where } G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**Action-value** $Q(s, a)$：在状态 $s$ 下采取动作 $a$ 之后的期望 return

$$Q(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$$

**Advantage** $A(s, a) = Q(s, a) - V(s)$：动作 $a$ 比平均水平好多少？为正 → 强化，为负 → 抑制，为零 → 不更新。

在 RLHF 中，通常 $\gamma = 1$（不做折扣），因为优化的单位是完整的 completion。

---

## $\Psi_t$ 可以是什么

<!-- cite-right: schulman2015high -->

$\Psi_t$ 的常见选择（reward 也可以用 $\gamma$ 折扣）：

| | $\Psi_t$ | 描述 | 方差 | 偏差 |
|:-:|-----------|-------------|:--------:|:----:|
| 1. | $R(\tau) = \sum_{t=0}^{T} r_t$ | trajectory 总 reward | 最高 | 无 |
| 2. | $\sum_{t'=t}^{T} r_{t'}$ | 从 $t$ 起的未来 return（即 return，$G_t$） | 高 | 无 |
| 3. | $G_t - b(s_t)$ | 带 baseline 的 return | 较低 | 无 |
| 4. | $Q^{\pi}(s_t, a_t)$ | 状态-动作 value function | 中 | 视情况 |
| 5. | $A^{\pi}(s_t, a_t) = Q - V$ | Advantage function | 低（$V$ 足够好时） | 无 |
| 6. | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD 残差 | 低 | 有一些 |

*baseline* $b(s_t)$ 是从 reward 信号中减去的任意值，用于降低方差。下文将说明这不会改变期望梯度。

---


<!-- layout: section-break -->

## 推导 policy gradient

---

## 设定：对期望求梯度

理想情况下，我们想直接得到 $\nabla_\theta J(\theta)$，但这做不到（状态采样分布本身依赖于 $\theta$）。

写成积分形式，目标函数为：

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] = \int_\tau p_\theta(\tau) R(\tau) \, d\tau$$

---

## 设定：对期望求梯度

理想情况下，我们想直接得到 $\nabla_\theta J(\theta)$，但这做不到（状态采样分布本身依赖于 $\theta$）。

写成积分形式，目标函数为：

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] = \int_\tau p_\theta(\tau) R(\tau) \, d\tau$$

直接对其求梯度：

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau$$

我们可以从 $p_\theta(\tau)$ 中采样 trajectory，但无法从 $\nabla_\theta p_\theta(\tau)$ 中采样，因为它不是概率分布。

推导依靠几个关键技巧，让我们能够近似或计算它。

---

## 应用"从 trajectory 计算梯度"的思想

将 log-derivative 恒等式（稍后详述）代入梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau
&& \text{differentiate the objective} \\
&= \int_\tau p_\theta(\tau) R(\tau) \nabla_\theta \log p_\theta(\tau) \, d\tau
&& \text{log-derivative identity (chain rule)} \\
&= \mathbb{E}_{\tau \sim p_\theta}\!\left[R(\tau) \nabla_\theta \log p_\theta(\tau)\right]
&& \text{expectation form}
\end{aligned}
$$

现在我们可以用 Monte Carlo 采样来估计它了！

---

## 应用"从 trajectory 计算梯度"的思想

唯一不那么显然的一步是中间那一行：

$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau)
&= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \\
\Rightarrow \quad
\nabla_\theta p_\theta(\tau)
&= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
\end{aligned}
$$

这就是 **log-derivative 恒等式**，由 $\log$ 的链式法则推出。

它把概率的梯度改写成 log 概率的形式，后者处理起来容易得多。

---

## 应用从 trajectory 计算梯度的思路

$p_\theta(\tau)$ 这一项去哪了？

它变成了期望内部的**采样分布**：

$$\int p_\theta(\tau) f(\tau)\, d\tau = \mathbb{E}_{\tau \sim p_\theta}[f(\tau)]$$

随后 Monte Carlo 通过从 policy 中采样 trajectory 来估计该期望：

$$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] \approx \frac{1}{N}\sum_{i=1}^{N} f(\tau_i), \qquad \tau_i \sim p_\theta$$

所以代码中没有显式的 $p_\theta(\tau_i)$ 项：**概率更高的 trajectory 本来就会在采样的 batch 中更频繁地出现**。

从宏观上看，rollout 生成负责采样，而 loss 代码负责处理 $R(\tau_i)\,\nabla_\theta \log p_\theta(\tau_i)$。

---

## 展开 trajectory 的对数概率以计算 $\nabla_\theta$

trajectory 的概率可以分解为：

$$p_\theta(\tau) = d_0(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)$$

取对数：

$$\log p_\theta(\tau) = \log d_0(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

对语言模型而言，这就是熟悉的自回归模式：序列的对数概率等于各 token 对数概率之和。

---

## 展开 trajectory 的对数概率以计算 $\nabla_\theta$

<div class="text-sm">

$$\log p_\theta(\tau) = \log d_0(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

现在对 $\theta$ 求梯度：

- $\nabla_\theta \log d_0(s_0) = 0$ — 初始状态分布与 $\theta$ 无关
- $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ — 环境动态与 $\theta$ 无关
- 只有 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 保留下来！

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

</div>
---

## 展开 trajectory 的对数概率以计算 $\nabla_\theta$

<div class="text-sm">

$$\log p_\theta(\tau) = \log d_0(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

现在对 $\theta$ 求梯度：

- $\nabla_\theta \log d_0(s_0) = 0$ — 初始状态分布与 $\theta$ 无关
- $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ — 环境动态与 $\theta$ 无关
- 只有 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 保留下来！

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

这就是你在语言模型代码中反复看到下面这段写法的原因：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

自动微分会把这个求和后的对数概率转化为相应的逐 token 梯度之和！
</div>
---

## 从完整回报到未来 reward

时刻 $t$ 的动作无法影响过去的 reward。因此，不再用完整 trajectory 的 reward $R(\tau)$ 加权：

$$\nabla_\theta J = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} R(\tau) \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

我们可以把 $R(\tau)$ 替换为**return-to-go** $G_t = \sum_{t'=t}^{T} r_{t'}$ — 只包含从 $t$ 时刻起的未来 reward：

$$\nabla_\theta J = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

这不会改变期望梯度，但能去除当前动作不可能影响到的过去 reward 所带来的噪声。这就是我们分类中 $\Psi_t$ 从选项 1 → 选项 2 的一步。

---

## baseline 让估计器保持无偏

原始回报噪声很大。由于后续采样的动作，以及经典 RL 中的环境随机性，同一个动作可能出现在总 reward 差异很大的 trajectory 中。

baseline 的作用是对这一噪声信号进行**中心化**：我们不再问“回报是否很高？”，而是问“相对于该状态的预期，回报是更高还是更低？”


---

## baseline 让估计器保持无偏

原始回报噪声很大。由于后续采样的动作，以及经典 RL 中的环境随机性，同一个动作可能出现在总 reward 差异很大的 trajectory 中。

baseline 的作用是对这一噪声信号进行**中心化**：我们不再问“回报是否很高？”，而是问“相对于该状态的预期，回报是更高还是更低？”

使用中心化后的回报：

$$\Psi_t = G_t - b(s_t)$$

其中 baseline $b(s_t)$ 只依赖于状态，而不依赖于采样到的是哪个动作。

减去这个 baseline 不会改变期望梯度：

---

## 基线使估计量保持无偏

<div class="text-sm">

原始回报噪声很大。同一个动作可能出现在总奖励差异极大的轨迹中，原因是后续采样的动作，以及（在经典 RL 中）环境的随机性。

使用中心化的回报，$\Psi_t = G_t - b(s_t)$。

减去这个基线并不会改变期望梯度：

$$
\mathbb{E}[(G_t - b(s)) \nabla_\theta \log \pi_\theta(a \mid s)]
= \mathbb{E}[G_t \nabla_\theta \log \pi_\theta(a \mid s)]
- \mathbb{E}[b(s) \nabla_\theta \log \pi_\theta(a \mid s)]
$$

第一项就是原始估计量，第二项则为零：

$$
\begin{aligned}
\mathbb{E}[b(s) \nabla_\theta \log \pi_\theta(a \mid s)]
&= b(s) \cdot \sum_a \nabla_\theta \pi_\theta(a \mid s) \\
&= b(s) \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s)}_{= 1} \\
&= 0
\end{aligned}
$$

归一化分布的梯度之和为零——因此我们可以自由地减去**任意关于状态的函数**作为基线。这就是为什么 $\Psi_t$ 的选项 3–5 能在不引入偏差的情况下降低方差。

</div>

---

## 回顾：$\Psi_t$ 可以取什么

<!-- cite-right: schulman2015high -->

$\Psi_t$ 的常见选择（奖励也可以按 $\gamma$ 折扣）：

| | $\Psi_t$ | 描述 | 方差 | 偏差 |
|:-:|-----------|-------------|:--------:|:----:|
| 1. | $R(\tau) = \sum_{t=0}^{T} r_t$ | 整条轨迹的总奖励 | 最高 | 无 |
| 2. | $\sum_{t'=t}^{T} r_{t'}$ | 从 $t$ 起的未来回报（即 return，$G_t$） | 高 | 无 |
| 3. | $\sum_{t'=t}^{T} r_{t'} - b(s_t)$ | 带基线的回报 | 较低 | 无 |
| 4. | $Q^{\pi}(s_t, a_t)$ | 状态-动作价值函数 | 中 | 视情况 |
| 5. | $A^{\pi}(s_t, a_t) = Q - V$ | Advantage function | **最低** | 无 |
| 6. | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD 残差 | 低 | 有一些 |

*基线* $b(s_t)$ 是从奖励信号中减去的任意值，用于降低方差——稍后我们会说明为什么它是无偏的。

---


## 策略梯度估计量

结合 log-derivative trick、return-to-go 和基线减法：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

这就是 **policy gradient theorem（策略梯度定理）**。本讲的每个算法，都是在特定的 $\Psi_t$ 与正则化选择下对它的一种具体实现。

对于语言模型，这意味着求和遍历的是 completion 中生成的 token，而 $\Psi_t$ 决定每个 token 的权重。

---

## 策略梯度推导回顾

$$
\begin{aligned}
J(\theta)
&= \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]
&& \text{objective} \\
&= \int_\tau p_\theta(\tau) R(\tau)\, d\tau
&& \text{integral form} \\
\nabla_\theta J(\theta)
&= \int_\tau \nabla_\theta p_\theta(\tau) R(\tau)\, d\tau
&& \text{differentiate} \\
&= \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
&& \text{final estimator}
\end{aligned}
$$

- log-derivative trick 把 $\nabla p_\theta(\tau)$ 变成了可采样的期望
- 展开 $\log p_\theta(\tau)$ 得到一系列策略 log-prob 项之和
- $\Psi_t$ 是学习信号：return、advantage 或 TD 式估计
- 在代码中：采样 rollout，对 token log-prob 求和，用 $\Psi_t$ 加权，再取平均

---

<!-- layout: section-break -->

## 为语言模型搭建 RLHF

---

<!-- columns: 40/60 -->
## RLHF 训练循环

在深入具体算法之前，我们先回顾它们所嵌入的完整 RLHF 流程。

|||

![RLHF 训练流程——RL 步骤利用 reward model 的信号来优化 policy。](assets/rlhf-overview.png)

---

<!-- columns: 50/50 -->
## 参考模型

**参考模型（reference model）** $\pi_\text{ref}$ 是 RL 训练开始时 policy 的冻结副本（通常是 SFT checkpoint）。

它只有一个作用：**锚定 policy，使其在优化过程中不会偏离太远。**

没有它，policy 可能会利用 reward model——找到得分很高、但退化或重复的输出（reward hacking）。

|||

![](assets/rlhf-overview.png)

---

<!-- columns: 50/50 -->
## KL divergence 作为正则化

KL 惩罚衡量当前 policy 在每个 token 上偏离 reference 的程度：

$$\text{KL}_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_\text{ref}(a_t \mid s_t)$$

实际操作中：
1. **生成**：用 $\pi_\theta$ 生成一个 completion
2. **计算 log-probs**：分别在 $\pi_\theta$ 和 $\pi_\text{ref}$ 下计算这些相同 token 的 log-probs
3. **相减**：得到每个 token 的 KL

塑形后的奖励为：中间 token 为 $\tilde{r}_t = -\beta \, \text{KL}_t$，最后一个 token 为 $\tilde{r}_T = R(\tau) - \beta \, \text{KL}_T$。

|||

![](assets/rlhf-overview.png)

---

<!-- columns: 50/50 -->
## RLHF 与 RLVR 中的 KL

**RLHF**（reward model 信号）：KL 惩罚至关重要——reward model 是学出来的代理，没有正则化时 policy 会利用其任何不完美之处。

**RLVR**（可验证奖励，例如数学正确性）：KL 通常会减弱甚至完全去掉。奖励就是 ground truth，可被利用的空间更小。一些 RLVR 设置（例如 DeepSeek R1）干脆去掉 reference model，从而节省显存。

趋势：奖励信号越可靠，对 KL 正则化的需求就越低。

|||

![](assets/rlhf-overview.png)

---

## 两个“旧”policy——不要混淆

| | $\pi_{\theta_\text{old}}$ | $\pi_\text{ref}$ |
|---|---|---|
| **是什么** | 上一次 rollout 时的 policy | RL 训练开始时的 policy（SFT checkpoint） |
| **更新** | 每个 batch（或每 $K$ 步） | 从不更新（冻结） |
| **用于** | importance-sampling 比率 $\rho_t$ | KL 惩罚 |
| **若去掉** | 每个 batch 只能做 1 步梯度更新（on-policy） | 有 reward hacking 风险 |

在某些实现中，“old logprobs”指的是生成时的 logprobs——模型只是在这些 logprobs 计算之后又被更新了，并不是另一份单独的模型副本。

---

<!-- layout: section-break -->

## REINFORCE

---

## REINFORCE

<!-- cite-right: williams1992simple -->

REINFORCE 是 policy gradient 最简单的实例。

它是 **policy gradient 的 Monte Carlo 形式**：采样 trajectory，计算其 return，并用这些采样得到的 return 为 log-prob 梯度加权。



---

## REINFORCE

<!-- cite-right: williams1992simple -->

REINFORCE 是 policy gradient 最简单的实例。

它是 **policy gradient 的 Monte Carlo 形式**：采样 trajectory，计算其 return，并用这些采样得到的 return 为 log-prob 梯度加权。

> 这个名称是一个缩写："REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility"（奖励增量 = 非负因子 × 偏移强化 × 特征资格）。

三个组成部分：

1. **Nonnegative factor**（$\alpha$）：learning rate
2. **Offset reinforcement**（$G_t - b(s_t)$）：为稳定性而中心化的 return
3. **Characteristic eligibility**（$\nabla \log \pi$）：如何把学习信号归因到每个 token



---

## REINFORCE

<!-- cite-right: williams1992simple -->

REINFORCE 是 policy gradient 最简单的实例。

它是 **policy gradient 的 Monte Carlo 形式**：采样 trajectory，计算其 return，并用这些采样得到的 return 为 log-prob 梯度加权。

> 这个名称是一个缩写："REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility"（奖励增量 = 非负因子 × 偏移强化 × 特征资格）。

三个组成部分：

1. **Nonnegative factor**（$\alpha$）：learning rate
2. **Offset reinforcement**（$G_t - b(s_t)$）：为稳定性而中心化的 return
3. **Characteristic eligibility**（$\nabla \log \pi$）：如何把学习信号归因到每个 token

更新规则：

$$\Delta\theta = \alpha \, (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$


---

## baseline 问题

没有 baseline 时，梯度用原始 return $G_t$ 为每个动作加权。这并不能告诉你某个动作是否**比预期更好或更差**——只能说明整体结果有多好。

原始 return 把动作的质量和状态的质量混在了一起，因此梯度方差很大。在糟糕状态下的好动作，与在良好状态下的坏动作，可能产生相近的 return。

---

## baseline 的解决方案

减去 baseline $b(s)$ 可以使信号中心化：此时梯度权重为 $(G_t - b)$，它回答的是“在这个状态下，这个动作比预期更好还是更差？”

$$\mathbb{E}_{a \sim \pi(\cdot \mid s)}[b(s) \nabla_\theta \log \pi_\theta(a \mid s)] = b(s) \cdot \mathbb{E}_{a \sim \pi(\cdot \mid s)}[\nabla_\theta \log \pi_\theta(a \mid s)] = 0$$

由于 $b(s)$ 不依赖于采样到的是哪个动作，它可以提到期望外面并抵消。期望梯度保持不变——但方差会大幅下降。

---

## 带 baseline 的 REINFORCE

完整的 REINFORCE 梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

这里，$G_t - b(s_t)$ 本身就是一个 advantage 估计：实际 return 比状态 $s_t$ 下的预期高出多少。

常见的 baseline：
- 批次内的**平均 reward**（最简单）
- 近期 reward 的**滑动平均**
- **学习得到的 value function** $V_\phi(s)$（可选——通向 actor-critic 方法的桥梁）

基础的 REINFORCE 不需要 critic，只需要 Monte Carlo return 和一个简单的 baseline。加入学习得到的 $V_\phi$ 可以进一步降低方差，但需要额外训练第二个模型，复杂度更高（……也更接近 PPO）。

---

## REINFORCE 架构

![语言模型的基础 REINFORCE 架构。shaped reward 由 reward model 得分与相对 reference model 的 KL 惩罚组合而成。](assets/reinforce_tikz.png)

---

<!-- layout: section-break -->

## REINFORCE leave-one-out (RLOO)

---

## RLOO：leave-one-out baseline

<!-- cite-right: ahmadian2024back -->

**核心思想**：每个 prompt 生成 $K$ 个 completion，用其余 $K-1$ 个 reward 作为 baseline：

$$b(s, a_k) = \frac{1}{K-1}\sum_{i=1,\, i \neq k}^{K} R(s, a_i)$$

第 $k$ 个 completion 的 advantage：

$$\hat{A}(s, a_k) = R(s, a_k) - b(s, a_k)$$

这是一个**按 prompt 计算**的 baseline，能自然地反映 prompt 的难度：难的 prompt 下所有 completion 的 reward 都低，因此 baseline 也低。

*细节：与不带 critic 的 REINFORCE 一样，这个序列级的 advantage 会广播到 completion 中的每个 token。*

---

## RLOO 计算示例

一个 prompt 的 $K = 4$ 个 completion，reward 为 $[0.8, 0.3, 0.6, 0.5]$：

| Completion | Reward | Baseline（其余的平均值） | Advantage |
|:----------:|:------:|:------------------------:|:---------:|
| 1 | 0.8 | $(0.3 + 0.6 + 0.5)/3 = 0.467$ | $+0.333$ |
| 2 | 0.3 | $(0.8 + 0.6 + 0.5)/3 = 0.633$ | $-0.333$ |
| 3 | 0.6 | $(0.8 + 0.3 + 0.5)/3 = 0.533$ | $+0.067$ |
| 4 | 0.5 | $(0.8 + 0.3 + 0.6)/3 = 0.567$ | $-0.067$ |

Completion 1（最好）得到强化，completion 2（最差）被抑制，completion 3 和 4 只得到小幅更新。

---

## REINFORCE / RLOO 小结

**REINFORCE**：最简单的 policy gradient。需要 baseline 来降低方差，不需要 value function。

**RLOO**：REINFORCE 加上巧妙的按 prompt 的 leave-one-out baseline。每个 prompt 的多个 completion 免费提供了 baseline。

两者都是后续所有方法的基础：

- 直观上可以联系到 GRPO 和其他分组 completion 算法
- 不需要 value function（节省显存/算力）
- 实现和调试都很简单
- 实践中是很强的 baseline [@ahmadian2024back; @wang2024helpsteer2p]

---

## RLOO 架构

![语言模型的 RLOO 架构。结构与 REINFORCE 相同，但每个 prompt 的多个 completion 提供了 leave-one-out baseline。](assets/rloo_tikz.png)

---

<!-- layout: section-break -->

## 近端策略优化 (PPO)

---

## REINFORCE 这么简单，为什么还要 PPO？

本讲中的所有算法都是 **on-policy** 的：每个批次都用当前 policy 生成新的 rollout，然后基于这些 rollout 更新。 
这与 off-policy 的 RL 方法（例如 DQN）不同，后者会存储并重放旧的经验。


---

## REINFORCE 这么简单，为什么还要 PPO？

本讲中的所有算法都是 **on-policy** 的：每个批次都用当前 policy 生成新的 rollout，然后基于这些 rollout 更新。 
这与 off-policy 的 RL 方法（例如 DQN）不同，后者会存储并重放旧的经验。

问题在于：原始的 policy gradient 对步长很敏感。更新太大，policy 可能崩溃；太小，训练又慢得难以忍受。TRPO [@schulman2015trust] 用硬性的 trust region 约束解决了这个问题，但需要代价高昂的二阶优化。


---

## REINFORCE 这么简单，为什么还要 PPO？

本讲中的所有算法都是 **on-policy** 的：每个批次都用当前 policy 生成新的 rollout，然后基于这些 rollout 更新。 
这与 off-policy 的 RL 方法（例如 DQN）不同，后者会存储并重放旧的经验。

问题在于：原始的 policy gradient 对步长很敏感。更新太大，policy 可能崩溃；太小，训练又慢得难以忍受。TRPO [@schulman2015trust] 用硬性的 trust region 约束解决了这个问题，但需要代价高昂的二阶优化。

近端策略优化 **(PPO)** [@schulman2017proximal] 通过一个简单的 clipped objective 获得了类似 TRPO 的稳定性——而且由于 clipping 使更新保持保守，你可以对每批 rollout 安全地进行多次梯度步，从而提高 sample efficiency。

---

## 更保守的梯度能带来什么？

从这批数据中榨取更多信号！ 
但这会带来新的问题：

- 在多次梯度更新中，我们如何约束更新幅度？
- 当数据已经偏离 on-policy 时，如何计算 policy gradient？

---

## PPO 核心思想 1：约束更新

<!-- cite-right: schulman2017proximal -->

过大的梯度步长会破坏 policy（不稳定、过度优化等）。

解决方案：**trust region**——限制 policy 在单次更新中可以移动的幅度。

---

## PPO 核心思想 2：重要性采样

我们希望在同一批数据上执行多次梯度步骤，但这些数据来自旧 policy $\pi_{\theta_\text{old}}$。

定义**重要性采样比率**（importance sampling ratio）：

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$

- 若 $\rho_t = 1$：当前 policy 与旧 policy 对该动作的看法一致
- 若 $\rho_t > 1$：当前 policy 赋予的概率高于旧 policy
- 若 $\rho_t < 1$：当前 policy 赋予的概率低于旧 policy

该比率对旧 policy 的样本重新加权，用来估计新 policy 的梯度。
将这一比率代入前面的 policy gradient 形式，就得到一个可以在旧 policy 数据上优化的代理目标函数（surrogate objective）。

---

## 代理目标函数

使用重要性采样后，policy gradient 变为：

$$J(\theta) = \mathbb{E}_t\!\left[\rho_t(\theta) \hat{A}_t\right]$$

**遗留问题**：若没有约束，最大化该目标可能迈出任意大的步长——比率 $\rho_t$ 可能远远偏离 1，使估计变得不可靠。

---

## PPO 的 clipped 目标函数（整合各部分）

PPO 对比率进行 clip 以防止过大的更新——这是受 trust region 思想启发的实用代理目标函数。原论文 [@schulman2017proximal] 称之为 $L^{CLIP}$，但它是我们要**最大化**的目标，因此我们使用 $J$：

$$J(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_t(\theta) \hat{A}_t,\; \text{clip}(\rho_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_t\right)\right]$$

其中 $\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ 是重要性采样比率。

其中 $\varepsilon$ 通常取 0.1–0.2。$\min$ 选取**更保守**的估计。

---

## 正 advantage（$\hat{A}_t > 0$）

<!-- footnote-right: $\rho_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$（policy 比率） -->

$$J(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_t(\theta) \hat{A}_t,\; \text{clip}(\rho_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_t\right)\right]$$

该动作在这个状态下优于平均水平——我们希望**提高**它的概率。分三种子情况：

1. $\rho_t < 1 - \varepsilon$：该动作在新 policy 下**更不可能**被采取。  
  目标：$\rho_t \hat{A}_t$——**正常梯度**，推高概率

2. $1 - \varepsilon \leq \rho_t \leq 1 + \varepsilon$：该动作的概率**大致不变**。  
  目标：$\rho_t \hat{A}_t$——**正常梯度**，推高概率

3. $\rho_t > 1 + \varepsilon$：该动作在新 policy 下**已经更可能**被采取。  
  目标：$(1+\varepsilon)\hat{A}_t$——梯度为**零**，无需更新（**被 CLIP**）

注意：在第一个梯度步骤中，$\rho_t = 1$（policy 尚未改变），因此比率从 1 附近开始，情况 1/3 只会在后续步骤中出现。

---

## 负 advantage（$\hat{A}_t < 0$）

<!-- footnote-right: $\rho_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$（policy 比率） -->

$$J(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_t(\theta) \hat{A}_t,\; \text{clip}(\rho_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_t\right)\right]$$

该动作在这个状态下差于平均水平——我们希望**降低**它的概率。分三种子情况：

1. $\rho_t < 1 - \varepsilon$：该动作在新 policy 下**已经更不可能**被采取。  
  目标：$(1-\varepsilon)\hat{A}_t$——梯度为**零**，无需更新（**被 CLIP**）

2. $1 - \varepsilon \leq \rho_t \leq 1 + \varepsilon$：该动作的概率**大致不变**。  
  目标：$\rho_t \hat{A}_t$——**正常梯度**，压低概率

3. $\rho_t > 1 + \varepsilon$：该动作在新 policy 下**更可能**被采取。  
  目标：$\rho_t \hat{A}_t$——**正常梯度**，压低概率

注意：在第一个梯度步骤中，$\rho_t = 1$（policy 尚未改变），因此比率从 1 附近开始，情况 1/3 只会在后续步骤中出现。

---

## 零 advantage（$\hat{A}_t = 0$）

该动作恰好符合预期。损失为零——**不更新**。

---

## Clipping 小结

在所有情况下，当 policy 在正确方向上**已经移动足够多**时，clipping 会停止梯度：

| Advantage | trust region 之内 | trust region 之外 |
|-----------|-------------------|---------------------|
| $\hat{A}_t > 0$ | 正常梯度（强化） | 若 $\rho_t > 1+\varepsilon$，梯度为零 |
| $\hat{A}_t < 0$ | 正常梯度（抑制） | 若 $\rho_t < 1-\varepsilon$，梯度为零 |
| $\hat{A}_t = 0$ | 不更新 | 不更新 |

Clipping 是**按 advantage 符号单侧生效**的：当 policy 已经在有利方向上移动足够多时，它会限制进一步移动，但绝不会阻碍在不利方向上的修正。

---

## PPO clipping 可视化

![PPO 目标函数在正、负 advantage 下的可视化。trust region 是比率位于 $1 \pm \varepsilon$ 之内的区域。](assets/ppo-viz-4x.png)

---

## Value function（critic）

PPO 在训练 policy 的同时还会训练一个 **value function** $V_\phi(s)$——它表示从状态 $s_t$ 出发的期望未来回报：

$$V_\phi(s_t) = \mathbb{E}\left[\sum_{k=0}^{T-t} \gamma^k r_{t+k} \;\middle|\; s_t\right]$$

- 使用独立的参数 $\phi$（通常由 reward model 或 SFT 模型初始化）
- 在每个 token 位置预测经 KL 调整后的未来回报
- 以 post-KL 回报为目标，通过 MSE 训练：$\mathcal{L}_V = \frac{1}{2}(V_\phi(s_t) - G_t)^2$

value function 充当 advantage 估计中学习得到的 baseline。

---

## PPO-RLHF 如何获得 token 级的 credit

PPO 式 RLHF 通常只有一个针对整段 completion 的标量 reward。它如何变成逐 token 的训练信号？

1. **KL 惩罚调整中间 token**：在每个 token $t$ 处，从 reward 中减去 $\beta \cdot \text{KL}_t$ → 得到逐 token 的 reward 信号
2. **最后一个 token 获得 RM 分数**：$r_T = R(\tau) - \beta \, \text{KL}_T$，其余 token 的 reward 为 $r_t = -\beta \, \text{KL}_t$
3. **GAE 向后传播 credit**：value function 加上 TD residual，根据这些调整后的 reward 为每个 token 分配 advantage

这就是 PPO-RLHF 把序列级 reward 变成 token 级训练信号的方式。

---

## Advantage 估计

最简单的 advantage：$\hat{A}_t = G_t - V_\phi(s_t)$

**为什么 advantage 有帮助**：
- **中心化**可以降低方差——我们问的是“这个 action 比平均水平好多少？”，而不是“总回报有多好？”
- **Credit assignment**——有了逐 token 的 value，每个 token 都有自己的 advantage 信号

这种 Monte Carlo 估计简单且无偏，但方差可能很高。Temporal Difference（TD）方法和 Generalized Advantage Estimation（GAE）用一点偏差换取更低的方差。

---

## TD residual

<!-- cite-right: schulman2015high -->

temporal difference（TD）residual 衡量实际 reward 比 value 预测高出多少：

$$\delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

这就是 **1 步 advantage 估计**：方差低（使用学习得到的 $V_\phi$），但偏差可能很高（如果 $V_\phi$ 不准确）。

---

## GAE：$K$ 步 advantage

我们可以扩展到 $K$ 步：

$$\begin{aligned}
\hat{A}_t^{(1)} &= \delta_t^V && \text{(1-step: low variance, high bias)} \\[4pt]
\hat{A}_t^{(2)} &= \delta_t^V + \gamma \delta_{t+1}^V && \text{(2-step)} \\[4pt]
\hat{A}_t^{(k)} &= \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V && \text{(k-step: more variance, less bias)}
\end{aligned}$$

当 $k \to \infty$ 时，我们恢复出完整的 Monte Carlo advantage $G_t - V_\phi(s_t)$（无偏差，方差最高）。

---

## GAE：指数加权

GAE 对所有 $K$ 步估计做指数加权平均：

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

其中 $\lambda \in [0, 1]$ 控制偏差与方差的权衡。

---

## GAE：偏差与方差的权衡

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

| $\lambda$ | 行为 | 方差 | 偏差 |
|:---------:|----------|:--------:|:----:|
| $0$ | 纯 TD（1 步） | 最低 | 最高 |
| $0.95$ | LLM 的典型默认值 | 平衡 | 平衡 |
| $1$ | Monte Carlo advantage | 最高 | 无 |

语言模型中的 $\gamma$ 通常取 $1.0$（不做折扣）。以上排序假设 $V_\phi$ 是准确的——实际中，训练不佳的 critic 会让即使是低 $\lambda$ 的估计也不可靠。

---

## PPO-RLHF：完整的 policy 目标函数

PPO-RLHF 的高层示意图是将 clipped PPO 目标与 KL 正则项相结合：

$$J(\theta) \approx L^{CLIP}(\theta) - \beta \, D_{\text{KL}}\!\left[\pi_\theta \| \pi_{\text{ref}}\right]$$

**两层正则化**：
1. **Clipping**——限制每个 batch 中 policy 的移动幅度（trust region）
2. **KL 惩罚**——限制整个训练过程中相对 reference policy 的总漂移

二者作用不同，并不冗余。

---

## PPO 训练循环（概念）

PPO 训练循环：

1. **Generate**：采样 prompt，用当前 policy $\pi_\theta$ 生成 completion
2. **Score**：用 reward model 计算 reward，并加上相对 $\pi_\text{ref}$ 的 KL 惩罚
3. **Estimate advantages**：用学习得到的 value function $V_\phi$ 计算 GAE
4. **Update**（K 个 epoch）：在同一个 batch 上计算 clipped policy gradient 加 value function loss
5. **Repeat**：新的 batch，同步 $\pi_{\theta_\text{old}} \leftarrow \pi_\theta$

典型设置：重新生成之前，每个 batch 做 K = 2–4 步梯度更新。

---

## PPO：内存中的四个模型

PPO 需要**四个**模型：

| 模型 | 作用 | 是否更新？|
|-------|---------|:--------:|
| Policy $\pi_\theta$ | 生成 completion | 是 |
| Value function $V_\phi$ | 估计每个 token 的期望回报 | 是 |
| Reference policy $\pi_\text{ref}$ | KL 惩罚的锚点 | 冻结 |
| Reward model $r_\psi$ | 为 completion 打分 | 冻结 |

这非常占内存——也是 GRPO 等更简单的替代方案出现的重要动机。

---

## PPO 架构

![语言模型的 PPO 架构。增加了一个训练出来的 value model 和 GAE，用于逐 token 的 advantage 估计。](assets/ppo_tikz.png)

---

<!-- layout: section-break -->

## GRPO 与现代变体

---

## GRPO：面向推理时代的 PPO

<!-- cite-right: shao2024deepseekmath -->

Group Relative Policy Optimization（GRPO）由 DeepSeekMath [@shao2024deepseekmath] 针对数学推理提出，此后成为语言模型 RL 的首选算法。它保留了 PPO 的 clipped objective，但完全去掉了 value function。

**核心思想**：对每个 prompt 生成 $G$ 个 completion，用这一组的 reward 统计量作为 baseline——不需要学习 critic。

- 比 PPO 更易实现和调试
- 节省内存（少一份模型副本）
- 非常适合 RLVR：有明确的验证信号，且 reward 方差较大（例如答对为 1、答错为 0）
- 随 DeepSeek R1 走红，并成为此后一系列工作的基础

---

## GRPO advantage

对于一组 $G$ 个 completion，其 reward 为 $R_1, \ldots, R_G$：

$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \ldots, R_G)}{\text{std}(R_1, \ldots, R_G)}$$

Z-score 归一化：高于平均的 completion 获得正 advantage，低于平均的获得负 advantage。

completion $i$ 中的每个 token 都获得**相同**的 advantage（序列级，而非逐 token）。

---

## GRPO objective

Clipped ratio（与 PPO 相同）+ 组内归一化的 advantage + 直接放在 loss 中的 KL 惩罚：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|a_i|}\sum_{t=1}^{|a_i|}\!\left(\min\!\left(\rho_{i,t} \hat{A}_i,\, \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) - \beta \, D_{\text{KL}, i,t}\right)$$

$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \ldots, R_G)}{\text{std}(R_1, \ldots, R_G)}$$

其中 clipping 作用于**逐 token**：$\rho_{i,t} = \frac{\pi_\theta(a_{i,t} \mid s_t)}{\pi_{\theta_\text{old}}(a_{i,t} \mid s_t)}$，但 $\hat{A}_i$ 由该 completion 中的所有 token 共享（序列级 advantage，逐 token ratio）。


---

## GRPO 与 PPO 对比

| | **PPO** | **GRPO** |
|---|---------|----------|
| **Value function** | 学习得到的 $V_\phi$ | 无 |
| **Advantage** | 通过 GAE 逐 token 计算 | 序列级，组内 z-score |
| **KL 惩罚** | 在 reward 中（先于 advantage 计算） | 在 loss 中（默认，但可选） |
| **内存中的模型数** | 4（policy、value、ref、RM） | 3（policy、ref、RM），不用 KL 时为 2 |
| **复杂度** | 较高 | 较低 |
| **常见用途** | 通用 RLHF | 推理 / RLVR（DeepSeek R1） |

GRPO 就是去掉 value function 的 PPO，改用统计量作为 baseline。

---

## RLOO 与 GRPO 对比

两者都为每个 prompt 使用多个 completion。关键区别在于 GRPO 采用了 PPO 式的 clipping：

| | **RLOO** | **GRPO** |
|---|---------|----------|
| **Baseline** | Leave-one-out 均值 | 组均值（做 z-score） |
| **更新方式** | REINFORCE（无 clipping） | PPO 式 clipped ratio |
| **KL 惩罚** | 可选（在 reward 中） | 在 loss 中（默认，但可选） |
| **Advantage** | $R_k - \frac{1}{K-1}\sum_{j \neq k} R_j$ | $\frac{R_i - \text{mean}}{\text{std}}$ |

原理相同（与同伴比较），机制不同。如果不做 std 归一化，GRPO 式的 advantage 估计在相差一个缩放常数的意义上等价于 RLOO。

---

## GSPO：序列级 ratio

<!-- cite-right: zheng2025gspo -->

**问题**：在长序列上聚合逐 token 的 importance ratio 在数值上不稳定。单个 ratio 很大的 token 就可能主导整个更新。

**GSPO** 使用几何平均——为每个 response 生成一个经过长度归一化的 importance weight：

$$\rho_i(\theta) = \exp\!\left(\frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s)}{\pi_{\theta_\text{old}}(a_{i,t} \mid s)}\right)$$

无论序列多长，几何平均都能保持在合理的数值范围内——梯度信号相近，数值性质更好。完整的 objective 与 GRPO 类似，只是换成了序列级 ratio：

$$J_{\text{GSPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \min\!\left( \rho_i(\theta) \hat{A}_i,\, \text{clip}(\rho_i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i \right)$$

现在，clipping 范围 $\varepsilon$ 作用于逐 token 的平均尺度，因此在不同的 completion 长度之间具有可比性。

---

## CISPO：Clipped importance sampling

<!-- cite-right: minimax2025minimaxm1scalingtesttimecompute -->

CISPO 对**importance weight 本身**做 clipping，而不是对 objective 做 clipping，并使用 stop-gradient：

$$J_{\text{CISPO}}(\theta) = \sum_{i,t} \text{sg}\!\left(\hat{\rho}_{i,t}\right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s_t)$$

$$\hat{\rho}_{i,t} = \text{clip}\!\left(\frac{\pi_\theta(a_{i,t} \mid s_t)}{\pi_{\theta_\text{old}}(a_{i,t} \mid s_t)},\; 1-\varepsilon^{-},\; 1+\varepsilon^{+}\right)$$

**与 PPO 的关键区别**：每个 token 仍然会获得梯度信号——weight 只是限制了它被放大的程度。非对称的边界（$\varepsilon^{+} > \varepsilon^{-}$）允许更激进的、提升 reward 的更新，从而鼓励探索。

---

## GRPO 架构

![语言模型的 GRPO 架构。每个 prompt 对应多个 completion，使用 group 归一化的 advantage，无需 value model。](assets/grpo_tikz.png)

---

<!-- layout: section-break -->

## 结论

---

## 本讲总结

**在大多数 RLHF 设置中，数据质量和奖励信号质量起主导作用；算法选择主要决定稳定性、效率和工程负担。** 所有方法优化的都是**同一个** policy gradient 目标：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_t \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

它们的差异在于：

1. **$\Psi_t$ 是什么**：total return、advantage、group-normalized advantage
2. **如何限制更新幅度**：不裁剪（REINFORCE）、目标函数裁剪（PPO/GRPO）、权重裁剪（CISPO）
3. **内存与计算**：2–4 个模型，是否训练 value function

---

## RL 速查表

所有核心 RL loss function 的一页参考表见：

**[rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet)**

---

## 下一讲：RL 实现

第 4 讲将把这些算法变成可运行的代码：

- **核心 loss 实现**：用语言模型搭建 RL 算法
- **Loss 聚合**：按序列与按 token 归一化的区别及其重要性
- **基础设施搭建**：需要存储什么（log-probs、values、ref log-probs），以及不存会出什么问题
- **权衡取舍**：监控训练、识别 reward hacking、修复常见故障

---

<!-- rows: 85/15 -->
## 谢谢

提问 / 讨论

联系方式：nathan@natolambert.com

通讯：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===


```builtwith
repo: natolambert/colloquium
```
