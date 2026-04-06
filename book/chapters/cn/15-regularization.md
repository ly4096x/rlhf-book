<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "过度优化"
prev-url: "14-over-optimization"
page-title: Regularization
search-title: "第15章：Regularization"
next-chapter: "评估"
next-url: "16-evaluation"
---

# Regularization

在整个 RLHF 优化过程中，会使用许多 regularization 步骤来防止对 reward model 的过度优化。
在这些情境中，过度优化的表现是模型输出无意义的文本。
优化"失控"的一些例子包括：模型输出逻辑上可跟随的数学推理，但答案极度错误；文本重复；语言切换；或大量特殊字符。
本章介绍用于控制模型优化的各种方法。

目前最流行的变体（在撰写本文时已被大多数 RLHF 实现所采用）是当前 policy 与参考 policy 在生成样本上的 KL distance。
"KL distance"是一个口语化术语，用于表达训练过程中的*优化距离*，尽管 KL divergence——衡量两个概率分布差异的底层数学方法——并不满足成为真正距离度量所需的形式化属性（相比于"两个分布之间的数值差异度量"，将其称为距离更为简便）。
许多其他 regularization 技术在文献中出现后，又在该研究系列的下一个模型迭代中消失。
也就是说，核心 KL distance（基于生成样本）之外的 regularization 通常用于稳定实验设置，而这些设置在下一代模型中往往可以被简化。
尽管如此，理解 RLHF 中用于约束优化的工具仍然非常重要。

*在本章中，我们用 $x$ 表示提示词，用 $y$ 表示补全内容。这种表示方式在语言模型文献中很常见，其中的方法作用于完整的提示词-补全对，而非单个 token。*

当在带有 reward model $r_\theta$ 的 RLHF 框架中使用时，一般公式如下：

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

其参考实现为：

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL 优化中的 KL Divergence

有关数学定义，请参阅附录 A（定义）。
KL divergence 衡量一个概率分布偏离另一个概率分布的程度——当 KL 为零时，两个分布产生相同的输出。
其定义如下：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

在 RLHF 中，关注的两个分布通常是新模型版本的分布（设为 $P(x)$）和参考 policy 的分布（设为 $Q(x)$）。
不同的优化器使用不同的 KL 方向。在本书中，最常用的"KL Penalty"被称为相对参考 policy 的反向 KL。在实践中，这简化为一种蒙特卡洛估计：从 RL 模型中采样 token，并从参考模型计算概率。直觉上，这种反向 KL 具有一种数值属性：当新模型 $P$（即 $\pi_{\text{RL}}$）在原始参考模型赋予低概率的区域放置大量概率质量时，会施加较大的惩罚。

另一种 KL 方向在机器学习中也常被使用，例如在某些 RL 算法的内部 trust region 计算中。直觉上，当新模型的更新*未将*概率赋予 $Q$（即 $\pi_{\text{ref}}$）中的高概率区域时，该惩罚会对新模型施加惩罚。这更接近于用于蒸馏或行为克隆的目标。

### 参考模型与生成内容

KL penalty 最常见的实现方式是将训练期间生成的 token 与静态参考模型之间的距离进行比较。
其直觉是：你希望训练后的模型保持接近原始模型的风格。
这个参考模型通常是经过指令微调的模型，但也可以是之前的 RL checkpoint。
通过简单替换，我们采样的模型变为 $\pi_{\text{RL}}(x)$ 和 $\pi_{\text{ref}}(x)$，如上文 @eq:kl_standard 所示（在标准定义中通常对应 $P$ 和 $Q$，当用于 RL KL penalty 时）。
这种 KL divergence penalty 早在大型语言模型流行之前就首次被应用于对话 agent [@jaques2017sequence]，而 KL control 很快被确立为 fine-tuning 预训练模型的核心技术 [@jaques2020human]。

### 实现示例

在实践中，KL divergence 的实现通常采用近似方法 [@schulman2016klapprox]，从而使实现大为简化。
根据上述定义，当直接从分布 $P(x)$ 采样时，KL 的求和可以转化为期望。
在这种情况下，分布 $P(x)$ 是当前正在训练的模型（而非参考模型）的生成分布。
这样，KL divergence 的计算变为：

$$
\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

这种方式实现起来更为简单，特别是在直接处理语言模型训练中频繁使用的对数概率时。

```python
# 第1步：从你的 policy 中采样（或以其他方式生成）一个序列
generated_tokens = model.generate(inputs)

# 第2步：在两个模型下对生成的序列进行打分
#    对于自回归语言模型，通常做法是：
#      inputs_for_scoring = generated_tokens[:, :-1]
#      labels           = generated_tokens[:, 1:]
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# 转换为对数概率，然后对齐标签以索引到 logits 中
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# 收集实际下一个 token 的对数概率
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# 现在可以对这些值求和（或取平均）以得到序列对数概率，
# 并计算 KL：
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

一些示例实现包括 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 和 [Hamish Ivison 的 Jax 代码](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。


## 隐式 Regularization

本章其他部分描述的是*显式* regularization：KL penalty、预训练梯度和 margin loss，这些都是从业者有意添加到训练目标中的方法。
越来越多的实证研究表明，基于 RL 的后训练还提供了*隐式* regularization——一种对记忆化和灾难性遗忘的内在抵抗力，这种抵抗力源于 on-policy 优化本身的结构。
这是由于损失更新的性质所致，即使没有任何用于控制 RL 训练的显式工具（如 KL penalty 或 replay buffer）。

### SFT 记忆，RL 泛化

后训练社区面临的核心问题是：在单一任务上训练时，模型是学习了一种可迁移到未见变体的可泛化规则，还是仅仅记忆了训练分布的表层模式？
Chu 等人 2025 [@chu2025sft] 通过一项受控实证研究回答了这一问题，该研究直接隔离了后训练方法——SFT 与 RL——对分布外（OOD）泛化的影响。
结论很明确：RL 学习的是可迁移的规则，而 SFT 则记忆训练数据，在分布偏移下出现崩溃。

该研究使用了两个内置规则变体的环境来理解其中的权衡：

- **GeneralPoints** 是一个算术纸牌游戏，模型接收四张扑克牌，需要用运算符（+、-、*、/）将其数值组合以达到目标数字（默认为 24）。OOD 测试会改变花牌的计分规则：训练时使用一种规则（J、Q、K 均计为 10），评估时使用另一种规则（J = 11，Q = 12，K = 13）。
- **GeneralPoints** 是一款算术纸牌游戏，模型需要获取四张扑克牌，并用运算符（+、-、*、/）将其数值组合以达到目标数字（默认为 24）。OOD 测试改变了花牌的计分方式：训练时使用一种规则（J、Q、K 均计为 10），评估时使用另一种规则（J = 11，Q = 12，K = 13）。

- **V-IRL** 是一个真实世界的视觉导航任务，模型需要遵循语言指令在城市街道中沿路线行进，并识别沿途的地标。OOD 迁移将动作空间从绝对方向（北、东）切换为相对方向（左、右）。

在所有任务变体中，随着训练计算量的增大，RL 始终提升 OOD 性能，而 SFT 尽管在分布内表现有所改善，OOD 性能却始终*下降*。
差异幅度令人瞩目：在仅用语言输入的 V-IRL 任务上，OOD 迁移是从绝对方向坐标到相对方向坐标，RL 将 OOD 逐步准确率从 80.8% 提升至 91.8%，而 SFT 则从 80.8% 崩溃至 1.3%。
SFT 模型不仅未能泛化，还破坏了基础模型原本具备的空间推理能力，退化为从指令短语到绝对方向的查找表。

### 在实践中保留：on-policy 数据缓解遗忘

上一节表明，在单一任务上，RL 能够泛化，而 SFT 只会记忆。
Chen 等人 2025 [@chen2025retainingdoingroleonpolicy] 提出了互补性问题：当*顺序*训练多个任务时，模型能否保留已掌握的知识？
他们发现，RL 在目标任务上取得了相当甚至更高的收益，同时遗忘程度远低于 SFT，并将这一优势追溯至两种目标所优化内容的根本差异。

为了理解两种方法行为如此不同的原因，我们可以通过 KL divergence 的视角来审视其目标函数。
本节首先说明两种常见的后训练方法可以对应到 KL divergence 的两个方向，然后解释将这两者用作损失函数时数值行为的差异如何转化为不同的模型行为。

KL divergence 定义为两个分布之间对数比的期望，$\mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]$，可以写成对数差的形式，分为两个方向：

- **Forward KL**：$\text{KL}(P \| Q) = \mathbb{E}_{x \sim P}[\log P(x) - \log Q(x)]$
- **Reverse KL**：$\text{KL}(Q \| P) = \mathbb{E}_{x \sim Q}[\log Q(x) - \log P(x)]$

其中 $P$ 是目标分布，$Q$ 是我们用参数 $\theta$ 建模的分布。
关键区别在于我们从哪个分布中采样：forward KL 从目标（或最优）分布 $P$ 中采样，而 reverse KL 从我们的 policy $Q$ 中采样。
在下面的推导中，$P$ 对应目标 $\pi_\star$（分析 SFT 时为训练数据分布，分析 RL 时为 reward 最优 policy），$Q$ 对应学习到的 policy $\pi_\theta$（即我们正在训练的内容）。
SFT 将目标放在前面——$\text{KL}(\pi_\star \| \pi_\theta)$——而 RL 则颠倒顺序——$\text{KL}(\pi_\theta \| \pi_\star)$——从而改变了我们从哪个分布中采样。
样本提供了学习所需的数据。目标函数，SFT 或 RL，从这些数据中塑造模型。

**SFT $\approx$ Forward KL。** 从 forward KL 的定义出发：

$$
\text{KL}(\pi_\star \| \pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) - \log \pi_\theta(y \mid x) \right]
$$

将对数差上的期望拆分为两项：

$$
= \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) \right] - \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\theta(y \mid x) \right]
$$

第一项 $\mathbb{E}[\log \pi_\star(y \mid x)]$ 仅依赖于数据分布，等于负熵 $-H(\pi_\star)$——一个不随 $\theta$ 变化的常数。
第二项 $-\mathbb{E}[\log \pi_\theta(y \mid x)]$ 是数据集上的负对数似然，即标准的 SFT 交叉熵损失 $\mathcal{L}_\text{SFT}(\theta)$。代入后：

$$
= \underbrace{-H(\pi_\star)}_\text{const} + \mathcal{L}_\text{SFT}(\theta) \propto \mathcal{L}_\text{SFT}(\theta)
$$ {#eq:sft_forward_kl}

由于熵项相对于 $\theta$ 是常数，两个损失函数具有相同的梯度和相同的最小值——最小化 SFT 损失等价于最小化 **forward KL** divergence $\text{KL}(\pi_\star \| \pi_\theta)$。

**RL $\approx$ Reverse KL。** 从标准的 KL regularization RL 目标出发：

$$
\max_\pi \; \mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}\!\left(\pi(\cdot \mid x) \| \pi_\text{ref}(\cdot \mid x)\right)
$$ {#eq:rl_objective_retaining}

提出 $-\beta$ 将最大化转化为最小化：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]
$$ {#eq:rl_min_form}

引入配分函数 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$ 将 reward 倾斜参考归一化为有效分布，并加减 $\log Z(x)$，内层期望变为 KL divergence：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}\!\left(\pi(\cdot \mid x) \;\middle\|\; \frac{1}{Z(x)} \pi_\text{ref}(\cdot \mid x) \exp\!\left(\tfrac{1}{\beta} r(x,y)\right) \right) - \log Z(x) \right]
$$ {#eq:rl_kl_form}

由于 $\log Z(x)$ 不依赖于 $\pi$，且 KL divergence 非负，当且仅当两个分布相同时等于零，因此当 $\pi$ 等于 reward 倾斜分布时，KL 被最小化至零。
reward $r(x,y)$ 下的最优 policy 因此为：

$$
\pi_\star(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$ {#eq:optimal_policy_retaining}

现在我们可以直接展示与 reverse KL 的联系。展开 $\text{KL}(\pi_\theta \| \pi_\star)$ 并代入 $\log \pi_\star(y \mid x) = \log \pi_\text{ref}(y \mid x) - \log Z(x) + \frac{1}{\beta} r(x, y)$：

$$
\begin{aligned}
\text{KL}(\pi_\theta \| \pi_\star) &= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\star(y \mid x) \right] \\
&= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x) + \log Z(x) - \frac{1}{\beta} r(x, y) \right] \\
&= - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) + \underbrace{\log Z(x)}_\text{const} \\
&\propto - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) \\
&= -\frac{1}{\beta} \mathcal{J}_\text{RL}(\theta)
\end{aligned}
$$

等价地，最大化 RL 目标 $\mathcal{J}_\text{RL}(\theta)$ 与最小化 **reverse KL** divergence $\text{KL}(\pi_\theta \| \pi_\star)$ 是相同的。

该推导表明，SFT 和 RL 优化的是根本不同的目标：SFT 最小化 forward KL，RL 最小化 reverse KL。

![forward KL（SFT）与 reverse KL（RL）的遗忘动态。"旧"模式代表先验知识，"新"模式代表目标任务。Forward KL 将 policy 拉伸以覆盖目标，并将概率质量从旧模式抽离（右上），而 reverse KL 将新模式向目标移动，而不干扰旧模式（右下）。来自 Chen 等人 2025，经作者许可。](images/retaining_by_doing_mode_intuition.png){#fig:retaining-mode-intuition}

KL divergence 的两个方向产生不同的优化压力。

Forward KL 在目标分布有概率质量而模型没有的地方对模型进行惩罚，这往往鼓励**模式覆盖**——模型广泛分布概率以覆盖目标的所有主要模式。
原因如下：forward KL 中的期望是在 $\pi_\star$ 下取的，因此它对模型未能给目标有概率质量的区域分配概率的情况给予严重惩罚。

Reverse KL 仅在模型实际放置概率质量的区域对模型进行惩罚，这往往鼓励**模式寻求**：模型可以集中于一个高概率模式，同时忽略其他模式。
此处期望是在 $\pi_\theta$——模型自身的分布——下取的，因此 $\pi_\theta(y \mid x) \approx 0$ 的区域对损失贡献很小，即使 $\pi_\star$ 在那里分配了大量概率质量。
同时，它也会惩罚模型在目标分布没有概率质量的地方放置概率质量。

基于这一区别，我们可能会天真地预期 SFT 的遗忘*少于* RL：模式覆盖的 forward KL 应当在目标的所有模式上保持概率质量，保留旧知识，而模式寻求的 reverse KL 可能会坍缩到单一的高 reward 模式并放弃其他模式。
然而，实际情况恰恰相反。
这种直觉假设了单峰 policy，但预训练 LLM 包含多个模式——对于多峰分布，动态会翻转。

考虑一个具有两个模式的 policy：一个"旧"模式代表先验知识，一个"新"模式代表目标任务（@fig:retaining-mode-intuition）。
Forward KL（SFT）试图覆盖目标分布的两个模式，这迫使 policy 拉伸并将概率质量*从*旧模式重新分配，破坏其形态并导致遗忘。
Reverse KL（RL）相比之下，只需要在某个高 reward 区域放置概率质量，因此它可以将其采样的新模式移向目标，而完全不触动旧模式，使先验知识保持完整。

RL 的模式寻求行为——reverse KL 的结构属性——保留了模型先验知识的广度，并实现了更好的泛化。

总结如下：

- **SFT（Forward KL）**：$\text{KL}(\pi_\star \| \pi_\theta)$——样本来自目标 $\pi_\star$，即人工编写补全内容的固定数据集。对于每个样本，我们问：我们的模型 $\pi_\theta$ 给这个样本分配了多少概率？模型从不生成任何内容；它只是学习模仿。这种模式覆盖压力迫使 policy 广泛重新分配概率质量，这可能会破坏先验知识。

- **RL（Reverse KL）**：$\text{KL}(\pi_\theta \| \pi_\star)$——样本来自我们自己的 policy $\pi_\theta$。对于模型生成的每个补全，我们问：这与 reward 最优 policy $\pi_\star$ 有多接近？由于模型仅在自己的生成内容上训练，更新保持在它已经放置概率质量的局部区域——reward 信号告诉它哪些生成内容需要强化，在不干扰分布其余部分的情况下将概率移向 $\pi_\star$。

### RL 的剃刀：为什么 Online RL 遗忘更少

上一节表明，on-policy 采样驱动了 RL 对遗忘的抵抗力，并将其机制追溯至 forward 与 reverse KL divergence 的动态。
对于任何给定的任务，存在许多不同的 policy 都能实现高性能。
Shenfeld 等人 2026 [@shenfeld2026rls] 从互补视角探讨了 RL 的泛化问题，提出了 **RL's Razor** 论题，其核心表述如下：

> 在解决新任务的众多高奖励方案中，on-policy 方法（如 RL）天然偏向于在 KL divergence 上更接近原始 policy 的解。

![偏向 KL 最小解可减少遗忘。（左）在能解决新任务的 policy 中，RL 收敛于 KL 距离基础模型最近的那些。（右）与 SFT 相比，在新任务性能相当的情况下，这种 KL 偏向带来了更高的旧任务保留率。来自 Shenfeld、Pari 和 Agrawal 2026。许可证 CC-BY。](images/rl_razor_motivation.png){#fig:rl-razor-motivation}


作者发现，对过去任务的遗忘程度与 fine-tuned policy 偏离初始模型的程度（以 KL divergence 衡量）直接成正比：

$$
\text{Forgetting} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[\text{KL}\!\left(\pi_0(\cdot \mid x) \| \pi(\cdot \mid x)\right)\right]\right)
$$ {#eq:rl_razor_forgetting}


作者通过多种 RL 和 SFT 训练方式的实验，实证证明遗忘程度与训练后 policy 和初始 policy 之间的 KL divergence 高度相关（$R^2 = 0.96$），**且该 KL 是在新任务数据上测量的**。
这一结果令人惊讶，因为 KL 是在*新任务*的输入分布上测量的，而非在旧任务的留出数据上，然而它仍能预测旧任务上的性能下降。
在实践中，这为我们提供了一种强有力的手段，可以直接通过基础模型与训练后 policy 之间的偏移来估计遗忘程度——即在新的专项数据上测量 KL 距离。

为了深入探究是什么驱动了 RL policy 中更小的 KL 偏移，作者从两个维度分解了 RL 与 SFT 之间的差异——on-policy 数据与 offline 数据之别，以及目标函数是否包含负梯度（RL 中当样本得分低于奖励基线时存在负梯度，而 SFT 仅对正确示范进行强化，不包含负梯度）。
值得注意的是，他们发现 on-policy 数据与 offline 数据的差异完全解释了泛化性能上的不同，而负梯度的有无则没有可察觉的影响。

直觉上，on-policy 方法采样的是模型本身已赋予非可忽略概率的输出，因此每次更新都被约束在接近当前分布的范围内。
相比之下，SFT 训练于一个固定的外部分布，该分布可能与模型当前产出的内容相距甚远，而每一个梯度步骤都会朝着那个遥远的目标靠拢，无论模型自身的倾向如何。

## 其他类型的 Regularization

在 post-training 文献中，许多顶尖模型都加入了其他 regularization 方法，以在各自的设置中达到领先性能。
以下两个例子旨在展示一些领先模型如何调整 post-training 设置以获得稳定的优化过程，而非作为在所有设置中都必然奏效的通用工具。
还有无数更具创意的解决方案有效且将被发现！

### Pretraining Gradients

另一种看待 regularization 的方式是：你可能有一个*数据集*，希望模型保持接近该数据集，正如 InstructGPT [@ouyang2022training] 中所做的那样，"以修复在公开 NLP 数据集上的性能退化"。
为了实现这一点，他们修改了 RLHF 的训练目标。
以 @eq:rl_start 为基础，我们可以将其转化为一个优化目标函数：从 RL policy 模型中采样，从用于 RLHF 的 RL 数据集中对 prompt $x$ 生成补全 $y$，得到：
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

随后，我们可以为标准自回归下一个 token 预测损失（即 pretraining 时所用的损失）添加一个额外奖励，其数据来自预训练语料库（或其他数据集）中采样的文档，以维持文本连贯性：

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right]
$$ {#eq:objective_pretraining}

近期研究提出使用负对数似然（negative log-likelihood）项来平衡 Direct Preference Optimization（DPO）的优化 [@pang2024iterative]。
鉴于 DPO 损失的成对性质，同样的损失修改方式也可应用于 reward model 训练，约束模型预测准确的文本（这一做法来自未发表论文的实验室传言）。

优化过程是对 DPO 的修改：
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

其中 $P_{\theta}$ 为可训练的 policy 模型，$P_{\text{ref.}}$ 为固定的参考模型（通常为 SFT 的 checkpoint），$(c_i^w, y_i^w)$ 和 $(c_i^l, y_i^l)$ 分别表示 prompt $x_i$ 对应的优胜补全与落败补全。
第一项是标准 DPO logistic 损失：通过对数似然比之差 $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$ 增大优胜与落败之间的差距，$\beta$ 控制该偏好信号偏离参考模型的强度。
第二项是对优胜补全施加的长度归一化负对数似然惩罚，权重为 $\alpha$，其作用是在绝对语言建模意义上保持优选文本的高似然性，而不仅仅是相对于被拒绝样本更优。

### Margin-based Regularization

在 RLHF 体系的其他部分，对优化的控制定义得没那么清晰。
大多数 reward model 除了标准对比损失函数之外没有额外的 regularization。
Direct Alignment Algorithms 通过 $\beta$ 参数以不同方式处理对 KL divergence 的 regularization（参见 [direct alignment 章节](https://rlhfbook.com/c/08-direct-alignment)）。

Llama 2 为 reward model 训练提出了一种 margin 损失 [@touvron2023llama]：

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

其中 $m(y_c, y_r)$ 是两个数据点 $y_c$ 和 $y_r$ 之间的 margin，代表两位标注者评分之差的数值。
这可以通过让标注者对输出进行数值评分，或使用量化排名方法（如 [Likert 量表](https://en.wikipedia.org/wiki/Likert_scale)）来实现。
