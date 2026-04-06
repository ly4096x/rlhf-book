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
