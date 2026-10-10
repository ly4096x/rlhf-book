<!-- source-commit: eecc49e -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "偏好数据"
prev-url: "11-preference-data"
page-title: Synthetic Data & Distillation
search-title: "第12章：Synthetic Data & Distillation"
meta-description: "现代 post-training 中广泛使用的 synthetic data、distillation、Constitutional AI 与 AI feedback 方法。"
next-chapter: "工具使用与函数调用"
next-url: "13-tools"
lectures:
  - video: "https://www.youtube.com/watch?v=6nyJ8y8ghsE&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=10"
    label: "第七讲：Synthetic Data 与现代 Post-training 方法"
---

# Synthetic Data 与 Distillation

来自*人类反馈*的强化学习（RLHF）深植于一个理念：让人类影响持续作用于我们所构建的模型之中。
当最初的模型成功通过 RLHF 完成训练时，人类数据是*唯一*可行的方式来实现这种改进。

人类是为训练问题生成足够高质量回复的唯一途径。
人类也是收集可靠且具体的反馈数据以训练 reward model 的唯一途径。

随着 AI 模型不断进步，这一假设迅速被打破。
synthetic data 的出现——其成本远低于人工且更易于迭代——通过降低实验和研究的成本，推动了 RLHF 的普及。
这使得 RLHF 成为早期的关注焦点，并进而发展为更广泛的、塑造模型的"post-training"方法。
本章对 synthetic data 如何替代或扩展 RLHF pipeline 的各个环节做一概括性综述。

## Synthetic Data 的作用

对 synthetic data 的一个常见批评是**模型坍缩（model collapse）**——即反复用模型自身的生成内容进行训练，会逐渐收窄有效的训练分布 [@shumailov2024ai]。
随着多样性下降，罕见事实和风格被低估，而微小错误在多次迭代中被放大，导致泛化能力下降。
在实践中，这类失败多与在未经过滤、重复且来源单一的输出上进行自训练相关；混合真实/人类数据、使用多样化的教师模型、去重以及严格的质量过滤可在很大程度上避免坍缩。
对于当今的前沿训练 pipeline，有证据表明 synthetic data 可以且应当大规模使用，而不会出现"坍缩故事"最极端版本所暗示的灾难性退化 [@gerstgrasser2024model] [@feng2024beyond]。

领先的模型**需要 synthetic data** 才能达到最佳性能。
现代 post-training 中的 synthetic data 涵盖训练的诸多环节——language model 被用于从种子示例生成新的训练 prompt [@wang2022self]、修改现有 prompt、为 prompt 生成 completion [@numina_math_7b]、提供 AI feedback 以创建偏好数据 [@cui2023ultrafeedback]、过滤 completion [@li2024superfiltering]，以及更多其他用途。
Synthetic data 是 post-training 的关键。

Synthetic data 能够产生如此深远影响的能力，随 GPT-4 级别模型的出现而涌现。
在早期 language model（如 Llama 2 和 GPT-3.5-Turbo）阶段，模型在生成或监督数据 pipeline 方面还不够可靠。
在此后短短一两年内，language model 在生成答案方面已远超人类。
在从 GPT-3.5 迈向 GPT-4 级别模型的过渡中，模型执行 LLM-as-a-judge 任务的能力也随之涌现。
GPT-4 及更强的模型在针对特定内容生成反馈或评分方面，鲁棒性和一致性都大幅提升。

自 ChatGPT 于 2022 年底发布以来，我们见证了众多具有重大影响的 synthetic dataset。
其中包括：UltraFeedback [@cui2023ultrafeedback]，首个知名的 synthetic 偏好数据集，开启了 DPO 革命；Stanford Alpaca，2023 年最早的对话风格 fine-tuning dataset 之一；Tülu 3 [@lambert2024t] 中聚焦特定技能（如数学、代码、指令遵循）的 synthetic dataset；以及 2025 年用于训练思维模型的 OpenThoughts 3 等众多 synthetic 推理 dataset [@guha2025openthoughts]。
当前入门工业级 post-training 的标准参考资料，大多涉及上述 Tülu 3 或 OpenThoughts 3 等 dataset，而快速入门指南则通常从规模更小、更简单的 Alpaca 开始，因为其训练速度更快。

另一个重大变化与 dataset 规模有关：fine-tuning dataset 的 prompt 数量大幅增加——Alpaca 约有 52K 条，OpenThoughts 和 Tülu 3 均超过 100 万条样本——同时回复长度也显著增长。
更长的回复与更多的 prompt，使得 Alpaca dataset 的训练 token 数量级达到 1000 万，而 Tülu 约为 5 亿（约 50 倍），OpenThoughts 3 则更大，约为 100 亿 token 的量级。

在整个转变过程中，synthetic data 并未在 pipeline 各环节均匀替代人类数据。
对于**指令数据（SFT）**，synthetic 生成基本已占据主导——从更强模型进行 distillation 所产生的 completion 质量，在大规模场景下已超越大多数人工写手（除最难的前沿推理问题外，仍有例外）。
对于 **RLHF 中的偏好数据**，情况则更为复杂：学术研究表明 synthetic 偏好数据性能相当，但前沿实验室仍将人类偏好数据视为竞争壁垒。
对于**评估**，分歧则呈现出不同形式：LLM-as-a-judge 能以较低成本对模型输出进行大规模*评分*，但基准测试和真实标签的底层创建仍依赖人工。
规律在于：synthetic data 在模型超越人类可靠性的领域占据主导，而人类在能力前沿、建立真实标签以及引导训练方面仍不可或缺。

## 利用 Synthetic Data 进行 Distillation

关于 synthetic data 在 language model 中作用的讨论，distillation 这一术语已成为最有影响力的表达方式。
Distillation 一词源自深度学习文献中师生知识蒸馏（Knowledge Distillation，KD）的技术定义 [@hinton2015distilling]。

![传统知识蒸馏（knowledge distillation）通过 KL 散度损失训练较小的 student 模型，使其匹配较大 teacher 模型的软概率分布。两个模型同时处理相同输入，温度缩放（$\tau > 1$）软化分布以揭示更多类别关系信息。](images/knowledge_distillation_tikz.png){#fig:knowledge-distillation data-dark-src="images/knowledge_distillation_tikz-dark.png"}

口语上，distillation 泛指使用更强模型的输出来训练较小模型。

![LLM post-training 中的 synthetic data 生成：将 prompt 传入强模型以生成 completion，并将其配对构成训练 dataset。该 dataset 随后用于通过标准监督学习对较小模型进行 fine-tuning。更复杂的 pipeline 可能涉及多个模型编辑 completion、生成偏好对或进行质量过滤。](images/synthetic_data_distillation_tikz.png){#fig:synthetic-data-generation data-dark-src="images/synthetic_data_distillation_tikz-dark.png"}

在 post-training 中，这种广义的 distillation 概念通常体现为两种常见形式：

1. 作为贯穿 post-training 大部分流程的数据引擎：用于指令的 completion、偏好数据（或 Constitutional AI），以及用于 RL 的验证。
2. 将特定技能从更强模型迁移至较弱模型，通常针对数学推理或编程等特定技能实施。

随着 language model 在回答各类任务时愈发可靠地超越人类，第一种策略的应用日益广泛。
GPT-4 级别模型将这一范畴扩展至利用强模型 distillation 处理数学和代码等复杂任务（如前所述）。
在此背景下，distillation 促使实验室构建模型系列——通常会训练一个大型内部模型（如 Claude Opus 或 Gemini Ultra），该模型不对外公开发布，仅供内部用于训练更强的模型。
对于开放模型而言，常见做法是将封闭 API 模型的训练数据 distill 至规模较小、权重公开的模型中 [@tunstall2023zephyr]。
在此过程中，精心筛选高质量 prompt 并过滤来自 teacher 模型的回复，对于最大化性能至关重要。

将特定技能迁移至较小 language model，遵循相同的 distillation 原则——获取尽可能优质的训练数据。
在这一方向上，已有大量论文研究如何利用来自更强模型的有限 dataset 来改善对齐 [@zhou2023lima]、数学推理 [@shridhar2023distilling] [@hsieh2023distilling] 以及 test-time scaling [@muennighoff2025s1]。

本章其余部分介绍的 synthetic data 方法，都是在训练 pipeline 中直接使用 language model 输出来设计数据配方的不同方式。

## 通向 On-Policy 师生 Distillation 之路

尽管 distillation 总体上已成为 post-training language model 的标准做法，但随着 post-training 配方转向推理模型和 agentic 模型，人们对师生知识蒸馏这一特定子领域的兴趣也重新升温。
采用新形式知识蒸馏训练的领先模型包括：阿里巴巴的 Qwen3 [@yang2025qwen3]、小米的 MiMo-V2-Flash [@mimo2025flash]、智谱 AI 的 GLM-5 [@glm5team2026glm5]，以及 DeepSeek-V4-Pro [@deepseekai2026deepseekv4]。

Distillation 之所以属于本章，是因为现代 post-training 中许多 synthetic data 的用法在实践中都是受 distillation 启发的 pipeline：更强的模型产生标签、completion、logits、critique 或其他监督信号，学生模型则在该信号上进行训练。
与此同时，关于 distillation 的技术文献正在发展成一套独立的 post-training 方法，尤其是 on-policy 和 self-distillation 配方日益普遍。
目前我们先将其作为 synthetic data 工具箱的一部分在此介绍，但本书未来的版本可能需要专门为 distillation 开辟一章，使其与 instruction fine-tuning、reinforcement learning 等并列，作为一种训练工具。

### 为 LM 适配 Knowledge Distillation

最初的文献提出知识蒸馏，专门用于从一个已训练好的、更强和/或更大的 *teacher* 网络来训练 *student* 模型 [@hinton2015distilling]。
KD 以使用*软*训练标签而著称，这与 next-token prediction 配合 cross-entropy loss 等标准目标所使用的 one-hot 标签不同。
基于软标签的目标关注的是所有可能的下一个 token 或预测上的分布，而不仅仅是单个预测 token 是否正确，并训练 student 分布去匹配 teacher 分布。

KD 通常可应用于任何深度学习问题，例如预测输入的单个类别。
为了将其专门应用于自回归式的 language model，可以对 loss 进行分解，得到逐 token 的分布匹配 loss。
2016 年，Kim & Rush 将 KD 应用于让 student 模型从 teacher 模型生成的*序列*中学习 [@kim-rush-2016-sequence]。

令 $s$ 为源句子或 prompt，$u = (u_1,\ldots,u_J)$ 为 teacher 模型的完整输出序列，$\mathcal{V}$ 为输出词表（tokenizer 中可能的 token），$q$ 为 teacher 的下一 token 分布，$p$ 为 student 的分布。
这里我们用 $u$ 作为 teacher 完整输出序列的中性符号，而将 $a$ 保留给下文 on-policy/RL 记号中由 student 采样的 completion/action 序列。
注意，他们的论文称之为 word-level distillation，但由于该论文早于现代 sub-word tokenizer，对现代 language model 而言，最好将其理解为在 tokenizer 词表上的逐 token 分布匹配：

$$
\mathcal{L}_{\mathrm{WORD-KD}}
= -\sum_{j=1}^{J}\sum_{k=1}^{|\mathcal{V}|}
q(u_j = k \mid s, u_{<j})\log p(u_j = k \mid s, u_{<j}).
$$ {#eq:word_kd}

WORD-KD 是受 Hinton 启发的经典师生知识蒸馏在 language model 上的应用。这通常是在训练语料中已有的静态文本上进行的。

这具有普通 cross-entropy 的形式 $-\sum_z q(z)\log p(z)$。
在每个位置 $j$，teacher 分布 $q$ 为每个可能的下一个 token $k \in \mathcal{V}$ 分配概率，当 student 的分布 $p$ 对 teacher 认为可能的 token 给出较低概率时，student 就会受到惩罚。

Sequence-level distillation 则将 $\mathcal{U}$ 视为可能输出序列的空间，并使 student 在完整序列上匹配 teacher 分布。
由于对所有完整序列 $u \in \mathcal{U}$ 求和是不可行的（需要对指数级数量的潜在序列求和），Kim & Rush 用单个高概率 teacher 输出 $\hat{u}$ 上的点质量来近似 teacher 在序列上的分布。
这里 $\hat{u}$ 是用 teacher 模型通过 beam search 生成的序列，因此 $\hat{u} = \mathrm{BeamSearch}_q(s) \approx \arg\max_{u \in \mathcal{U}} q(u \mid s)$：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SEQ-KD}}(s)
= -\sum_{u \in \mathcal{U}} q(u \mid s)\log p(u \mid s)
\approx -\log p(\hat{u} \mid s) \\
= -\sum_{j=1}^{|\hat{u}|}\log p(\hat{u}_j \mid s, \hat{u}_{<j}).
\end{aligned}
$$ {#eq:sequence_kd}

SEQ-KD 朝现代方法迈出了一步，即由 teacher 模型生成 token 作为 student 的信号。这是解锁我们后面将看到的各种 on-policy distillation 风格的核心一步，也是使对所有可能序列的计算变得可处理所必需的。
在转向现代模型上流行的 KD 变体时，我们将这种训练方式称为 *offline* KD——即用于训练 student 模型的生成内容是事先生成好的。

在继续之前，有两点联系值得说明。

第一，有一系列流行模型是用 offline KD 训练的，例如分类器 DistilBERT [@sanh2019distilbert] 和 TinyBERT [@jiao2020tinybert]，它们将 language model 的其他改进与 offline distillation 结合起来（值得注意的是，它们不是*序列* distillation，因为这些 encoder 模型并未针对多 token 自回归预测进行蒸馏）。

第二，我们可以与第 15 章对 Kullback-Leibler（KL）divergence 的详尽介绍建立联系，因为上面使用的 cross-entropy 目标与 KL divergence 密切相关。
对于 teacher 分布 $q$ 和 student 分布 $p$，cross-entropy 定义为

$$
H(q,p) = -\sum_z q(z)\log p(z).
$$ {#eq:kd_cross_entropy}

这与 @eq:word_kd 以及 @eq:sequence_kd 的第一项形式相同。
Cross-entropy 也可以分解为 teacher 分布的熵与 KL divergence 之和：

$$
\begin{aligned}
H(q,p)
&= H(q) + D_{\mathrm{KL}}(q\|p) \\
&= -\sum_z q(z)\log q(z)
+ \sum_z q(z)\log\frac{q(z)}{p(z)}.
\end{aligned}
$$ {#eq:kd_forward_kl}

第一项 $H(q)$ 只取决于 teacher。
因此，当 teacher 固定并作为训练数据来源时，最小化 cross-entropy 等价于最小化从 teacher 到 student 的 forward KL，即 $D_{\mathrm{KL}}(q\|p)$。
这就是 offline KD 和类 SFT 训练所使用的 KL 方向。

### 从 Offline 到 On-Policy Distillation

这些 *offline* KD 算法存在一些局限，促使人们提出 on-policy 变体。
学习的 offline 性质意味着，student 模型可能面临 teacher 模型与 student 在 inference 时生成的序列之间的分布不匹配。
例如，forward KL 目标可能会促使 student 模型高估 teacher 分布中的低概率区域。
这些问题共同为 *on-policy* distillation（OPD）打开了局面。

这种 train-test 差距被称为 **exposure bias** [@arora-etal-2022-exposure] [@song2026surveyonpolicydistillationlarge]。
Offline KD 从 teacher 采样轨迹 $u \sim \pi_T(\cdot \mid s)$，并在所得前缀上最小化逐 token 的 KL，

$$
\mathcal{L}_{\mathrm{KD}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D},\, u \sim \pi_T(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(
\pi_T(\cdot \mid s, u_{<t})
\;\|\;
\pi_\theta(\cdot \mid s, u_{<t})
\right).
$$ {#eq:exposure_train}

而在 inference 时，student 是按照自己的 policy 进行 rollout，因此真正重要的量是沿*其自身*轨迹的期望任务 loss，

$$
\mathcal{L}_{\mathrm{eval}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D}_{\mathrm{test}},\, a \sim \pi_\theta(\cdot \mid s)}
\ell_{\mathrm{task}}(s, a)
$$ {#eq:exposure_test}

这里，$\ell_{\mathrm{task}}(s, a)$ 表示针对已完成的 student 回复的任何下游任务 loss，例如答案错误、测试用例失败，或 judge/rubric loss。
Exposure bias 是不等式 $\pi_T(\cdot \mid s) \neq \pi_\theta(\cdot \mid s)$ 的直接后果：训练期间访问的前缀 $(s, u_{<t})$ 与测试时访问的前缀 $(s, a_{<t})$ 来自不同的状态访问分布，因此 student 是在一组与其实际行动所处状态不同的状态上受到监督的。

转向 on-policy distillation 的核心思想是：我们可以调整优化方式，从 student 模型中采样，并衡量其与 teacher 分布之间的距离，而不是从 teacher 模型中采样。
MiniLLM 指出需要转向 reverse KL 优化（我们在第 15 章中直观解释了为何这个目标可能更好），并提出在在线 policy-gradient RL 框架中使用 KD loss 函数 [@gu2024minillm]。
同期的其他工作 [@agarwal2024policy] 展示了 on-policy KD 的前景，并将"从 student 生成、由 teacher 评分"的迭代过程与 RL 文献中的模仿学习工作联系起来。
为了建立这一联系：DAgger 是此类模仿学习算法之一，它迭代地训练一个在环境中用其学得的 policy 行动的 agent，并由一个 oracle policy 告知它本应采取什么行动，这些反馈随后可用于更新其 policy [@ross2011reduction]。

这一差距的代价可以通过推动 DAgger 的监督模仿学习界来量化。
在最初的离散动作设定中，假设学得的 policy 在 teacher 诱导的训练分布上与 teacher 的期望逐步动作误差在 $\epsilon$ 以内，其中 $\mathbb{I}[\cdot]$ 是指示函数，条件为真时取 1，否则取 0，

$$
\mathbb{E}_{s_t \sim d_{\pi_T}}\!\left[
\mathbb{I}\!\left(\pi_\theta(s_t) \neq \pi_T(s_t)\right)
\right] \leq \epsilon.
$$ {#eq:dagger_perstep}

监督模仿学习的分析 [@ross2011reduction] 表明，沿 student 采样的长度为 $L$ 的轨迹累积的期望 loss 可能随 $L$ 呈二次增长 [@song2026surveyonpolicydistillationlarge]：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\sum_{t=1}^{L} \ell\!\left(s, a_{<t}\right)\right] \leq O(\epsilon L^2).
$$ {#eq:dagger_trajectory}

对于 LLM，这个离散动作的界应被视为一种类比，而非理论保证。
在实践中，LLM 在长时域上预测完整的 next-token 分布，因此 @eq:dagger_perstep 中的 0-1 动作不一致假设并不能干净地适用。
Prompt 或前缀自然对应于状态，采样的 token 对应于动作，但 token-level distillation 通常用 KL 或 cross-entropy 等分布式 loss 来衡量，因此经典的 DAgger 数学并不能精确迁移。

这种 $O(\epsilon L^2)$ 式的误差累积对现代 LLM 尤为显著，因为它们经常生成长达数千个 token 的序列。
单个次优 token 就会使前缀略微偏离分布，而模型从未见过这种被扰动的前缀，更容易再次出错，导致文本质量下降或产生幻觉。
On-policy distillation 通过*迭代地*从当前 student 采样 completion，并由 teacher 在所访问的状态上进行监督，来解决这一问题。
Student 直面自己的错误，在其所访问的特定分布外状态上获得 teacher 反馈，并学会恢复行为。
在 DAgger 的交互式模仿学习分析下，这一迭代过程可将误差累积从 $O(\epsilon L^2)$ 降低到 $O(\epsilon L)$ [@ross2011reduction]。
对于 LLM，这解释了 OPD 背后的动机：精确的界未必能干净地适用于每一种 token-level distillation 设置，但 on-policy 方法在实践中的成功支持了其背后的直觉。

对于 on-policy distillation，令 $s$ 为 prompt，$a = (a_1,\ldots,a_L)$ 为从当前 student policy $\pi_\theta(\cdot \mid s)$ 采样的 completion，$s_t = (s, a_{<t})$ 为第 $t$ 步的 token-level 状态。
Teacher policy $\pi_T$ 是固定的，因此该目标比较的是 student 的下一 token 分布与 teacher 在 student 所诱导状态上的分布。
由于期望是从 $\pi_\theta$ 采样的，且 student 分布位于 $D_{\mathrm{KL}}(\pi_\theta \| \pi_T)$ 的左侧，所以这是一个 reverse-KL 目标：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_T(\cdot \mid s_t)\right).
$$ {#eq:opd_reverse_kl}

这里我们转而使用期望记号，正如第 6 章（介绍基础 RL policy-gradient 算法）中广泛使用的那样，因为该优化是通过采样轨迹并数值估计梯度来求解的。
向采样框架的这一转变，自然过渡到基于 RL 的现代 LLM 训练基础设施，后者被设计为在从当前训练的 policy 生成 token 与进行学习更新之间快速交替。

事实上，近期的 OPD 实现更进一步地将 KD 与 RL 整合，即直接把 KD 距离作为 RL 优化中的 reward 信号。
一种典型实现是，将 reverse KL 距离中逐 token 贡献的负值作为 RL 算法中的 advantage [@lu2025onpolicy]。
对于状态 $s_t$ 下采样的 token $a_t$，token-level 的 log-probability 差可以写成一个类 advantage 信号：

$$
A_t^{\mathrm{OPD}}
= \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t).
$$ {#eq:opd_kl_advantage}

使用逐 token KL 贡献的负值，将最小化问题转化为最大化信号：teacher 评价高于 student 的采样 token 获得正 advantage，而 teacher 评价低于 student 的 token 获得负 advantage。
Teacher 的 log-prob 差就像稠密的 token-level 反馈，可能比稀疏的 verifiable reward 或 reward model 输出提供更有用的学习反馈。

### 现代 OPD 变体

这一设置还可以进一步扩展：使用多个 teacher 模型来教授同一个最终模型，或在生成内容中插入额外信息以帮助模型识别错误。
首先，我们介绍如何将多个 teacher 整合进单次训练。
这些 teacher 可以是特定的专家模型（例如针对数学或代码等领域），也可以是先前的中间训练 checkpoint。
对于每个 teacher，可以针对训练 batch 中每个 prompt 或任务类型选择一个贡献权重，从而构成 Multi-Teacher On-Policy Distillation（MOPD）[@mimo2025flash]。
对于多个 teacher，令 $\pi_{T_k}$ 为第 $k$ 个 teacher，$w_k(s)$ 为其依赖于 prompt 的混合权重（满足 $\sum_k w_k(s) = 1$），则 reverse KL loss 为：

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t \sum_k w_k(s) D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_{T_k}(\cdot \mid s_t)\right).
$$ {#eq:mopd_objective}

在大规模 post-training 中，这可以让配方随着组织规模的扩大而进一步扩展。
多个团队可以分别打造高质量的专家模型，这些模型随后可作为最终 student 模型的 teacher 模型，正如 [@deepseekai2026deepseekv4] 和 [@mimo2025flash] 所做的那样。

OPD 还可以与本书研究的其他领域以多种方式结合，例如将 reverse KL 作为 advantage，与 GRPO 的组内归一化等其他 advantage 计算形式并用，从而实现更复杂的 reward shaping。
KD 方法在 post-training 方法中较为特殊，因为它们通常要求 student 和 teacher 共享同一个 tokenizer，因为监督信号可能是来自另一个 LLM 的逐 token 反馈。

扩展方法，例如 On-Policy Self-Distillation（OPSD），让 language model 自己或借助外部工具验证 completion，使其充当拥有特权信息（privileged information）的 teacher，从而无需显式更强的 teacher 也能提升自身性能 [@zhao2026selfdistilled]（OPSD 训练概览见 @fig:sdpo）。
例如，Cursor 以对 RL 轨迹给出有针对性文本反馈的形式使用 self-distillation，训练其 Composer 2.5 coding 模型 [@cursor2026composer25]，该模型基于 Kimi K2.5 进行 finetune。
下面是一个简化的直觉说明，因为在实践中，以下设置会与代码正确性等其他 loss 函数结合使用。
在这一设置中，Cursor 让模型用一个包含常见 bug 列表的 judgement prompt 来审查 RL 轨迹。
当遇到 bug 时，judgement 模型会在 RL 中修改生成的序列——插入一个提示，供模型在未来学习——然后继续进行 distillation loss。
这包含一个循环：首先在 RL 中用标准 language model 生成方式生成 completion，然后运行 judge 模型并可选地插入提示 token，最后为新的 completion 生成 logprobs 以应用 knowledge distillation loss。
token 空间中的提示足以帮助模型纠正自己的输出，即使是在性能的绝对前沿上进行改进（关于如何最好地构建和使用这些提示——通常称为*特权信息*（privileged information）[@penaloza2026privileged]——目前有大量有意义的研究正在进行）。

这使得 on-policy distillation 成为一种核心的 post-training 方法，适用于将多种技能整合到一个通用模型中，或在专门的部署中推动性能前沿。

![三种 distillation 机制，按 rollout 的来源和监督的流向进行比较。**Sequence KD**（左）：teacher 离线生成输出，student 通过 cross-entropy（CE）loss 训练以匹配该输出。**On-policy distillation（OPD）**（中）：student 以 on-policy 方式生成 rollout（例如在 RL 框架内），由独立的 teacher 对每个访问到的 token 打分，用逐 token 的 KL divergence（KL）训练 student。**On-policy self-distillation（OPSD）**（右）：同一个模型身兼两职——在上下文中加入特权信息（提示）构成 teacher 轨迹，并用 KL loss 将无提示的生成向其蒸馏，无需单独的 teacher 模型。](images/distillation_directionality_tikz.png){#fig:distillation-directionality data-dark-src="images/distillation_directionality_tikz-dark.png"}

![字符串反转任务上的 on-policy self-distillation（OPSD）。同一个 policy $\pi_\theta$ 对同一个由 student 采样的 completion $y$ 进行两次前向传播：**teacher** 传播以问题加一个正确的同组示范为条件（黄色），**student** 传播仅以问题为条件（绿色）。两次传播之间的逐 token reverse KL（对 teacher 做 stop-gradient）将仅以问题为条件的 policy 拉向其以示范为条件的自身；高亮的列是分布差异最大的、被错误采样的 token。](images/sdpo_tikz.png){#fig:sdpo data-dark-src="images/sdpo_tikz-dark.png"}

### 建议实验

配套代码 `code/distillation/` 实现了 SDPO [@hubotter2026reinforcement]，即 @fig:sdpo 所示的 on-policy self-distillation 设置（同期的 OPSD 论文 [@zhao2026selfdistilled] 与之密切相关）：同一个 policy 既充当以示范为条件的 teacher，又充当仅以问题为条件的 student，并用逐 token 的 reverse KL 进行训练。
它运行在一个小型字符串反转任务上，使得 on-policy distillation 循环的成本足够低，可以在单张 GPU 上完整观察。

1. **运行 SDPO 字符串反转示例。**

   ```bash
   cd code/
   uv run python -m distillation.train --config distillation/configs/sdpo.yaml
   ```

   关注 `reward`、`loss` 和 `skipped`，以及循环中打印的 teacher/student rollout 样本。
   `skipped` 计数是被轮询到的 prompt 中，其采样组内不含任何正确 rollout 的数量；随着 student 的提升，被跳过的 prompt 会越来越少，`reward` 则趋近于 1。

2. **调整 on-policy 相关参数。**
   复制 `distillation/configs/sdpo.yaml`，在保持任务不变的情况下扫描 `num_rollouts`、`kl_top_k` 和 `prompts_per_step`。
   每个 prompt 的 rollout 越多，越容易找到正确的同组示范（降低 `skipped`），代价是每步生成量更大；`kl_top_k` 则在 reverse KL 所匹配的 teacher 分布比例与计算开销之间进行权衡。

## AI Feedback

在 RLHF 快速发展后不久，来自 AI 反馈的强化学习（RLAIF）作为一种替代方案应运而生——AI 可以近似 pipeline 中人类数据的部分，从而加速实验或研究进展。
AI feedback 广义上是指一系列技术，利用 AI 来增强或生成数据，以解释某一输入的质量（可用于不同的训练方法或评估），其起点是成对偏好 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing]。
使用 RLAIF 来完全替代或增强人类反馈的动机有很多。
在 RLHF 流程中，AI feedback 以其在偏好数据收集和相关 reward model 训练阶段的作用最为著名（Constitutional AI 是其中一种具体实现方式）。
在本章中，我们聚焦于通用 AI feedback 及其在 RLHF 训练 pipeline 中的这一具体用法，并在本书后续章节中介绍理解或使用 synthetic data 的更多方式。

随着 AI feedback 日趋成熟，其应用已超越简单替代人类偏好标签的范畴。
使廉价偏好数据收集成为可能的 LLM-as-a-judge 基础设施，同样支撑了可扩展的评估（见第 16 章），以及近年来基于 rubric 的 reward——将 RL 训练延伸至无可验证答案的领域，这是本章后续将探索的前沿方向。

### 平衡 AI 与人类 Feedback 数据

相较于人类，AI 模型在生成特定数量反馈方面的成本要低得多——截至 2026 年，一条人类偏好数据的成本约为 1 美元或更高（某些情况下每条 prompt 甚至超过 10 美元），而使用前沿 AI 模型（如 GPT-4o）获取 AI feedback 的成本则低于 0.01 美元。
此外，人力成本大体保持不变，而领先模型在这些任务上的性能持续提升，单位性能成本不断下降。
这种成本差异，使得原本因成本过高而被拒之门外的大量研究者，也能进入 RLHF 方法实验的市场。

除价格因素外，AI feedback 在性能上引入了与人类反馈不同的*权衡*，相关研究在学界仍在持续推进。
AI feedback 在我们所训练的 language model 的评估中扮演着更为突出的角色——其低廉的成本使其得以应用于各种大规模任务，而在这些场景中，人类数据的成本（或时间延迟）几乎无法承受。
上述所有话题深度交织——AI feedback 数据永远不会完全取代人类数据，即便是在评估方面；而用于评估的 AI feedback 数量将远超用于训练的，因为进行评估的人远多于训练模型的人。

AI feedback 数据在哪些具体领域和应用场景——即对话、安全、推理、数学等——中优于人类数据，目前尚无定论。
RLAIF 的早期研究表明，AI feedback 可以完全取代人类数据，并将其誉为有效的替代方案 [@lee2023rlaif]，尤其当评估仅聚焦于对话任务时更是如此 [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels]。
ChatGPT 出现后研究 RLHF 的早期文献，评估体系较为单一，聚焦于模型作为有用助手跨领域的"对齐"程度（详见第 17 章）。
后续研究呈现出更为细腻的图景——在更广泛的评估集（例如包含部分推理任务）上，最优均衡点是将一批难以准确标注的数据点路由给人类，而大部分数据则交由 AI feedback 处理 [@miranda2024hybrid] [@xu2025rlthf]。
目前虽缺乏针对更广领域 RLHF 中人类与 AI feedback 数据平衡的专项研究，但已有众多技术报告表明 RLHF 总体上能提升这一广泛评估集的表现——其中一些使用 DPO（如 Ai2 的 Tülu 3 [@lambert2024t]、Olmo 3 [@teamolmo2025olmo3]，以及 HuggingFace 的 SmolLM 3 [@bakouch2025smollm3]），另一些则使用在线 RLHF pipeline（如 Nvidia 的工作，将 Scale AI 的人类偏好数据与基于 LLM 的反馈混合使用（通过 helpsteer 系列工作 [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]）：Nemotron Nano 3 [@nvidia2025nemotron3nano]、Nemotron-Cascade [@wang2025nemotron]，以及 Llama-Nemotron 推理模型 [@bercovich2025llamanemotron]）。

总体而言，尽管 AI feedback 及相关方法对该领域显然极具价值，但人类数据尚未被这些成本更低的替代方案完全取代。
目前存在诸多假设，但尚未有研究证明人类数据是否能在实际产品设置中实现对模型更精细的控制，或是否对 character training 等新兴训练方法有独特作用（character training 是一套新兴技术，可精确控制模型的个性，详见第 17 章）。
对于入门者而言，AI feedback 应作为第一选择；但对于扩展至更大规模运营的 pipeline，最终纳入人类反馈的过渡几乎是必然的。

RLAIF 这一术语由 Anthropic 在论文 *Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional] 中提出，该论文标题中两种方法（Constitutional AI 与 AI Feedback）之间的关系，在 AI 社区中曾引发初期困惑。
自 Constitutional AI（CAI）论文发布及 RLAIF 正式化以来，RLAIF 已成为 post-training 和 RLHF 文献中的默认方法——相关示例已多到难以一一列举。
两者的关系应理解为：CAI 是开创更广泛 RLAIF 领域的奠基性示例。

关于人类数据与 AI feedback 数据差异的一个经验法则如下：

1. 人类数据具有高噪声、低偏差的特点。这意味着数据的收集和过滤可能更为困难，但一旦整理妥当，将提供非常可靠的信号。
2. Synthetic 偏好数据具有低噪声、高偏差的特点。这意味着 AI feedback 数据更易于上手，但可能对模型产生棘手且意外的系统性二阶效应，并被系统性地反映在数据中。

本书重点介绍了大量学术研究成果，展示了如何在 RLHF 工作流中替换为 AI 偏好数据并取得强劲的评估分数 [@miranda2024hybrid]，但更广泛的行业趋势表明，RLHF 文献与更不透明的最佳实践之间存在隔阂。
在整个行业中，人类数据通常被视为一道重要的护城河和主要的技术优势。

### 为判断任务构建专用 LLM

随着 RLAIF 方法日趋普及，许多人开始思考：生成回答的模型与生成批判或评分的模型是否应该有所区分。
具体而言，LLM-as-a-judge 的校准问题受到了质疑。
多项研究表明，LLM 作为评估者时存在不一致性 [@wang2023large]，且倾向于偏好自身生成的回答（被称为自我偏好偏差）[@panickssery2024llm]。

针对这些偏差，许多人提出了这样一个问题：是否可以专门训练一个独立模型来承担这一标注任务？
已有多个模型以替代前沿模型充当数据标注工具为目标而发布，例如批判模型 Shepherd [@wang2023shepherd] 和 CritiqueLLM [@ke2023critiquellm]，以及用于评估回答性能的模型 Auto-J [@li2023generative]、Prometheus [@kim2023prometheus]、Prometheus 2 [@kim2024prometheus] 和 Prometheus-Vision [@lee2024prometheus]，但这些模型并未在已有文献记录的训练流程中得到广泛采用。
部分研究发现，通过重复采样扩大推理规模 [@brown2024large] [@zhao2025sample] [@kalra2025verdict]、self-refinement [@madaan2023self] 或竞标赛排名 [@pace2024west] 能够获得更接近真实判断的估计，或生成质量更高的偏好对。
其他校准技术则使模型的生成能力与判断能力协同演进 [@wu2024meta]。
普遍的共识是：尽管偏差确实存在，但领先的语言模型已针对这一任务进行了大量训练——因为无论是 AI 实验室的内部运营还是客户的广泛使用都有此需求——因此通常无需自行训练评判模型，除非你的任务涉及大量未公开于互联网的私有信息。

## Constitutional AI

Constitutional AI（CAI）方法被 Anthropic 用于其 Claude 模型，是最早有文献记载的、大规模将 synthetic data 用于 RLHF 训练的案例。
Constitutional AI 通过两种方式生成 synthetic data：

1. 对经过 fine-tuning 的指令数据进行批判，使其遵循一套原则，例如"该回答是否鼓励暴力？"或"该回答是否真实？"当模型对问题生成回答时，会对照 constitution 中的原则列表进行检验，并不断优化回答。随后，模型在这一最终 dataset 上进行 fine-tuning。
2. 通过让语言模型判断哪个补全更优——结合 constitution 中随机抽取的某条原则作为背景——来生成成对偏好数据（类似于基于原则引导的 reward model 研究 [@sun2024salmon]）。此后，RLHF 照常进行，使用的是 synthetic data，这也是 RLAIF 这一名称的由来。

总体而言，CAI 以其第二部分（即偏好数据）最为人所知，但其为指令数据引入的方法在后训练阶段的通用数据过滤和 synthetic data 生成方法中被广泛采用。

CAI 可以形式化如下。

Bai et al. 2022 利用一套人工编写的原则（称为 *constitution*），使用独立的 LLM 生成用于 fine-tuning 的人工偏好数据和指令数据 [@bai2022constitutional]。
Constitution $\mathcal{C}$ 是一组书面原则，指明批判阶段需关注的具体方面。
指令数据的构建方式是：反复从 $\mathcal{C}$ 中抽取原则 $c_i$，要求模型修改其对提示 $x$ 的最新输出 $y^i$，使其符合 $c_i$。
这将产生一系列指令变体 $\{y^0, y^1, \cdots, y^n\}$，对应于批判过程中使用的原则 $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$。
最终数据点为提示 $x$ 与某个 $n$ 对应的最终补全 $y^n$ 的组合。

偏好数据的构建方式类似但更为简单：使用 $\mathcal{C}$ 中的原则子集作为 feedback model 的背景。
Feedback model 接收提示 $x$、一组原则 $\{c_0, \cdots, c_n\}$，以及来自先前 RLHF dataset 的两个补全 $y_0$ 和 $y_1$（分别标记为答案 (A) 和 (B)）。
新数据点的生成方式是：让语言模型选择输出 (A) 或 (B) 中质量更高且更符合所述原则的一个。
在早期模型中，可以通过提示模型输入 `The answer is: `，然后比较 (A) 或 (B) 对应的 token 概率来实现；但现在更常见的做法是由模型先阐明推理过程再给出答案，这通常被称为一种 generative reward model [@mahan2024generative]。

### CAI 延伸阅读

Constitutional AI 有许多相关研究方向和扩展，但其中鲜有被明确记载为 RLHF 和后训练流程中清晰改进的方法。

- OpenAI 发布了 Model Spec [@openai2024modelspec]，这是一份阐明其模型预期行为的文件，并表示正在探索让模型直接参照该文件进行对齐的方法（可视为 CAI 的近亲）。OpenAI 持续更新其 spec，并使用一种称为 Deliberative Alignment [@guan2024deliberative] 的方法训练了其推理模型（如 o1），使模型在对齐过程中参照这些安全或行为策略。
- Anthropic 持续在模型训练中使用 CAI，不断更新 Claude 所使用的 constitution [@Anthropic2023ClaudesConstitution]，并探索群体集体如何在模型原则上达成共识，以及当外部群体自行创建原则后再与 Anthropic 共享以训练模型时，这将如何改变模型行为 [@ganguli2023]。
- 开源社区探索了将 CAI 应用于开放 dataset 的复现工作 [@Huang2024cai]，以及探索语言模型之间对话数据生成的研究 [@lambert2024self]。
- 其他工作则将原则驱动的偏好或 feedback 与不同的优化方法相结合。
Sun et al. 2023 [@sun2023principledriven] 将原则作为 reward model 的背景信息，并用于训练 Dromedary 模型 [@sun2024salmon]。
Glaese et al. 2022 [@glaese2022improving] 使用原则来提高 RLHF 过程中人工判断的准确性。
Liu et al. 2025 [@liu2025inference] 训练了一个 reward model，使其能在推理时自动生成原则，并以此给出最终评分。
Franken et al. 2024 [@franken2024self] 将遵循原则形式化为互信息最大化问题，预训练模型无需任何标注即可学习。

## Rubrics：针对特定提示的 AI Feedback 训练方法

AI feedback 在训练中的作用于 2024 年底至 2025 年间显著增强，彼时该领域正寻求以可验证奖励扩展强化学习的途径（见第七章）。
Rubrics 的概念应运而生，旨在为那些没有明确可验证答案的提示提供近似可验证的评判标准。
这将允许模型对一个问题尝试生成多个答案，并通过 RL 朝着最优答案更新。
这一思路与本章讨论的其他方法密切相关，其有效运作可能始于 LLM 评判能力和 synthetic data 实践在业界整体提升之后。
如今，以 rubrics 作为奖励的 RL 已被证明能在科学推理、事实性等技能上带来有意义的提升 [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics]。

下方展示了一个 rubric 示例及其对应的提示 [@liu2025openrubrics]：
```text
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**: 
1. The response includes exactly five distinct artifacts as requested. [Hard Rule] 
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule] 
3. The response provides a brief description of each artifact's historical significance. [Hard Rule] 
4. The response provides a brief description of each artifact's mysterious origins or unexplained aspects. [Hard Rule] 
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule] 
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle] 
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle] 
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle] 
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle] 
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

`[Hard Rule]` 和 `[Principle]` 是用于标注某条 feedback 优先级的特定标签。也可以使用其他方式表示重要程度，例如简单的优先级数字。

Rubric 的生成通常针对训练数据中的每个提示单独进行，这在准备阶段会积累可观的 synthetic data 成本。
为缓解这一问题，通常先按领域套用一个通用 rubric 作为起点，再由监督语言模型为每个提示分配细粒度的 rubric 评分，以引导训练所用的 feedback。
下方展示了一个用于为科学任务生成 rubric 的示例提示 [@gunjal2025rubrics]：

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. 
Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains. 
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. 
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. 

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.
Now, given the question and reference_answer, generate the rubric as described. 
The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.
```

另一个更简单的示例如下 [@rezaei2025onlinerubrics]：

```text
SYSTEM:
You generate evaluation rubrics for grading an assistant's response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don't rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1-10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

如你所见，这些提示词可以非常详细，并且会针对训练设置进行调整。

将 RL 训练与 rubrics 结合的做法将持续演进，超越其早期应用，延伸至指令遵循 [@he2025advancedif]、deep research [@shao2025drtulu]、对 deep research agent 的评估 [@sharma2025researchrubrics]，以及长文本生成 [@ruan2025expertlongbench]。
