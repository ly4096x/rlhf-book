---
title: "第 7 讲：合成数据与现代 post-training 方法"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 7 讲"
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
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only — leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
---

<!-- Source note: build with `make teach`, which copies assets/ into the output. A single-file `colloquium build -o ...` does NOT copy assets/, so the figures 404 in that standalone build. -->
<!-- Notation note: chapter 12 is careful about u = teacher trajectory (u~pi_T, forward KL) vs a = student-sampled action (a~pi_theta, reverse KL). Keep q=pi_T, p=pi_theta consistent. KL is written D_{\mathrm{KL}} throughout this deck (matching the chapter). -->

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 7 讲：合成数据与现代 post-training 方法

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 12 章：合成数据。</p>

---

<!-- animate: bullets -->
## 合成数据为何后来居上

最早用 RLHF 训练模型时，人类数据是获得高质量回复和可靠反馈的*唯一*途径。随着模型变强，这一前提很快就不再成立。

- **更便宜、更快的迭代** -- 合成数据降低了一次 post-training 实验的成本，让那些被人类数据流水线拒之门外的人也能入场。数据的收集速度也快得多，这也催生了我们如今听到的 RSI 论点。
- **能力门槛** -- post-training 中的合成数据，直到 GPT-4 级别的模型出现才真正奏效。Llama 2 和 GPT-3.5-Turbo 还不够可靠，无法*生成*或*监督*数据；LLM-as-a-judge 的能力是在从 GPT-3.5 → GPT-4 的跃升中涌现的。
- **post-training 的重心** -- 如今，领先模型*需要*合成数据才能达到前沿水平。Distillation（蒸馏）是指把更强模型的能力迁移给学生模型的通用说法。

---

<!-- columns: 50/50 -->
## 本讲内容

我们梳理合成数据如何取代或扩展了 post-training 流程的大部分环节 -- 然后从零推导 **on-policy distillation**，作为技术核心。

**定义：** 合成数据是指任何用于训练 AI 模型、且其生成过程在某种程度上来自另一个 AI 模型的训练数据。范围非常广！

|||

```box
title: 大致的计划
tone: accent
content: |
  1. 合成数据的**角色**
  2. 借助合成数据的**通用蒸馏**
  3. 通向 **on-policy distillation** 之路（技术核心）
  4. **AI feedback** 与 **Constitutional AI**
  5. **Rubrics** -- 针对具体 prompt 的 AI feedback
```

---

<!-- rows: 50/50 -->
## 第 7 讲：所处位置

<!-- row-columns: 32/36/32 -->

```box
title: 概览
tone: muted
compact: true
content: |
  1. 引言
  2. 相关工作
  3. 训练概览
```

|||

```box
title: 核心训练流程
tone: muted
compact: true
content: |
  4. 指令微调
  5. Reward model
  6. 强化学习
  7. 推理
  8. 直接对齐
  9. Rejection sampling
```

|||

```box
title: 数据与偏好
tone: accent
compact: true
content: |
  10. 偏好的本质
  11. 偏好数据
  12. **合成数据与 CAI**
```

===

<!-- row-columns: 32/36/32 -->

```box
title: 实践考量
tone: muted
compact: true
content: |
  13. Tool use
  14. 过度优化
  15. 正则化
  16. 评估
  17. 产品与 Character
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


<!-- rows: 60/40 -->
## 回顾：合成数据在流程中的位置

<!-- row-columns: 48/52 -->
post-training 流程包含许多环节：

1. 收集 / 生成 **prompt**
2. 为 SFT 生成 **completion**
3. 为 RLHF 收集 **偏好**
4. 为 RLVR 计算 **reward**
5. 过滤、清洗，以及以上所有环节

|||

来自其他语言模型的合成数据如今为每个环节提供输入：从种子生成 prompt、生成 completion、标注偏好，以及为 RL 验证答案。

我们还会介绍围绕这些思路出现的几种训练方法。

===

![现代 post-training 会运行多轮反馈与优化；合成数据如今为大部分循环提供数据。](assets/rlhf-complex.png)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 1 部分：合成数据的角色

---

<!-- animate: bullets -->
## 合成数据的用途

现代 post-training 中的“合成数据”贯穿整个流程——同一个模型可以被复用于多种角色：

- **从种子样例生成新 prompt** [@wang2022self]
- **修改 / 扩展** 已有 prompt
- **为 prompt 生成 completion** [@numina_math_7b]
- **提供 AI 反馈** 以构建偏好数据 [@cui2023ultrafeedback]
- **过滤 completion** 以保证质量 [@li2024superfiltering]
- **验证** 答案，作为 RL 的 reward

---

## 人类数据与 synthetic data 的平衡

Synthetic data 并未在整个流程中均匀地取代人类数据。

- **指令数据（SFT）：** synthetic 基本已经*胜出*——在大规模场景下，distillation 优于大多数人类写手。
- **偏好数据（RLHF）：** *好坏参半*——学术研究表明其表现相当，但前沿实验室仍将人类偏好数据视为竞争壁垒。
- **评测：** LLM-as-a-judge 能低成本地扩展*打分*，但 benchmark 和 ground-truth 标签仍需要人类作为依据/相关性校验。

人类数据的整理在很大程度上决定了模型的前沿水平并为早期进展播下种子，随后再用 synthetic data 将其规模化。

在 ChatGPT 发布前后，人类数据是进步的核心驱动力。

---

<!-- columns: 58/42 -->
## 模型坍缩：一个过时的担忧

![递归式自训练会逐代收窄分布，尾部最先消失。来源：Shumailov et al. (2024)。](assets/model-collapse-shumailov.png)

|||

一种常见批评：反复用模型自身的生成内容进行训练，会收窄有效的训练分布 [@shumailov2024ai]。

其论点如下：
*随着多样性下降，稀有的事实和风格代表性不足，微小的错误会在迭代中不断累积。*

---

## 模型坍缩：一个过时的担忧

但坍缩主要是*未经过滤、单一模型、自训练*循环的失败。实践中可以通过以下方式避免：

- 混入真实/人类数据，
- 使用多样化的教师模型，
- 去重，
- 严格的质量过滤。

有证据表明，synthetic data 可以——也应该——大规模使用，而不会出现最极端的坍缩说法所描述的灾难性退化 [@gerstgrasser2024model] [@feng2024beyond]。

---

## 经典的早期 synthetic 数据集及其规模

几个数据集定义了各自的时代：**UltraFeedback** [@cui2023ultrafeedback]（开启了 DPO 革命）、**Stanford Alpaca**（早期 chat SFT）、**Tülu 3** [@lambert2024t]（RLVR 之前的 SFT 数据集）以及 **OpenThoughts 3** [@guha2025openthoughts]（通用推理数据集）。

| | Prompt 数 | Token 数（约） |
| :--- | :---: | :---: |
| **Stanford Alpaca**（2023） | 52K | ~10M |
| **Tülu 3**（2024） | ~1M | ~500M |
| **OpenThoughts 3**（2025） | 1.2M | ~10B |

---

<!-- columns: 50/50 -->
## “Distillation”的两种含义

**技术含义（知识蒸馏，Knowledge Distillation）：** 训练一个较小的*学生*模型去匹配更强的*教师*模型的完整输出分布——使用*软*标签，而非 one-hot 目标 [@hinton2015distilling]。本讲稍后会详细介绍。

|||

**口语含义（当下的用法）：** “用更强模型的输出去训练一个较弱的模型。”人们说“中国的模型从 GPT/Claude 蒸馏而来”之类的话时，指的就是这个意思。


---

## 题外话：今年 Interconnects 上关于 distillation 的文章

```box
title: 延伸阅读
tone: surface
content: |
  - [**How much does distillation really matter for Chinese LLMs?**](https://www.interconnects.ai/p/how-much-does-distillation-really)（2026 年 2 月）- 中国开源模型的进步中，究竟有多少真正可以追溯到对前沿模型 API 的 distillation。
  - [**The distillation panic**](https://www.interconnects.ai/p/the-distillation-panic)（2026 年 5 月）- 为什么对前沿实验室基于恐惧的 distillation 言论迅速做出反应，可能对更广泛的 AI 生态不利。
```

---

<!-- img-align: center -->
<!-- cite-right: hinton2015distilling -->
## Distillation 1：经典知识蒸馏

![知识蒸馏通过 KL divergence 训练学生模型去匹配更大教师模型的软概率分布。两个模型看到相同的输入，温度缩放（$\tau > 1$）会软化分布，从而暴露类别之间的关系。](assets/knowledge_distillation_tikz.png)

---

<!-- columns: 52/48 -->
<!-- valign: center -->
## Distillation 2：synthetic data 生成流水线

![Prompt 被送入强模型以生成 completion，这些 completion 与 prompt 配对成训练数据集，用于通过标准监督学习 fine-tune 较小的模型。更复杂的流水线会编辑 completion、生成偏好对，或进行质量过滤。](assets/synthetic_data_distillation_tikz.png)

是的，这个图有意画得非常简单。

|||

```box
title: 推理模型带来了什么变化
tone: muted
content: |
  早期这件事很简单：在强模型的 API 输出上训练即可。

  推理模型让它变难了——你需要的是*推理轨迹（reasoning traces）*，而闭源 API 不再返回它们。Distillation 转向了：

  - **更激进的 jailbreaking**，以恢复被隐藏的轨迹（有争议）。
  - **更多使用开源模型**——拥有权重就能拿到完整的轨迹。
```

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 2 部分：通往 on-policy distillation（OPD）之路

---

<!-- valign: top -->
## 设置与记号

知识蒸馏（knowledge distillation）由 Hinton、Vinyals & Dean 提出 [@hinton2015distilling]，其原始形式使用**软**标签——即下一个 token 的完整分布——而不是 next-token prediction 的 one-hot 目标。要把它应用到自回归 LM，需要将损失按 token 分解。

设：

- $s$ -- 源 prompt
- $u = (u_1, \ldots, u_J)$ -- **teacher** 生成的完整输出序列
- $\mathcal{V}$ -- 输出词表（tokenizer）
- $q$ -- teacher 的下一个 token 分布，$p$ -- student 的

**记号约定**：$u$ 是 *teacher* 的轨迹；后文 on-policy / RL 记号中，我们用 $a$ 专指 *student* 采样得到的 completion。$u$ 与 $a$ 的区分正是 "on-policy" 的关键所在。

---

<!-- valign: top -->
## 词级（逐 token）蒸馏

**LLM 的标准师生蒸馏** [@kim-rush-2016-sequence] -- 把经典的 Hinton 软标签思想应用到每个 token 位置：

$$
\mathcal{L}_{\mathrm{WORD\text{-}KD}}
= -\sum_{j=1}^{J}\sum_{k=1}^{|\mathcal{V}|}
q(u_j = k \mid s, u_{<j})\,\log p(u_j = k \mid s, u_{<j}).
$$

在每个位置上，这都是在已有的训练语料 $\mathcal{V}$ 上匹配 teacher 对整个词表的*完整*下一个 token 分布 -- 是软标签，而不是 one-hot 目标。内层的 $\sum_{k=1}^{|\mathcal{V}|}$ 遍历*每一个可能的下一个 token*。

对每个 token 匹配一个分布听起来代价很高，但它是**可计算的**：每个位置只需 $|\mathcal{V}|$ 个概率 -- 与普通交叉熵相同的 $O(J\,|\mathcal{V}|)$ 代价。对整个*序列*做匹配才是难点（见下一页）。

<!-- footnote-right: 这里的 “word-level” 指基于 tokenizer 的逐 token。 -->

---

<!-- valign: top -->
## 序列级蒸馏

词级 KD 给出的是逐 token 的软分布。论文中序列级蒸馏的目标，是把这一思路应用到 teacher 自己生成的数据上，从而提供新的训练数据/信号，并提升性能！

（WORD-KD 是 Kim & Rush 论文中的 baseline。）

---

<!-- valign: top -->
## 序列级蒸馏

序列级 KD 转而用众数来近似 teacher 在*整个序列*空间 $\mathcal{U}$ 上的分布（这是对指数级多个序列的求和，不可计算）：由 *teacher* 生成一个高概率输出 $\hat{u} = \mathrm{BeamSearch}_q(s)$，student 用普通的 NLL 在其上训练：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SEQ\text{-}KD}}(s)
&= -\sum_{u \in \mathcal{U}} q(u \mid s)\log p(u \mid s) && \text{sum over all sequences: intractable}\\[4pt]
&\approx -\log p(\hat{u} \mid s) && \text{teacher} \approx \text{point mass on } \hat{u}\\[4pt]
&= -\sum_{j=1}^{|\hat{u}|}\log p(\hat{u}_j \mid s, \hat{u}_{<j}) && \text{autoregressive factorization}
\end{aligned}
$$

```box
tone: muted
content: |
  **补充**：第一批 offline-KD 模型是分类器——DistilBERT [@sanh2019distilbert] 和 TinyBERT [@jiao2020tinybert]——它们把 offline distillation 与 LM 的其他进展结合起来。由于这些是 embedding 模型，它们并非*序列*蒸馏，但与该领域的相关发展一脉相承。
```

---

<!-- valign: top -->
## SEQ-KD 与 SFT 之间的联系

首先回顾 teacher $q$ 与 student $p$ 的交叉熵——与 KD 损失形式相同：

$$
\begin{aligned}
H(q,p) &= -\sum_z q(z)\log p(z) && \text{definition}
\end{aligned}
$$

<!-- step -->

加上再减去 teacher 自身的对数概率 $\sum_z q(z)\log q(z)$ -- 它完全不涉及 $p$：

$$
\begin{aligned}
H(q,p) &= \underbrace{-\sum_z q(z)\log q(z)}_{H(q)} \;+\; \sum_z q(z)\log q(z) - \sum_z q(z)\log p(z) && \text{add and subtract}
\end{aligned}
$$

<!-- step -->

后两项共享权重 $q(z)$，因此可以合并为一个对数比（对数运算法则）：

$$
\begin{aligned}
H(q,p) &= H(q) \;+\; \sum_z q(z)\log\frac{q(z)}{p(z)} = H(q) + D_{\mathrm{KL}}(q\|p) && \text{definition of KL}
\end{aligned}
$$

---

<!-- valign: top -->
## SEQ-KD 与 SFT 之间的联系

我们刚刚证明了：交叉熵可以分解为 teacher 的熵加上一个 KL：

$$ H(q,p) = H(q) + D_{\mathrm{KL}}(q\|p) $$

<!-- step -->

$H(q)$ 只取决于固定的 teacher，因此最小化交叉熵*就是*最小化 forward KL：

$$ \boxed{\ \min_p H(q,p)\ \equiv\ \min_p D_{\mathrm{KL}}(q\|p)\quad\text{(forward KL: the direction of offline KD and SFT)}\ } $$

<!-- step -->

因此，序列级 KD 归结为在 teacher 生成文本上做 SFT -- 即 "offline KD"，这些生成内容通常由 teacher 模型提前生成好。


---

<!-- valign: top -->
## Offline KD 中的 exposure bias：训练 / 测试不匹配

Offline KD 采样 **teacher** 的轨迹 $u \sim \pi_T$，并逐 token 匹配（这里 $q = \pi_T$，$p = \pi_\theta$）：

$$
\mathcal{L}_{\mathrm{KD}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D},\, u \sim \pi_T(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(\pi_T(\cdot \mid s, u_{<t}) \,\|\, \pi_\theta(\cdot \mid s, u_{<t})\right).
$$

但在测试时，student 按自己的 policy 进行 rollout（$\ell_{\mathrm{task}}(s, a)$ 是该测试域上的 loss 函数）：

$$
\mathcal{L}_{\mathrm{eval}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D}_{\mathrm{test}},\, a \sim \pi_\theta(\cdot \mid s)}\ \ell_{\mathrm{task}}(s, a).
$$

由于 $\pi_T \neq \pi_\theta$，训练前缀与测试前缀来自不同的状态分布——**exposure bias** 指的就是 student 容易累积误差的倾向 [@arora-etal-2022-exposure] [@song2026surveyonpolicydistillationlarge]。

---

<!-- valign: top -->
## DAgger 类比：误差累积

On-policy distillation 与**模仿学习**（imitation learning）相通：DAgger 让 agent 在自己的 rollout 上训练，并由 oracle（teacher）标注它*本应*采取的动作 [@ross2011reduction]。

假设 student 在 teacher 诱导的状态上与 teacher 的单步误差不超过 $\epsilon$：

$$ \mathbb{E}_{s_t \sim d_{\pi_T}}\!\left[\mathbb{I}\!\left(\pi_\theta(s_t) \neq \pi_T(s_t)\right)\right] \leq \epsilon. $$

<!-- step -->

监督式模仿学习的分析 [@ross2011reduction] 表明，从 student 采样的长度为 $L$ 的轨迹上累积的期望 loss 可能随 $L$ 呈二次增长 [@song2026surveyonpolicydistillationlarge]：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\sum_{t=1}^{L} \ell\!\left(s, a_{<t}\right)\right] \leq O(\epsilon L^2).
$$

<!-- step -->

对 LLM 而言，这更多只是一种类比——token loss 是分布式的（KL），而不是 0-1 的动作不一致（本页第一个公式中的指示函数取值为 0 或 1）。

---

<!-- animate: bullets -->
## 从 off-policy 到 on-policy

从 student 而非 teacher 采样，可以大幅减少我们前面讨论的各种分布误差。
- 在 offline KD 中，一个次优 token 就可能让 student 的生成略微偏离分布（**out-of-distribution**）；模型在训练中从未见过该 token，因此更容易再次出错。
- On-policy distillation 迭代地从 student 采样，并由 teacher 在 student *自己*访问到的状态上进行监督。
- 在 DAgger 的交互式分析下，误差累积从 $O(\epsilon L^2)$ 降为 $O(\epsilon L)$ [@ross2011reduction]。
- **MiniLLM** 在 policy gradient 框架中引入了 reverse-KL 目标 [@gu2024minillm]；同期工作将 on-policy KD 与模仿学习联系起来 [@agarwal2024policy]（这篇论文更接近如今使用的现代蒸馏形式）。

---

<!-- valign: top -->
## On-policy distillation 的目标函数

令 $a = (a_1, \ldots, a_L)$ 为从 **student** $\pi_\theta(\cdot \mid s)$ 采样的 completion，token 级状态为 $s_t = (s, a_{<t})$。Teacher $\pi_T$ 固定不变：

$$
\boxed{\
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{s,\, a \sim \pi_\theta(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid s_t) \,\|\, \pi_T(\cdot \mid s_t)\right)
\ }
$$

这就进入了第 6 章（policy gradients）的采样 / 期望框架——也自然地衔接到交替进行生成与更新的现代 RL 训练基础设施。 

从 *student* 采样同时也翻转了我们所最小化的 KL 方向（forward $D_{\mathrm{KL}}(\pi_T \,\|\, \pi_\theta)$ → reverse $D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_T)$——见下一页）。

---

<!-- rows: 18/82 -->
## Forward KL 与 reverse KL

从 student 采样 completion，会使 $\pi_\theta$ 位于 KL 的**左侧**——这就翻转了它的方向（估计 KL 及其方向，取决于你从哪个分布采样）：

===

<!-- row-columns: 50/50 -->
**Offline KD / SFT**（forward KL）——期望取自 *teacher*，$z \sim \pi_T$（**off-policy**：固定的 teacher 数据集）：

$$ D_{\mathrm{KL}}(\pi_T \,\|\, \pi_\theta) = \mathbb{E}_{z \sim \pi_T}\!\left[\log\frac{\pi_T(z)}{\pi_\theta(z)}\right] $$

*模式覆盖*（mass-covering）——按 **teacher** 的概率质量加权：凡是 teacher 有概率质量而 $\pi_\theta \to 0$ 之处，log 比值都会爆炸，因此 student 必须覆盖 teacher 可能说出的*所有内容*。

|||

**On-policy distillation**（reverse KL）——期望取自 *student*，$z \sim \pi_\theta$（**on-policy**：从你正在训练的模型中采样）：

$$ D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_T) = \mathbb{E}_{z \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(z)}{\pi_T(z)}\right] $$

*模式寻找*（mode-seeking）——按 **student 自身**的概率质量加权：只在*它*把概率放在 teacher 不认可之处时才受惩罚，因此会塌缩到 teacher 的模式上。（为什么 reverse KL 往往更好：见第 15 章。）

---

<!-- valign: top -->
## 将 KD 视为 RL advantage

近期的实现直接把 KD 距离当作 reward：用每个 token 的负 reverse-KL 贡献作为 advantage [@lu2025onpolicy]。状态 $s_t$ 处的 reverse KL 是对 *student* 采样 token 的期望：

$$ D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid s_t) \,\|\, \pi_T(\cdot \mid s_t)\right) = \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\!\left[\log \pi_\theta(a_t \mid s_t) - \log \pi_T(a_t \mid s_t)\right]. $$

在实践中你从不需要对整个词表求和：单个采样 token 就是该 KL 的无偏估计 $\hat{D}_{\mathrm{KL}}$ [@schulman2020klapprox]，而每个 token 的 advantage 就是它的相反数：

$$ A_t^{\mathrm{OPD}} = -\hat{D}_{\mathrm{KL}} = -\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_T(a_t \mid s_t)\right) = \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t). $$

---

## 将 KD 视为 RL advantage
在实践中你从不需要对整个词表求和：单个采样 token 就是该 KL 的无偏估计 $\hat{D}_{\mathrm{KL}}$ [@schulman2020klapprox]，而每个 token 的 advantage 就是它的相反数：

$$ A_t^{\mathrm{OPD}} = \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t). $$

<!-- step -->
- teacher 认为概率更高的 token → 正 advantage；概率更低的 → 负 advantage。
- 形式相同，KL 方向相反：$\log \pi_T - \log \pi_\theta$ *看起来*像 forward-KL 项——使其成为 **reverse** KL 的是采样 $a_t \sim \pi_\theta$（student），而不是符号。
- Teacher 的 log-prob 差距是稠密的 token 级反馈——可能比稀疏的可验证 reward 或单个标量 reward-model 分数更丰富。
- 这可以融入现代 RL 机制——例如与 GRPO 的组内归一化结合使用，以实现更复杂的 reward shaping。

---

<!-- valign: top -->
## 多教师 on-policy distillation（MOPD）

使用多个教师——领域专家（数学、代码）或更早的 checkpoint——每个教师对每个 prompt 有一个混合权重 $w_k(s)$（满足 $\sum_k w_k(s) = 1$）[@mimo2025flash]：

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
= \mathbb{E}_{s,\, a \sim \pi_\theta(\cdot \mid s)}
\sum_t \sum_k w_k(s)\, D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid s_t) \,\|\, \pi_{T_k}(\cdot \mid s_t)\right).
$$

在大规模场景下，这让不断壮大的组织可以分工协作：许多小组各自训练专家教师，之后再蒸馏进同一个最终的学生模型（DeepSeek-V4-Pro [@deepseekai2026deepseekv4]、MiMo-V2-Flash [@mimo2025flash]）。

更多内容，请看我与 Finbarr Timbers 的[对话](https://www.youtube.com/watch?v=sbXEPxIazqY&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=9)，其中梳理了 MOPD 在 2026 年前沿模型训练方案中的应用。

---

<!-- valign: top -->
<!-- animate: bullets -->
## Self-distillation：推进前沿

在最前沿，已经没有更强的模型可供蒸馏。**On-Policy Self-Distillation（OPSD**）绕开了这个问题：教师就是*以特权信息为条件的同一个模型*——也就是学生模型在推理时拿不到的提示（hint）[@zhao2026selfdistilled]。self-distillation 的梯度会让模型认识到 hint 之后的 token 是错误的，并通过 OPD 风格的损失吸收这一教训。

**Cursor 的 Composer 2.5**（基于 Kimi K2.5）就是这样训练的 [@cursor2026composer25]（强烈推荐观看[这段视频](https://x.com/dwarkesh_sp/status/2062353335529935114)！）：

- 一个 judge 对照常见 bug 清单审查 RL 轨迹。
- 一旦发现 bug，就在序列中插入一个 hint——即模型在测试时看不到的特权信息。
- 模型朝着自己带 hint 的续写计算 KD 损失，学会在*没有提示*的情况下达到同样的结果。
- token 空间中的一个 hint 就足以实现自我纠正——而*如何*构造这种信号是一个活跃的研究方向（“privileged information”）[@penaloza2026privileged]。

---

## On-policy distillation 正变得非常流行

随着重心转向推理模型和 agentic 模型，teacher-student KD 迎来了复兴。采用新型知识蒸馏方式训练的主流模型包括：

- **Qwen3**（阿里巴巴）[@yang2025qwen3]
- **MiMo-V2-Flash**（小米）[@mimo2025flash]（提出了 MOPD）
- **DeepSeek-V4-Pro** [@deepseekai2026deepseekv4]

一个注意事项：逐 token 的 KD 要求学生和教师共用同一个 tokenizer——这在 post-training 方法中并不常见，也是它在实验室自己的模型家族内部表现出色的部分原因。

---

<!-- img-align: center -->
<!-- valign: center -->
## 从离线 KD 到 self-distillation

![**Sequence KD**：教师离线生成 rollout，学生用交叉熵损失去匹配这些固定的目标。](assets/distillation_directionality_1.png)

---

<!-- img-align: center -->
<!-- valign: center -->
## 从离线 KD 到 self-distillation

![**On-policy distillation** 颠倒了生成者：rollout 来自学生（例如在 RL 循环中），由另一个教师用逐 token 的 KL 为每个访问到的 token 打分。](assets/distillation_directionality_2.png)

---

<!-- img-align: center -->
<!-- valign: center -->
## 从离线 KD 到 self-distillation

![**On-policy self-distillation**：同一个模型身兼两职——在上下文中加入特权信息（hint）即可生成教师轨迹，无需单独的教师模型。](assets/distillation_directionality_3.png)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 3 部分：AI feedback 与 Constitutional AI

---

## 基于 AI 反馈的强化学习（RLAIF）

RLHF 兴起后不久，**RL from AI Feedback（RLAIF**）随之出现——用 AI 来近似人类数据这一步，最初从成对偏好开始 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing]。

在最初对其效果存在争论之后，它最终成为了默认做法。成本是其中一个显而易见的优势（估算）：

- 一条*人类*偏好数据：每个 prompt **\$1 -- \$10+**。
- *AI* 反馈（例如 GPT-4o）：每个 prompt **< \$0.01**。

人力成本大体持平；而模型的性价比不断提高。 
这使得此前负担不起的人群也能开展 RLHF 实验。

---

## Bias-variance 权衡

在早期 RLHF 与 RLAIF 的争论中，我听到过一条口头上的经验法则，它有一定的直觉合理性。

**人类数据**——*高噪声、低偏差。*

收集和过滤都更困难，但处理得当后能提供非常可靠的信号。


**合成偏好数据**——*低噪声、高偏差。*

更容易上手，但可能带有棘手的二阶效应，这些效应会*系统性地*固化在数据中。

---

## 平衡人类反馈与 AI 反馈

关于人类偏好数据与 AI 偏好数据之间的最终界限，目前没有明确的文献。一些背景包括：

- 早期 RLAIF 文献声称 AI 反馈可以完全取代人类数据——尤其是在聊天任务上 [@lee2023rlaif] [@cui2023ultrafeedback]。
- 后续工作更为细致：在更广泛的评测（包括推理）上，最佳组合是把困难的数据点交给人类，而把大部分数据交给 AI [@miranda2024hybrid] [@xu2025rlthf]。
- 还没有研究在*所有*领域上刻画人类/AI 的平衡，这凸显出开放的学术工作在获取高质量人类数据方面的普遍局限。
- **业界现实**：人类偏好数据仍被视为相当重要的护城河。这也可能更多来自 prompt 分布、隐式反馈等。人类数据仍在使用。

---

## 构建专用的 judge 模型

如果你大量使用 AI 反馈或 LLM-as-a-judge 评测，就会出现一个问题：是否应该为此专门准备一个模型。问题在于——它们效果究竟如何？

- 已有一些研究考察 LLM 在这些反馈任务上的表现。结果包括：LLM 作为评估者并不一致 [@wang2023large]，并且表现出**自我偏好偏差（self-preference bias）**——偏爱自己的生成结果 [@panickssery2024llm]。
- 确实存在专门的 judge / critic 模型——Prometheus [@kim2023prometheus]、Prometheus 2 [@kim2024prometheus] 等——但在已公开的 post-training 方案中并未被广泛采用。
- 均衡状态：前沿模型本身已经为评判做了大量训练，因此你很少需要自己的 judge 模型——*除非*你的任务涉及公共互联网上没有的私有数据。

---

## CAI：最早的大规模 synthetic RLHF 数据

Constitutional AI（CAI）是 Anthropic 为 Claude 模型打造的 post-training 方法，也是有文献记载的最早的、大规模将 synthetic data 用于 RLHF 的案例 [@bai2022constitutional]。CAI 指的是他们早期 Claude 模型所用的一组特定技术，此后已发生了很大变化（不过 Anthropic 仍在使用 constitution——是的，确实容易混淆）。

**RLAIF** 这一术语同样出自这篇论文，由此引发了对两者关系的困惑。

> CAI 是开启整个 RLAIF 领域的典型例子。CAI ⊂ RLAIF。

CAI 以两种方式生成 synthetic data——一种用于指令，一种用于偏好。 
其中广为人知、影响也更大的部分是偏好。

---

<!-- img-align: center -->
<!-- cite-right: bai2022constitutional -->
## Constitutional AI：原始示意图

![两个阶段。监督阶段（上）：模型依据 constitution 对自己经过红队测试的回复进行批评和修订，并 fine-tuning 得到 SL-CAI。RL 阶段（下，RLAIF）：基于 AI 比较数据训练的偏好模型驱动 RL，得到最终的 RL-CAI 模型。来源：Bai et al. (2022)。](assets/cai-overview.png)

---

<!-- valign: top -->
## 第 1 阶段：批评与修订 → SFT 数据

**constitution** $\mathcal{C}$ 是一组由人类编写的原则（例如 *“这个回答是否鼓励暴力？”*、*“这个回答是否真实？”*）。

<!-- step -->

模型反复采样一条原则 $c_i \in \mathcal{C}$，并针对 prompt $x$ 修订其最新输出 $y^i$，使之符合 $c_i$：

$$
\{c_0, c_1, \ldots, c_{n-1}\}\ \longrightarrow\ \{y^0, y^1, \ldots, y^n\}
\qquad\Longrightarrow\qquad
\text{SFT point } (x, y^n).
$$

随后模型在这份精炼后的数据集上进行 fine-tuning。 
这些批评方法也被广泛用于数据过滤和 synthetic data 生成。

---

<!-- valign: top -->
## 第 2 阶段：AI 偏好标注 → RLAIF

通过向 feedback 模型提供以下内容来构造偏好：

- 一个 prompt $x$，
- 一个原则子集 $\{c_0, \ldots, c_n\}$，
- 两个标记为 (A) / (B) 的 completion $y_0, y_1$。

模型选出质量更高、更符合该原则的回答。之后 RLHF 照常进行——因此称为 *RLAIF*。

<!-- step -->

```box
tone: muted
content: |
  补充：这也与 generative reward model 以及 LLM-as-a-judge 领域的进展等文献相关。见第 5 章 / 第 2 讲。

  - **早期：** 用 `The answer is: ` 作为 prompt，读取 A / B 中哪一个的 token 概率更高。
  - **现代：** **generative reward model** 先解释其推理过程，再做出选择 [@mahan2024generative]（参见由原则引导的 reward model [@sun2024salmon]）。
```

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 4 部分：Rubrics——针对特定 prompt 的 AI feedback

---

## rubrics 为何流行起来？

Rubrics 已成为在长尾领域扩展 RL 的常用工具。它们也被用于特定领域的评估，以及任何需要把领域专业知识“训练进”模型的场景。
- 一种把 RL with verifiable rewards（第 7 章）的思路扩展到*没有*明确可验证答案的任务的方法。
- 为 prompt 编写近乎可验证的标准，生成多个回答，并朝最好的回答做 RL 更新。
- 随着 LLM judge 和 synthetic data 实践成熟，于 2024 年末至 2025 年兴起。这很可能也与 RL 在前沿 post-training 中变得更为普及有关。
- 已在科学推理和事实性方面带来提升 [@gunjal2025rubrics] [@viswanathan2025checklists] [@rezaei2025onlinerubrics] [@liu2025openrubrics]。

---

<!-- valign: top -->
## 一个 rubric 示例

对于没有唯一正确答案的 prompt，依据带标签的标准打分 [@liu2025openrubrics]。下面是一个简化的 rubric 示例：

```text
Prompt: As a museum curator, suggest five obscure artifacts for a
"Mysteries of the Ancient World" exhibit ...

Rubric:
1. Includes exactly five distinct artifacts.            [Hard Rule]
2. Each from a different culture and time period.       [Hard Rule]
3. Brief description of each artifact's significance.   [Hard Rule]
6. Communicates clearly and is well-organized.          [Principle]
8. Uses engaging language that stimulates curiosity.    [Principle]
```

`[Hard Rule]` = 原子化、必须通过的检查项；`[Principle]` = 较为宽松的质量标准。标签编码了优先级（用数字也可以）。rubric 的各个子部分共同构成得分。

---

<!-- valign: top -->
## 通过 meta-prompt 为每个 prompt 生成

Rubrics 针对每个 prompt 单独生成，以保证质量并对 over-optimization 保持稳健。例如：先有按领域的基础 rubric，再由监督用的 LM 针对每个 prompt 加以细化 [@gunjal2025rubrics]。

```text
You are an expert rubric writer for science questions ...
Choose 7-20 rubric items based on question complexity.
Each item: title (2-4 words), description (category prefix +
  what to look for), weight.
  - Essential : critical facts; omission invalidates the answer  (1-5)
  - Important : key reasoning / completeness                     (1-5)
  - Optional  : nice-to-have depth or style                      (1-5)
  - Pitfall   : common mistakes to penalize                    (-1,-2)
Output: a JSON array of {title, description, weight}.
```

（有删节——真实的 meta-prompt 很长，并针对训练设置做过调优。）

---

## Rubric 的未来方向

基于 rubric 的 RL 是 AI 反馈驱动训练的前沿，应用已超出最初的场景：

- **高级指令遵循** [@he2025advancedif]
- **Deep research** agent [@shao2025drtulu]
- **评估**研究类 agent [@sharma2025researchrubrics]
- **长文本生成**，配合结构化 checklist [@ruan2025expertlongbench]

非常通用的工具！

---
<!-- layout: section-break -->

## 结论

---

## 合成数据是研究者打造优秀模型最常用的单一工具（基础设施之外）

这里介绍的技术会继续变得越来越复杂，看着它们发展非常有趣。
我刚开始写这本书时，搭建一些合成数据工作流还很费劲！
难免会踩坑，但总体而言，它如今已是成熟的工作流。

关于如何做到最好，还有很多细小的开放问题，但这往往是对手头技术问题的领域特定反映。

---

## 课程大纲

1. 引言与训练概览 -- 第 1-3 章
2. IFT、reward model、拒绝采样 -- 第 4、5、9 章
3. RL 理论 -- 第 6 章（上）
4. RL 实现与实践 -- 第 6 章（下）
5. Reasoning -- 第 7 章
6. 直接对齐算法 -- 第 8 章
7. 合成数据 -- 第 12 章
8. 偏好与偏好数据 -- 第 10/11 章

...

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
