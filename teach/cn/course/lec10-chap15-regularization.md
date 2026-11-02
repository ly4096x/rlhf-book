---
title: "第 10 讲：RL 中的正则化、RL 为何泛化、SFT 为何遗忘"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 10 讲"
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
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* Per-line math reveals: left-align display math and tighten spacing so
     stacked single-line equations read as one aligned derivation. */
  .slide.math-steps .katex-display { text-align: left; margin: 0.35em 0; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 10 讲：RL 中的正则化、RL 为何泛化、SFT 为何遗忘

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 15 章。</p>

---

<!-- layout: section-break -->
<!-- align: center -->

## 这些优化器如何改变模型的分布？我们又该如何控制？

---

<!-- columns: 45/55 -->
## 回顾：RLHF 流程

RL 这一步在最大化 reward model 给出的奖励的同时，**会对偏离 reference model 的行为施加惩罚**。

本讲就聚焦于这个惩罚——以及有它和没有它时会发生什么。

|||

![RLHF 训练流程——RL 这一步针对 reward model 优化 policy，并通过 KL penalty 使其保持在 reference model 附近。](assets/rlhf-overview.png)

---

<!-- columns: 50/50 -->
## RLVR 同样有正则化，但最佳实践不同

同样的 RL 循环，不同的奖励来源：用验证函数取代 reward model。

推理模型（在使用工具之前）常常去掉 KL penalty 以促进学习。

随着大规模工具使用的出现，正则化正在重新流行——但针对的是偏离*采样分布*的漂移，而不是相对 reference model 的 KL penalty（本讲末尾会详细展开）。

|||

![RLVR 用验证函数取代 reward model，但 RL 循环——以及正则化问题——是一样的。](assets/rlvr-system.png)

---

<!-- columns: 40/60 -->
## 本讲内容

我们如何显式地控制模型上的优化压力。

我们所用优化器的数学形式如何塑造模型的形态。

|||

```box
title: 本讲安排
tone: accent
content: |
  1. 控制 RL 的 **KL penalty**
  2. **KL divergence 的两个方向**
  3. **为什么 RL 比 SFT 泛化得更好**
  4. **其他相关工作**
```

---

<!-- columns: 50/50 -->

## 题外话：请先看第 9 讲

```iframe
src: https://rlhfbook.com/teach/course/lec9-chap14-appb-overoptimization/
title: Lecture 9 slides
height: 300
style: "border:none;display:block;width:100%;aspect-ratio:16/9;height:auto"
```

[幻灯片](https://rlhfbook.com/teach/course/lec9-chap14-appb-overoptimization/) -- 没错，这就是真正的幻灯片，嵌在幻灯片里。

|||

```iframe
src: https://www.youtube.com/embed/y04JhXpiI4s?list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y
title: Lecture 9 video
height: 300
style: "border:none;display:block;width:100%;aspect-ratio:16/9;height:auto"
```

[在 YouTube 上观看](https://www.youtube.com/watch?v=y04JhXpiI4s&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第一部分：显式 KL 惩罚

---

<!-- animate: bullets -->
## 对参考模型的 KL 惩罚用于控制 reward

LLM 中经典的 reward 函数：

$$ r = r_\theta - \lambda_{\text{KL}} \, D_{\mathrm{KL}}\!\left( \pi_{\text{RL}}(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x) \right) $$

- RL 中的 KL 控制早于 LLM：先是对话 agent [@jaques2017sequence]，随后是预训练模型的 fine-tuning [@jaques2020human]。 
- 这个惩罚是一个 **reverse KL**——通过从 *policy* 采样、再用参考模型打分来估计。它会惩罚 policy 把概率质量放在参考模型不会放的地方。
- "KL 距离"即所花费的*优化距离*（口语说法——KL 并不是真正的度量）。
- 实践中，上式中的 $\lambda$ 通常写作 $\beta$。 

---

<!-- columns: 45/55 -->
## 实践中如何测量 KL

从 $P$ 采样，把定义变成一个期望：

$$ D_{\mathrm{KL}}(P \,\|\, Q) = \mathbb{E}_{x \sim P}\left[ \log P(x) - \log Q(x) \right] $$

- 实践者会在训练中关注 KL 曲线——KL 非常大通常意味着有 bug 或模型已经崩坏。
- 低方差估计器（$k_1, k_2, k_3$）曾在 [Q&A 2](https://www.youtube.com/watch?v=gB-bYUECpzE&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=11) 中讨论过 [@schulman2020klapprox]。

|||

```python
# sample from the policy
tokens = model.generate(inputs)

# score sampled tokens under both models
logprobs     = log_softmax(model.forward(tokens).logits)
ref_logprobs = log_softmax(ref_model.forward(tokens).logits)

# log-probs of the tokens actually generated
token_lp     = gather(logprobs, tokens)
ref_token_lp = gather(ref_logprobs, tokens)

# sequence-level difference approximates KL
kl_approx = token_lp.sum(-1) - ref_token_lp.sum(-1)
```

---

## 真实训练中的曲线长什么样

![](assets/olmo2-grpo-zero-reward-kl.png)

一次 OLMo-2-7B 的 GRPO 训练，直接在*基座*模型上做 RL（R1-Zero 风格，无 SFT），来自 open-instruct（[公开 W&B 日志](https://wandb.ai/ai2-llm/open_instruct_public/reports/OLMo-2-7B-GRPO-Fast-Zero--VmlldzoxMjA0MjU4MQ)）：在约 1M 个 episode 中，可验证 reward 不断攀升，而记录的与参考模型的 KL 先上升、随后波动。这次训练的 **β = 0**。
这是一个健康的形态。

---

<!-- cite-right: ziegler2019fine -->
<!-- animate: bullets -->

## 静态还是动态 KL 惩罚？β 最初是一个反馈控制器

第一篇在语言模型上做 RLHF 的论文并没有固定 β。它设定了一个**目标 KL**，让控制器去追踪它：

$$ e_t = \operatorname{clip}\!\left( \frac{\mathrm{KL}(\pi_t, \pi_{\text{ref}}) - \mathrm{KL}_{\text{target}}}{\mathrm{KL}_{\text{target}}},\, -0.2,\, 0.2 \right), \qquad \beta_{t+1} = \beta_t \left(1 + K_\beta\, e_t\right) $$

- 一个"对数空间的比例控制器"（原文用语），其中 $K_\beta = 0.1$：KL 过高 → β 增大，把 policy 拉回来；过低 → β 减小，放开 policy。不同随机种子的训练会落在*同一个* KL 预算上，使实验之间可以比较。
- 这个想法比 RLHF 更早：PPO 原始的**自适应 KL 惩罚**变体围绕目标值将 β 翻倍或减半 [@schulman2017proximal]，而后来的约束 RL 通过在惩罚乘子上使用完整的 **PID 控制器**，明确了控制论的框架 [@stooke2020responsive]。
- 现代实践又转回使用较小的**静态** β——在许多 RLVR 推理配方中，甚至完全不用 KL 项。

---

<!-- cite-right: ziegler2019fine -->

## 静态还是动态 KL 惩罚？β 最初是一个反馈控制器

第一篇在语言模型上做 RLHF 的论文并没有固定 β。它设定了一个**目标 KL**，让控制器去追踪它：

$$ e_t = \operatorname{clip}\!\left( \frac{\mathrm{KL}(\pi_t, \pi_{\text{ref}}) - \mathrm{KL}_{\text{target}}}{\mathrm{KL}_{\text{target}}},\, -0.2,\, 0.2 \right), \qquad \beta_{t+1} = \beta_t \left(1 + K_\beta\, e_t\right) $$

看代码：[原始控制器](https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L115-L124)（lm-human-preferences，2019 年）· [TRL 的 `AdaptiveKLController`](https://github.com/huggingface/trl/blob/v0.11.4/trl/trainer/utils.py#L54-L69)（v0.11.4——在现代重写中已被删除）· [如今的 open-instruct](https://github.com/allenai/open-instruct/blob/5b2ebfa12381925bb431845d588dbc9ebead20a7/open_instruct/grpo_utils.py#L104-L105)：静态的 `beta = 0.05`，[直接作用在 loss 中](https://github.com/allenai/open-instruct/blob/5b2ebfa12381925bb431845d588dbc9ebead20a7/open_instruct/grpo_fast.py#L718)。

---

<!-- layout: section-break -->
<!-- align: center -->

## 第二部分：RL 优化是一种 reverse KL 最小化

---

## reward 惩罚与优化形态是两回事

我们一开始讲的是 RL 设置上的一个**惩罚**：加在 reward 上的一项，带有一个你来调节（或控制）的系数。

接下来这部分讲的是 **RL 优化的形态**，以及它与 KL 的关系。
归根结底是"KL 的哪个方向"——这由*样本从哪里来*决定：

<!-- step -->

- **SFT** 从*数据（或另一个教师模型）*采样 → 最小化其 loss 恰好就是最小化一个 **forward** KL。
- **RL** 从*自身*采样 → *在*开启惩罚的情况下，最大化目标恰好就是朝着一个被 reward 倾斜的参考分布做 **reverse-KL 最小化**。这只在优化中包含 KL 惩罚时才成立（并不适用于所有 RLVR 结果）……但 on-policy 采样仍然会让 RL *偏向* KL 最小的解。后面会详细讲。

---

<!-- valign: top -->
<!-- title: center -->
<!-- cite-right: chen2025retainingdoingroleonpolicy -->
## RL 就是 reverse KL

从 KL 正则化目标出发（也就是我们的 RL 训练器所优化的目标）：

$$
\max_\pi\; \mathcal{J}_{\text{RL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)}\left[ r(x, y) \right] - \beta\, D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right)
$$

<!-- step -->

除以 $-\beta$ 并归一化后，目标变成单个 KL——当且仅当 $\pi$ 等于最优 policy $\pi_\star$ 时取得最小值。第 6 讲走过同样的推导（DPO 的起点）：

$$ \pi_\star(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left(\tfrac{1}{\beta}\, r(x,y)\right) $$

<!-- step -->

可以把 $\pi_\star$ 理解为**reward 倾斜后的参考模型**（这个名字是 AI 建议的，我想不出更好的）：取 $\pi_{\text{ref}}$，把每个 completion 的概率乘以 $\exp(r/\beta)$，再重新归一化（这正是 $Z(x)$ 的全部作用）。这种“倾斜”把概率质量推向高 reward 的 completion，同时仍停留在参考模型的支撑集之内——$\beta$ 大时几乎不倾斜，$\beta$ 小时则集中在 reward 最高的 completion 上。

---

<!-- valign: top -->
<!-- title: center -->
<!-- class: math-steps -->
<!-- cite-right: chen2025retainingdoingroleonpolicy -->
## RL 就是 reverse KL

现在展开到 $\pi_\star$ 的 reverse KL，代入 $\log \pi_\star = \log \pi_{\text{ref}} - \log Z(x) + \tfrac{1}{\beta} r(x,y)$：

$$ D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_\star) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}\left[ \log \pi_\theta(y \mid x) - \log \pi_\star(y \mid x) \right] \qquad \text{definition; samples from } \pi_\theta $$

<!-- step -->

$$ \phantom{D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_\star)} = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}\left[ \log \pi_\theta - \log \pi_{\text{ref}} + \log Z(x) - \tfrac{1}{\beta}\, r(x,y) \right] \qquad \text{substitute } \log \pi_\star $$

<!-- step -->

$$ \phantom{D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_\star)} = -\tfrac{1}{\beta}\,\mathbb{E}\left[r(x,y)\right] + D_{\mathrm{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right) + \underbrace{\log Z(x)}_{\text{constant}} \qquad \text{regroup the terms} $$

<!-- step -->

$$ \phantom{D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_\star)} \propto -\tfrac{1}{\beta}\, \mathcal{J}_{\text{RL}}(\theta) \qquad \text{drop the constant} $$

<!-- step -->

$$ \boxed{\ \max_\theta\, \mathcal{J}_{\text{RL}}(\theta) \iff \min_\theta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_\star)\ } $$

加上惩罚项之后，RL 并不只是*用到*了 reverse KL——整个目标**本身就是**一个 reverse KL，指向 reward 倾斜后的参考 policy。

---

<!-- valign: top -->
<!-- title: center -->
## SFT 是 forward KL

再看对照对象。SFT 在固定数据集上训练——样本来自*数据分布*，记作 $\pi_{\mathcal{D}}$，这对应 KL 的另一个方向：

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_{\mathcal{D}} \,\|\, \pi_\theta) &= \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_{\mathcal{D}}(y \mid x) - \log \pi_\theta(y \mid x) \right] && \text{definition; samples are the data}
\end{aligned}
$$

<!-- step -->

$$
\begin{aligned}
&= \underbrace{\mathbb{E}_{(x,y) \sim \mathcal{D}}\left[ \log \pi_{\mathcal{D}}(y \mid x) \right]}_{-H(\pi_{\mathcal{D}}),\ \text{constant in } \theta} \; - \; \mathbb{E}_{(x,y) \sim \mathcal{D}}\left[ \log \pi_\theta(y \mid x) \right] && \text{split the expectation}
\end{aligned}
$$

<!-- step -->

$$
\begin{aligned}
&= -H(\pi_{\mathcal{D}}) + \mathcal{L}_{\text{SFT}}(\theta) \;\propto\; \boxed{\ \mathcal{L}_{\text{SFT}}(\theta)\ } && \text{the NLL term is the SFT loss}
\end{aligned}
$$

梯度相同，最小值也相同：最小化 SFT loss *就是*最小化到数据分布的 forward KL。

---

<!-- rows: 18/82 -->
## SFT 与 RL 是 KL 的两个方向

这与第 7 讲（on-policy distillation）非常相似，两个方向的 KL 都出现了。Post-training 的数学一再重演：

===

<!-- row-columns: 50/50 -->
**Forward KL** -- 监督微调（即标准 KL）：

$$ D_{\mathrm{KL}}(\pi_{\mathcal{D}} \,\|\, \pi_\theta) = \mathbb{E}_{y \sim \pi_{\mathcal{D}}}\!\left[\log \tfrac{\pi_{\mathcal{D}}(y)}{\pi_\theta(y)}\right] $$

样本来自**目标**（固定数据集）。 

*模式覆盖*：只要目标有概率质量而 $\pi_\theta \to 0$，loss 就会爆炸——模型必须铺开以覆盖所有内容。

|||

<!-- step -->

**Reverse KL** -- 强化学习：

$$ D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_\star) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log \tfrac{\pi_\theta(y)}{\pi_\star(y)}\right] $$

样本来自**policy 自身**。 

*模式寻求*：只在它放置概率质量的地方受到惩罚，因此会集中在高 reward 的模式上。

---

<!-- rows: 18/82 -->
## 回顾第 7 讲：蒸馏中的同样两个方向

蒸馏版本，原样取自第 7 讲——用 teacher $\pi_T$ 取代目标分布。从 student 采样 completion，正是让 $\pi_\theta$ 出现在 KL 的**左侧**：

===

<!-- row-columns: 50/50 -->
**离线 KD / SFT**（forward KL）——期望取自 *teacher*，$z \sim \pi_T$（**off-policy**：固定的 teacher 数据集）：

$$ D_{\mathrm{KL}}(\pi_T \,\|\, \pi_\theta) = \mathbb{E}_{z \sim \pi_T}\!\left[\log\frac{\pi_T(z)}{\pi_\theta(z)}\right] $$

*Mass-covering*——按 **teacher** 的概率质量加权：只要 teacher 有概率质量而 $\pi_\theta \to 0$，对数比值就会爆炸，因此 student 必须覆盖 teacher *可能说出的一切*。

|||

<!-- step -->

**On-policy 蒸馏**（reverse KL）——期望取自 *student*，$z \sim \pi_\theta$（**on-policy**：采样的是你正在训练的模型）：

$$ D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_T) = \mathbb{E}_{z \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(z)}{\pi_T(z)}\right] $$

*Mode-seeking*——按 **student 自身**的概率质量加权：只有当*它*把概率放在 teacher 不认可的地方时才会受到惩罚，因此会坍缩到 teacher 的众数（mode）上。第 7 讲把*为什么 reverse KL 更好*留到了本讲。

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 3 部分：为什么 RL 的泛化更好

---

<!-- cite-right: chu2025sft -->
## “SFT 记忆，RL 泛化”

受控研究：在一个任务上做 post-training，再在规则变化下评估（即对问题做小幅修改，以测试分布外表现）[@chu2025sft]。

- **GeneralPoints**：用 $+,-,\times,\div$ 组合四张牌得到 24；测试时切换为人头牌规则（训练：J/Q/K = 10；测试：11/12/13）。
- **V-IRL**：视觉导航；测试时从绝对方向（北/东）切换为相对方向（左/右）。

在 V-IRL 上，RL 将分布外准确率从 **80.8% 提升到 91.8%**；SFT 则使其从 **80.8% 跌至 1.3%**——破坏了 base model 原本就具备的空间推理能力。

仅凭 on-policy 的结构，基于 RL 的 post-training 就带有*隐式*正则化；SFT 则更脆弱。

论文：[arxiv.org/abs/2501.17161](https://arxiv.org/abs/2501.17161)

---

## 哪个方向应该遗忘得更少？

直觉上的理解：forward KL 是 *mass-covering*，所以 SFT 应该保留每一个 mode——而 mode-seeking 的 RL 应该坍缩到某一个 mode 上，遗忘其余的。

这对吗？

---

<!-- img-fill -->
<!-- cite-right: chen2025retainingdoingroleonpolicy -->
## 哪个方向应该遗忘得更少？

这种直觉假设 policy 是单峰的——而 LLM 是多峰的。论文：[arxiv.org/abs/2510.18874](https://arxiv.org/abs/2510.18874)

![](assets/retaining_by_doing_mode_intuition.png)

---

<!-- rows: 50/50 -->
<!-- cite-right: shenfeld2026rls -->
## RL 的剃刀：同等性能下更低的 KL 漂移

> *“在新任务的众多高 reward 解中，RL 等 on-policy 方法天然偏向于在 KL divergence 意义上更接近原始 policy 的解。”*

- 遗忘与 KL 漂移相关：$\text{Forgetting} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[D_{\mathrm{KL}}\!\left(\pi_0 \,\|\, \pi\right)\right]\right)$，$R^2 = 0.96$——在**新任务的数据**上测得。这是一个低成本的遗忘预测指标。
- 消融实验：**on-policy 数据完全解释了差异**——负梯度没有可察觉的影响。论文：[arxiv.org/abs/2509.04259](https://arxiv.org/abs/2509.04259)

===

![在能解决新任务的 policy 中，RL 会收敛到 KL 上最接近 base model 的那些——在新任务性能相当时，对先前任务的保留更好。Shenfeld et al., 2026 (CC-BY)。](assets/rl_razor_motivation.png)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 4 部分：控制优化的其他工具

---

<!-- animate: bullets -->
## 实践中的其他正则化

其中大多数都是脚手架：为稳定某一特定设置而加入，在下一代模型中就被简化掉了。

- **预训练 next-token 预测梯度**（InstructGPT）：在目标中加入 $\gamma\, \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}}\left[\log \pi_{\text{RL}}(x)\right]$，“以修复在公开 NLP 数据集上的性能退化”[@ouyang2022training]。
- **DPO 搭配 NLL**：$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}} + \alpha\, \mathcal{L}_{\text{NLL}}$ 使 chosen 文本在绝对意义上保持高似然，而不只是相对更优 [@pang2024iterative]。
- reward model 的 **Margin loss**（Llama 2）：$-\log \sigma\!\left(r_{\theta}(y_c) - r_{\theta}(y_r) - m(y_c, y_r)\right)$，其中 margin $m$ 来自标注者评分差——即第 8 讲中的 Likert 量表 [@touvron2023llama]。



---

<!-- animate: bullets -->
## 2026：trust region / KL 惩罚正在退场

工具使用正在改变正则化需要承担的角色。在 agentic 配方中，对 reference 的 KL 惩罚正在消失：

- GLM-5 直接移除了它——“以加速 RL 改进”[@glm5team2026glm5]。Kimi 同样不使用：K2 → K3 的配方完全没有 KL 惩罚，也没有 reference policy [@moonshot2026kimik3]。
- 在我们的 TMax terminal-agent 配方中，我们测量了其中的权衡：较小的 KL 减轻了坍缩的严重程度，但降低了 reward，因此最终配方取 $\beta = 0$ [@ivison2026tmax]。
- TMax 论文概括了我们正在对付的数值问题的变化：20 多轮的轨迹、async/partial rollout，以及训练与推理引擎之间的不匹配，使得来自*采样器*的漂移成为关键失效点，而不是偏离初始化的漂移。（在 TMax 中，不稳定性在超过 10 个 assistant 轮次后增加，在 5 轮以下则不存在 [@ivison2026tmax]。）

---

<!-- rows: 22/78 -->
<!-- cite-right: qi2026dppo -->
## 采样分布上的 trust region

**DPPO**（Divergence Proximal Policy Optimization）不再使用 PPO 的逐 token ratio clip，而是根据 rollout 策略与训练策略之间*直接估计*的散度（二值 Total Variation）来 mask token [@qi2026dppo]。

===

<!-- row-columns: 55/45 -->
逐 token 的 ratio 只是对该散度的单样本噪声估计。GLM 的 IcePop [@glm5team2026glm5] 和 Kimi 的 log-ratio 区间 [@moonshot2026kimik3] 做的是同一件事：梯度被 mask，而不是被 clip。

这正是当下正则化研究的焦点——算法层面的更新，而非 KL 惩罚。

|||

![DPPO 论文中的动机：对于同样的 rollout token，逐 token 的概率 ratio（左）在低概率处会爆炸，而直接估计的 TV 散度（右）则保持稳定。Qi et al., 2026。](assets/dppo-ratio-vs-tv.png)

---

## 要点总结

- KL 惩罚是显式的控制手段：一种在策略自身样本上估计的 **reverse KL**。 
- KL 正则化的 RL 目标就是 reverse KL 最小化——向经过 reward 倾斜的参考策略做 mode-seeking——而 SFT 是 forward 方向，向数据做 mass-covering。
- 即使没有惩罚项，on-policy RL 也存在隐式正则化——SFT 记忆，RL 泛化。 

可验证奖励比 reward model 更不容易出现 overoptimization！因此，正则化正在发生变化，KL 惩罚所起的作用越来越小。

---

<!-- columns: 50/50 -->
## 课程回顾

0. 预备知识复习
1. 概览 *(第 1-3 章)*
2. IFT、reward model 与 rejection sampling *(第 4、5、9 章)*
3. RL：动机与数学 *(第 6 章)*
4. RL：实现与实践 *(第 6 章)*
5. 推理模型的崛起 *(第 7 章)*
6. Direct preference optimization *(第 8 章)*

|||

7. 合成数据与现代 post-training *(第 12 章)*
8. 偏好与偏好数据 *(第 10-11 章)*
9. Over-optimization 与 RLHF 的坏名声 *(第 14 章，附录 B)*
10. **正则化** *(第 15 章)* -- *今天*
11. **评测** *(第 16 章)* -- *下一讲（暂定）*
12. **工具使用基础** *(第 13 章)* -- 希望能讲？

---

<!-- rows: 85/15 -->
## 谢谢大家

欢迎提问和讨论！

如果你有空订阅或把我的内容分享给朋友，对扩大影响力帮助极大。

联系方式：nathan@natolambert.com

Newsletter：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

```builtwith
repo: natolambert/colloquium
```
