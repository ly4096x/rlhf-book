---
title: "人类反馈强化学习（RLHF）与 post-training 导论"
author: "Nathan Lambert"
date: "March 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: ""
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 人类反馈强化学习（RLHF）与 post-training 导论

<div class="colloquium-title-eyebrow">SALA 2026</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
<p>厄瓜多尔基多<br>2026 年 3 月 11 日</p>
</div>

<p class="colloquium-title-note">简要概览 RLHF、RLVR 以及现代 language model 的 post-training 配方。</p>

---

<!-- columns: 45/55 -->
## 什么是 language model？

核心性质：
- language model 为文本分配概率。
- 词块被拆分为 **token**，这是模型内部的表示形式。
- 给定前面的 token，模型预测下一个 token。不断重复这一过程，就能一步一步地生成补全（这被称为**自回归**）。

|||

![Transformer 的原始模型架构图。2017 年。](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 什么是（现代）language model？

现代 language model：
- 拥有数十亿到数万亿参数。
- 大多源自 Transformer 架构，该架构推广了**自注意力**机制以及全稠密层的使用。
- 预测和处理的远不止文本：Gemini 和 ChatGPT 可以处理图像、音频和视频。

|||

![Transformer 的原始模型架构图。2017 年。](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2017 年：Transformer 诞生

- **2017：** Transformer 诞生

|||

![Transformer 的原始模型架构图。2017 年。](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2018 年：GPT-1、ELMo 和 BERT

- 2017：Transformer 诞生
- **2018：** GPT-1、ELMo 和 BERT 发布

|||

![GPT-1 的 language model 架构。2018 年。](assets/gpt-1.png)
<!-- cite-left: radford2018gpt, peters2018elmo, devlin2018bert -->

---

<!-- rows: 30/70 -->
## 2019 年：GPT-2 与 scaling laws

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- **2019：** GPT-2 与 scaling laws

===

![著名的 scaling laws 图。2020 年。](assets/scaling-laws.png)
<!-- cite-right: radford2019gpt2, kaplan2020scaling -->

---

<!-- columns: 45/55 -->
## 2020 年：GPT-3 令人惊讶的能力

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- **2020：** GPT-3 令人惊讶的能力

|||

![GPT-3 以拓展 in-context learning 和 few-shot prompting 的理念而闻名。截图来自论文。](assets/few-shot.png)
<!-- cite-left: brown2020gpt3 -->

---

<!-- columns: 45/55 -->
## 2021：Stochastic Parrots

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 的惊人能力
- **2021：** Stochastic Parrots

|||

![](assets/stochastic-parrots.png)
<!-- cite-right: bender2021stochastic -->

---

<!-- columns: 45/55 -->
## 2022：ChatGPT

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 的惊人能力
- 2021：Stochastic Parrots
- **2022：** ChatGPT

|||

![](assets/chatgpt.webp)
<!-- cite-right: openai2022chatgpt -->

---

<!-- columns: 45/55 -->
## 2023：GPT-4 与前沿规模

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 的惊人能力
- 2021：Stochastic Parrots
- 2022：ChatGPT
- **2023：** GPT-4 与前沿规模

|||

![一张图片，称 NVIDIA CEO 黄仁勋疑似泄露了 GPT-4 是一个约 2T 参数的 MoE 模型。](assets/jensen-gpt4.jpeg)
<!-- cite-right: openai2023gpt4 -->

---

<!-- columns: 45/55 -->
## 2024：o1 与推理模型

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 的惊人能力
- 2021：Stochastic Parrots
- 2022：ChatGPT
- 2023：GPT-4 与前沿规模
- **2024：** o1 与推理模型

|||

![OpenAI o1 发布时那张著名的 test-time scaling 曲线图。](assets/o1-test-time.png)
<!-- cite-right: openai2024o1 -->

---

<!-- columns: 45/55 -->
## 2025：o3、Claude Code 与 agents

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 的惊人能力
- 2021：Stochastic Parrots
- 2022：ChatGPT
- 2023：GPT-4 与前沿规模
- 2024：o1 与推理模型
- **2025：** o3、Claude Code 与 agents

|||

![](assets/claude-code.png)
<!-- cite-right: openai2025o3, anthropic2025claudecode, openai2025agents -->

---

<!-- columns: 50/50 -->
## 预训练：next-token prediction

- 在来自网页、书籍、代码和文档的数万亿 token 文本上训练
  - 模型通常在 5-50+ 万亿 token 上训练
  - 1T 文本 token 约对应 3-5 TB 数据
  - 各实验室收集并过滤的数据量是模型实际使用量的 10-20 倍
  - 模型的整体数据漏斗规模约在 PB 量级
- 目标：预测每个序列中的下一个 token
- 结果：强大、灵活、有用的模型

|||

![](assets/pretraining_next_token_tikz.png)

---

## base model 补全文本

预训练之后，我们得到的是一个美化版的自动补全模型，例如：^[Base model 也正通过 midtraining 和更好的数据配比变得更加灵活。]

<div class="colloquium-spacer-md"></div>

```conversation
messages:
  - role: user
    content: "The president of the United States in 2006 was"
  - role: assistant
    model: "Llama 3.1 405B Base"
    content: "George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006..."
```

---

## 后训练让模型像聊天机器人一样回答

最早的现代后训练（或经过 RLHF 调优）模型，把续写格式转变为始终遵循“回答问题的风格”。
下面是早期对话模型的一个示例：

<div class="colloquium-spacer-md"></div>

```conversation
messages:
  - role: user
    content: "The president of the United States in 2006 was"
  - role: assistant
    model: "Tülu 3 405B"
    content: "George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009."
```

---

<!-- columns: 40/60 -->
## ChatGPT 让 RLHF 模型变得更加易用

模型的回答迅速演变，具备了：
- 更好的**格式**：直接、对话式的回答
- 更好的**风格**：有帮助、简洁、使用 Markdown 等
- 更好的**产品**：人们真的可以每天使用

|||

```conversation
size: 0.775
messages:
  - role: user
    content: "I'm giving a talk on RLHF tomorrow. Can you help me structure it?"
  - role: assistant
    content: |
      Absolutely — here's a simple structure:

      **1. Start with the basics**
      - What a language model is
      - Why next-token prediction is not enough

      **2. Explain post-training**
      - Instruction tuning
      - Preference tuning / RLHF
      - RLVR for verifiable tasks

      **3. End with why it matters**
      - Better products like ChatGPT
      - Better reasoning in models like o1
      ...
```

---
<!-- layout: section-break -->

## 那么，基于人类反馈的强化学习（RLHF）到底是什么？

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## 哪个后空翻更好？

![](assets/christiano-backflip-human.webp)

|||

![](assets/christiano-backflip-reward.webp)

---

<!-- cite-right: christiano2017, ziegler2019fine, ouyang2022training, bai2022constitutional -->
## 为什么会出现 RLHF？

- 许多目标人类很容易**判断**，却很难写成精确的奖励函数
- 在语言模型中，我们想要的往往是隐式的：**遵循意图**、**有帮助**、**无害**
- 预训练优化的是**下一个 token 预测**，而不是助手行为
- 偏好比较把这些人类判断转化为可扩展的训练信号

RLHF 让我们能够优化那些可以**评估**的行为，即使我们很难**明确指定**奖励。

---

<!-- columns: 40/60 -->
<!-- cite-right: christiano2017 -->
## 语言模型之前的 RLHF

- **TAMER**（Knox & Stone, 2008）— 人类给智能体的动作打分，从而学习奖励
- **Christiano et al. 2017** — 在 Atari 轨迹偏好上做 RLHF
- **Ziegler et al. 2019** — 首次在语言模型上做 RLHF

|||

![Christiano et al. 2017 系统图的复现版本。](assets/rlhf_schematic_tikz.png)

---

<!-- cite-right: ziegler2019fine -->
## 语言模型的 RLHF：比较两个 completion

<!-- columns: 50/50 -->

```conversation
messages:
  - role: user
    content: "Explain the moon landing to a 6-year-old."
  - role: assistant
    model: "Completion A"
    content: "The Apollo program culminated in a successful lunar landing in 1969. Astronauts used a spacecraft to descend to the moon's surface and collect samples before returning to Earth."
```

|||

```conversation
messages:
  - role: user
    content: "Explain the moon landing to a 6-year-old."
  - role: assistant
    model: "Completion B"
    content: "People built a special rocket to go to the moon. Two astronauts landed there, walked around, and came home safely to tell everyone what they saw."
```

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## 左：人类反馈；右：手工设计的奖励函数

![](assets/christiano-backflip-human.webp)

|||

![](assets/christiano-backflip-reward.webp)

---

<!-- cite-right: sutton2018reinforcement -->
<!-- columns: 65/35 -->

## 经典 RL

强化学习问题通常写作**马尔可夫决策过程（Markov Decision Process, MDP）**：
- 状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}$
- 转移动力学 $P(s_{t+1}\mid s_t, a_t)$
- 奖励函数 $r(s_t, a_t)$ 与折扣因子 $\gamma$
- 优化整条轨迹上的累积回报

$$\text{MDP } (\mathcal{S}, \mathcal{A}, P, r, \gamma)$$

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

|||

![](assets/rl.png)

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## 经典 RL 与 RLHF 对比

<div class="text-sm">

**经典 RL**
- 智能体在具有状态 $s_t$ 的环境中采取动作 $a_t$ 
- 奖励是环境在每一步给出的已知函数 $r(s_t, a_t)$
- 优化整条轨迹上的累积回报（总步数为 $T$）

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

<div class="colloquium-spacer-md"></div>

**RLHF**
- 没有环境——prompt 从数据集中采样
- 奖励是从人类偏好中**学习**得到的（一个代理）
- **回复级别**的奖励（bandit 式，而非逐 token）
- 通过 **KL 惩罚**进行正则化，使策略保持接近基础模型

$$J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

</div>

|||

![](assets/rlhf.png)

---

<!-- columns: 50/50 -->

## 基于*可验证*奖励的强化学习

当答案可以被直接检查时，把同样的 RL 算法应用到 LLM 上，无需训练 reward model：
- 例如数学：检查最终答案。  
  代码：运行测试。
- 没有学习得到的 reward model——**没有代理目标**
- 使 RL 计算量能够在推理任务上扩展
- 解锁了**推理时扩展（inference time scaling）**：每个问题在生成阶段投入更多计算，性能随计算量呈对数线性提升
- RLVR 由 **Tülu 3** [@lambert2024t] 命名，并由 **DeepSeek R1** [@guo2025deepseek] 推广

|||

![](assets/rlvr-system.png)

---

## 对比经典 RL、LLM RLHF 与 RLVR

| | 经典 RL | RLHF | RLVR |
|---|---|---|---|
| **奖励** | 环境 | 学习得到（代理） | 可验证（精确） |
| **状态转移** | 有 | 无 | 无 |
| **奖励粒度** | 逐步 | 逐回复 | 逐回复 |
| **主要挑战** | 探索与利用的权衡 | 过度优化 | 任务泛化 |
| **示例** | CartPole | 聊天风格调优 | 数学推理 |

---


<!-- rows: 60/40 -->
## 现代 RLHF 的发展历程

![](assets/rlhf_timeline_tikz.png)

===

<!-- row-columns: 50/50 -->

- **Ziegler 2019** [@ziegler2019fine] —— 首个应用于语言模型的 RLHF
- **InstructGPT** [@ouyang2022training] —— ChatGPT 背后的经典 RLHF 流程
- **Constitutional AI** [@bai2022constitutional] —— 为 Claude 引入了早期的 AI 反馈方法

|||

- **DPO** [@rafailov2024direct] —— 无需 reward model 的直接偏好优化（DPO）
- **Llama 3** [@dubey2024llama] 与 **Tülu 3** [@lambert2024t] —— 现代多阶段训练流程
- **DeepSeek R1** [@guo2025deepseek] —— 推广了 RLVR

---

<!-- valign: center -->
<!-- cite-right: ouyang2022training -->

## InstructGPT 的三步 RLHF 流程


![InstructGPT 中的三步 RLHF 流程图，它在数年间成为 RLHF 的“标准”做法。](assets/instructgpt.jpg)


---

<!-- columns: 45/55 -->
<!-- cite-right: ouyang2022training -->
## 第 1/3 步：指令微调（IFT）

后训练的基础。也称为 **监督微调（SFT）**：
- 从预训练语言模型出发
- 收集*期望的*助手行为示范
- 在 prompt-response 对上用标准监督学习进行训练。  
  （batch size、学习率等设置有所不同）
- 模型现在可以回答问题了。  
  借助 IFT，可以很容易地把基座模型快速适配到许多领域。

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=
- \sum_{(x, y^\star)} \sum_{t=1}^{|y^\star|}
\log \pi_\theta \!\left(y^\star_t \mid x, y^\star_{<t}\right)
$$

|||

```conversation
size: 0.9
messages:
  - role: system
    content: "You are a helpful, harmless assistant. A system message like this can be used to steer the model to specific personas or behaviors."
  - role: user
    content: "Write me a short poem about an optimistic goldfish."
  - role: assistant
    content: "Bright little goldfish\nFinds a sunrise in each wave\nSmall bowl, endless hope"
```

---

<!-- columns: 45/55 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## 第 2/3 步：Reward modeling

概览：
- 针对同一个 prompt，收集两个模型输出之间的**比较**
- RLHF 的名字来自对 completion 之间的*人类*反馈的收集，但如今其中很大一部分是 AI 反馈
- 训练一个 reward model $r_\phi(x, y)$，使其给被偏好的 completion 打更高的分

|||


概率模型认为：当一个回答获得更高的 reward 分数时，它应当胜出：

$$
P(y_w \succ y_l \mid x)
=
\sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

训练则是最小化“被偏好回答胜过被拒绝回答”的负对数似然：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
- \log \sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

符号说明：
- $x$ 是 prompt
- $y_w$ 是**获胜**的回答
- $y_l$ 是**落败**的回答
- $r_\phi(x, y)$ 是训练得到的 reward model

---

<!-- columns: 45/55 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## 第 2/3 步：Reward modeling

```box
title: 核心思想
tone: accent
content: |
  RLHF 中使用的 reward，是模型预测一段给定文本在一个 pair/batch 中成为“获胜”或“被选中”的 completion 的概率。巧妙！
```

|||


概率模型认为：当一个回答获得更高的 reward 分数时，它应当胜出：

$$
P(y_w \succ y_l \mid x)
=
\sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

训练则是最小化“被偏好回答胜过被拒绝回答”的负对数似然：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
- \log \sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

符号说明：
- $x$ 是 prompt
- $y_w$ 是**获胜**的回答
- $y_l$ 是**落败**的回答
- $r_\phi(x, y)$ 是训练得到的 reward model

---

<!-- columns: 50/50 -->
<!-- cite-right: ouyang2022training -->
<!-- footnotes: right -->
## 第 3/3 步：基于 reward model 的 RL

一切汇聚于此（RLHF 也因此得名）：
- 从数据集 $\mathcal{D}$ 中采样一批 prompt $x_i$
- 用正在训练的模型生成 completion $y_i \sim \pi_\theta(\cdot \mid x_i)$
- 用 reward model $r_\phi(x_i, y_i)$ 为其打分
- 加入 **KL penalty**，使 policy 与 SFT/参考模型保持接近。^[KL divergence 衡量当前 policy 与参考模型的差异程度。对于离散输出，$D_{\mathrm{KL}}(\pi \,\|\, \pi_{\mathrm{ref}})=\mathbb{E}_{y \sim \pi}\!\left[\log \pi(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right]$。人们常口头称之为模型之间的“KL 距离”，尽管它并不是严格意义上的度量。]
- 用 policy-gradient RL 算法更新 policy（InstructGPT 和 ChatGPT 中使用的是 Proximal Policy Optimization，即 PPO）

$$
J(\pi)
=
\mathbb{E}\!\left[r_\phi(x, y)\right]
- \beta D_{\mathrm{KL}}\!\left(\pi \,\|\, \pi_{\mathrm{ref}}\right)
$$

|||

![](assets/rlhf-overview.png)

---

<!-- rows: 35/65 -->
<!-- title: center -->

## RLHF 目标函数详解

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
\underbrace{r_\phi(x, y)}_{\text{maximize the reward}}
- \underbrace{\beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)}_{\text{but don't change the model too much}}
$$


===


<!-- row-columns: 50/50 -->

<div style="text-align: left;">

<!-- Prompt $x$ 来自数据集，而不是环境。

policy $\pi_\theta(y \mid x)$ 生成完整的回复 $y$，reward model $r_\phi(x, y)$ 则评估人类是否会喜欢该回复。 -->

</div>

|||

<div style="text-align: left;">

参考模型 $\pi_{\mathrm{ref}}$ 使 policy 锚定在 SFT 模型附近。

$D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)$ 衡量在 prompt $x$ 上新 policy 偏离该参考模型的程度。

$\beta$ 控制**改进行为**与**保持接近**模型已有知识之间的权衡。

</div>

---

<!-- rows: 27/73 -->
<!-- cite-right: rafailov2024direct -->
## 能否更直接地优化这个目标？

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
r_\phi(x, y)
- \beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

===

<!-- row-columns: 60/40 -->

**Direct Preference Optimization (DPO)**

- 推导出了朝向上式最优解 $\pi^*$ 的梯度 
- 无需单独的 reward model（通过训练一个隐式 reward model 实现）
- 直接在针对某个 prompt（$x$）的偏好回复（$y_w$）与被拒绝回复（$y_l$）上训练

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
- \log \sigma \!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
$$

|||

---

<!-- rows: 27/73 -->
<!-- cite-right: rafailov2024direct -->
## 能否更直接地优化这个目标？

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
r_\phi(x, y)
- \beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

===

<!-- row-columns: 60/40 -->

**Direct Preference Optimization (DPO)**

- 推导出了上述方程最优解 $\pi^*$ 的梯度 
- 不再需要单独的 reward model（通过训练一个隐式的 reward model）
- 直接在针对同一 prompt ($x$) 的偏好回复 ($y_w$) 与被拒绝回复 ($y_l$) 上训练

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
- \log \sigma \!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
$$

|||

```box
title: DPO 之所以大受欢迎，是因为它
tone: accent
content: |
  - 实现起来简单得多
  - 运行成本低得多
  - 能达到最终性能的约 80% 甚至更多
  - 我用它构建了 Zephyr-Beta、Tülu 2/3、Olmo 2/3 等模型
```

---

<!-- valign: center -->
## 训练配方的演变

| | InstructGPT (2022) | Tülu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **指令数据** | ~10K | ~1M | 100K+ |
| **偏好数据** | ~100K | ~1M | On-policy |
| **RL 阶段** | ~100K 个 prompt | ~10K (RLVR) | N/A |

总体趋势是各个阶段都使用多得多的算力，同时重心越来越向 RLVR 转移。

---

<!-- rows: 60/40 -->
## 早期：InstructGPT

![](assets/rlhf-basic.png)

===

早期的 RLHF 有一套文档齐全、足够简单的方法。
- **InstructGPT** 让经典的三阶段配方成为标准：
  SFT、reward modeling，然后针对 reward model 做 RL。*OpenAI 甚至暗示最初的 ChatGPT 就是这样训练的！*
- 这成为现代 post-training 大部分工作的思想模板。

---

<!-- rows: 50/50 -->
## 从 RLHF 到 post-training

![](assets/rlhf-complex.png)

===

最初的“RLHF”配方，演变成了一系列复杂的步骤，用来得到最终最好的模型（例如 Nemotron 4 340B、Llama 3.1）。
- 现代系统保留了同一个核心思想，即组合使用多个优化器，发挥各自的优缺点，但增加了更多阶段、更多数据和更多过滤。
- 这一趋势仍在延续，配方也此消彼长，因为 RLVR 和模型合并等工具，以不同方式改变着可行的范围。

---

## 从 RLHF 到“post-training”

自 ChatGPT 问世以来，这个领域大致经历了几个不同的阶段：
1. 2023：用简单的 SFT 打造更好的聊天机器人，并复现 RLHF 的基本原理（Alpaca、Vicuna 等）
2. 2024：DPO 主导开源模型，训练阶段不断增多（Zephyr-beta、Tülu 2 等）
3. 2025：RLVR 与复杂的训练配方（Tülu 3、Olmo 3、Nemotron 3、R1 等）
4. 2026：Agentic 训练、multi-turn RL 等

---

## 从 RLHF 到“post-training”

自 ChatGPT 问世以来，这个领域大致经历了几个不同的阶段：
1. 2023：用简单的 SFT 打造更好的聊天机器人，并复现 RLHF 的基本原理（Alpaca、Vicuna 等）
2. **2024：DPO 主导开源模型，训练阶段不断增多**（Zephyr-beta、Tülu 2 等）
3. 2025：RLVR 与复杂的训练配方（Tülu 3、Olmo 3、Nemotron 3、R1 等）
4. 2026：Agentic 训练、multi-turn RL 等

2024 年间，随着训练阶段超越了 InstructGPT 式的配方、DPO 广泛普及，整个领域的重心转向了 post-training，RLHF 在很大程度上被视为众多工具之一（甚至可能根本用不上）。

---

<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima -->

RLHF 给人的印象是，它对最终语言模型的贡献很小。

> “模型的知识和能力几乎完全在预训练中习得，而 alignment 只是教它在与用户交互时应使用哪种格式的子分布。”

*LIMA: Less Is More for Alignment* (2023)


|||

---


<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima,muennighoff2024olmoe,ai2_olmoe_ios_2025 -->

RLHF 的名声是：它对最终语言模型的贡献很小。

> "模型的知识和能力几乎完全是在预训练期间学到的，而对齐只是教它在与用户交互时应使用哪种格式的子分布。"

*LIMA: Less Is More for Alignment*（2023）

|||

这种认为对齐（或 RLHF）只是在教"格式"的看法，有时会让人觉得 post-training 对模型只做了很小的改动，甚至把 finetuning 描述为"*只是风格迁移*"。

在数万亿 token 的网络文本上训练出的 base model，已经见过并学习了极其广泛的样例。
这一阶段的模型所蕴含的潜在能力，远超早期 post-training 配方所能展现的程度。

问题在于：post-training 如何与这些能力相互作用？

---

<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima,muennighoff2024olmoe,ai2_olmoe_ios_2025 -->

RLHF 的名声是：它对最终语言模型的贡献很小。

一个例子，**OLMoE**——同一 base model 系列，仅更新 post-training：
- [`OLMoE-1B-7B-0924-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)（2024 年 9 月）：平均评测得分 **38.44**
- [`OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)（2025 年 1 月）：平均评测得分 **45.62**

Base model 决定了*上限*。Post-training 的任务一直是**触达这个上限**。

---

<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima,vergarabrowne2026operationalising -->

RLHF 的名声是：它对最终语言模型的贡献很小。

> "模型的知识和能力几乎完全是在预训练期间学到的，而对齐只是教它在与用户交互时应使用哪种格式的子分布。"

*LIMA: Less Is More for Alignment*（2023）

> "浅层对齐假说（SAH）认为，大语言模型的大部分知识是在预训练期间学到的，而 post-training 只是把这些知识呈现出来。"

*Operationalising the Superficial Alignment Hypothesis via Task Complexity*（2026）

|||

3 年后的第二篇论文，与我对 post-training 的直觉相吻合。 

```box
title: 我把这称为 post-training 的**激发理论（Elicitation Theory）**，即我们试图把模型中最有用的知识提取出来。
tone: accent
```

---


<!-- layout: section-break -->

## 超越激发：post-training 的 RL 规模化时代

---

## OpenAI 借 o1-preview 展示的里程碑式 scaling 曲线

<!-- img-align: center -->
<!-- cite-right: openai2024o1 -->

![](assets/o1.webp)

---

## o1：测试时 scaling

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

推理算力（生成的 token 数）与下游性能之间呈对数线性关系。

- 这是模型的一种基本性质，借助 RLVR 才以如今广为人知的形式被释放出来
- 实现方式多种多样：一条长 chain of thought（CoT）序列、多个并行 agent，或两者混合
- 改进 inference-time scaling 会改变曲线的斜率和截距

|||

![](assets/o1-test-time.png)

---

## o1：训练时 scaling（借助 reinforcement learning！）

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

o1 发布中常被低估的一部分（也包括后续的 reasoning/agentic 模型）。
- 扩大 reinforcement learning 的算力，同样会带来对数线性的性能回报！
- 核心问题：扩大 RL *训练*规模，只是从 base model 中激发出更多能力，还是真的在教会新能力？

由此形成了训练语言模型的双向 scaling 格局——预训练与 post-training。 
scaling 的第三个位置在 inference 阶段（那里没有权重更新）。

|||

![](assets/o1-train-time.png)


---

## Cursor Composer 1.5：RL scaling

<!-- img-align: center -->
<!-- cite-right: cursor2026composer15 -->

![](assets/cursor-composer-1_5-scaling.png)

---

## DeepSeek-R1-Zero：RL 规模化

<!-- img-align: center -->
<!-- cite-right: guo2025deepseek -->

![](assets/deepseek-r1-zero-figure1-training.png)

---

<!-- columns: 30/70 -->
## Olmo 3.1：延长 RL 训练

<!-- cite-right: teamolmo2025olmo3 -->

迄今为数不多的“完全开放”的大规模 RL 训练之一。
- 训练一个通用的 32B reasoning 模型。
- 完整的 RL 训练在 224 块 GPU 上耗时约 **28 天**。
- 整个训练过程中性能提升非常稳定，事实上我们不得不停下来的时候，它还在继续上升！

|||

![](assets/olmo31-extended-rl.jpeg)

---

## 现状小结

post-training 和 RLHF 正以前所未有的速度变化。
- 语言模型正在变得“原生支持工具调用”，现在的重点是工具、harness（即你如何告诉模型使用这些工具），而不仅仅是权重
- RLHF 和人类偏好并没有消失，但演进得慢得多，也已不在业界关注的中心
- 随着 coding agent 的出现，构建语言模型和做研究的方式正在快速变化

---

<!-- rows: 50/50 -->
## 本次分享约为一门完整课程的第 1 讲

<!-- row-columns: 34/33/33 -->

```box
title: 导论（本次分享）
tone: accent
compact: true
content: |
  1. 引言
  2. 关键相关工作
  3. 训练概览
```

|||

```box
title: 核心训练流程
tone: muted
compact: true
content: |
  4. Instruction Tuning
  5. Reward Models
  6. 强化学习
  7. 推理
  8. 直接对齐
  9. Rejection Sampling
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

<!-- row-columns: 34/33/33 -->

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

完整的课程幻灯片即将发布在 [rlhfbook.com/course](https://rlhfbook.com/course) 和 YouTube @natolambert！

---

<!-- rows: 85/15 -->
## 谢谢大家

很抱歉未能亲自到场！

联系方式：nathan@natolambert.com

Newsletter：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===


```builtwith
repo: natolambert/colloquium
```
