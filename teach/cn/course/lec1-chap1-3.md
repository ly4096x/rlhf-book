---
title: "第一讲：概览"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第一讲"
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

# 第一讲：概览

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 1-3 章</p>

---

<!-- columns: 45/55 -->
## 什么是语言模型？

核心特性：
- 语言模型为文本分配概率。
- 词块被拆分为 **token**，这是模型内部的表示形式。
- 给定之前的 token，模型预测下一个 token。不断重复即可逐步生成一段 completion（这称为**自回归**，autoregressive）。

|||

![Transformer 的原始模型架构图。2017 年。](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 什么是（现代）语言模型？

现代语言模型：
- 拥有数十亿到数万亿参数。
- 主要源自 Transformer 架构，该架构推广了 **self-attention** 机制以及全稠密层。
- 预测和处理的远不止文本：Gemini 和 ChatGPT 可处理图像、音频和视频。

|||

![Transformer 的原始模型架构图。2017 年。](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2017 年：Transformer 诞生

- **2017 年**：Transformer 诞生

|||

![Transformer 的原始模型架构图。2017 年。](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2018 年：GPT-1、ELMo 和 BERT

- 2017 年：Transformer 诞生
- **2018 年**：GPT-1、ELMo 和 BERT 发布

|||

![GPT-1 的语言模型架构。2018 年。](assets/gpt-1.png)
<!-- cite-left: radford2018gpt, peters2018elmo, devlin2018bert -->

---

<!-- rows: 30/70 -->
## 2019 年：GPT-2 与 scaling laws

- 2017 年：Transformer 诞生
- 2018 年：GPT-1、ELMo 和 BERT 发布
- **2019 年**：GPT-2 与 scaling laws

===

![著名的 scaling laws 曲线图。2020 年。](assets/scaling-laws.png)
<!-- cite-right: radford2019gpt2, kaplan2020scaling -->

---

<!-- columns: 45/55 -->
## 2020 年：GPT-3 的惊人能力

- 2017 年：Transformer 诞生
- 2018 年：GPT-1、ELMo 和 BERT 发布
- 2019 年：GPT-2 与 scaling laws
- **2020 年**：GPT-3 的惊人能力

|||

![GPT-3 因拓展了 in-context learning 和 few-shot prompting 的理念而闻名。截图来自论文。](assets/few-shot.png)
<!-- cite-left: brown2020gpt3 -->

---

<!-- columns: 45/55 -->
## 2021：Stochastic Parrots

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 令人惊讶的能力
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
- 2020：GPT-3 令人惊讶的能力
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
- 2020：GPT-3 令人惊讶的能力
- 2021：Stochastic Parrots
- 2022：ChatGPT
- **2023：** GPT-4 与前沿规模

|||

![一张图片显示，NVIDIA CEO 黄仁勋据称泄露了 GPT-4 是一个约 2T 参数的 MoE 模型。](assets/jensen-gpt4.jpeg)
<!-- cite-right: openai2023gpt4 -->

---

<!-- columns: 45/55 -->
## 2024：o1 与推理模型

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 令人惊讶的能力
- 2021：Stochastic Parrots
- 2022：ChatGPT
- 2023：GPT-4 与前沿规模
- **2024：** o1 与推理模型

|||

![OpenAI o1 发布时著名的 test-time scaling 曲线图。](assets/o1-test-time.png)
<!-- cite-right: openai2024o1 -->

---

<!-- columns: 45/55 -->
## 2025：o3、Claude Code 与智能体

- 2017：Transformer 诞生
- 2018：GPT-1、ELMo 和 BERT 发布
- 2019：GPT-2 与 scaling laws
- 2020：GPT-3 令人惊讶的能力
- 2021：Stochastic Parrots
- 2022：ChatGPT
- 2023：GPT-4 与前沿规模
- 2024：o1 与推理模型
- **2025：** o3、Claude Code 与智能体

|||

![](assets/claude-code.png)
<!-- cite-right: openai2025o3, anthropic2025claudecode, openai2025agents -->

---

<!-- columns: 50/50 -->
## 预训练：Next-token prediction

- 在来自网页、书籍、代码和文档的数万亿 token 文本上训练
  - 模型通常在 5-50+ 万亿 token 上训练
  - 1T 文本 token 约对应 3-5 TB 数据
  - 各实验室收集并过滤的数据量是模型实际使用量的 10-20 倍
  - 模型所面向的完整数据漏斗规模达到 PB 量级
- 目标：预测每个序列中的下一个 token
- 结果：强大、灵活、有用的模型

|||

![](assets/pretraining_next_token_tikz.png)

---

## base model 补全文本

预训练之后，我们得到的只是一个“高级自动补全”模型，例如：^[通过 midtraining 和更好的数据配比，base model 也正变得越来越灵活。]

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

## Post-training 让模型像聊天机器人一样回答

最早的现代 post-training（或经 RLHF 调优）模型，把续写格式转变为始终遵循“回答问题的风格”。
早期对话模型的样子如下：

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

## 对训练阶段的直觉理解

不同的训练阶段如何改变模型：

- **Pretraining**：构建模型的世界知识、语言流畅度和广泛能力
- **指令微调 / SFT**：教模型以问答格式作答，并且常常教它复述特定的 token 序列
- **偏好微调 / RLHF**：使用对比损失（contrastive loss）整体修改 completion，让模型更丰富、更灵活
- **RLVR**：提升模型在可验证问题上的能力，这可以转化为更复杂的 agentic 行为

---

## 对训练阶段的直觉理解

不同的训练阶段如何改变模型：

- **Pretraining**：构建模型的世界知识、语言流畅度和广泛能力
- **指令微调 / SFT**：教模型以问答格式作答，并且常常教它复述特定的 token 序列
- **偏好微调 / RLHF**：使用对比损失（contrastive loss）整体修改 completion，让模型更丰富、更灵活
- **RLVR**：提升模型在可验证问题上的能力，这可以转化为更复杂的 agentic 行为

这对 post-training 意味着：
- 基于 RL 的 post-training 为模型提供更丰富的、response 级别的对比反馈，使其更灵活、更有吸引力
- 要达到最佳性能，SFT 仍然是必要的基础

---

<!-- columns: 40/60 -->
## ChatGPT 的出现让 RLHF 使模型变得更易用

模型的回答迅速演变为具有：
- 更好的**格式**：直接的对话式回答
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

## 那么，究竟什么是基于人类反馈的强化学习（RLHF）？

---

<!-- cite-right: sutton2018reinforcement -->
<!-- columns: 65/35 -->

## 经典强化学习（RL）

强化学习问题通常写成**马尔可夫决策过程（MDP）**：
- 状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}$
- 转移动态 $P(s_{t+1}\mid s_t, a_t)$
- reward 函数 $r(s_t, a_t)$ 与折扣因子 $\gamma$
- 优化一条轨迹上的累积回报

$$\begin{gathered}
\text{MDP } (\mathcal{S}, \mathcal{A}, P, r, \gamma) \\[6pt]
J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]
\end{gathered}$$

|||

![](assets/rl.png)

---

<!-- columns: 65/35 -->
<!-- cite-right: sutton2018reinforcement -->
## 通俗理解 RL

强化学习基础：
- 强化学习是**试错学习**  
  在长期 reward 中权衡探索与利用
- **状态（State）**：智能体当前所处的情境
- **动作（Action）**：智能体接下来做什么
- **Reward**：衡量该动作好坏的信号
- **Policy**：选择动作的策略
- **Trajectory**：由状态、动作和 reward 组成的序列：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
- **Trajectory 概率**：  
  $P(\tau\mid\pi) = p(s_0)\prod_{t=0}^T \pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)$

|||

![](assets/rl.png)

---

<!-- rows: 60/40 -->
<!-- cite-right: sutton2018reinforcement -->

<!-- row-columns: 50/50 -->

## 一个简单的 RL 示例：恒温器

智能体在许多 episode 中学习何时开启或关闭加热器
- **State**：当前室温
- **Action**：开启或关闭加热器
- **Reward**：室温保持在目标温度附近时为正
- **Policy**：决定下一步做什么的规则

||| 

示例 policy：

$$
\pi(a_t = \text{on} \mid s_t) =
\begin{cases}
1 & \text{if } s_t < 70^\circ\text{F} \\
0 & \text{otherwise}
\end{cases}
$$

===

![恒温器 RL 示例中的轨迹概率示例。](assets/thermostat_equation.png)

---

<!-- columns: 45/55 -->
<!-- cite-right: sutton2018reinforcement -->
## CartPole：一个标准的 RL 任务

- **State**：小车位置、速度、杆的角度、角速度
$$s_t = (x_t,\; \dot{x}_t,\; \theta_t,\; \dot{\theta}_t)$$

- **Action**：向左或向右推动小车
$$a_t \in \{\text{left},\; \text{right}\}$$

- **Reward**：杆每保持直立一步，奖励 +1
$$r(s_t, a_t) = \begin{cases} 1 & \text{if } |\theta_t| < 12° \text{ and } |x_t| < 2.4 \\ 0 & \text{otherwise (episode ends)} \end{cases}$$

|||

![](assets/cartpole.png)

---

<!-- columns: 45/55 -->
<!-- cite-right: sutton2018reinforcement -->
## CartPole：动力学

每个动作都会改变系统的物理状态。完整的状态更新如下：

$$\begin{aligned}
\ddot{x}_t &= \frac{F + m_p l (\dot{\theta}_t^2 \sin\theta_t - \ddot{\theta}_t \cos\theta_t)}{m_c + m_p} \\[6pt]
\ddot{\theta}_t &= \frac{g \sin\theta_t - \cos\theta_t \cdot \ddot{x}_t}{l}
\end{aligned}$$

其中 $m_c$ 是小车质量，$m_p$ 是杆的质量，$l$ 是杆的长度，$g$ 是重力加速度，$F$ 是施加的力。

这就是为什么经典 RL 是一个**多步控制问题**——每个动作都会改变下一个状态，奖励在整条 trajectory 上不断累积。

|||

![](assets/cartpole.png)

---

<!-- cite-right: christiano2017, ziegler2019fine, ouyang2022training -->
## 为什么会有 RLHF？

在经典 RL 中，reward function 是已知的——CartPole 每一步给 +1。但对许多任务来说，奖励**很难写出来**：

- 判断哪首诗更好很容易，但写出一部杰作很难
- 一眼就能看出哪个回答有帮助，但要把"有帮助"写成公式却很难
- 预训练优化的是**下一个 token 的预测**——最可能的续写并不总是最有用的

RLHF 让我们能够针对可以**评估**的行为进行优化，即使我们很难**明确指定**奖励。

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## 哪个后空翻更好？

![](assets/christiano-backflip-human.webp)

|||

![](assets/christiano-backflip-reward.webp)

---

<!-- columns: 40/60 -->
<!-- cite-right: christiano2017 -->
## 语言模型之前的 RLHF

- **TAMER**（Knox & Stone，2008）——人类为智能体的动作打分，从而学习奖励
- **Christiano et al. 2017**——在 Atari trajectory 偏好上做 RLHF
- **Ziegler et al. 2019**——首次在语言模型上做 RLHF

|||

![对 Christiano et al. 2017 系统示意图的复现。](assets/rlhf_schematic_tikz.png)

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## 左：人类反馈；右：手工设计的奖励函数

![](assets/christiano-backflip-human.webp)

|||

![](assets/christiano-backflip-reward.webp)

---

<!-- rows: 60/40 -->
## 现代 RLHF 之路

![](assets/rlhf_timeline_tikz.png)

===

<!-- row-columns: 50/50 -->

- **Ziegler 2019** [@ziegler2019fine] — 首个应用于语言模型的 RLHF
- **InstructGPT** [@ouyang2022training] — ChatGPT 背后的经典 RLHF 配方
- **Constitutional AI** [@bai2022constitutional] — 在 Claude 中引入了早期的 AI 反馈方法

|||

- **DPO** [@rafailov2024direct] — 无需 reward model 的直接偏好优化（DPO）
- **Llama 3** [@dubey2024llama] 与 **Tülu 3** [@lambert2024t] — 现代多阶段配方
- **DeepSeek R1** [@guo2025deepseek] — 推广了 RLVR

---

## 其他早期的代表性语言模型 RLHF 工作

- **Stiennon et al. 2020** [@stiennon2020learning] — 从人类反馈中学习摘要。将 Ziegler 等人的工作用 PPO 扩展到长文本摘要
- **WebGPT** [@nakano2021webgpt] — 用 RLHF 训练模型浏览网页并回答问题，同时给出引用
- **Anthropic HH** [@bai2022training] — 用 RLHF 训练有帮助且无害的助手。提出了在开放研究中被广泛使用的 “HH” 数据集
- **GopherCite** [@menick2022teaching] — 教语言模型用经过验证的引文来支持回答（DeepMind）
- **Sparrow** [@glaese2022improving] — 通过有针对性的人类判断改进对话智能体的对齐（DeepMind）。在 RLHF 中加入了基于规则的约束
- **InstructGPT** [@ouyang2022training] — 驱动 ChatGPT 的经典三步 RLHF 配方

---

<!-- rows: 30/70 -->
## RLHF 对比数据长什么样

人类（或 AI）标注者针对同一个 prompt 看到两个回答并选出更好的一个——这个偏好对会成为 reward model 的训练数据。

===

<!-- row-columns: 50/50 -->

```conversation
size: 0.85
messages:
  - role: user
    content: "Explain why the sky is blue in one sentence."
  - role: assistant
    model: "Response A ✓"
    content: "The sky is blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight are scattered more by atmospheric molecules than longer wavelengths."
```

|||

```conversation
size: 0.85
messages:
  - role: user
    content: "Explain why the sky is blue in one sentence."
  - role: assistant
    model: "Response B"
    content: "The sky appears blue because of the way light interacts with the atmosphere and stuff, it's basically just physics."
```

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## 经典 RL 与 RLHF

<div class="text-sm">

**经典 RL**
- 智能体在具有状态 $s_t$ 的环境中执行动作 $a_t$
- 奖励是环境给出的、每一步已知的函数 $r(s_t, a_t)$
- 优化整条轨迹上的累积回报（总步数 $T$）

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

<div class="colloquium-spacer-md"></div>

**RLHF**
- 没有环境——prompt 从数据集中采样
- 奖励是从人类偏好中**学习**得到的（一种代理）
- **回答级别**的奖励（bandit 式，而非逐 token）
- 用 **KL 惩罚**做正则化，使模型不偏离基础模型太远

$$J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

</div>

|||

![](assets/rlhf.png)

---

<!-- cite-right: ouyang2022training -->

## InstructGPT 的三步 RLHF 配方


![InstructGPT 中的三步 RLHF 流程图，它在数年间成为 RLHF 的“标准”做法。](assets/instructgpt.jpg)


---

<!-- columns: 45/55 -->
<!-- cite-right: ouyang2022training -->
## 第 1/3 步：指令微调（IFT）

后训练的基础。也称为 **Supervised Fine-tuning（SFT）**：
- 从预训练 language model 出发
- 收集*期望*的助手行为示范
- 在 prompt-response 对上用标准监督学习训练。  
  （batch size、learning rate 等设置有所不同）
- 模型现在能够回答问题。  
  用 IFT 可以很容易地将 base model 快速适配到众多领域。

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
- 针对同一个 prompt，收集两个模型输出之间的**对比**
- RLHF 的名字来自收集人类对不同 completion 的反馈，但如今大部分反馈来自 AI
- 训练 reward model $r_\phi(x, y)$，使其对更受偏好的 completion 打出更高分

|||


概率模型认为：回答获得的 reward 分数越高，就越应当胜出：

$$
P(y_w \succ y_l \mid x)
=
\sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

训练则是最小化“受偏好回答胜过被拒绝回答”的负对数似然：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
- \log \sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

符号说明：
- $x$ 是 prompt
- $y_w$ 是**获胜**回答
- $y_l$ 是**落败**回答
- $r_\phi(x, y)$ 是训练得到的 reward model

---

<!-- columns: 45/55 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## 第 2/3 步：Reward modeling

```box
title: 核心思想
tone: accent
content: |
  RLHF 中使用的 reward，就是模型预测某段文本在一对/一批 completion 中成为“获胜”或“被选中”那个的概率。巧妙！
```

|||


概率模型认为：回答获得的 reward 分数越高，就越应当胜出：

$$
P(y_w \succ y_l \mid x)
=
\sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

训练则是最小化“受偏好回答胜过被拒绝回答”的负对数似然：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
- \log \sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

符号说明：
- $x$ 是 prompt
- $y_w$ 是**获胜**回答
- $y_l$ 是**落败**回答
- $r_\phi(x, y)$ 是训练得到的 reward model

---

<!-- columns: 50/50 -->
<!-- cite-right: ouyang2022training -->
<!-- footnotes: right -->
## 第 3/3 步：针对 reward model 的 RL

一切汇聚于此（RLHF 也由此得名）：
- 从数据集 $\mathcal{D}$ 中采样一批 prompt $x_i$
- 用正在训练的模型生成 completion $y_i \sim \pi_\theta(\cdot \mid x_i)$
- 用 reward model $r_\phi(x_i, y_i)$ 为它们打分
- 加入 **KL 惩罚**，使 policy 与 SFT／参考模型保持接近。^[KL divergence 衡量当前 policy 与参考模型之间的差异程度。对于离散输出，$D_{\mathrm{KL}}(\pi \,\|\, \pi_{\mathrm{ref}})=\mathbb{E}_{y \sim \pi}\!\left[\log \pi(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right]$。人们常口头称之为模型之间的“KL 距离”，尽管它并不是严格意义上的度量。]
- 用 policy gradient RL 算法更新 policy（InstructGPT 和 ChatGPT 中使用的是 Proximal Policy Optimization，即 PPO）

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

## 拆解 RLHF 目标函数

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
\underbrace{r_\phi(x, y)}_{\text{maximize the reward}}
- \underbrace{\beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)}_{\text{but don't change the model too much}}
$$


===


<!-- row-columns: 50/50 -->

<div style="text-align: left;">

<!-- Prompts $x$ come from a dataset, not an environment.

The policy $\pi_\theta(y \mid x)$ generates a full response $y$, and the reward model $r_\phi(x, y)$ scores whether humans would like that response. -->

</div>

|||

<div style="text-align: left;">

参考模型 $\pi_{\mathrm{ref}}$ 将 policy 锚定在 SFT 模型附近。

$D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)$ 衡量在 prompt $x$ 上，新 policy 相对该参考模型偏离了多远。

$\beta$ 控制**改进行为**与**贴近模型已有知识**之间的权衡。

</div>

---

<!-- rows: 27/73 -->
<!-- cite-right: rafailov2024direct -->
## 能否更直接地优化它？

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
r_\phi(x, y)
- \beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

===

<!-- row-columns: 60/40 -->

**Direct Preference Optimization (DPO)**

- 推导出了朝向上述方程最优解 $\pi^*$ 的梯度 
- 无需单独的 reward model（通过训练一个隐式 reward model 实现）
- 直接在针对某个 prompt（$x$）的偏好回答（$y_w$）与被拒绝回答（$y_l$）上训练

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
## 能否更直接地优化？

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
r_\phi(x, y)
- \beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

===

<!-- row-columns: 60/40 -->

**Direct Preference Optimization（DPO）**

- 推导出了上述方程最优解 $\pi^*$ 的梯度方向 
- 无需单独的 reward model（通过训练一个隐式的 reward model 实现）
- 直接在针对某个 prompt（$x$）的偏好回复（$y_w$）与被拒回复（$y_l$）上训练

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
title: DPO 为何如此流行
tone: accent
content: |
  - 实现起来简单得多
  - 运行成本低得多
  - 能达到最终性能的 ~80% 甚至更多
  - 我用它训练过 Zephyr-Beta、Tülu 2/3、Olmo 2/3 等模型
```

---

## Rejection sampling：最简单的偏好优化

生成大量 completion，打分，然后在最优者上做 fine-tuning：

1. **生成**：从当前模型为每个 prompt 采样 $N$ 个 completion
2. **打分**：将每个 completion 送入 reward model $r_\phi(x, y)$
3. **选择**：保留得分最高的 completion
4. **Fine-tuning**：在筛选后的数据集上使用标准 SFT loss

$$\mathcal{L}_{\mathrm{RS}} = \mathcal{L}_{\mathrm{SFT}}\!\left(\theta;\; \{(x_i, y_i^{\star})\}\right) \quad \text{where } y_i^{\star} = y_{i,\, \arg\max_{j} r_\phi(x_i, y_{i,j})}$$

简单、稳定且应用广泛：Llama 2 [@touvron2023llama] 和 DeepSeek R1 [@guo2025deepseek] 都包含 rejection sampling 阶段。

---

## 偏好调优全景

| | Rejection Sampling | 在线 RL（PPO） | DPO |
|---|---|---|---|
| **机制** | 先过滤，再 SFT | 生成、打分、更新 policy | 直接对偏好做梯度更新 |
| **Reward model** | 需要 | 需要 | 隐式（无单独 RM） |
| **On-policy 数据** | 是（从当前模型生成） | 是（每步都生成） | 否（固定的偏好数据集） |
| **复杂度** | 低 | 高 | 低 |

三者优化的是同一个底层目标——区别在于它们**如何**将 policy 推向更高 reward 的 completion。关于哪种方法最终性能最好，目前争论不少，通常 RL 胜出，但证据并不一致。

---

<!-- cite-right: gao2023scaling -->
## 注意：代理目标与过度优化

Reward model 是**代理（proxy）**，而非真实标准。即使训练良好的 RM，也只是与真实的用户满意度*相关*。

**Goodhart 定律**：“当一个指标变成目标时，它就不再是一个好指标。”

实践中的表现：
- **Reward hacking**：RM 分数不断攀升，但实际质量下降
- **冗长偏好（verbosity bias）**：越长的回复得分越高，导致模型变得啰嗦
- **Sycophancy**：模型说用户爱听的话，而不是准确的话
- **过度拒答（over-refusal）**：模型拒绝合理的请求（例如“how to kill a linux process”）

KL penalty $\beta$ 是主要的防线——它限制 policy 偏离 reference model 的程度。但过度优化是所有基于偏好的训练中的**根本性矛盾**。

---

## 训练方案的演变

| | InstructGPT（2022） | Tülu 3（2024） | DeepSeek R1（2025） |
|---|---|---|---|
| **指令数据** | ~10K | ~1M | 100K+ |
| **偏好数据** | ~100K | ~1M | On-policy |
| **RL 阶段** | ~100K 个 prompt | ~10K（RLVR） | 无 |

总体趋势是在所有阶段投入多得多的算力，同时重心更多转向 RLVR。

---

<!-- rows: 60/40 -->
## 早期：InstructGPT

![](assets/rlhf-basic.png)

===

早期的 RLHF 有一套记录详尽、足够简单的做法。
- **InstructGPT** 使经典的三阶段方案成为标准：
  SFT、reward modeling，然后针对 reward model 做 RL。*OpenAI 甚至暗示最初的 ChatGPT 就是这样训练的！*
- 这成为现代 post-training 大部分工作的思想模板。

---

<!-- rows: 50/50 -->
## 从 RLHF 到 post-training

![](assets/rlhf-complex.png)

===

最初的“RLHF”方案，演变成了一系列复杂的步骤，以得到最终最好的模型（例如 Nemotron 4 340B、Llama 3.1）。
- 现代系统保留了同样的核心思想——使用多个各有优劣的优化器——但增加了更多阶段、更多数据和更多过滤。
- 这一趋势仍在延续，各种方案此消彼长，因为 RLVR 和 model merging 等工具以不同方式改变着可行的范围。

---

## 从 RLHF 到“post-training”

自 ChatGPT 问世以来，这个领域大致经历了几个不同的阶段：
1. 2023：用简单的 SFT 打造更好的聊天机器人，并复现 RLHF 的基本原理（Alpaca、Vicuna 等）
2. 2024：DPO 主导开源模型，训练阶段不断扩展（Zephyr-beta、Tülu 2 等）
3. 2025：RLVR 与复杂的训练配方（Tülu 3、Olmo 3、Nemotron 3、R1 等）
4. 2026：Agentic 训练、multi-turn RL 等

---

## 从 RLHF 到“post-training”

自 ChatGPT 问世以来，这个领域大致经历了几个不同的阶段：
1. 2023：用简单的 SFT 打造更好的聊天机器人，并复现 RLHF 的基本原理（Alpaca、Vicuna 等）
2. **2024：DPO 主导开源模型，训练阶段不断扩展**（Zephyr-beta、Tülu 2 等）
3. 2025：RLVR 与复杂的训练配方（Tülu 3、Olmo 3、Nemotron 3、R1 等）
4. 2026：Agentic 训练、multi-turn RL 等

2024 年内，该领域的重心转向了 post-training：训练阶段的演进超出了 InstructGPT 式的配方，DPO 广泛普及，RLHF 在很大程度上被视为众多工具之一（甚至可能根本用不上）。

---

<!-- columns: 50/50 -->
## “只是风格迁移”
<!-- cite-right: zhou2023lima -->

人们一直认为，RLHF 对最终语言模型的贡献微乎其微。

> “模型的知识和能力几乎完全是在 pretraining 阶段学到的，而 alignment 只是教它在与用户交互时应使用哪一类格式的子分布。”

*LIMA: Less Is More for Alignment*（2023）


|||

---


<!-- columns: 50/50 -->
## “只是风格迁移”
<!-- cite-right: zhou2023lima,muennighoff2024olmoe,ai2_olmoe_ios_2025 -->

人们一直认为，RLHF 对最终语言模型的贡献微乎其微。

> “模型的知识和能力几乎完全是在 pretraining 阶段学到的，而 alignment 只是教它在与用户交互时应使用哪一类格式的子分布。”

*LIMA: Less Is More for Alignment*（2023）

|||

这种认为 alignment（或 RLHF）只是在教“格式”的观点，曾让人觉得 post-training 只会对模型做出微小的改动，并把 finetuning 说成“*只是风格迁移*”。

在数万亿 token 的网络文本上训练出的 base model，已经见过并学习了极其广泛的样本。
此阶段的模型所蕴含的潜在能力，远超早期 post-training 配方所能释放的程度。

问题在于：post-training 与这些能力之间是如何相互作用的？

---

<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima,muennighoff2024olmoe,ai2_olmoe_ios_2025 -->

人们一直认为，RLHF 对最终语言模型的贡献微乎其微。

以 **OLMoE** 为例——基础模型系列相同，仅更新了 post-training：
- [`OLMoE-1B-7B-0924-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)（2024 年 9 月）：平均评测得分 **38.44**
- [`OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)（2025 年 1 月）：平均评测得分 **45.62**

Base model 决定了*上限*，而 post-training 的任务是**触达这个上限**。 

简单的 post-training 往往远不足以榨出模型的全部性能（尤其是在进展飞快的时候）。

---

<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima,vergarabrowne2026operationalising -->

人们一直认为，RLHF 对最终语言模型的贡献微乎其微。

> “模型的知识和能力几乎完全是在 pretraining 阶段学到的，而 alignment 只是教它在与用户交互时应使用哪一类格式的子分布。”

*LIMA: Less Is More for Alignment*（2023）

> “表层对齐假说（SAH）认为，大语言模型的大部分知识都是在 pre-training 期间学到的，而 post-training 只是把这些知识呈现出来。”

*Operationalising the Superficial Alignment Hypothesis via Task Complexity*（2026）

|||

3 年后的第二篇论文，与我对 post-training 的直觉相吻合。 

```box
title: 我把这称为 post-training 的**激发理论（Elicitation Theory）**：我们要做的，是把模型中最有用的知识提取出来。
tone: accent
```

---


<!-- columns: 50/50 -->
## 对 post-training 的直觉
<!-- cite-right: zhou2023lima,vergarabrowne2026operationalising -->

人们一直认为，RLHF 对最终语言模型的贡献微乎其微。

> “模型的知识和能力几乎完全是在 pretraining 阶段学到的，而 alignment 只是教它在与用户交互时应使用哪一类格式的子分布。”

*LIMA: Less Is More for Alignment*（2023）

> “表层对齐假说（SAH）认为，大语言模型的大部分知识都是在 pre-training 期间学到的，而 post-training 只是把这些知识呈现出来。”

*Operationalising the Superficial Alignment Hypothesis via Task Complexity*（2026）

|||

3 年后的第二篇论文，与我对 post-training 的直觉相吻合。 

```box
title: 我把这称为 post-training 的**激发理论（Elicitation Theory）**：我们要做的，是把模型中最有用的知识提取出来。
tone: accent
```
<br>
Pretraining 为这辆车打造了底盘——post-training 则是从中榨取最大性能的精细手艺。

---

<!-- columns: 50/50 -->

## 基于*可验证*奖励的强化学习

当答案可以被直接检验时，对 LLM 应用同样的 RL 算法，无需训练 reward model：
- 例如数学：检查最终答案。
  代码：运行测试。
- 没有学习得到的 reward model —— **没有 proxy 目标**
- 使 RL 算力可以在推理任务上持续扩展
- 解锁了 **inference time scaling**：在生成阶段为每个问题投入更多算力，性能会随算力呈对数线性提升
- RLVR 由 **Tülu 3** [@lambert2024t] 命名，并由 **DeepSeek R1** [@guo2025deepseek] 推广

|||

![](assets/rlvr-system.png)

---

## 对比经典 RL、LLM RLHF 与 RLVR

| | 经典 RL | RLHF | RLVR |
|---|---|---|---|
| **奖励** | 环境 | 学习得到（proxy） | 可验证（精确） |
| **状态转移** | 有 | 无 | 无 |
| **奖励粒度** | 每一步 | 每个 response | 每个 response |
| **主要挑战** | 探索与利用的权衡 | 过度优化 | 任务泛化 |
| **示例** | CartPole | 聊天风格调优 | 数学推理 |

---

<!-- layout: section-break -->

## 超越 elicitation：post-training 的 RL 扩展时代

---

## OpenAI 借 o1-preview 给出的里程碑式 scaling 曲线

<!-- img-align: center -->
<!-- cite-right: openai2024o1 -->

![](assets/o1.webp)

---

## o1：Test-time scaling

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

推理算力（生成的 token 数量）与下游性能之间呈对数线性关系。

- 这是模型的一种基本属性，通过 RLVR 才以如今广为人知的形式被解锁
- 实现方式多种多样：一条很长的 chain of thought（CoT）序列、多个并行的 agent，或两者的混合
- 改进 inference-time scaling 会改变曲线的斜率和截距

|||

![](assets/o1-test-time.png)

---

## o1：Training-time scaling（借助强化学习！）

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

这是 o1 发布中常被低估的一部分（对未来的推理/agentic 模型同样如此）。
- 扩展强化学习算力，同样会带来对数线性的性能回报！
- 核心问题：扩展 RL *训练*，究竟只是从 base model 中激发出更多能力，还是真的教会了新能力？

由此形成了训练语言模型的双向 scaling 格局 -- 既有 pretraining，也有 post-training。 
第三个 scaling 维度在推理阶段（那里没有权重更新）。

|||

![](assets/o1-train-time.png)


---

## Cursor Composer 1.5：RL scaling

<!-- img-align: center -->
<!-- cite-right: cursor2026composer15 -->

![](assets/cursor-composer-1_5-scaling.png)

---

## DeepSeek-R1-Zero：RL scaling

<!-- img-align: center -->
<!-- cite-right: guo2025deepseek -->

![](assets/deepseek-r1-zero-figure1-training.png)

---

<!-- columns: 30/70 -->
## Olmo 3.1：延长 RL 训练

<!-- cite-right: teamolmo2025olmo3 -->

迄今为数不多的"完全开放"的大规模 RL 训练之一。
- 训练一个通用的 32B 推理模型。
- 完整的 RL 训练在 224 块 GPU 上耗时约 **28 天**。
- 整个训练过程中性能提升非常稳定，事实上在我们不得不停止时，它仍在上升！

|||

![](assets/olmo31-extended-rl.jpeg)

---

## 这将我们带向何处

Post-training 和 RLHF 的变化速度，可能比以往任何时候都快。
- 语言模型正在成为"原生支持工具调用"的系统，如今更重要的是工具、harness（你如何告诉模型使用这些工具），以及远不止权重本身的诸多要素
- RLHF 和人类偏好并没有消失，但演进得慢得多，也已不在业界关注的中心
- 随着 coding agent 的出现，构建语言模型和开展研究的方式正在快速变化

---

<!-- rows: 50/50 -->
## 第 1 讲：概览

<!-- row-columns: 34/33/33 -->

```box
title: 第 1 讲
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
  4. Instruction tuning
  5. Reward model
  6. 强化学习
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

<!-- row-columns: 34/33/33 -->

```box
title: 实践考量
tone: muted
compact: true
content: |
  13. Tool use
  14. 过度优化
  15. 正则化
  16. 评估
  17. 产品与角色特性
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
  - rlhfbook.com
  - 共享参考资料和后续讲座课件均发布于此
```

---

<!-- rows: 85/15 -->
## 谢谢大家

提问 / 讨论

联系方式：nathan@natolambert.com

Newsletter：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===


```builtwith
repo: natolambert/colloquium
```
