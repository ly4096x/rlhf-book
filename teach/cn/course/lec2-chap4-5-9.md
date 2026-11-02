---
title: "第 2 讲：IFT、reward model 与 rejection sampling"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 2 讲"
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

# 第 2 讲：IFT、reward model 与 rejection sampling

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">后训练课程。第 4、5、9 章 </p>

---

<!-- rows: 50/50 -->
## 第 2 讲：IFT、reward model 与 rejection sampling

<!-- row-columns: 32/36/32 -->

```box
title: 概览
tone: muted
compact: true
content: |
  1. 引言
  2. 相关工作
  3. 训练概述
```

|||

```box
title: 核心训练流程
tone: accent
compact: true
content: |
  4. **指令微调**
  5. **Reward model**
  6. 强化学习
  7. 推理
  8. 直接对齐
  9. **Rejection sampling**
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
  - [GitHub 仓库](https://github.com/natolambert/rlhf-book)
```

---

## 为什么把这三章放在一起？

这几章介绍的是入门 RLHF 与 post-training 的较简单工具。
核心讲座将聚焦 RL，但我们需要先打好基础。

---

## 为什么把这三章放在一起？

这几章介绍的是入门 RLHF 与 post-training 的较简单工具。
核心讲座将围绕 RL 展开，但我们需要先打好基础。

此外，这几章构成了一条自然的流程——**从预训练模型到偏好调优模型（含 reward model）的最简完整路径**。 
这是完整 RLHF 之前自然的一步：

1. **指令微调**教模型遵循指令（格式）
2. **Reward model** 从人类偏好中学习对质量打分（信号）
3. **拒绝采样**利用这些分数筛选出更好的训练数据（优化，之后将被 RL 取代）

---


<!-- layout: section-break -->

## 第一部分：指令微调

---

<!-- rows: 55/45 -->
## 指令微调处于什么位置？

指令微调（IFT），也称监督微调（SFT），是 post-training 的第一步。

- 以只会预测下一个 token 的预训练模型为起点
- 教它在对话格式中**遵循指令**
- 是所有后续偏好调优的基础（以及其他方法，例如面向数学/代码的以 RL 为主的方案）

===

![RLHF 流程——指令微调是第一步。](assets/rlhf-basic.png)

---

## 指令微调有什么作用？

指令微调非常强大！截至本文撰写时：
- 无数专用模型仅靠 IFT/SFT 就被创造出来，并产生了巨大影响
- 在你的领域内，只需 1-10K 条高质量样本就能见效
- 可以扩展到数百万条 prompt，通常仍能带来显著提升（例如 OpenThoughts3 [@guha2025openthoughts]）
- 它因为不够炫酷而常被忽视甚至轻视，但所有 post-training 都应该从看看 IFT 能走多远开始！

---

## 两条研究脉络的汇合

指令微调源于两条并行的研究脉络：

1. **统一的 text-to-text 框架**：T5 [@raffel2020exploring] 表明，把每个 NLP 任务都表述为“文本输入，文本输出”，效果出奇地好
2. **规模化 + 指令遵循**：FLAN [@wei2021finetuned]、T0 [@sanh2021multitask] 和 Natural Instructions [@mishra2021cross] 表明，在带有明确指令的多样化任务上训练，能提升 zero-shot 泛化能力

---

## 两条研究脉络的汇合

指令微调源于两条并行的研究脉络：

1. **统一的 text-to-text 框架**：T5 [@raffel2020exploring] 表明，把每个 NLP 任务都表述为“文本输入，文本输出”，效果出奇地好
2. **规模化 + 指令遵循**：FLAN [@wei2021finetuned]、T0 [@sanh2021multitask] 和 Natural Instructions [@mishra2021cross] 表明，在带有明确指令的多样化任务上训练，能提升 zero-shot 泛化能力

**SFT 在 ChatGPT 之后迅速兴起**：Alpaca [@alpaca]、OpenAssistant [@kopf2023openassistant]、Tulu [@wang2023camelsgo] 等众多工作，帮助把指令微调变成了一套可广泛复现的开放配方

---

## Chat template：指令微调的结构

模型需要一种结构化格式来管理**谁在说话**以及**要生成什么**。 
早期的 chat template 定义了三种角色：

- **System**：背景指令（人设、约束）
- **User**：人类的消息
- **Assistant**：模型的回复

```
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

模型会一直生成，直到产出文本结束 token（此处为 `<|im_end|>`）。

---

<!-- columns: 48/52 -->
## 对话如何被渲染为 template 文本

**Conversation 对象：**

大多数 post-training 数据的存储方式（与模型无关）：
```python
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {
        "role": "user",
        "content": "How many helicopters can a human eat in one sitting?",
    },
]
```

|||

**渲染后的 chat template 文本：**

模型眼中的数据长这样（取决于 tokenizer）：
```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

`tokenizer.apply_chat_template(messages)` 会在 tokenization 之前执行这一转换。

---

<!-- columns: 48/52 -->
## 同一示例的可视化

模型看到的内容：
```text
<|im_start|>system
You are a friendly chatbot who always
responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat
in one sitting?<|im_end|>
<|im_start|>assistant
```

|||

用户看到的内容：
```conversation
size: 0.9
messages:
  - role: system
    content: "You are a friendly chatbot who always responds in the style of a pirate"
  - role: user
    content: "How many helicopters can a human eat in one sitting?"
  - role: assistant
    content: "..."
```

---

## Chat template 的底层实现：Jinja 代码

Chat template 实现为存储在 tokenizer 配置中的 **Jinja 代码片段**。 
这是将 Python dict 列表转换为模型所见 token 序列的原始代码：

```jinja
{{ bos_token }}
{% for message in messages %}
    {{ '<|im_start|>' + message['role'] + '\n'
       + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```

完整的 template 还会强制 role 交替（`user`/`assistant`/`user`/...），并处理可选的 `system` 消息。 
在代码中通过 `tokenizer.apply_chat_template(messages)` 应用。
例如，参见 [`Olmo-3-7B-Instruct`](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/chat_template.jinja) 的模板。

---

## Jinja chat template 的痛点

哎呀。

---

## Jinja chat template 的痛点

使用 chat template 有很多困难之处。
- 人类不易直接读懂
- 细微的问题，或与 tokenizer 不匹配，都可能导致训练失败
- 随着 reasoning model 和 tool use 的出现，复杂度不断上升

---

## 不同模型的 chat template 各不相同

不同模型家族使用不同的 special token，但结构相同。

**Zephyr [@tunstall2023zephyr]：**

```text
<|system|>
You are a friendly chatbot...</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

**Tulu [@lambert2024t]：**

```text
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm
functioning as expected. How can I assist you today?<|endoftext|>
```

这些都通过存储在 tokenizer 配置中的 Jinja template 应用（`apply_chat_template`）。

---

## 多轮对话

通过交替 role，对话可以自然地延续：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

整段历史被打包成一个 token 序列——生成时，模型会把之前所有轮次都当作上下文。

---

## 另一种方案：OpenAI 的 Harmony 格式

OpenAI 在发布 [gpt-oss](https://openai.com/index/introducing-gpt-oss/) 的同时推出了 [**Harmony**](https://github.com/openai/harmony)，用基于 Rust 的渲染器取代 Jinja，并将输出拆分为多个 **channel**：

- `analysis` — 内部推理 / chain-of-thought（对用户隐藏）
- `commentary` — 工具调用在此输出
- `final` — 面向用户的回复

```text
<|start|>assistant<|channel|>analysis<|message|>I need to check the weather...<|end|>
<|start|>assistant<|channel|>commentary to=functions.get_weather
<|constrain|>json<|message|>{"location":"SF"}<|call|>
<|start|>assistant<|channel|>final<|message|>It's 65°F and sunny in SF.<|return|>
```

为什么？Jinja 无法干净地处理工具调用（`tojson` 转义、边界模糊）。Harmony 把这部分复杂性转移到专门的库（PyPI 上的 `openai-harmony`）中，而不是模板字符串里。

---

## Harmony Python 示例

```python
from openai_harmony import (Role, Message, Conversation,
    SystemContent, load_harmony_encoding, HarmonyEncodingName)

# Build messages
system = Message.from_role_and_content(Role.SYSTEM, SystemContent.new())
user = Message.from_role_and_content(Role.USER, "What is 2 + 2?")

# Assemble a conversation
convo = Conversation.from_messages([system, user])

# Render to tokens using the OSS encoding
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

# Decode back to text
print(enc.decode_utf8(tokens))
```

`render_conversation_for_completion` / `render_conversation_for_training` 这种拆分取代了 Jinja 模板中的 `add_generation_prompt`。

---

## 模型究竟学到了什么？

<!-- cite-right: ouyang2022training -->

**Prompt masking**：在 IFT 期间，模型只学习预测 **assistant 回复**，而不是用户消息。

- System 和 user token（通常）会在 loss 中被 mask 掉
- 只有 assistant completion token 对梯度更新有贡献
- 模型学的是*如何回答*，而不是*如何提问*

对于多轮对话，有两种策略：

1. **仅最后一轮**：mask 掉除最后一条 assistant 回复之外的所有内容
2. **所有 assistant 轮次**：只 mask user/system token，在每条 assistant 回复上训练

---

## 指令微调的最佳实践

<!-- cite-right: zhou2023lima, lambert2024t -->

**数据质量比数量更重要：**

- 高质量的 completion 至关重要——prompt 本来就被 mask 了，所以模型从回复中学习
- 约 1M 个 prompt 就足以得到优秀的、可用于 RLHF 的模型
  - 继续扩大规模仍有帮助，但收益递减
  - 随着 reasoning model 的出现，SFT prompt 的数量有所*减少*。这是一个开放的研究问题
- 最好的 prompt 与下游任务的分布一致

---

## 指令微调的最佳实践

<!-- cite-right: zhou2023lima, lambert2024t -->

**数据质量比数量更重要：**

- 高质量的 completion 至关重要——prompt 本来就被 mask 了，所以模型从回复中学习
- 约 1M 个 prompt 就足以得到优秀的、可用于 RLHF 的模型
  - 继续扩大规模仍有帮助，但收益递减
  - 随着 reasoning model 的出现，SFT prompt 的数量有所*减少*。这是一个开放的研究问题
- 最好的 prompt 与下游任务的分布一致

**训练细节与预训练不同：**

- **Batch size**：小得多（例如 256，而预训练为 1024–2048 条序列）
- **Learning rate**：低 1-2 个数量级（例如 $1 \times 10^{-5}$ vs. $3 \times 10^{-4}$），以防在更窄的数据分布上过拟合
- **Loss function**：同样是 cross-entropy，但只作用于未被 mask 的（assistant）token

---

## IFT 的数据规模

所需指令数据的规模已快速演变：

- **ChatGPT 之后的早期**：约 1 万条高质量人工生成的样本就可能达到 state-of-the-art [@ouyang2022training; @no_robots]
- **当前实践**：大规模合成数据集效果最好，但质量过滤必不可少

扩大 prompt 数量很快带来了更多性能提升。如今，reasoning model 在 SFT 中通过每个 prompt 使用更多 token（以及更长的上下文长度）来训练，消耗了更多算力和 token。

---

## IFT 的数据规模

所需指令数据的规模已快速演变：

- **ChatGPT 之后的早期**：约 1 万条高质量人工生成的样本就可能达到 state-of-the-art [@ouyang2022training; @no_robots]
- **当前实践**：大规模合成数据集效果最好，但质量过滤必不可少

扩大 prompt 数量很快带来了更多性能提升。如今，reasoning model 在 SFT 中通过每个 prompt 使用更多 token（以及更长的上下文长度）来训练，消耗了更多算力和 token。

Tülu 3 [@lambert2024t] 构建了一个 state-of-the-art 的 SFT 数据集，规模约 3 亿 token。

约一年后，Olmo 3 [@teamolmo2025olmo3] 的 state-of-the-art *reasoning* SFT 数据集约有 200 亿 token。其他工作已将其扩大了 10 倍以上，而其根本极限（以及与 RL 的相互作用）仍属未知。

---


## Post-training 的要素：Prompts

成功的 post-training 始于针对目标技能的**有意义的评测**，以及能代表这些技能的**典型查询 prompts**。

所有 post-training 阶段都需要符合任务分布的 prompts。以下是 [@lambert2024t] 中的 prompt 数量示例：

- **Supervised fine-tuning**：约 100 万条 prompts
- **Preference fine-tuning**：约 100 万条（与 SFT 部分重叠会有帮助）
- **Reinforcement fine-tuning**：约 1 万至 10 万条（可用数据较少）

这些数字可能有很大出入，但关键在于：**prompts 是每个阶段的起始材料**。
当然，近期的工作已经把各类 RL 训练阶段的规模大幅扩大了 :D！

---

## 构建 SFT 数据：合成 completions

**合成数据**已成为构建 SFT 数据集的主流方法 -- 沿用 [@wang2022self] 的做法：

1. 从 $N$ 条高质量（通常由人工撰写）的 prompts 出发
2. 让强大的 LM 生成这些指令的改写版本
3. 用另一个（或同一个）强大的 LM 生成 completions
4. 结果：轻松获得 10 倍以上的训练数据

**回复的质量**是相对简单的部分 —— 强模型（如 GPT-4o、Llama 3.1 405B）能为大多数指令生成不错的 completions。  

对于分布外或全新的任务，**人类数据**仍然不可或缺。录制本课时，这主要是医疗、法律等"知识工作"类任务。

---

## SFT 设计流程

两条可重复、可并行的路线：

**数据混合：**
- 取现有数据集，与当前的混合配方合并，观察性能
- 投入大量精力**删除**数据，同时保持性能
- 先完全专注于混合，再做数据整理

**数据整理：**
- 找出模型落后的评测
- 为这些技能构建新的针对性数据
- [可选] 根据质量或正确性过滤回复

这两条路线循环迭代：混合已有数据、评测、整理缺失的部分，再次混合。

---

<!-- layout: section-break -->

## 第二部分：Reward models

---

<!-- columns: 45/55 -->
## Reward model 在 RLHF 中的角色

<!-- cite-right: christiano2017 -->

在 RLHF 中，reward model 扮演**环境**的角色 —— 它返回奖励信号，告诉 policy 表现得如何。

与标准 RL 的关键区别：在 RLHF 中，我们可以从人类偏好中**控制并学习**这个奖励函数，而不是由环境固定给出。

Reward model 将复杂、主观的人类判断压缩为单一的标量分数。

|||

![RLHF 中的 reward model 扮演环境奖励信号的角色。](assets/rlhf-overview.png)

---

<!-- cite-right: sutton2018reinforcement -->
<!-- columns: 65/35 -->

## 回顾：经典强化学习

强化学习问题通常写作**马尔可夫决策过程（MDP）**：
- 状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}$
- 转移动态 $P(s_{t+1}\mid s_t, a_t)$
- 奖励函数 $r(s_t, a_t)$ 与折扣因子 $\gamma$
- 优化整条轨迹上的累积回报

$$\begin{gathered}
\text{MDP } (\mathcal{S}, \mathcal{A}, P, r, \gamma) \\[6pt]
J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]
\end{gathered}$$

|||

![](assets/rl.png)

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## 回顾：经典 RL 与 RLHF

<div class="text-sm">

**经典 RL**
- Agent 在具有状态 $s_t$ 的环境中采取动作 $a_t$
- 奖励是环境在每一步给出的已知函数 $r(s_t, a_t)$
- 优化整条轨迹上的累积回报（总步数 $T$）

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

<div class="colloquium-spacer-md"></div>

**RLHF**
- 没有环境 —— prompts 从数据集中采样
- 奖励是从人类偏好中**学习**得到的（一个代理）
- **回复级别**的奖励（bandit 式，而非逐 token）
- 通过 **KL 惩罚**进行正则化，使其接近 base model

$$J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

</div>

|||

![](assets/rlhf.png)

---

<!-- rows: 30/70 -->
## 回顾：RLHF 比较数据长什么样

人类（或 AI）标注者看到同一 prompt 下的两个回答，并选出更好的一个——这个偏好对会成为 reward model 的训练数据。

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

## 偏好的 Bradley-Terry 模型

<!-- cite-right: BradleyTerry -->

**概率模型**是我们假设与真实判断方式相符的一种数学形式——然后再用数据拟合它的参数。 
经典的 reward model 使用的是 **Bradley-Terry 模型**（1952）。 

---

## 偏好的 Bradley-Terry 模型

<!-- cite-right: BradleyTerry -->

**概率模型**是我们假设与真实判断方式相符的一种数学形式——然后再用数据拟合它的参数。 
经典的 reward model 使用的是 **Bradley-Terry 模型**（1952）。 
给定两个项目 $i$ 和 $j$，评判者偏好 $i$ 胜于 $j$ 的概率为：

$$P(i > j) = \frac{p_i}{p_i + p_j}$$

每个项目都有一个潜在强度 $p_i > 0$。令 $p_i = e^{r_i}$ 进行重参数化，就可以使用无界分数 $r_i \in \mathbb{R}$——这正是神经网络天然的输出：

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)$$

只有**分数之差**才重要——给所有分数加上同一个常数，偏好不会改变。

这并不是唯一可能的模型——但它在早期 RLHF 中行之有效，并沿用至今。它是一个有用的近似，而不是自然定律。

---


## 从 Bradley-Terry 到 reward model

给定 prompt $x$、被选中的 completion $y_c$ 和被拒绝的 completion $y_r$，我们用 reward model $r_\theta$ 对两者打分：

$$P(y_c > y_r \mid x) = \frac{\exp(r_\theta(y_c \mid x))}{\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))}$$

我们要找到使该概率最大化的 $\theta$——即模型应给人类偏好的 completion 打出更高的分数。

---

## 推导 reward model 的 loss

<!-- cite-right: ouyang2022training -->

从最大化偏好概率出发：

$$\theta^* = \arg\max_\theta \frac{\exp(r_\theta(y_c \mid x))}{\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))}$$

---

## 推导 reward model 的 loss

<!-- cite-right: ouyang2022training -->

从最大化偏好概率出发：

$$\theta^* = \arg\max_\theta \frac{\exp(r_\theta(y_c \mid x))}{\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))}$$

分子分母同时除以 $\exp(r_\theta(y_c \mid x))$：

$$= \arg\max_\theta \frac{\exp(r_\theta(y_c \mid x)) \;/\; \exp(r_\theta(y_c \mid x))}{\left(\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))\right) \;/\; \exp(r_\theta(y_c \mid x))}$$

---

## 推导 reward model 的 loss

<!-- cite-right: ouyang2022training -->

从最大化偏好概率出发：

$$\theta^* = \arg\max_\theta \frac{\exp(r_\theta(y_c \mid x))}{\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))}$$

分子分母同时除以 $\exp(r_\theta(y_c \mid x))$：

$$= \arg\max_\theta \frac{\exp(r_\theta(y_c \mid x)) \;/\; \exp(r_\theta(y_c \mid x))}{\left(\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))\right) \;/\; \exp(r_\theta(y_c \mid x))}$$

分子变为 $1$，而分母中 $\frac{\exp(r_\theta(y_r \mid x))}{\exp(r_\theta(y_c \mid x))} = \exp(r_\theta(y_r \mid x) - r_\theta(y_c \mid x))$：

$$= \arg\max_\theta \frac{1}{1 + \exp(r_\theta(y_r \mid x) - r_\theta(y_c \mid x))}$$

---

## 推导 reward model 损失

<!-- cite-right: ouyang2022training -->

从最大化偏好概率出发：

$$\theta^* = \arg\max_\theta \frac{\exp(r_\theta(y_c \mid x))}{\exp(r_\theta(y_c \mid x)) + \exp(r_\theta(y_r \mid x))}$$

分子分母同时除以 $\exp(r_\theta(y_c \mid x))$：

$$= \arg\max_\theta \frac{1}{1 + \exp(r_\theta(y_r \mid x) - r_\theta(y_c \mid x))}$$

改写为 $\frac{1}{1 + \exp(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)))}$。这正是 sigmoid 函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$：

$$= \arg\max_\theta \; \sigma\!\left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right)$$

---

## 推导 reward model 损失

<!-- cite-right: ouyang2022training -->

$$\theta^* = \arg\max_\theta \; \sigma\!\left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right)$$

训练时在数据集上最大化**对数**似然——要先取 $\log$ *再*求平均，因为 $\mathbb{E}[\log P] \neq \mathbb{E}[P]$：

$$\theta^* = \arg\max_\theta \; \mathbb{E}_{(x, y_c, y_r) \sim D}\left[\log \sigma\!\left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right)\right]$$

取反得到（单个样本的）损失：

$$\mathcal{L}(\theta) = - \log \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right)$$

---

## 损失的两种等价形式

第一种形式，来自 InstructGPT [@ouyang2022training]：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$

第二种形式，来自 Anthropic 的工作 [@askell2021general]：

$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$

二者等价：利用 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$ 可得 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta})$。

两者本质上都是**二元交叉熵**——reward model 学习的是对哪个 completion 更受偏好进行分类。

---

<!-- rows: 35/65 -->
## Reward model 训练可视化

模型在每个 completion 的 EOS token 处计算一个标量分数。对比损失只取决于 chosen 与 rejected 之间的**分数差**。

===

![训练偏好 reward model 需要成对的 chosen 与 rejected completion。](assets/pref_rm_training.png)
<!-- cite-right: ouyang2022training -->

---

## Reward model 架构

最常见的实现：在语言模型之上接一个**线性头**，输出单个标量。

```python
class BradleyTerryRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.lm(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        ).hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        seq_repr = hidden[batch_idx, lengths]
        return self.head(seq_repr).squeeze(-1)
```

---

## 损失很简单

有了上面的模型，训练损失只需三行：

```python
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

**实践提示**：reward model 通常只训练 **1 个 epoch**，以避免对偏好数据过拟合。

---

<!-- layout: section-break -->

## Reward model 变体

---

## 偏好 margin 损失

<!-- cite-right: touvron2023llama -->

当标注者给出 **Likert 量表评分**（如 1-5）时，偏好的强弱程度可以为训练提供信息。Llama 2 提出了 margin 项 $m(y_c, y_r)$：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$

例如，若 chosen 得 5 分、rejected 得 2 分，则 $m = 3$。

这会促使模型对偏好程度很强的样本对产生**更大的分数差距**。

**注意**：Llama 3 去掉了 margin 项——团队观察到在大规模下改进已趋于有限。

---

## K-wise loss 与平衡比较

<!-- cite-right: ouyang2022training, zhu2024starling, zhu2023principled -->

**InstructGPT** 对每个 prompt 的多个比较进行平衡以防止过拟合（训练时将所有 prompt 放在同一个 batch 中来实现这一点）：

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$

**K-wise loss**（Plackett-Luce 模型，用于 Starling 7B/34B）处理 $K$ 个 completion 的完整排序：

$$P(\sigma^i|s^i,a_0^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta}(s^i,a_{\sigma^i(j)}^i))}$$

当 $K = 2$ 时，它退化为 Bradley-Terry。

---

## Outcome reward model（ORM）

<!-- cite-right: cobbe2021gsm8k -->

对于**推理任务**，我们通常可以获得可验证的正确性。ORM 使用 completion 级别的正确性标签（在 completion 的各个 token 上重复）来训练一个逐 token 的 head：

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))]$$

其中 $r \in \{0,1\}$ 是 completion 级别的正确性标签，训练时会广播到 completion 的所有 token 上。

**与 Bradley-Terry 的关键区别**：不需要成对比较，只需为每个回答提供正确/错误标签。并且模型输出的是**逐 token 的正确概率**，而不是 EOS token 处的单个分数。

---

<!-- columns: 50/50 -->
## BT reward model 与 ORM：前向传播

**Bradley-Terry RM**：在**最后一个 token** 处打分

```python
hidden = lm(**inputs,
    output_hidden_states=True
).hidden_states[-1]

# Extract EOS token representation
lengths = attention_mask.sum(dim=1) - 1
batch_idx = torch.arange(hidden.size(0))
seq_repr = hidden[batch_idx, lengths]

# Single scalar per completion
score = head(seq_repr).squeeze(-1)
#       shape: (batch,)
```

|||

**ORM**：在**每个 token** 处打分

```python
hidden = lm(**inputs,
    output_hidden_states=True
).hidden_states[-1]

# Score every token position
logits = head(hidden).squeeze(-1)
#        shape: (batch, seq_len)
```

两者都从相同的基础 LM hidden states 出发，区别在于 head 作用在**哪里**。

---

## 代码对比：BT reward model 与 ORM

<!-- columns: 50/50 -->

**Bradley-Terry RM loss**：对比式，需要成对数据：

```python
score_chosen = model(**inputs_chosen)
score_rejected = model(**inputs_rejected)

loss = -F.logsigmoid(
    score_chosen - score_rejected
).mean()
```

每个 completion 一个标量。Loss 取决于 chosen 与 rejected 之间的**差值**。

|||

**ORM loss**：逐 token，只需标签：

```python
logits = model(**inputs)
# shape: (batch, seq_len)

mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits[mask], labels[mask].float()
)
```

每个 token 一个分数。每个 completion token 都以同一个结果标签（$r = 0$ 或 $1$）为目标进行训练。

---

## ORM 训练图解

结果标签（$r = 0$ 或 $1$）会被**广播**到每个 completion token 上，prompt token 则被 mask 掉。模型从这一重复的信号中学习逐 token 的正确性预测。在更大的训练 batch 中，会有多样化的监督信号可供学习。

![训练 ORM：completion 级别的正确性标签通过 binary cross-entropy 作用于每个 completion token。](assets/orm_training.png)

---

## ORM 推理

在推理时，ORM 会在**每个 token 处**输出正确概率。这些 token 级别的分数随后可以聚合为回答级别的 verifier 分数，用于过滤或重排序。

![推理时，ORM 在 completion 上输出逐 token 的正确概率。](assets/orm_inference_flat.png)

---

## 常见误区："ORM" ≠ 基于正确/错误训练的 BT

一些论文用"outcome reward model"指代**基于正确与错误补全训练的 Bradley-Terry 模型**。这混淆了两种不同的架构：

| | **BT RM（基于正确/错误）** | **ORM（Cobbe et al.）** |
|---|---|---|
| **最少训练输入** | 两个补全（一对） | 一个补全 + 标签 |
| **输出** | EOS 处的单个标量 | 逐 token 的概率 |
| **损失** | $-\log\sigma(r_c - r_{ic})$ | 逐 token 的二元交叉熵 |
| **Head** | 最后一个 token 处打分 | 每个 token 处都打分 |

基于（正确，错误）对训练的 BT 模型仍然是**偏好 RM**——只不过用正确性作为偏好信号。真正的 ORM 具有本质上不同的输入输出结构。

---

<!-- rows: 55/45 -->
## 过程奖励模型（PRM）

<!-- cite-right: lightman2023let -->

PRM 对中间**推理步骤**打分，而不仅仅是最终结果：

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} \log p_\theta(z_{s_i} \mid x, s_i) \right]$$

标签只作用于**步骤边界**（例如双换行）：错误 (-1)、中性 (0)、正确 (+1)。训练时其余所有 token 都被 mask 掉。

===

![PRM 仅在步骤边界处提供监督。每个步骤获得一个 3 类标签。](assets/prm_training_inference.png)

---

<!-- columns: 50/50 -->
## 代码中的 PRM

**3 类 head**（不像 ORM 那样是二元的）：

```python
# head outputs 3 classes per token:
#   0 = incorrect, 1 = neutral, 2 = correct
head = nn.Linear(hidden_size, 3)

hidden = lm(**inputs,
    output_hidden_states=True
).hidden_states[-1]

logits = head(hidden)
#        shape: (batch, seq_len, 3)
```

|||

**仅在步骤边界处计算损失**：

```python
# Labels: class index at step-end tokens,
#         -100 everywhere else
mask = labels != -100
loss = F.cross_entropy(
    logits[mask], labels[mask]
)
```

对比 ORM：在*每个*补全 token 处使用二元标签。
对比 PRM：仅在*步骤边界*处使用 3 类标签。

---

## 奖励模型类型对比

| 模型 | 预测内容 | 训练数据 | 输出 |
|-------|----------|-----------|--------|
| **偏好 RM** | EOS token 处的质量 | 成对（chosen vs. rejected） | 单个标量 |
| **ORM** | 补全 token 上的结果信号 | 二元结果标签 | 逐 token 的概率 |
| **PRM** | 每一步的正确性 | 步骤级标注 | 步骤边界处的分数 |
| **Value Function** | 预期剩余回报 | on-policy rollout | 逐 token 的预期回报 |

**关键区别——ORM vs. Value Function：**

- **ORM** 用逐 token 的 head 预测离线标注的正确性：$p(\text{correct}_t)$
- **Value function** 预测预期*剩余*回报：$V(s_t) = \mathbb{E}[\sum_{k \geq t} \gamma^{k-t} r_k \mid s_t]$

架构相同，但语义和监督流程不同。关于 value function，会在 policy gradient 的讲次中进一步介绍。

---

## 小结

- **RM：**"整个回答有多好？"——标量值
- **ORM：**"这个回复是否有望正确？"——在补全 token 上训练的结果信号
- **PRM：**"推理步骤是否合理？"——逐步分数
- **Value Function：**"从这里起还剩多少 reward？"——RL 中用于计算 advantage 的 baseline

---

## 生成式奖励建模：LLM-as-a-judge

训练 reward model 的一种替代方案：**用 prompt 让 LLM** 来评判质量（prompt 来自 [@zheng2023judging]）。

```
[System]
Please act as an impartial judge and evaluate the quality of the
responses provided by two AI assistants to the user question below.
...
After providing your explanation, output your final verdict by strictly
following this format: "[[A]]" if assistant A is better, "[[B]]" if
assistant B is better, and "[[C]]" for a tie.
```

由此催生的 benchmark：MT-Bench、AlpacaEval、Arena-Hard、WildBench。

**现状**：在 RM 评测上，生成式 RM 往往**不及**训练得到的 reward model，但搭建成本更低，并被广泛用于评测和训练流程。评测与实际效用之间存在不匹配。

---

## Reward model 评估

我很高兴能通过第一个基准 [@lambert2024rewardbench] 推动这一领域的起步！
RM 评估领域已迅速扩展：

- **通用对话：** RewardBench、RewardBench2、RMB、RM-Bench
- **专项：** M-RewardBench（多语言）、RAG-RewardBench、RewardMATH
- **Process RM：** PRM Bench、ProcessBench
- **Agentic：** Agent-RewardBench
- **多模态：** MJ-Bench、Multimodal RewardBench、VL RewardBench

早期 RM 研究的主要工作是建立基准并识别行为模式。 
训练方法上的创新（aspect-conditioned 模型、高质量人类数据集、scaling 实验）仍是活跃的研究方向。

---

<!-- layout: section-break -->

## 第 3 部分：Rejection sampling

---

## 使用 reward model 最简单的做法

<!-- cite-right: gilks1992adaptive -->

**Rejection sampling** 是偏好 fine-tuning 中常用的基线方法。思路如下：

1. 生成大量候选 completion
2. 用 reward model 为它们打分
3. 只保留得分最高的
4. 在这些样本上做 fine-tuning

没有 policy gradient，没有在线 RL，只有**经过筛选的监督学习**。

这个名字来自计算统计学：从一个简单分布中采样，用启发式规则接受或拒绝样本，从而逼近复杂的目标分布。

---

## Rejection sampling 概览

四个阶段：
1. **选择 prompt 和 reward model**（复用 IFT 的 prompt，或新构建一批）
2. 用当前模型为每个 prompt **生成** $N$ 个 completion
3. 用 reward model 对所有 completion **打分**
4. 在得分最高的 completion 上 **fine-tune**（与 instruction tuning 使用相同的 SFT loss）

WebGPT [@nakano2021webgpt]、Anthropic 的 Helpful and Harmless [@bai2022training]、Llama 2 Chat [@touvron2023llama] 以及许多其他开创性工作都使用了这一方法。

这仍然只是**离线数据筛选**：先生成，再在筛选后的输出上训练。

---

## 数学形式：prompt、completion 与 reward

给定 $M$ 个 prompt，每个对应 $N$ 个 completion：

$$X = [x_1, x_2, \ldots, x_M]$$

$$Y = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,N} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,N}
\end{bmatrix}
\quad
R = \begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,N} \\
r_{2,1} & r_{2,2} & \cdots & r_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
r_{M,1} & r_{M,2} & \cdots & r_{M,N}
\end{bmatrix}$$

每个 reward：$r_{i,j} = \mathcal{R}(y_{i,j} \mid x_i)$

---

## 选择方法 1：每个 prompt 取最优

对每个 prompt 独立地选出得分最高的 completion：

$$S(R) = [\arg\max_{j} r_{1,j},\ \arg\max_{j} r_{2,j},\ \ldots,\ \arg\max_{j} r_{M,j}]$$

**示例**（5 个 prompt，4 个 completion）：

$$R = \begin{bmatrix}
\mathbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \mathbf{0.8} & 0.6 & 0.5 \\
\mathbf{0.9} & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & \mathbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \mathbf{0.6}
\end{bmatrix}
\quad S(R) = [1, 2, 1, 3, 4]$$

结果：每个 prompt 对应一个 completion，所有 prompt 都有代表。

---

## 选择方法 2：全局最优的若干对

将 reward 矩阵展平，在全局范围内选出得分最高的 $K$ 对：

$$S_K(R_{\text{flat}}) = \text{argsort}(R_{\text{flat}})[-K:]$$

这里，$R_{\text{flat}}$ 是由 reward 矩阵展平得到的长度为 $MN$ 的向量。每个被选中的展平索引随后会映射回其原始的 `(prompt, completion)` 对。

**同一示例，全局取前 5：**

$$R = \begin{bmatrix}
\mathbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \mathbf{0.8} & 0.6 & 0.5 \\
\mathbf{0.9} & 0.3 & 0.4 & \mathbf{0.7} \\
0.2 & 0.5 & \mathbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$

现在 prompt 3 得到**两个** completion（0.9 和 0.7），而 prompt 5 **一个也没有**。这种方式优化的是绝对质量，但可能偏向简单的 prompt。

---

## 实现细节

核心超参数都很直观：

- **Temperature**：0.7-1.0（completion 需要多样性）
- **每个 prompt 的 completion 数量**：10-30+（太少 = 筛选噪声大）
- **Fine-tuning**：在选出的 completion 上做标准 SFT（损失相同，但学习率等细节可能与最初的 IFT 不同）

**实用技巧**：在批量 RM 推理前先按长度对 completion 排序，以减少 padding token 的计算（这样每个 batch 中的样本长度相近，有些框架会自动处理）。

**开放问题**：如何在多阶段流水线中安排 RS、是否使用多个模型的生成结果、最优的 prompt 选择。目前还没有完全开放的复现，这有点让人困惑。我的直觉是，训练 reward model 有一些微妙的技巧。

---

## Best-of-N 采样：不做 fine-tuning 的 rejection sampling

<!-- cite-right: liu2023statistical -->

**Best-of-N（BoN）** 遵循同样的生成并打分流程，但跳过了 fine-tuning 步骤：

$$S(R) = \arg\max_{j \in [1,N]} r_j$$

- 在**推理时**使用，用来挑出最好的 completion
- **不会**修改模型——它只是一种采样技术
- 常被用作 PPO 等在线 RL 方法的基线对比

BoN 是最简单的 reward 引导方法：多生成几个，挑最好的。也可以用验证器 / LLM-as-a-judge 来代替传统的 reward model。

---

## 目前为止的完整流程

把所有环节串起来——从预训练模型到偏好微调模型的一条简单路径：

1. **Instruction fine-tune** 预训练模型 → 学会对话格式
2. **收集偏好数据** → 人工标注员比较成对的回答（数据部分会在训练讲座之后详细介绍）
3. **在偏好数据上训练 reward model** → 学会为质量打分
4. **Rejection sampling** → 生成大量 completion，保留最好的，再做 fine-tuning

这是一个很强的基线。

后续讲座将介绍更高级的优化方法：PPO（针对学到的 reward 信号优化 policy），以及 Direct Preference Optimization（DPO 及其算法家族，直接从偏好对中优化，而不是过滤训练数据）。

---

<!-- rows: 50/50 -->
## 下一站：RL！🍒

<!-- row-columns: 32/36/32 -->

```box
title: 概览
tone: muted
compact: true
content: |
  1. 引言
  2. 关键相关工作
  3. 训练概览
```

|||

```box
title: 核心训练流程
tone: accent
compact: true
content: |
  4. **Instruction tuning**
  5. **Reward models**
  6. Reinforcement learning
  7. 推理
  8. 直接对齐
  9. **Rejection sampling**
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

```box
title: 课程主页
tone: surface
compact: true
content: |
  - [rlhfbook.com](https://rlhfbook.com)
  - [GitHub 仓库](https://github.com/natolambert/rlhf-book)
```

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
