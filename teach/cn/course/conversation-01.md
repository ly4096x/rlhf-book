---
title: "Post-Training 配方的演变"
author: "Nathan Lambert × Finbarr Timbers"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "对话 1"
  right: "Lambert × Timbers {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
---

<!-- layout: title-banner -->

# 前沿 Post-Training 配方综述

<div class="colloquium-title-eyebrow">一场对话 · rlhfbook.com/course</div>
<div class="colloquium-title-rule"></div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert × Finbarr Timbers</p>
<p>2026 年 6 月</p>
</div>

<p class="colloquium-title-note">从经典的三步 RLHF 配方，到多教师 on-policy 蒸馏（MOPD）的兴起。</p>

---

## Post-training 的经典配方演变

过去一年中，post-training 配方的形态变化，比此前三年加起来还要大。

- **2022–2023（InstructGPT）：** 单一流程 — SFT → reward model → RL。
- **2024（Llama 3、Tülu 3 等）：** 开源配方将 SFT → DPO → 可验证奖励 RL 规范化。闭源配方则采用多阶段 RLHF。
- **2025（DeepSeek R1）：** 推理 RL（R1）让大规模 RL 成为核心。
- **2026（MiMo Flash V2）：** 配方分化为*众多专家模型*，再合并回一个模型。
---

## 新趋势：MOPD

**多教师 On-Policy 蒸馏（Multi-teacher On-Policy Distillation，MOPD）** 是 2026 年前沿模型中普遍出现的模式。

1. 训练 **N 个领域专家教师**（每个：先 SFT，再在相关领域上做 RL）。
2. 训练**一个通用学生**，让它采样*自己的*轨迹（这就是最终的 post-training 模型）。
3. 在每条 rollout 上，逐 token 地最小化与*相关*教师输出分布之间的 **reverse-KL**。


<!-- footnote-right: 渊源：MiMo Flash v2 首次提出 → DeepSeek V4 与 Nemotron 3 Ultra 将其扩展到 >10 个教师。 -->

---

## MOPD 为何出现？

- **RL 变得昂贵，且容易产生冲突。** 在一次训练中混合数学、代码和 agentic RL，最终会让各项能力相互折损。
- **专家模型成本低 / 在组织上易于扩展。** 单一领域上的 SFT 再 RL 已被充分理解，且易于并行。随着 post-training 日益复杂，跨组织扩展是一大优势。
- **On-policy 蒸馏已经成熟。** 随着 RLVR 的复兴，相关文献与经验持续涌现。 

<!-- footnote-right: 来源：[DeepSeek V4 §5.1](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)、[MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) -->

---
<!-- layout: section-break -->
<!-- title: center -->

## 走到今天的路

---

## InstructGPT（2022 年 3 月）— 经典三步法

<!-- columns: 58/42 -->

![InstructGPT：在示范数据上做 SFT → 在比较数据上训练 reward model → PPO](assets/instructgpt-recipe.png)

|||

- 在人类示范数据上做 **SFT**
- 在人类比较数据上训练 **Reward model**
- 针对 reward model 进行 **PPO**

<!-- footnote-right: 来源：[InstructGPT (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) -->

---

## Llama 2（2023 年 7 月）— 多阶段 RLHF

<!-- columns: 60/40 -->

![Llama 2：预训练 → SFT → 结合 rejection sampling 与 PPO 的迭代式 RLHF](assets/llama2-recipe.png)

|||

- 先 **SFT**，再进行多轮**迭代式 RLHF**
- 每一轮：**rejection sampling** → **PPO**
- **两个** reward model — 分别针对**有用性**与**安全性**


<!-- footnote-right: 来源：[Llama 2 (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288) -->

---

## Llama 3（2024 年 7 月）— 复杂的多阶段配方，搭配更简单的优化器

<!-- rows: 70/30 -->
![Llama 3 post-training：reward model → rejection sampling → SFT → DPO，多轮迭代，最佳模型输入下一轮](assets/llama3-recipe.png)

===

- 每一轮：**reward model** → 每个 prompt 采样 **K 个** → **rejection sampling** → **SFT** → **DPO**
- 没有在线 RL — RM 仅用于**过滤**；共运行 **6 轮**，最佳模型作为下一轮的起点

<!-- footnote-right: 来源：[Llama 3 (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) -->

---

## Tülu 3（2024 年 11 月）— 简洁的三阶段 post-training

<!-- rows: 62/38 -->
![Tülu 3：整理 prompt → SFT → DPO → RLVR，并配有 held-out 评测套件](assets/tulu3-recipe.png)

===

精选 prompt → **SFT** → **DPO** → **RLVR** 

（RL with verifiable rewards——这一缩写由这篇论文首次提出）

<!-- footnote-right: 来源：[Tülu 3 (arXiv:2411.15124)](https://arxiv.org/abs/2411.15124) -->

---

## Olmo 3（2025 年 12 月）— Tülu 3 recipe 的推理升级版

<!-- rows: 60/40 -->
![Olmo 3 模型流程：预训练 → 中期训练 → 长上下文，随后 Think / Instruct / RL-Zero 三条分支各自经历 SFT → DPO → RLVR](assets/olmo3-recipe.png)

===

<!-- footnote-right: 来源：[Olmo 3 (arXiv:2512.13961)](https://arxiv.org/abs/2512.13961) -->

---

## DeepSeek R1（2025 年 1 月）— 以 RL 为核心

<!-- columns: 55/45 -->

![DeepSeek-R1 多阶段流程：R1-Zero，然后 cold-start SFT → reasoning RL → rejection-sampling SFT → final RL](assets/deepseek-r1-recipe.png)

|||

**配方：**

- **R1-Zero** — 直接在 base 模型上做纯 RL（GRPO），*不使用 SFT*；用来为完整训练流程**播下 reasoning 行为的种子**，而不是一个独立产品
- **R1** — cold-start SFT → reasoning RL → rejection-sampling SFT → final RL → 蒸馏到 dense 模型
- **recipe 的一大转变：** 以大规模 RLVR 作为主要驱动力，用 SFT 来蒸馏和精炼 RL 学到的行为

<!-- footnote-right: DeepSeek 系列之一。来源：[DeepSeek R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) · [Interconnects: R1 recipe for o1](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1) -->

---

## V3 之后的 DeepSeek 演进

<!-- animate: bullets -->

- [**V3**](https://arxiv.org/abs/2412.19437) · 2024 年 12 月 — SFT + GRPO RL。
- [**R1**](https://arxiv.org/abs/2501.12948) · 2025 年 1 月 — 多阶段 RL；reasoning *涌现*。
- [**V3.1**](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) · 2025 年 8 月 — 单一模型内混合 think / non-think。
- [**V3.2**](https://arxiv.org/abs/2512.02556) · 2025 年 12 月 — 通过 RL 训练 6 个 specialist → SFT 蒸馏 → 一次混合 GRPO。
- [**V4**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) · 2026 年 4 月 — 10+ 个领域专家 → MOPD。

<!-- footnote-right: 每个模型均链接到其报告。V3.1 的细节来自其 model card（没有完整报告）；V3-0324 是同一 recipe 上的质量小幅提升，因此省略。 -->

---

<!-- layout: section-break -->
<!-- title: center -->

## 2026 年风格的 recipe！

---

## MiMo Flash v2（2026 年 1 月）— MOPD 的起点

<!-- rows: 64/36 -->
![MiMo Flash v2 post-training：SFT → 领域 teacher → multi-teacher on-policy distillation](assets/mimo-v2-flash-recipe.png)

===

**阶段：** Stage 1 SFT → Stage 2 训练约 6 个领域专家 teacher（采用较早期的 post-training recipe）→ Stage 3 通过 MOPD 合并为单一 student。

首次清晰地将 **multi-teacher on-policy distillation** 作为整合步骤提出——用“从专家蒸馏”取代单一的整体式 RL 阶段。

<!-- footnote-right: 来源：[MiMo-V2-Flash report (arXiv:2601.02780)](https://arxiv.org/abs/2601.02780) -->

---

## Nemotron 3 Ultra（2026 年 6 月）— 两轮迭代，多位 teacher

<!-- columns: 58/42 -->

![Nemotron 3 Ultra：两轮迭代的 multi-teacher on-policy distillation](assets/nemotron3-recipe.png)

|||

**阶段：** SFT → **multi-teacher on-policy distillation**，进行**两轮迭代**，使用 **>10 个 teacher**，覆盖 reasoning、代码、数学和 agentic 等领域。

**新颖之处：** 跨不同领域的多轮 MOPD——先蒸馏，再从更新后的 teacher 重新蒸馏。

<!-- footnote-right: 来源：[NVIDIA Nemotron 3 Ultra technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) -->

---

## MAI-Thinking-1（2026 年 6 月）— 比 V4 更接近 R1

<!-- rows: 48/52 -->
![MAI-Thinking-1：specialist RL climbs → trace-distillation SFT → 整合 → 最终 climb](assets/mai-thinking1-recipe.png)

===

**阶段：** mid-trained base → **3 轮 specialist RL “climbs”**（如 STEM）→ **trace-distillation SFT** 以整合这些 climbs → 最后一轮 RL climb → MAI-Thinking-1。

**比起 V4，更接近 DeepSeek R1** — 多阶段 RL 加 trace-distillation SFT 做整合，*而非* on-policy MOPD。并不是唯一没用 MOPD 的实验室！


<!-- footnote-right: 来源：[MAI-Thinking-1](https://microsoft.ai/news/introducing-mai-thinking-1/) -->

---

## Kimi K2.5（2026 年 1 月）— agentic、多模态

<!-- columns: 56/44 -->

![Kimi K2.5 Agent Swarm：自主调度的并行 agent 编排](assets/kimi-k25-recipe.png)

|||

**阶段：** **纯文本 SFT** → 覆盖 coding、vision、reasoning、agentic 任务的**文本–视觉联合 RL**。

（未提及 MOPD）

<!-- footnote-right: 来源：[Kimi K2.5 technical report](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) · [blog](https://www.kimi.com/blog/kimi-k2-5.html) -->

---

## GLM-5（2026 年 2 月）——按能力分阶段的 RL

<!-- columns: 58/42 -->

![GLM-5 流程：Base → SFT → Reasoning RL → Agentic RL → General RL，并带有跨阶段蒸馏](assets/glm5-recipe.png)

|||

**阶段：** Base → SFT → **Reasoning RL** → **Agentic RL** → **General RL**。

MOPD 还没有普及，但势头正猛！

<!-- footnote-right: 来源：[GLM-5 technical report (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763) -->

---

<!-- layout: section-break -->
<!-- title: center -->

## 那么，接下来我们将走向何方？
