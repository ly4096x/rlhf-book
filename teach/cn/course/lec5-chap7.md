---
title: "第 5 讲：推理模型的兴起"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 5 讲"
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

# 第 5 讲：推理模型的兴起

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程，第 7 章</p>

---

<!-- rows: 50/50 -->
## 第 5 讲：推理模型的兴起

<!-- row-columns: 32/36/32 -->

```box
title: 概览
tone: muted
compact: true
content: |
  1. 引言
  2. 重要相关工作
  3. 训练概览
```

|||

```box
title: 核心训练流程
tone: accent
compact: true
content: |
  4. 指令微调
  5. 奖励模型
  6. 强化学习
  7. **推理**
  8. 直接对齐
  9. 拒绝采样
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
  17. 产品与 Character
```

|||

```box
title: 附录
tone: muted
compact: true
content: |
  A. 关键定义
  B. 风格基准
  C. 参考文献
```

|||

```box
title: 课程安排
tone: surface
compact: true
content: |
  1. 概览（第 1-3 章）
  2. IFT、RM、RS（第 4、5、9 章）
  3. RL 理论（第 6 章上）
  4. RL 实践（第 6 章下）
  5. **推理（第 7 章）**
```

---

## 从 RL 到推理

第 3-4 讲介绍了语言模型的核心 policy gradient RL 实现：PPO、GRPO、loss 聚合、异步训练等。

本讲：推理模型的兴起、推理专用的早期技术，以及对 2025 年核心模型的回顾等。

---

<!-- columns: 38/62 -->
## LeCun 蛋糕

<!-- cite-right: synced2019cake -->

在 NeurIPS 2016 上，Yann LeCun 提出了蛋糕比喻：预测式 / 无监督学习是蛋糕本体，监督学习是糖霜，RL 是樱桃。

在现代语言模型中，这个类比已经完整成立：

- **自监督学习**（互联网数据）= 蛋糕的主体
- **监督微调**（指令）= 糖霜
- **强化学习**（RLHF，之后是 RLVR）= 顶上的樱桃

|||

![](assets/lecun-cake-analogy.png)

<!-- step -->

--> 有了 agent，RL 在算力上可能不止是樱桃，但它依然是训练循环的基本形态！

---

<!-- rows: 60/40 -->
## RLHF vs RLVR：reward 如何改变一切

<!-- row-columns: 50/50 -->

**RLHF** -- 主观打分：

> *请解释经济学中的机会成本。*
>
> 打分需要判断清晰度、准确性和完整性——全是习得的偏好，没有明确答案。

|||

**RLVR** -- 可验证打分：

> *所有小于 20 的素数之和是多少？*
>
> `extracted_answer == 77` → Reward = 1
>
> *编写 `fib(n)`，返回第 n 个斐波那契数。*
>
> `assert fib(10) == 55` → 所有测试通过 → Reward = 1

===

post-training 方法是一样的（即算法相同），但构建模型有了一套新的直觉。

<!-- step -->

**RLVR 为推理模型打开了大门。** 现在 RL 训练会使用混合 reward、rubric 以及各种工具来训练有用的模型。这里清晰的区分只是出于教学目的，实践中很少见。

---

## RLVR 反馈循环

<!-- cite-right: lambert2024t -->
<!-- img-align: center -->

![RLVR 以 RL 反馈循环的形式呈现。验证函数取代 reward model，对 completion 打分。](assets/rlvr-system.png)

---

<!-- columns: 50/50 -->
## 回顾：RL 训练 vs 推理时扩展

<!-- cite-right: openai2024o1 -->

![训练时算力扩展：更多 RL 训练可提升 policy。](assets/o1-train-time.png)

|||

![测试时算力扩展：更多推理算力可提升性能。](assets/o1-test-time.png)


---

## 回顾：thinking token 长什么样

<!-- cite-right: guo2025deepseek -->

**标准模型**直接回答：

> *所有小于 20 的素数之和是多少？*
>
> 小于 20 的素数有 2、3、5、7、11、13、17、19。它们的和是 **77**。

**推理模型**在回答前会生成 `<think>` token：

> `<think>` I need to find all primes less than 20. Starting from 2: 2 is prime. 3 is prime. 4 = 2×2, not prime. 5 is prime... Let me add them up: 2+3=5, 5+5=10, 10+7=17, 17+11=28, 28+13=41, 41+17=58, 58+19=77. `</think>`
>
> 答案是 $\boxed{77}$。
>
> **验证**：`extracted_answer == 77` → Reward = 1

对于更难的问题，thinking 可以长达**数千个 token**。

---

## 本讲关键术语

<!-- columns: 50/50 -->

**评估指标**：

- **pass@1**：每道题只采样一次时的准确率
- **pass@K**：生成 $K$ 个 completion，报告其中是否有*任意*一个正确——衡量覆盖率

如果 $n$ 次尝试中有 $c$ 个正确答案：

$$
\widehat{\mathrm{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

|||

**算法 / 架构术语**：

- **DAPO**：Dynamic Advantage Policy Optimization——GRPO 的一种放宽 clipping 的变体 [@yu2025dapo]。早期重要的 RLVR 论文
- **CISPO**：Clipped Importance Sampling PO——裁剪重要性采样（IS）权重，而非逐 token 的比率，出自 MiniMax-M1 [@minimax2025minimax_m1]

---


<!-- layout: section-break -->

## 推理模型的寒武纪大爆发

---

## 推理模型的寒武纪大爆发

<!-- img-align: center -->

![](assets/reasoning-model-wall.png)

---

## 此前的研究

RLVR 背后的关键思想并不新——在 o1/R1 让它们成为主流（并真正有效）之前，早已有人探索过：

<!-- animate: bullets -->

- **STaR** [@zelikman2022star]（以及 **Quiet-STaR** [@Zelikman2024QuietSTaRLM]）：利用真实答案奖励的自学式推理（2022-2024）。采样带有不同推理过程（rationale）的 CoT，保留答案正确的，再进行训练。
- **TRICE** [@hoffman2023training]：受 MCMC 启发的推理轨迹优化（算法更复杂）。
- **VinePPO** [@VinePPO]：在 GSM8K/MATH 上使用二元数学奖励的 PPO，通过大量 rollout 和蒙特卡洛估计实现。
- **Tülu 3** [@lambert2024t]：用 PPO 优化数学正确性，同时保持广泛的能力。

<!-- step -->

随后出现的模型以更简单的方式放大了这些方法，并转移了 post-training 的重心。

---

## 为什么 RL 现在奏效了？

<!-- animate: bullets -->

- **稳定性更容易处理**：仍是一流的研究问题（熵坍缩、长程信用分配），但工具和配方已足够成熟，可被广泛采用
- **基础模型足够好**：多方证据表明，RL 推理训练直到 2024 年前后的模型才变得可行——需要一个能力门槛
- **可验证领域提供干净的信号**：数学和代码给出明确无歧义的奖励，避免了 RLHF 中的 reward hacking 问题

---

## 本讲内容

1. 开创性的模型
2. 我们从中学到了什么
3. 近期历史带给我们的启示

---

<!-- rows: 39/61 -->
## DeepSeek R1（2025 年 1 月 20 日）：催化剂

<!-- cite-right: guo2025deepseek -->

一次出人意料地迅速的 o1 复现。

**R1-Zero**：在基础模型上进行纯 RL，没有 SFT 热启动。表明仅靠大规模 RL *本身*就能诱导出 chain-of-thought 推理。

**完整的 R1 配方**：冷启动 SFT → 大规模 RL → 蒸馏出较小的模型。

===

![DeepSeek R1-Zero 的训练奖励曲线，展示了涌现出的推理行为。](assets/deepseek-r1-zero-figure1-training.png)

---

<!-- columns: 45/55 -->
## Kimi 1.5（2025 年 1 月 20 日）：扩展课程学习

<!-- cite-right: team2025kimi -->

Kimi 1.5 与 R1 同在 1 月的这一波发布，强调 **RL 规模加课程学习（curriculum）**。

- 在中文和英文推理数据上进行 PPO/GRPO 风格的 RL
- 难度调度和在线过滤，保证梯度始终有用
- 逐步延长长度，在支持长 CoT 的同时减少过度思考
- 技术报告很详细，但**没有开源 k1.5 权重**

|||

![Kimi 1.5 的 RL 训练动态：各基准上的准确率与回复长度同步增长。](assets/kimi-15-training-dynamics.png)

---

<!-- rows: 43/57 -->
## Open-Reasoner-Zero（2025 年 2 月 18 日）：极简复现

<!-- cite-right: hu2025openreasonerzero -->

首个面向 r1 风格、基于基础模型的 RL 训练的公开数据集和代码。

- 完全开源：**训练代码、精选的 RL 数据和模型权重**
- 使用带 GAE（$\lambda=1, \gamma=1$）的原始 PPO 和简单的基于规则的奖励
- 没有 KL 惩罚
- 3 月 31 日扩展：更易用的脚本和 12.9 万道精选题目

===

![论文中 Open-Reasoner-Zero 的训练时扩展曲线。](assets/open-reasoner-zero-train-scale.png)

---

<!-- columns: 45/55 -->
## Qwen 3（2025 年 4 月 29 日）：混合推理模型

<!-- cite-right: yang2025qwen3 -->

新的开源权重标准（RIP Llama 4）。

- 可切换的 thinking：`/think` 和 `/no_think` 模式（多数实验室都发现这在训练上相当头疼）
- Thinking budget 长度控制
- 大模型的训练配方沿用 R1 式的多阶段流程
- 轻量模型采用由强到弱的蒸馏：先是 off-policy 输出，再是 on-policy 的 teacher-logit 匹配

Llama-Nemotron [@bercovich2025llamanemotron] 是另一个可切换推理模式的开源权重推理模型，公开了 post-training 数据和训练代码库，但配方不同。

|||

![论文中 Qwen 3 在不同 thinking budget 下的性能曲线。](assets/qwen3-thinking-budget.png)

---

<!-- columns: 45/55 -->
## MiMo（2025 年 4 月 30 日）：端到端的推理流程

<!-- cite-right: xia2025mimo -->

小米公开了从预训练到 post-training 的**完整流程**。

针对推理能力对训练配方做的调整：
- 预训练阶段采用三阶段数据混合（25T tokens）
- 预训练阶段采用 Multi-Token Prediction（MTP）
- 多领域 RL，防止在单一任务类型上过度优化

这是少数在 post-training 中公开清晰中间 checkpoint 的开源权重推理模型之一！

|||

![MiMo-7B-RL 在代码（LiveCodeBench）和数学（AIME）上都超过了 o1-mini 和更大的 RL 基线 -- 面向推理的预训练决定了 RL 的上限。](assets/mimo-rl-curves.png)

---

<!-- columns: 40/60 -->
## OpenThoughts 3（2025 年 6 月 5 日）：SFT 数据配方依然重要

<!-- cite-right: guha2025openthoughts -->

基础性的开放推理数据（至今仍相当强）！

- 涵盖数学、代码和科学的 120 万条公开样本
- QwQ-32B 的推理轨迹，以及 1,000 多组受控的数据流水线实验
- OpenThinker3-7B 仅靠 SFT 就达到了很强的推理性能

社区需要在强大、开放的推理轨迹上投入更多（并弄清什么样的 teacher 才算好）。

|||

![OpenThoughts3 在 AIME 2025、LiveCodeBench 和 GPQA Diamond 上的 scaling 曲线。](assets/openthoughts3-scaling.png)

---

<!-- columns: 40/60 -->
## MiniMax-M1（2025 年 6 月 16 日）：一篇有趣的 RL 论文！

<!-- cite-right: minimax2025minimax_m1 -->

MiniMax M1 的论文真正经受住了时间的考验。在这些广受欢迎的模型之上：

1. CISPO 对 importance-sampling 权重做裁剪，而不是丢弃高更新量的 token。这类算法一直沿用至今。
2. 提出了如今广为人知的 FP32 LM head 推理-训练不匹配图。

|||

![修复前后，训练引擎与推理引擎给出的逐 token 概率对比。以低精度运行 LM head 会让两者产生偏差 -- 这是一种会让 RL 不稳定的隐蔽的训练/推理不匹配 -- 而仅将 LM head 升为 FP32 就能让二者重新对齐。](assets/minimax-m1-logprob-mismatch.png)

---

<!-- columns: 40/60 -->
## GLM-4.5（2025 年 7 月 28 日）：推理扩展到 agent 任务

<!-- cite-right: zeng2025glm45 -->

GLM-4.5 报告是较早关注 agent 行为的报告之一。除此之外，它是夏季浪潮中“又一个”非常强的开源中国模型。

- 总参数 355B，激活参数 32B
- 支持 thinking 和直接回答两种模式
- 通过 expert-model 迭代加 RL，提升 agent、推理和通用对话能力

|||

![GLM-4.5 在 agent、推理和代码 benchmark 上的平均表现。](assets/glm45-arc-benchmarks.png)

---

<!-- columns: 40/60 -->
## Olmo 3 Think（2025 年 11 月 20 日）：完全开放的推理模型

<!-- cite-right: teamolmo2025olmo3 -->

对推理模型完整生命周期最全面的开放文档。

公开内容：各阶段、checkpoint、数据、基础设施、超参数。

有趣的 DPO 发现，以及更多好玩的内容。更多细节请看[这个演讲](https://www.youtube.com/watch?v=uaZ3yRdYg8A)。


|||

![模型报告中 Olmo 3.1 Think 的延长 RL 训练生命周期。](assets/olmo31-extended-rl.jpeg)

---

<!-- columns: 45/55 -->
## DeepSeek V3.2（2025 年 12 月 1 日）：推理走向 agent 化

<!-- cite-right: deepseekai2025v32 -->

DeepSeek V3.2 将 R1 的配方推进到**工具使用和 agent 环境**。

- V3.2-Exp 的开源权重 MoE 继任者
- V3.2 将 thinking 直接整合进工具使用，不再是单独的推理模型
- Speciale 变体以最大化推理性能为目标

|||

![DeepSeek V3.2 benchmark 图表，对比推理和 agent 能力。](assets/deepseek-v32-benchmark.png)

---

<!-- rows: 43/57 -->
## Nemotron 3 Nano（2025 年 12 月 15 日）：高效的 agentic 推理

<!-- cite-right: nvidia2025nemotron3nano -->

又一个非常开放（且来自美国）的模型，引领了 2026 年更强的美国开放权重模型。

- 总参数 31.6B，每次前向传播约激活 3.2B
- Mamba-Transformer 混合 MoE 架构
- 后训练包括 SFT、多环境 RLVR 和 RLHF

===

![技术报告中 Nemotron 3 Nano 的准确率与吞吐量对比。](assets/nemotron-3-nano-comparison.png)

---

<!-- columns: 45/55 -->
## MiMo-V2-Flash（2025 年 12 月 16 日）：把多个教师蒸馏成一个

<!-- cite-right: mimo2025flash -->

其突出贡献在后训练：**Multi-Teacher On-Policy Distillation (MOPD)**。

- 为每个领域（数学、代码、搜索、推理）训练一个专家教师，再将它们蒸馏到同一个学生模型
- 学生从**稠密的 token 级 reward** 中学习，而不只是序列级的结果 reward——样本效率更高，并且在每个领域都能追平最强的教师
- 底层模型是为快速 rollout 而设计的 309B/15B MoE，但这里的重点是训练方案

|||

![MOPD 后训练：来自各领域教师的稠密 token 级 reward 将学生提升到教师水平的准确率，而序列级 ORM 则停滞不前。](assets/mimo-v2-flash-mopd.png)

---

<!-- columns: 50/50 -->
## 近期的美国开放模型

[**NVIDIA Nemotron 3 Super**](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/)（2026 年 3 月 10 日）

- 120B/12B MoE；LatentMoE、MTP、原生 NVFP4 预训练
- 完全开放：checkpoint、训练数据和训练方案

|||

[**Arcee Trinity-Large-Thinking**](https://www.arcee.ai/blog/trinity-large-thinking)（2026 年 4 月 1 日）

- 面向长程、多轮工具使用调优的前沿推理模型
- Apache 2.0 权重；SFT + RL 流程


---

## 近期的中国开放模型

- [**Kimi K2.5**](https://www.kimi.com/blog/kimi-k2-5)（2026 年 1 月 27 日）→ [**K2.6**](https://www.kimi.com/blog/kimi-k2-6)（2026 年 4 月 20 日）：原生多模态与 agent swarm，随后是更强的长程编程能力
- [**GLM-5**](https://z.ai/blog/glm-5)（2026 年 2 月 12 日）→ [**GLM-5.1**](https://docs.z.ai/guides/llm/glm-5.1)（2026 年 4 月 7 日）：agentic 工程，随后是更长、更稳定的自主运行
- [**DeepSeek V4 Preview**](https://api-docs.deepseek.com/news/news260424)（2026 年 4 月 24 日）：开放 Pro 和 Flash 模型，双推理模式，高效的架构创新，以及 1M token 上下文

前沿推进得很快——从“推理模型”到**长程、使用工具的 agent 底座**。

---

## 这片格局告诉了我们什么

<!-- animate: bullets -->

1. RL 的基础方法很快就在开源代码库和数据上被复现
2. 从数学/代码推理任务到 agentic 模型的转变紧随其后。像 Olmo 3 这样的模型（我参与构建，知道它来得稍晚）在这方面就更显突出
3. 为了让长上下文推理更高效，模型层面出现了大量创新，尤其来自中国的实验室
4. 许多持久的基本功（FP32 LM head、CISPO 式损失等）很快就出现了，但需要大量调试才能收敛

---

<!-- layout: section-break -->

## 常见的实现模式

---

## 方案决策（1/4）

<!-- animate: bullets -->

1. **离线难度过滤** -- 为每个 prompt 预先采样 $N$ 个 completion，保留通过率约 20-80% 的 prompt（梯度所在之处）。*Seed-Thinking 1.5 [@seed2025seed], Open-Reasoner-Zero [@hu2025openreasonerzero], Phi-4 Reasoning [@abdin2025phi4], INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained], MiMo [@xia2025mimo], Skywork OR-1 [@he2025skyworkor1], Olmo 3 [@teamolmo2025olmo3]*
2. **在线过滤与课程学习** -- 跳过现在过于简单或过于困难的 prompt；把更难的问题留到后面。*Kimi 1.5 [@team2025kimi], Magistral [@mistral2025magistral], Llama-Nemotron [@bercovich2025llamanemotron], INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained], MiMo [@xia2025mimo]*
3. **零梯度过滤 + 主动采样** -- 丢弃所有 $G$ 个 completion 全部通过或全部失败的分组（advantage 为 0），然后补满 batch。*Olmo 3 [@teamolmo2025olmo3]*。让 loss 非常平滑稳定！
4. **KL 惩罚：曾被舍弃，现在又悄然回归** -- 单轮数学 RLVR 通常设 $\beta=0$（可验证 reward 不易被过度优化）。*Magistral [@mistral2025magistral], Open-Reasoner-Zero [@hu2025openreasonerzero], Skywork OR-1 [@he2025skyworkor1]* 但长程、off-policy 的 agentic 训练正在重新引入 KL/参考项以保证稳定——例如 Cursor [Composer 2](https://cursor.com/blog/composer-2-technical-report)。
5. **放宽/非对称 clipping** -- 放宽上侧 clip 边界以保留探索性更新（DAPO [@yu2025dapo]）。*Magistral [@mistral2025magistral], INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained]*

---

## 方案决策（2/4）

<!-- animate: bullets -->

6. **CISPO** -- 对 importance sampling 权重做 clip，而不是屏蔽高更新幅度的 token（MiniMax-M1 [@minimax2025minimax_m1]）；ScaleRL [@khatri2025art] 在其 scaling 研究中选它作为最佳损失。回顾 importance sampling 与 clipping，见 [第 3 讲](https://www.youtube.com/watch?v=K_Sj_-1BUMM)。
7. **归一化的选择** -- 如何对 advantage 和 loss 求平均，会悄悄改变目标。在**整个 batch** 上而非每个 prompt 分组内归一化 advantage，避免少数过易/过难的 prompt 主导训练（Magistral [@mistral2025magistral], MiMo [@xia2025mimo]）；对**总 token 数**而非每条序列求平均 loss，以消除长度偏差（Dr. GRPO [@liu2025understanding]）。关于逐 token 与逐序列损失聚合的深入讲解，见 [第 4 讲](https://www.youtube.com/watch?v=i-AIMpZHgeg)。
8. **格式 reward** -- 奖励有效的 `<think>` 块和可提取的答案；关乎可用性，而非正确性。*DeepSeek R1 [@guo2025deepseek], Open-Reasoner-Zero [@hu2025openreasonerzero], Magistral [@mistral2025magistral], Skywork OR-1 [@he2025skyworkor1]*
9. **语言一致性 reward** -- 惩罚推理过程中的语言切换。*DeepSeek R1 [@guo2025deepseek], Magistral [@mistral2025magistral]*。模型过去常在推理时切换成中文或法语。
10. **长度控制** -- 渐进式长度扩展——即训练时的最大长度，以此奖励简洁的思考（Kimi 1.5 [@team2025kimi]）；小幅长度惩罚（INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained]）；过长过滤以提升吞吐。

---

<!-- columns: 55/45 -->
## 方案决策（3/4）

<!-- cite-right: teamolmo2025olmo3 -->

<!-- animate: bullets -->

11. **inference 是瓶颈** -- completion 又长又不定长；Olmo 3 的 learner GPU 约 **75%** 的时间处于空闲，inference 算力是训练的 **5-14 倍**。
12. **Off-policy / 异步更新** -- actor 持续生成，learner 持续消费；rollout 会略微过时，但吞吐大幅提升。*Seed-Thinking 1.5 [@seed2025seed], INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained]* -- 深入讲解：Olmo 3 [幻灯片](https://docs.google.com/presentation/d/17Evr5AqnwjtSZ4H5vpnoEvGmI9fRBO19SN2dGHg7tPE/edit)、[录像](https://www.youtube.com/watch?v=uaZ3yRdYg8A)
13. **Test-time scaling** -- 多数投票（DeepSeek R1 [@guo2025deepseek], Phi-4 [@abdin2025phi4]）；selector 模型（DeepSeek-GRM [@liu2025inference]）；pass@K 与 pass@1 之间的差距显示了提升空间。

|||

![分布式 RL 训练基础设施。](assets/distributed-rl.png)

---

## 方案决策（4/4）

<!-- animate: bullets -->

14. **发现 -- 纯文本 RL 能提升多模态** -- 在多模态训练*之后*做文本推理 RL，可以提升视觉任务。*MiMo-VL, Magistral [@mistral2025magistral]*
15. **发现 -- midtraining 决定 RL 的上限** -- 预训练中数学/代码的占比限定了 RL 所能达到的水平。*MiMo [@xia2025mimo]*

---

## 开放性问题

<!-- animate: bullets -->

- RL 训练是在**发现**新能力，还是在**激发**预训练已经学到的能力？
- 没有更好的预训练数据，推理训练还能走多远？预训练似乎并没有“死”。
- Agentic RL（工具使用 + 推理）是否需要根本不同的训练配方？
- 我们能否系统地研究推理 RL 的 scaling 特性？[@khatri2025art]
- 1 年、2 年、5 年和 10 年后，预训练与 RL 之间的算力分配会是什么样子？

---

## RL 训练库

- [**slime**](https://github.com/THUDM/slime) -- 基于 SGLang 的原生 post-training 框架，用于扩展 RL；GLM 系列模型使用了它
- [**SkyRL**](https://github.com/NovaSky-AI/SkyRL) -- 面向 LLM 的模块化全栈 RL 库，支持异步和多轮 agent 工作流
- [**veRL**](https://github.com/volcengine/verl) -- 源自 HybridFlow 工作、灵活且面向生产的 RL 训练库
- [**OpenRLHF**](https://github.com/OpenRLHF/OpenRLHF) -- 基于 Ray 的 RLHF 和 RLVR 训练框架
- [**TRL**](https://github.com/huggingface/trl) -- Hugging Face 的 post-training 库，trainer API 易于上手
- [**Open Instruct**](https://github.com/allenai/open-instruct) -- Ai2 的开源 post-training 配方与研究代码

---

## 课程大纲

1. 引言与训练概览 -- 第 1-3 章
2. IFT、reward model、rejection sampling -- 第 4、5、9 章
3. RL 理论 -- 第 6 章（上）
4. RL 实现与实践 -- 第 6 章（下）
5. **推理 -- 第 7 章**
6. 直接对齐算法 -- 第 8 章
7. ...

---

<!-- rows: 85/15 -->
## 谢谢

欢迎提问与讨论。

**Nathan Lambert**

rlhfbook.com | interconnects.ai

===

```builtwith
repo: natolambert/colloquium
```
