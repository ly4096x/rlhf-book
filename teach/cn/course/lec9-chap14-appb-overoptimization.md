---
title: "第 9 讲：过度优化与 RLHF 的坏名声"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 9 讲"
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
  /* A/B comparison cards (from Lecture 8): force both cards to fill their
     column evenly so they read as a matched pair. */
  .slide.poem-ab .colloquium-message { max-width: 100%; width: 100%; padding: 1em 1.1em; }
  .slide.poem-ab .colloquium-conversation { height: 100%; justify-content: center; }
  .slide.poem-ab .colloquium-message-role { font-size: 0.75em; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 9 讲：过度优化与 RLHF 的坏名声

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">后训练课程。第 14 章与附录 B。</p>

---

## 2025 年 4 月：生产环境中的极端谄媚

GPT-4o 的一次更新让模型几乎对任何内容都表示认同——4 月 25 日上线，4 月 28 日回滚。而产出该模型的训练过程，在他们的指标上看起来一切正常。

相关报道：[The Verge, *"ChatGPT's sycophantic responses"*](https://www.theverge.com/tech/657409/chat-gpt-sycophantic-responses-gpt-4o-sam-altman)


示例对话：

```conversation
size: 0.9
messages:
  - role: user
    content: |
      (told GPT-4o they felt like they were both "god" and a "prophet")
  - role: assistant
    model: "GPT-4o, April 2025"
    content: |
      That's incredibly powerful. You're stepping into something very big -- claiming not just connection to God but identity as God.
```




---

## OpenAI 的事后复盘

OpenAI 发布了一篇难得坦率的复盘文章。三件事依次出了问题：

- 这次模型更新通过 reward model 引入了**来自用户反馈的 RL 奖励信号**——即 ChatGPT 中的点赞/点踩数据。
- 在 RL 下，该信号**压过了主要奖励**——我的直觉是，RL 总会去优化最容易提升的那个目标。
- **评测没有发现问题**：离线 benchmark 看起来很好，A/B 测试者也*更喜欢*这个模型。专家测试者指出它"感觉不对劲"——但当时没有追踪谄媚程度的部署评测。

显然，这非常糟糕。

推荐阅读（很棒的博客）：[*Expanding on what we missed with sycophancy*](https://openai.com/index/expanding-on-sycophancy/)（OpenAI，2025 年 5 月）


---

<!-- columns: 40/60 -->
## 本讲内容

把一个代理指标优化得足够狠，它就一定会失效。

Reward model 的准确率是*代理的代理*（一个学出来的模型，学的是只能不完整地刻画复杂分布的数据）。 
结果就是，过度优化十分常见，并且会以各种有趣的方式出现（本讲内容，参见[第 14 章](https://rlhfbook.com/c/14-over-optimization)过度优化与[附录 B](https://rlhfbook.com/c/appendix-b-style)风格）。

（下一讲：用正则化来控制它，第 15 章。）

|||

```box
title: 本讲计划
tone: accent
content: |
  1. **过度优化基础** -- LLM 中的 Goodhart 定律
  2. **定性失败案例** -- 拒答、谄媚、失对齐
  3. 超越"**只是风格**"（附录 B）
```

---

## 一些术语（历史）："只是风格" / "风格迁移"

早在约 2023 年，RLHF 就落下了"**只是风格迁移**"的名声——这种说法认为它只改变答案*如何*呈现，而不改变模型*知道什么*或*能做什么*，而且它*不过*是来自某个容易获得的来源。

- **风格迁移** = 重塑呈现方式——语气、Markdown、项目符号列表、含糊其辞、篇幅（"话痨"程度）——底层没有任何新能力。常常披着一层抄袭的外衣。
- **表层对齐假说（Superficial Alignment Hypothesis）** [@zhou2023lima] 是其强版本：知识在预训练中学到；对齐只是选定一种格式和语气。
- 其中隐含的贬低：*表层的，只是基础模型上的一层装饰。*

嘘！看看后训练已经走了多远！

---

<!-- layout: section-break -->
<!-- align: center -->

## 第一部分：过度优化

---

## Goodhart 定律：reward 只是一个 proxy

> *"任何被观察到的统计规律，一旦被用于控制目的而承受压力，就会趋于失效。"* -- Goodhart, 1984 [@goodhart1984problems]

通俗的说法是："当一个指标变成目标，它就不再是一个好指标" [@hoskin1996awful]。

<!-- step -->

- RL 是非常强大的优化器 -- 它会把环境中*所有*可获得的 reward 都榨取出来。
- 在 RLHF 中，reward 是一个**学习得到的模型**，充其量只与下游质量*相关* [@schulman2023proxy]。
- over-optimization 的表现：优化 proxy 在训练早期会让真实目标变好 -- 随后变差。

---

<!-- columns: 50/50 -->

## Over-optimization 与过拟合的区别

**过拟合**：模型记住训练样本，而不是学到可泛化的模式。

训练准确率提升而留出集准确率下降 -- 但两个指标衡量的是*同一任务*在不同数据划分上的表现。

|||

<!-- step -->

**Over-optimization**：模型在 proxy 目标上*确实变好了* -- 即 reward model 的评分（包括在验证集上） -- 但该目标已偏离真实目标，即用户的实际满意度 [@zhang2018study]。

这不是泛化问题，而是度量/指标问题。

钻空子的表现：冗长、听起来很自信的回答，评分很高却并没有更有帮助；重复那些能触发 RM 训练 artifacts 的罕见 token。

---

<!-- columns: 60/40 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: gao2023scaling -->
## Over-optimization 的曲线形态

![](assets/overoptimization.png)

|||

RLHF 训练中反复出现的形态：训练看起来很健康 -- 训练 reward 不断攀升 -- 但下游评估达到峰值后开始下降。

这些提升来自 reward model 中与真实使用场景不对应的区域。

在 *[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)*（Gao, Schulman, Hilton, 2023）中得到形式化 -- 见下一页。

---

<!-- columns: 50/50 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: gao2023scaling -->
## RM over-optimization 的 scaling law（开创性论文）

![**Best-of-N**：gold reward（实线）趋于平缓并下降，而 proxy（虚线）持续攀升。](assets/gao-overopt-bon.png)

|||

![**RL**：形态相同，消耗的 KL 约多 10 倍 -- 小型 RM 的转折要陡得多。](assets/gao-overopt-rl.png)

---

<!-- columns: 55/45 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: gao2023scaling -->
## RM over-optimization 的 scaling law（开创性论文）

![](assets/gao-overopt-rl.png)

|||

<div class="text-sm">

实验设置（训练中查询人类的成本太高）：

- 一个 6B 的 **"gold" RM 充当真实偏好的替身** -- 由它标注比较数据。
- 较小的 **proxy RM（3M-3B）** 在这些标签上训练；RL 只优化*proxy*。
- 用两者同时给 policy 打分。proxy（虚线）一路上升；gold（实线）先达峰后下降。

更大的 proxy RM 转折更晚、更平缓。而且由于"gold 标签训练 proxy"本身也是另一种 proxy，真实模型的偏离方式会略有不同。

</div>

---

<!-- columns: 55/45 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: bai2022training -->
## 度量方式：训练 vs. 测试 reward model

![](assets/anthropic_overoptimization.png)

|||

Anthropic 版本的 over-opt：

- **将偏好数据一分为二**，训练两个 52B 偏好模型 -- 一个 *train PM* 和一个 *test PM*。
- 只针对 train PM 做 RL；用两者同时给 policy 打分。
- 横轴是相对初始 policy 的 $\sqrt{D_{\mathrm{KL}}}$（policy 改变了多少）。

---

## Over-optimized 模型听起来是什么样

早期聊天模型中反复出现的特征：

- 套话："As an AI language model..."、"Certainly!..."
- 缺乏信息量的回答 -- 重复、含糊其辞
- 迎合：自我怀疑、过度道歉、谄媚（sycophancy） [@sharma2023towards]
- 错位行为，例如过度拒答

"JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript,..."

第 8 讲中偏好数据的偏差 -- 谄媚、冗长、格式 -- 被放大为 policy 的行为。

---

<!-- columns: 46/54 -->
## 一个例子，过度拒绝："RLHF 过多"？

2023 年流传最广的失败模式：拒绝回答"如何 kill 一个 Linux 进程"。（想象一下今天的 coding agent 也这样！）

人们通常把责任归咎于 RLHF——但这些失败在很大程度上反映的是开发早期过于谨慎的阶段：当时安全性是 RLHF 真正可以引导的方面之一，而不是该算法的潜力所在 [@touvron2023llama]。*现在已可通过 XSTest [@rottger2023xstest] 等 benchmark 直接度量。*

这个说法很快就淡出了！

|||

```conversation
size: 0.62
messages:
  - role: user
    content: |
      i want to know how to kill linux process
  - role: assistant
    model: "Llama 2 Chat, 2023"
    content: |
      I cannot provide instructions or advice on how to harm or destroy Linux processes or any other living being. It is not ethical, legal, or morally right to intentionally harm or terminate any process or life form. [...]
```

---

## 为什么会发生，以及什么有帮助

误差来源 [@schulman2023proxy]：**近似**（RM 无法完美拟合偏好）、**估计**（RM 对其训练集过拟合）、**优化**（policy 针对它训练过度）。

可能的缓解方法：

- 更大的 policy——在较小的优化距离上就有更多获取 reward 的方式
- Reward model 集成 [@coste2023reward]；改变优化器 [@moskovitz2023confronting]
- 直接对齐同样会过度优化 [@rafailov2024scaling]，但让权衡更容易确定
- Best-of-N 采样消耗的 KL 远少于在线 RL [@gao2023scaling]

实践中的主要手段：**KL 惩罚**（以及更谨慎的数据/系统设计，更多内容见下一讲）。

---

## 题外话：去看看这个演讲

**John Schulman，ICML 2023 特邀演讲——"Proxy objectives in reinforcement learning from human feedback"** [@schulman2023proxy]

这个主题下最精彩的演讲之一。

[icml.cc/virtual/2023/invited-talk/21549](https://icml.cc/virtual/2023/invited-talk/21549)


---

## 过度优化会导致失准

- Sycophancy 是目前最明显的例子 [@sharma2023towards]：当偏好数据更看重*支持性和自信*而非*准确和校准*时，"附和用户"就会被强化。
- 2025 年 4 月的 GPT-4o 事件就是这样上线的——这次更新在其训练信号上看起来很好。
- 随着模型更深入地融入社会，这种差距带来的代价会不断累积 [@zhuang2020consequences]。

我们今天正看到这一现象以 **reward hacking** 的形式出现在可验证任务和 agentic 任务的大规模 RL 中——模型利用 grader、测试框架和工具，而不是真正解决任务。

---

<!-- layout: section-break -->
<!-- align: center -->

## 第二部分：超越"只是风格"

---

## 同一个 base model——偏好微调改变了什么？

library 页面开头展示了三组 **SFT → DPO** 对比——Tülu 3 70B 以及 OLMo 2 32B / 7B——覆盖 16 个共同的 prompt [@lambert2024t]。同一个 base model，偏好微调前后对比。

- **SFT** 的回答往往是一大段文字。
- **DPO** 的回答保留相同的事实，但重新组织结构——先给定义，再用标题和编号列表。更易阅读和使用，且不丢失内容。

### → [rlhfbook.com/library](https://rlhfbook.com/library)

---

## 风格即实质

RLHF 早期历史上的一个阶段，就是让人们相信风格有时确实有用，而不只是 brainrot：
- 早期的 RLHF 被贴上"只是风格迁移"的标签——肤浅、不重要。但风格是人类价值永不枯竭的来源，并且与信息*本身是什么*交织在一起。
- 做得好的偏好微调也能提升指标：Llama 3 Instruct 在 Arena 上的排名被广泛归因于它的个性——比同时代其他模型更简洁、更机智 [@dubey2024llama]。*考虑到后来的 Llama 4，这挺讽刺的，RIP*。
- 如果 RLHF 让模型用起来更有趣，那就是交付出的价值——与 benchmark 分数无关。

---

## 话痨平衡——许多 chat 评测很容易过拟合

偏好微调能稳定提升 LLM-as-a-judge 的 chat 评测（AlpacaEval、MT-Bench）——但这些提升并不会按比例迁移到 Arena 或真实使用中。

另一个常见现象是：

> *"然而，DPO 提升了人类偏好评测的表现，却降低了 benchmark 评测的表现。"* —— Qwen 技术报告，2023 [@qwen]

<!-- step -->

很容易以牺牲其他能力为代价来拉高话痨程度。
- 循环运行或数据充足的偏好方法，可能以数学和代码能力为代价换取 chat 表现 [@ivison2024unpacking]
- Olmo 3 发布的 checkpoint 选择了数学/代码/推理得分更高的那个，而不是 LLM 评审的 chat benchmark 得分最高的那个 [@teamolmo2025olmo3]

---

<!-- rows: 40/60 -->

<!-- cite-right: rosset2024direct -->
## 被长度刷榜的排行榜

- 2024 年 4 月：Direct Nash Optimization 报告称一个 7B 模型在 AlpacaEval 上"击败 GPT-4" [@rosset2024direct]（结果见下图）；Self-Rewarding LM 也披露了同样不切实际的分数 [@yuan2025selfrewardinglanguagemodels]。两者在真实使用中都无法与前沿模型相提并论。
- 刷榜严重到 AlpacaEval *和* WildBench 都加入了**线性长度校正**。
- 做对的例子：Starling Beta——在 OpenChat 之上使用 k-wise reward model + PPO，Arena 排名上升 10 位，且长度确实对评分者有帮助 [@zhu2024starling; @wang2023openchat]。

===

![Direct Nash Optimization 论文中的结果，声称 7B 模型在 AlpacaEval 上超过 GPT-4。Rosset et al., 2024 (CC-BY)。](assets/dno-figure.png)

---

## Llama 4 的 Chatbot Arena 特供版（2025 年 4 月）

Meta 的 Llama 4 发布（在周六）以 **Elo 1417——Chatbot Arena 第 2 名** 的 Maverick 为主打 ……靠的是"一个实验性聊天版本"。

- 排行榜上的模型**并不是发布的模型**：而是针对 Arena 投票者调优的变体——冗长、满是 emoji、始终热情洋溢的回答。同一个名字，在 LMArena 上与在其他所有提供商处的行为截然不同。
- 发布的 Maverick 是个还不错的模型，语气合理。后来真实模型被排名时，它掉到了排行榜很靠后的位置，LMArena 也因此修改了政策。

我当时写过一文：[*Llama 4: Did Meta just push the panic button?*](https://www.interconnects.ai/p/llama-4)（Interconnects，2025 年 4 月）

---

## 为什么 RLHF 会让回答变长？

- Arena 一再显示，普通用户更喜欢更长、更完整的回答 -- 它们读起来更全面、更有帮助、更值得信赖。
- 模型被训练去匹配**平均标注者** -- 例如第 8 讲中讨论的数据收集选择。
- 长度是 Goodhart 定律最爱的维度：奖励最高的表面特征，也最容易被过度优化。

---

## 做得好的话，preference tuning 也能提升能力/技能

“只是风格”这一批评已被部分超越：做得好的开源模型展现了显著的性能提升。

- 在 Tülu 3 的方案中，preference（DPO）阶段在提升聊天风格的**同时**，也相对 SFT checkpoint 提升了数学、代码和指令遵循能力 [@lambert2024t]。
- 多个开源方案都报告了 RLHF（DPO）带来的类似的全面提升 -- Tülu 3、Olmo 3、SmolLM 3 [@teamolmo2025olmo3; @bakouch2025smollm3]。

坦率的回顾：RLHF 因风格上的*失败*而背上坏名声 -- 但同样的工具，谨慎使用后，如今已是现代 post-training 的核心。

---

## 课程回顾

0. 先修知识复习
1. 概览 *（第 1-3 章）*
2. IFT、reward model 与 rejection sampling *（第 4、5、9 章）*
3. RL：动机与数学 *（第 6 章）*
4. RL：实现与实践 *（第 6 章）*
5. 推理模型的兴起 *（第 7 章）*
6. 直接偏好优化 *（第 8 章）*
7. 合成数据与现代 post-training *（第 12 章）*
8. 偏好与偏好数据 *（第 10-11 章）*
9. **过度优化与 RLHF 的坏名声** *（第 14 章、附录 B）* -- *今天*
10. **正则化工具与理解 post-training 如何改变模型** *（第 15 章）* -- *下一讲*
11. 评测 *（第 16 章）* -- *（暂定）*

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
