---
title: "第 12 讲：前沿模型评测的演进"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 12 讲"
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
  .slide--title-sidebar h1 .title-subtitle {
    display: block;
    margin-top: 0.6em;
    font-size: 0.45em;
    font-weight: 400;
    letter-spacing: 0;
    opacity: 0.75;
  }
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* Vertically center every content slide by default (special layouts and
     explicit valign-top/bottom directives still win). */
  .slide.active:not([class*="slide--"]):not(.valign-top):not(.valign-bottom) {
    justify-content: center;
  }
  /* Verbatim prompt dumps: compact monospace so full prompts fit a slide. */
  .slide pre {
    font-size: 0.52em;
    line-height: 1.38;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 12 讲：前沿模型评测的演进 <span class="title-subtitle">从 few-shot prompting 到 agentic sandbox</span>

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 16 章。</p>

---

## 衡量进展的方式会随时间而变

不同类型的模型需要不同的评测策略。

模型越来越聪明，我们就需要不断重新发明 AI 的最前沿。

评测是 AI 进步的核心，也是我们认清模型真实面貌的关键。

---

<!-- columns: 48/52 -->
## 前沿模型评测比以往任何时候都更难

Opus 4.6 和 GPT-5.3-Codex 在同一周发布（2026 年 2 月），benchmark 上的领先幅度微乎其微——根本无法决定该用哪个模型。

真正拉开差距的是在使用中才发现的东西：易用性、产品契合度、长时间 agentic 任务中的行为。在前沿水平上，基于 benchmark 的发布反响几乎无关紧要——必须靠持续的测试和清晰的表述来完成比较。

在 Interconnects 上阅读：[后 benchmark 时代](https://www.interconnects.ai/p/opus-46-vs-codex-53)

|||

![](assets/interconnects-post-benchmark-era.png)

---

<!-- columns: 38/62 -->
<!-- cite-right: kwa2025measuring -->
## ……而且我们想衡量的任务耗时越来越长

前沿模型能够完成的任务长度（成功率 50%）**大约每 7 个月翻一番**——从几秒钟的问题，到人类专家需要数小时的任务（[Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)）。

如今衡量前沿水平，意味着要把耗时数小时的专家级任务反复运行多次。

|||

![前沿模型的任务完成 time horizon。图来自 METR（Time Horizon 1.1），CC-BY。](assets/metr-time-horizon-11.png)

---

<!-- columns: 45/55 -->
## 本讲内容

本课程中的每一项训练决策——数据配比、超参数、哪个 checkpoint 上线——都是依据 benchmark 数字做出的。

今天：我们如何走到现代的 benchmark 方法与研究。

|||

```box
title: 本讲安排
tone: accent
content: |
  1. **各个时代** -- 每种模型是如何被提示、评分和做 benchmark 的
  2. **再谈 agentic evals** -- 今天的分数背后的整个系统
  3. **能信这个数字吗** -- 方差、污染与刷分
```

---

<!-- layout: section-break -->
<!-- align: center -->

## 第一部分：post-training 评测的各个时代

---

<!-- animate: bullets -->
## Benchmark 反映各个时代的训练目标

理解 evals 的关键：热门 benchmark 是**其所处时代 post-training 最佳实践的映射**。

- **Chat 时代** *(2022-23)* -- 基础知识与对话风格
- **多技能时代** *(2023-24)* -- post-training 提升的不再只是对话，还有更多技能（数学、代码、事实性、安全等）
- **推理与工具时代** *(2024-26)* -- 高难度的数学、编程与推理问题，长 chain of thought
- **Agent 与真实工作** *(当下)* -- 在产品和 harness 中完成端到端的知识工作任务

---

<!-- rows: 20/80 -->
<!-- cite-right: brown2020language, robinson2023leveraging -->
## Base model（post-training 之前）：few-shot prompting

Base model 无法直接回答裸问题或格式化问题 -- 因此 eval prompt 会附带模式示例（3 到 8 个以上的 in-context 样本），让模型接着续写。经典 evals：**5-shot MMLU**、**8-shot GSM8K**。

===

<!-- row-columns: 50/50 -->

```text
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4.
What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)
```

|||

```text
### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous
on a closed interval [a,b], then f must attain both a
maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```
---

<!-- columns: 50/50 -->
<!-- cite-right: brown2020language, teamolmo2025olmo3 -->
## 评分方式：log-likelihood 与 exact match

**Log-likelihood scoring**：比较模型赋予每个答案选项的概率 -- 可以只看选项字母 `A`，也可以看完整的答案字符串。无需采样，完全确定。这是 pretraining evals 的标准做法，因为此时的模型往往无法以规整的格式作答。

|||

<!-- step -->

**Generation + exact match**：采样一个 completion，再提取答案。更贴近真实使用场景 -- 自 2024 年前后起成为 post-training 的标准做法。聚合多个 completion/样本即得到 majority voting；例如 **pass@k** 就是常用工具。

生成与答案提取带来了答案提取格式的 bug，而在如今的 agentic model 上，这类问题只会更加复杂。

---

<!-- columns: 46/54 -->
<!-- cite-right: chen2021codex -->
## pass@k 背后的数学

pass@k = **$k$ 个样本中至少有一个**解出问题的概率。

朴素做法 -- 恰好生成 $k$ 个，记录其中是否有通过的 -- 对每个问题来说就像一次**高方差**的抛硬币，而把小样本的成功率代入 $1-(1-\hat{p})^k$ 则是**有偏**的。

Codex 论文给出的修正：采样 $n \geq k$ 个 completion，统计通过的个数 $c$，再对所有问题平均一个无偏估计量。

|||

<!-- step -->

$$\text{pass@}k = \mathop{\mathbb{E}}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

- $\binom{n-c}{k}\big/\binom{n}{k}$ 是从 $n$ 个样本中（不放回地）抽取 $k$ 个、且**全部失败**的概率
- $n$ 越大 → 在同一 $k$ 下估计越紧；论文在 $k \leq 100$ 时使用 $n=200$
- 各项参数相互影响：**更高的 temperature 可能损害 pass@1，却有助于 pass@100** -- 因此所报告的 "pass@1" 取决于 $n$ 和采样设置，而不仅仅是模型本身

---

<!-- columns: 40/60 -->
<!-- footnote-right: 幻灯片来源：Florian Brand -->
## 早期的 pipeline 很简单

输入 prompt，输出 completion，然后打分。 
几乎所有可能出错的地方，都在于**如何格式化 prompt**。

|||

![](assets/eval-system-v1.png)

---

<!-- columns: 38/62 -->
<!-- cite-right: wei2022chain, kojima2022large -->
## 思维链（CoT）的出现推动了更难问题上的进展

展示中间步骤的 few-shot 示例，能让模型在回答之前先进行推理。
当人们还在用 prompt 驱动 base model 时，加入 CoT 让数学和推理得分大幅跃升！
而这还是在现代 post-training 出现之前。

很快，只需在 prompt 末尾加上 *"Let's think step by step"*，就能近似实现这种行为。

|||

```text
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain-of-thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They...
```

---

## 进入聊天时代：zero-shot 指令遵循

指令调优（FLAN [@wei2021finetuned]、T0 [@sanh2021multitask]）以及随后的 RLHF，改变了人们使用模型的预期：模型学会了直接回答问题。回过头看，这是件大事！但当时并非默认做法。

现在，模型的输入可以是这样的：
```text
User: "What is the capital of France?"
Assistant:
```

<!-- step -->

- **LLM-as-a-judge** 随着问题变得开放式而出现（评估也在模仿 RLHF 训练）
- 经典评估：**MT-Bench** [@zheng2023judging]、**AlpacaEval** [@dubois2024length]，以及社区规模的 [Chatbot Arena](https://lmarena.ai/) [@chiang2024chatbot]
- MMLU 等 MCQ 评估依然保留（但处于变动中，人们的用法各不相同）——现在以 zero-shot 方式作答，在温度为 0 下采样答案字母

---

<!-- columns: 55/45 -->
<!-- cite-right: lambert2024t, hendrycks2020measuring -->
## zero-shot prompting 的兴起花了不少时间！

我们训练 Tülu 3（2024 年夏）时，许多评估是 zero-shot 与 few-shot prompting 的混合。不过，当时整个领域关注的是多种能力（来自 Tülu 3）：

- **知识**：MMLU, PopQA, TruthfulQA
- **推理**：BigBenchHard, DROP
- **数学与代码**：MATH, GSM8K, HumanEval
- **指令遵循与安全**：IFEval 等

到 2025 年初，所有人都在使用 zero-shot prompting；如今，对 post-trained 模型使用 multi-shot prompting 已十分罕见，除非该模型并未针对此进行训练（真正的 in-context learning）。

|||

```box
title: 示例题目（MMLU）
content: |
  What was GDP per capita in the United States in 1850 when adjusting for inflation and PPP in 2011 prices?

  A. About $300  
  B. About $3k  
  C. About $8k  
  D. About $15k
```

与其说是智力，不如说是互联网冷知识——但它很好地反映了 pretraining 的知识水平。

---

<!-- columns: 40/60 -->
<!-- cite-right: lambert2024t -->
## 鼓励模型进行推理

我们知道，如今的 reasoning model 会在回答前先思考。Tülu 3 的 MMLU prompt（reasoning model 出现之前的模型）→ 与 few-shot 时代相同的 MMLU 评估，现在改为长篇 CoT 并采用 exact-match 检查。

现代评估套件可以带有**针对各 benchmark 的 prompt**，用于调整格式等。 

|||

Tülu 3 的 MMLU prompt：
```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

---

<!-- columns: 40/60 -->
<!-- cite-right: lambert2024t -->
## 鼓励模型进行推理

我们知道，如今的 reasoning model 会在回答前先思考。Tülu 3 的 MMLU prompt（reasoning model 出现之前的模型）→ 与 few-shot 时代相同的 MMLU 评估，现在改为长篇 CoT 并采用 exact-match 检查。

现代评估套件可以带有**针对各 benchmark 的 prompt**，用于调整格式等。 

|||

采样设置也成为评估 prompt 的一部分：reasoning model 需要 **temperature > 0** 才能取得最佳得分——[Qwen 的 model card](https://huggingface.co/Qwen/Qwen3-32B) 直接写明 **"DO NOT use greedy decoding"**。请阅读 `generation_config.json`：推荐设置是"免费"的性能提升。


---

## 随着使用方式越来越开放，格式变得脆弱

- 格式不匹配可以让模型从 **60% 掉到接近 0** [@schulhoff2024prompt] -- 用 prompt 损失性能远比提升性能容易
- 答案提取很脆弱：死板的后缀（*"The answer is:"*），或在文本任意位置搜索答案的正则表达式
- 不同训练集的格式甚至互相冲突：NuminaMath [@li2024numinamath] 要求 `\boxed{42}`，MetaMath [@yu2023metamath] 要求 `The answer is: 42` -- **两者一起训练可能比单独用任何一个更差**
- 与格式无关的评分需要大量精力和反复调试，实践中往往很少见 -- LLM-judge 因其灵活性，甚至被广泛用作答案提取器

---

## Reasoning 与 tool-use 推动业界转向更难的任务

Reasoning 模型使旧的 eval 套件饱和，于是下一代 eval 出现了：

- **知识**：GPQA Diamond [@rein2023gpqa]、Humanity's Last Exam [@phan2025hle]、FrontierMath
- **数学**：近期的 AIME 竞赛
- **软件**：SWE-Bench（及其变体）、LiveCodeBench [@jain2024livecodebench]
- 题目来源从互联网转向**研究生、博士和教授** -- 出题成了专家劳动

---

<!-- columns: 50/50 -->
<!-- cite-right: phan2025hle -->
## Reasoning 与 tool-use 推动业界转向更难的任务

```box
title: "示例：Humanity's Last Exam"
content: |
  What was the rarest noble gas on Earth as a percentage of all terrestrial matter in 2002?

  Official answer: **Oganesson**
```

|||

<!-- step -->

这个官方答案在三个方面都是错的：oganesson **不是气体**（预测为固体）、**不是惰性元素**（预测具有反应活性）、也**不是地球上天然存在的**（人工合成，迄今只造出约 5 个原子 -- 2002 年首次合成）。

<!-- step -->

难度超过一定程度后，**核对标准答案成了瓶颈** -- 专家撰写不再等于正确。

即便如此，eval 中的错误标签仍是常见问题！大多数 eval 饱和在 90-95%，而不是 100%。

---

<!-- cite-right: openai2024swebench -->
## 今天：真实工作的 eval

- 前沿 eval 是**端到端的专业任务**：SWE-bench Verified、Terminal-Bench、[GDPVal](https://openai.com/index/gdpval/)、[APEX](https://arxiv.org/abs/2601.14242)
- 任务作者现在是**经验丰富的专业人士**：GDPVal 的任务来自平均拥有 **14 年**行业经验的专家；APEX 的专家在 Goldman、McKinsey 等公司平均有 7 年以上经验 -- 专家出题成了新的成本中心
- 而且模型不是"裸"评测的：它们运行在 **harness 和产品之中**（Claude Code、Codex CLI）-- 这正是上一讲的主题（11 - tool-use）

---

<!-- columns: 38/62 -->
## 每个时代都以同样的方式收场：饱和

Benchmark 是消耗品。随着分数逼近上限，只剩下最难（以及标注错误）的题目，benchmark 也就不再能区分模型。

|||

![主要 AI benchmark 随时间逐渐饱和。图源：Epoch AI，CC-BY。](assets/benchmark-performance.jpeg)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 2 部分：再谈 agentic eval

---

<!-- columns: 34/64 -->
<!-- cite-right: tbench2026 -->
<!-- footnote-right: 幻灯片来源：Florian Brand -->
## Agentic pipeline：模型只是其中一环

**Harness** -- 围绕模型的 prompt、工具和上下文管理循环 -- 运行在 **sandbox** 中：一个可复现的世界，包含任务所需的文件、工具和规则。

再加上硬件和超时设置，长达数小时的 trajectory 由正则表达式或 LLM judge 来评分。

|||

![](assets/eval-system-v3.png)

---

<!-- animate: bullets -->
<!-- footnote-right: 幻灯片来源：Florian Brand -->
## Harness 至关重要

- 前沿模型是**在自己的 harness 中训练的** -- 换一个 harness 评测会低估其能力
- 极端案例，来自 [ARC-AGI-3 报告](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)：在某个环境中，Opus 4.6 没有 harness 时得分 **0%**，使用手工打造的 harness 则达到 **97.1%**
- 这就是为什么"同一个模型、不同的 agent 产品"会得到差异巨大的分数

---

<!-- footnote-right: 幻灯片来源：Florian Brand -->
## 许多细小因素共同影响 agentic 分数

第 23 页方框中的每一个实现细节，都是某个人选定的旋钮（通常没有文档记录） -- 两个实验室运行"同一个 benchmark"，测到的东西可能有实质差异

- **inference engine 可能不同**：[vLLM 的复盘](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy)讲了部署 Kimi K2 的经历 -- 三个引擎 bug 使 tool-call 成功率**低于 20%**；修复后达到 **99.9%**。权重完全相同。很多团队都有这类问题
- **硬件层面的差异**：不同 GPU 之间的方差确实存在，而且令人头疼。有些 benchmark 会刻意测量这种方差（KernelBench 类任务需要特定 GPU）。在大规模 sandbox 下，一个有问题的 GPU actor 就可能卡住整个系统并拖垮 eval（见下文）
- **在高难度任务上，超时设置至关重要**：模型在难题上经常会超时 -- Terminal-Bench 2 用 3-5 倍长的超时重跑，使 GPT-5.2 提高了 [+6 到 +15 分](https://github.com/xdotli/gpt-5.2-tb2)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 3 部分：最终的 eval 数字可信吗？

---

<!-- columns: 55/45 -->
<!-- cite-right: teamolmo2025olmo3 -->
## 评测方差无处不在

在 temperature > 0 下采样意味着**对同一个模型重复运行同一个 eval，分数也会变动**。

在 Olmo 3 期间我们实测了：14 个模型、3 次运行的标准差，按 benchmark 统计 →

大多数推理时代的 evals 有 **0.25 到 1.5 分的噪声** -- 而这还是在没人改动 prompt 或采样设置之前。

agentic evals 的噪声可能更高！

更多内容见[附录 C：评测方差](https://rlhfbook.com/c/appendix-c-practical#evaluation-variance)。

|||

| | Benchmark | σ |
|---|-----------|-----|
| 高方差 | GPQA | 1.48 |
| | AlpacaEval 3 | 1.24 |
| | IFEval | 0.88 |
| 稳定 | ZebraLogic | 0.56 |
| | AIME 24 (avg@32) | 0.54 |
| 非常稳定 | LiveCodeBench (avg@10) | 0.29 |
| | MATH | 0.25 |
| | MMLU | 0.22 |

---

<!-- cite-right: teamolmo2025olmo3 -->
## 管理评测噪声

要管理评测噪声，需要做很多事情（有时这几乎就是全部工作）。例如：

- **用 avg@k 而不是 pass@k**：例如在 Tülu 3 中，LiveCodeBench 既有噪声*又*便宜 -- 重复运行 10 次后，它从高方差变成了非常稳定。这对任何 eval 都适用，但成本也相应成比例增加。
- 方差还会从基础设施渗入：**batch size、tensor-parallel 设置、长文本生成的数值精度**（见前面关于 infra 的幻灯片）
- 实用经验法则：两篇新闻稿之间 **约 1 分的差距就是噪声**

---

## 为什么各实验室之间的横向比较不可靠

这是我开始对公司发布的、与竞品对比的公告有些警惕的原因。OpenAI 和 Anthropic 相当公允，但很多公司在对比上更可疑。

- 每个实验室的 eval 体系都**针对自身内部需求调优**：关键 benchmark 的自定义 prompt、未公开的格式、不同的推理引擎
- 我们看到的是一个有时相当复杂的函数的输出
- 没人披露哪些公开 benchmark 是**留出 (held out) 的，哪些是被 hillclimb 的** -- 从外部完全看不到 train/dev/test 的规范性
- 推理时扩展 (inference-time scaling) 会混淆一切：更多 token 就能换来更高分数，而 token 预算很少被控制

---

<!-- footnote-right: 来源：[Artificial Analysis](https://artificialanalysis.ai/models) -->
## 当今的成本-性能 Pareto 前沿


![Intelligence Index 与单任务成本的对比，并绘出 Pareto 前沿。图来自 Artificial Analysis。](assets/aa-intelligence-cost-pareto.png)

---

## 评测在实验室内部究竟用来做什么

对 evals 来说，作为训练目标的意义大于作为监控视角。

- 各实验室会针对约 50 个以上的优先 evals 做 hillclimb，最后报告公开的评测集（只是子集，甚至可能不是训练目标）
- 好的内部 eval 真正的产出是**在模型之间做决策所需的统计功效**
- 有时“测试集”其实就是好数据（几年前有传闻，现在听到得少了）：MATH 和 GSM8K 的训练集质量很高，在某个阶段至关重要 -- 如果某个实验室不追踪该 eval，用它们来训练就是理性的选择
- 对于 benchmark 无法衡量的部分，人工 A/B 测试和 Elo 仍在回路中（回顾第 8 讲、其他关于 RLHF 的工作、Arena 等。依然重要！）

---

## 污染：在测试集上训练是有意为之吗？

有一个长期存在的研究领域，旨在弄清训练数据是有意还是无意地提高了分数。

- **去污染 (Decontamination)** = 在训练集和测试集之间做 n-gram / 子串搜索，以去除重叠，避免 eval 分数来自记忆而非泛化 [@singh2024evaluation]
- Tülu 3 发现一些流行的开源数据集存在污染（训练集 X 评测集）：UltraFeedback×TruthfulQA、Evol-CodeAlpaca×HumanEval、NuminaMath×MATH [@lambert2024t]
- 一个关于污染的微妙信号：用**随机奖励**做 RL 竟然能提升 Qwen 的 benchmark 成绩 [@shao2025spurious] -- 只能用基础模型中存在污染来解释；这是早期 RLVR 研究中的真实混淆因素（[博客文章](https://www.interconnects.ai/p/reinforcement-learning-with-random)）
- 生态系统的应对：改写扰动版 benchmark（同样的问题，换上新数字），以抓出在原题上训练过的模型 [@huang2025math]；或者使用私有排行榜/测试集（没法作弊！）

---

<!-- rows: 15/85 -->
## 如今的模型热衷 reward hack，还会钻 evals 的空子

Agent 热爱最大化 reward（这会演变成走捷径）。[NIST](https://www.nist.gov/caisi/cheating-ai-agent-evaluations) 和 [DebugML](https://debugml.github.io/cheating-agents/) 已记录了这些现实中的案例。当然，还有 OpenAI 某模型造成的 Hugging Face [入侵事件](https://huggingface.co/blog/security-incident-july-2026)。

===

<!-- row-columns: 50/50 -->

**观察到的手法：**

- 在 git 历史中挖掘**修复该 bug 的未来 commit** -- 一个开源模型在其 **24% 的 SWE-bench trajectory** 中这样做
- 通过**镜像站、网页存档和包注册中心**绕过 URL 黑名单
- 把**预期的测试输出硬编码**进代码（而不是修改代码）
- 滥用测试运行器的怪癖

|||

**防御措施：**

- 移除一切非必要的访问权限
- 为验证和测试运行提供**第二个独立的沙箱**
- 用第二个 LLM 监控第一个（成本高）

---

<!-- columns: 50/50 -->
## 工具：运行你自己的研究级 evals

**成熟的评测框架：**

- [Inspect](https://inspect.aisi.org.uk/) `★ 2.5k` -- 英国 AI Security Institute
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) `★ 13.5k` -- 自 GPT-3 时代起 EleutherAI 的标准
- [LightEval](https://github.com/huggingface/lighteval) `★ 2.5k` -- Hugging Face；驱动了 Open LLM Leaderboard
- [OLMES](https://github.com/allenai/olmes) `★ 390` -- Ai2 的可复现评测系统
- [HELM](https://crfm.stanford.edu/helm/) `★ 2.9k` -- Stanford CRFM
- [Eval Gauntlet](https://github.com/mosaicml/llm-foundry) `★ 4.4k` -- Mosaic，现属 Databricks

|||

**环境浪潮：**

- [olmo-eval](https://github.com/allenai/olmo-eval) `★ 64 (new)` -- Ai2 面向*模型开发回路内部*的 evals 工作台（2026）
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) `★ 4.4k` + [Environments Hub](https://www.primeintellect.ai/blog/environments) -- Prime Intellect 的库与社区 hub：2,500+ 个可验证环境，同时可作为 evals（AIME、Terminal-Bench 等）
- 发展方向：**训练和评测共用同一套环境代码** -- 写一次，既用来 hillclimb，也用来衡量

---

## 要点总结

- Benchmark 反映了**其所处时代的训练目标**——而每个时代都以饱和告终。
- 分数是**整个系统的属性**：prompt、sampling、engine、harness、sandbox、硬件、grader。模型只是其中一环。
- 预计会有 **±1 分的纯噪声**；跨实验室的比较至多只能作为方向性参考。
- Contamination、gaming 等都会扭曲单一数字——对于重要的决策，请**运行你自己的 evals**并掌控整个系统。
- 完整评测成本高昂！对现代 agentic 套件做深入评测，花费可能 **>$100K** *（来自 Florian Brand）*

---

<!-- columns: 50/50 -->
## 从这里继续

下面两个演讲对本讲的所有内容都有更深入的讲解：

- 我在 NeurIPS 上关于 evals 的演讲，穿插在 Olmo 3 的故事中：[幻灯片](https://docs.google.com/presentation/d/10DT7MM8vbKFrxPiwAfcyvea0V4d6XpFBhe6_Pmneu2A/edit) · [录像](https://youtu.be/uaZ3yRdYg8A?t=2465) *（evals 部分从 41:05 开始）*
- Florian Brand 的 [LLM benchmarks in the time of agents](https://www.youtube.com/watch?v=kmTMc-fVSXw&t=870s) *（从 14:30 开始）*——本讲 agentic evals 部分的灵感来源之一

|||

![](assets/neurips-evals-talk-title.png)


---

<!-- columns: 50/50 -->
## 课程回顾

0. 先修知识复习
1. 概览 *（第 1-3 章）*
2. IFT、Reward Model 与 Rejection Sampling *（第 4、5、9 章）*
3. RL：动机与数学 *（第 6 章）*
4. RL：实现与实践 *（第 6 章）*
5. Reasoning 模型的兴起 *（第 7 章）*
6. Direct Preference Optimization *（第 8 章）*
7. 合成数据与现代 post-training *（第 12 章）*

|||

8. 偏好与偏好数据 *（第 10-11 章）*
9. Over-Optimization 与 RLHF 的坏名声 *（第 14 章，附录 B）*
10. 正则化工具与理解 post-training 如何改变模型 *（第 15 章）*
11. Tool use、Function Calling 与通往 Agent 之路 *（第 13 章）*
12. **评测** *（第 16 章，附录 C）*——*今天*
13. 塑造模型性格与产品 *（第 17 章）*——*下一讲*

---

<!-- rows: 85/15 -->
## 谢谢

提问 / 讨论

联系方式：nathan@natolambert.com

Newsletter：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

```builtwith
repo: natolambert/colloquium
```
