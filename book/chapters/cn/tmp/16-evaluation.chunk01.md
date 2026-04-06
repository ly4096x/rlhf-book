<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "正则化"
prev-url: "15-regularization"
page-title: 评估
search-title: "第16章：评估"
next-chapter: "产品与特性"
next-url: "17-product"
---

# 评估

评估是一套用于理解本书所述训练过程的质量与影响的技术体系。
评估通常通过 benchmark 来呈现（常见 benchmark 包括 MMLU、GPQA、SWE-Bench、MATH 等），这些 benchmark 是离散的问题集或环境，旨在衡量模型的某一特定属性。
评估是一种不断演进的方法，因此本章将呈现 RLHF 领域评估的近期发展阶段，以及将延续至语言模型未来发展的共同主题。
理解语言模型评估（尤其是 post-training 阶段）的关键在于：当前主流的评估体系是对主流训练实践与目标的反映。
尽管具有挑战性的评估推动语言模型向新领域进步，但大多数评估仍围绕为新模型构建有效信号而设计。

在许多方面，本章旨在呈现 RLHF 早期历史中各主流评估体系的缩影，使读者能够理解其共同主题、实施细节与失败模式。

RLHF 与 post-training 的评估在其早期历史中经历了几个明显的阶段：

1. **早期对话阶段**：早期以 RLHF 或偏好调优训练的模型，主要针对能够捕捉模型对话能力的评估，尤其是相对于 GPT-4 等已知强模型的表现。早期典型例子包括 MT-Bench [@zheng2023judging]、AlpacaEval [@dubois2024length] 和 Arena-Hard [@li2024crowdsourced]。这些 benchmark 用 LLM-as-a-judge 取代了人工评估者，使用 GPT-4 等模型对回复进行打分——这是一种以低成本扩展人工评估标准的方式（见第12章）。模型的评估范围较窄，这些评估现在被归类为"对话"或"instruction following"领域。
2. **多技能时代**：随着时间推移，业界逐渐认识到 RLHF 不仅可用于提升对话能力，还能改善更多技能。例如，Tülu 评估套件涵盖了知识（MMLU [@hendrycks2020measuring]、PopQA [@mallen2023llm_memorization]、TruthfulQA [@lin2021truthfulqa]）、推理（BigBenchHard [@suzgun2022challenging]、DROP [@dua2019drop]）、数学（MATH [@hendrycksmath2021]、GSM8K [@cobbe2021gsm8k]）、编程（HumanEval [@chen2021codex]、HumanEval+ [@evalplus]）、Instruction Following [@zhou2023instructionfollowingevaluationlargelanguage] 以及安全性（多项评估的综合）。这反映出 post-training 已被视为超越安全性与对话的多维度解决方案。
3. **推理与工具使用**：当前 post-training 的主流是聚焦于具有挑战性的推理与工具使用问题。这包括知识密集型的高难度任务，如 GPQA Diamond [@rein2023gpqa] 和 Humanity's Last Exam [@phan2025hle]；复杂的软件工程任务，如 SWE-Bench+ [@aleithan2024swebenchplus] 和 LiveCodeBench [@jain2024livecodebench]；以及以近期 AIME 竞赛为代表的高难度数学题。

在此之外，新的领域还将不断涌现。
随着 AI 日益走向产业化，评估的激励结构正在转变，并呈现出多利益相关方的特征。
自 ChatGPT 发布以来，私有评估（如 Scale Leaderboard [@scale2024seal]）、社区驱动的评估（如 Arena [@chiang2024chatbot]）以及第三方评估机构（如 ArtificialAnalysis 和 Epoch AI）相继涌现。
本章将穿插介绍这些评估的实施方式与理解视角。

## Prompting 格式：从 Few-shot 到 Zero-shot 再到 CoT

对语言模型进行 **prompting** 首先是一种操作行为，但它也被视为一种可以专门练习和/或系统性训练的技艺 [@schulhoff2024prompt]。
Prompt 是向语言模型组织信息和上下文的方式。
对于常规交互，prompt 相对简单。
对于复杂场景，精心设计的 prompt 将决定特定用例的成败。

在评估方面，prompting 技巧对模型性能有着显著影响。
某些 prompting 技巧——例如下文讨论的格式问题——可能使模型性能从60%骤降至接近0。
类似地，prompt 的改变也能帮助模型在训练过程中学习得更好。
通俗地说，对模型进行良好的 prompting 有时能带来使用未来模型的主观体验，释放出超出正常使用范围的性能表现。

使用现代语言模型进行良好的 prompting，可能需要为模型准备整篇报告作为输入（通常包含数千个 token 的生成文本）。
这种行为源于语言模型性能的衡量与理解方式发生的诸多变化。

早期语言模型仅被用作智能自动补全工具。
为了以更开放的方式使用这些模型，人们会向模型展示多个示例，再附上一段不完整的提示语。这被称为 few-shot 或 in-context learning [@brown2020language]，当时并不涉及 instruction tuning 或 RLHF。
在常见评估中，其形式如下：

```text
# Few-Shot Prompt for a Question-Answering Task
You are a helpful assistant. Below are example interactions to guide your style:

### Example 1
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

### Example 2
User: "Who wrote the novel '1984'?"
Assistant: "George Orwell wrote '1984.'"

# Now continue the conversation using the same style.
User: "Can you explain what a neural network is?"
Assistant:
```

在这里，有多种方式可以评估答案。如果我们考虑 MMLU 风格的问题（模型需要从多个选项中选择答案）：

```text
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous on a closed interval [a,b], then f must attain both a maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```

要让语言模型在此处给出答案，可以根据某些采样参数生成一个 token 并判断其是否正确（即 A、B、C 或 D，上述格式参见 [@robinson2023leveraging] 中的提案），也可以查看每个 token 的对数概率，若正确答案的概率更高则判定为正确。

让我们深入了解这些评估细节。
前者通常被称为单次尝试的精确匹配（exact match），或聚合多个样本时的多数投票（majority voting）（pass@k 是编程评估中用于测试功能正确性的类似指标），后者则称为（条件）对数似然打分（log-likelihood scoring），其中的条件即为 prompt。
两者的核心区别在于：从底层概率分布中采样会自然引入随机性，而模型对其 token 输出的对数概率是静态的（忽略微小的数值差异）。

对数似然打分有两种可能的实现方式——第一种是查看字母 (A) 的概率，第二种是查看答案"The Mean Value Theorem"的概率。
这两种都是可用的指标，但预测答案字母比预测完整的、可能包含多个 token 的答案概率要简单得多。
对数似然打分在 pretraining 评估中更为常见，因为此阶段的模型缺乏 exact match 所需的问答格式，而 exact match 则是 post-training 的标准做法 [@teamolmo2025olmo3]。
