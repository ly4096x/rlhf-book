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
Exact match 存在不同的问题，例如需要严格的格式后缀（如 `The answer is:`），或者需要在生成的文本中任意位置检测答案（例如查找 `(C)` 或答案字符串本身）。
如果 evaluation 格式与模型的生成方式不匹配，得分可能会大幅下降。
使用语言模型进行 evaluation 时，最好确保格式不成为瓶颈，从而能够充分测试模型的完整能力。
实现与格式无关的 evaluation 需要大量的努力和调试才能做好，在实践中相当罕见。

回到 evaluation 的历史。
无论使用上述哪种设置，few-shot prompting 的一个常见挑战是模型不会遵循格式，这会被计为错误答案。
在设计 evaluation 领域时，上下文中使用的示例数量通常被视为一个设计参数，范围从 3 到 8 个甚至更多。

在 few-shot prompting 的演进过程中，出现了为模型提供 chain-of-thought 示例以供其遵循的想法。
这以上下文示例中包含书面推理过程的形式呈现，如下所示（这后来被显式 prompting 生成推理步骤所取代）[@wei2022chain]：

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

A: The cafeteria had 23 apples originally. They..
```

随着时间的推移，语言模型变得越来越强大，它们逐渐演进到 zero-shot evaluation，即所谓的"zero-shot learners"[@wei2021finetuned]。
Finetuned Language Net（FLAN）表明，针对特定任务进行 fine-tuning 的语言模型——作为现代指令调整的前身——能够泛化到它们未曾训练过的 zero-shot 问题 [@wei2021finetuned]（T0 中也发现了类似结果 [@sanh2021multitask]）。
这是 instruction fine-tuning（IFT）的出现，它是 RLHF 和 post-training 的重要前驱。
一个 zero-shot 问题如下所示：

```text
User: "What is the capital of France?"
Assistant:
```

从 2022 年起，时间线开始纳入关键的早期 RLHF 工作，例如 InstructGPT。
伴随这些模型而来的核心能力和使用场景转变是更加开放性的使用方式。
随着使用方式愈加开放，从模型中采样进行 evaluation 变得越来越流行，因为这更能反映实际使用情况——从技术上讲，这可以被称为基于生成的（exact-match）evaluation，但它还没有一个清晰的规范术语。
在 ChatGPT 之后这段时期到近年来，一些多项选择 evaluation 仍然在 RLHF 研究中被使用，因为任何向通用实践的过渡都需要相当长的时间，通常需要数年才能展开（例如，这类 evaluation：将温度设置为零并采样字符 A、B、C 或 D）。

随着推理模型在 2024 年底至 2025 年初的兴起，模型行为的一个重大变化是在每个答案之前增加了长 Chain-of-Thought（CoT）推理过程。
这些模型不再需要使用 [@kojima2022large] 中提出的经典短语"think step by step"进行 prompting。
evaluation 实践的下一次演进是基于生成的（exact-match）evaluation，结合 chain of thought 推理（因此几乎总是使用大于零的温度以获得最佳性能）。

例如，在某些设置中，针对每个问题或类别都设计了专门的 prompts 来帮助从模型中提取行为。
Tülu 3 是一篇早期具有里程碑意义的论文，详细介绍了用于在多项选择题上进行 CoT 答题的一些 prompts [@lambert2024t]。
以下是用于 MMLU 的示例 prompt，MMLU 是从单 token 答案采样过渡到带有 exact match 答案检查的长形式 CoT 的 evaluation 之一。

```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

这一点，尤其是当模型使用特殊格式将思考 tokens 与答案 tokens 分开时，促使了 evaluation 体系最近一次重大更新的出现。
Evaluation 正在向测试模型以生成方式响应并结合 chain-of-thought prompting 的方向发展。

## 为何许多外部 Evaluation 比较不可靠

AI 公司在模型发布公告中进行的语言模型 evaluation，只能与其他新闻稿在较大误差范围内进行比较——即稍好或稍差的模型应被视为等效——因为各家公司内部使用的 evaluation 流程在不同模型之间并不统一，也没有明确的文档记录。
例如，在 Olmo 3 项目中，作者发现在推理模型时代，大多数 post-training evaluation 在 evaluation 设置保持不变的情况下，标准差在 0.25 到 1.5 分之间 [@teamolmo2025olmo3]——使用不同的 prompts 或采样参数可能带来更大的分数变化。
各实验室在训练过程中对 evaluation 进行 hillclimb，以使模型更加实用，传统上使用训练集、开发集（即验证集）和保留的 evaluation 集（即测试集）的混合。
Hillclimbing 是一个口语化术语，用于描述使模型在一组目标 benchmarks 上逐步提升的实践。
对于社区用于比较领先模型的公开 evaluation，无法知晓哪些被用于训练，哪些被保留用于测试。

随着 evaluation 分数成为企业营销方案的核心组成部分，各公司内部的实现方式已经出现偏差。
有传言称，主要 AI 实验室为 GSM8k 或 MATH 等重要 evaluation 使用"自定义 prompts"。
这些实践演变迅速。

语言模型 evaluation 栈被视为营销工具，因为这些 evaluation 没有硬性的事实来源。
前沿实验室内部正在发生的是 evaluation 套件被调整以适应其内部需求。
当结果被共享时，我们得到的是实验室为其模型获得的数字形式的输出，但并非该函数的所有输入。
这些输入是非常敏感的配置，在 OpenAI、Meta、Anthropic 和 Google 之间各不相同。
即使是完全开放的 evaluation 标准也难以保证可重现性。
专注于自己的模型是获得接近可重复 evaluation 技术的唯一途径。
支撑这些营销行为的背后，从技术团队开始，存在着良好的初衷。

另一个在比较多个实验室 evaluation 时产生混淆的例子，是在 evaluation 比较中加入推理时扩展（inference-time scaling）。
推理时扩展表明，模型可以通过在推理时使用更多 tokens 来提升性能。
因此，通过推理时使用的 token 总数来控制 evaluation 分数非常重要，但目前还不是通行做法。

根据 post-training 中数据的格式化方式，模型在不同 evaluation 格式之间会存在显著差异。
例如，两个流行的开放数学数据集 NuminaMath [@li2024numinamath] 和 MetaMath [@yu2023metamath] 在训练中因答案格式的细微差异而相互冲突——Numina 将答案放在 `\boxed{XYZ}` 中，而 MetaMath 将答案放在 `The answer is: XYZ` 之后——同时在两者上训练可能导致性能比仅使用其中一个更差。
强大的模型被训练为能够适应多种格式，但它们通常有一种最擅长的格式。

最终，关于评估闭源模型的现状，我们得出以下几个关键要点：

- 我们不知道、也不一定拥有实验室正在攀升的关键测试集，因此某些 evaluation 只是代理指标。
- 前沿模型的推理正变得愈发复杂，涉及特殊系统 prompts、特殊 tokens 等，我们不知道这如何影响 evaluation，以及
- 我们不知道用于数字化报告闭源 evaluation 的所有格式和细节。

所有这些动态，加上过去几年 AI 模型的极速进步，产生了类似 @fig:benchmark-saturation 中那样的著名图表，其中每个时代流行的 benchmarks 都被非常迅速地解决。
描述这种在单个 benchmark 层面动态的常用术语是饱和（saturation）。
随着每个 benchmark 接近 100%，模型的进步开始放缓，因为只剩下更难（或者在许多情况下是标注错误的）数据点，这使其作为训练进度衡量指标（或两个模型之间比较）的可靠性降低。

![Epoch AI 的报告显示主要 AI evaluation 如何随时间迅速饱和（饱和是指给定 benchmark 达到完整性能，模型不再具有有意义的信号）。许可证 CC-BY。](images/benchmark-performance.jpeg){#fig:benchmark-saturation}

## 实验室如何在内部实际使用 Evaluation 来改进模型

对前沿语言模型的 evaluation，在今天既是一门艺术，也是一门科学，准确规定不同团队如何使用 evaluation 是不可能的。

不同团队选择不同的 evaluation 来保持独立性，即将其作为真正的测试集，但没有人披露他们选择了哪些。
例如，流行的推理 evaluation MATH 和 GSM8k 都有训练集，其中的 prompts 可以很容易地用来提高性能。
使用来自同一分布的 prompts 来提高性能，与通过训练通用数学数据来泛化到这些任务，是非常不同的。
事实上，这些*训练集*包含非常高质量的数据，因此模型可以从在其上进行训练中获益。
如果这些公司*并未*将相应的 evaluation 作为核心指标来追踪，那么在 evaluation 集上进行训练可能是一个实际可行的决策，因为高质量数据是模型开发的主要限制因素。

领先的 AI 实验室通过专注于少数几个关键 evaluation 来进行逐步攀升，并在最后报告核心公开集上的分数。
关键在于，他们用于追踪进展的部分 evaluation（例如 GPT-4 报告 [@achiam2023gpt] 中用于扩展规模的交叉熵损失预测数据集）通常并不公开。

post-training 的 evaluation 与人工评估高度相互依赖。
对于生成式语言模型的人工评估会产生 Elo 排名（在 Anthropic 早期论文如 Constitutional AI 中很流行），而对 reward model 的人工评估则反映一致性程度。
这些也可以通过在 A/B 测试窗口中向用户提供两个不同模型来获得（如[偏好数据章节](https://rlhfbook.com/c/11-preference-data)中所讨论的）。

他们选择关注的有限 evaluation 集合在 evaluation 与训练之间形成了紧密联系。
曾经有一段时间，MMLU 是重点关注的 evaluation 之一。
GPQA 在推理模型兴起期间极为流行，这得益于社区对科学能力的日益关注。
各实验室会修改 evaluation 以使其更适合自身需求，例如 OpenAI 发布了 SWE-Bench-Verified [@openai2024swebench]。
此外，每个前沿实验室还构建或购置了许多公众无法获取的内部 evaluation。

在内部改进 evaluation 对下游训练的关键影响在于**提升比较训练运行时的统计功效**。
通过更换 evaluation，这些实验室降低了其优先信号上的噪声，从而能够做出更明智的训练决策。

这一情况因现代语言模型训练栈中 post-training 的复杂性而进一步加剧。
如今对语言模型进行 evaluation 涉及相当数量的 token 生成（而非仅仅查看答案的对数概率），因此需要一定的计算开销。
普遍认为，前沿实验室会使用一些小技巧来提升许多任务上的性能——最常见的解释是针对特定 evaluation 使用一次性的 prompting。

## Contamination

当前语言模型实践中（即不仅限于 RLHF 和 post-training）的一个重大问题，是有意或无意地将 evaluation 数据集中的数据用于训练。
这被称为*数据集污染（dataset contamination）*，相应地，避免污染的实践称为*去污染（decontamination）*。
为了对数据集进行去污染，需要对训练集和测试集进行搜索，通过词/子词 token 上的 n-gram 重叠，或固定长度的字符子串匹配（例如 50 个字符）来查找匹配项 [@singh2024evaluation]。
数据被污染的方式有很多，但最常见的是在多个阶段从网络抓取训练数据时引入的。
benchmark 通常发布在被爬取的公开网页上，或者用户将问题输入给模型，这些内容随后可能出现在未来模型的候选训练数据中。

例如，在为 Tülu 3 进行 evaluation 套件去污染时，作者发现流行的开放数据集被 RLHF 的常用 evaluation 所污染 [@lambert2024t]。
这些重叠包括：UltraFeedback 对 TruthfulQA 的污染、Evol-CodeAlpaca 对 HumanEval 的污染、NuminaMath 对 MATH 的污染，以及 WildChat 对安全 evaluation 的污染。
这些污染是通过训练 prompt 与 evaluation 集中精确 prompt 之间的 8-gram 重叠发现的。

在其他情况下，模型被发现曾在与 benchmark 非常接近的数据上进行训练，例如保持数学题的文字不变而只修改数字，这可能导致 post-training 阶段出现异常行为，例如当模型在随机奖励上使用 RL 训练时 benchmark 却有所提升——这是一种人为设定，只有当模型存在某些类型的数据污染时，性能才应当提升。
这种 base model 污染（无法确切证明模型行为为何如此）已成为许多早期基于 Qwen 2.5 和 Qwen 3 base model 的 RLVR 工作中一个重要的干扰变量 [@shao2025spurious] [@wu2025reasoning]。

为了理解不披露或不发布训练数据的模型的污染情况，研究者会创建新版本的 benchmark，其中包含对原题稍加改动的问题（例如针对 MATH 的改动 [@huang2025math]），以此观察哪些模型是针对原始格式或题目进行训练的。
这些扰动 benchmark 上的高方差并不能证实污染，这一点难以证明，但可能表明某些模型是针对特定格式训练的，这种训练未必能转化为真实世界的性能。


## Tooling

目前有许多开源 evaluation 工具可供选择。
其中包括：

- 来自英国安全研究所的 Inspect AI [@inspectAI2024]，
- HuggingFace 的 LightEval [@fourrier2023lighteval]，为 Open LLM Leaderboard [@open-llm-leaderboard-v2] 提供支持，
- Eleuther AI 基于其 GPT-Neo-X 模型基础设施构建的 evaluation harness [@gao2023evalharness]（包含良好的 GPT-3 时代 evaluation 设置和配置）[@gpt-neox-20b]，
- AI2 基于 OLMES 构建的库 [@gu2024olmes]，
- Stanford 基础模型研究中心的 HELM [@liang2023helm]，
