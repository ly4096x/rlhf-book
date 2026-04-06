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
