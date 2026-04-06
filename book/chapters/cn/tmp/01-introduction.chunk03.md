这些早期模型的benchmark完全依赖于主观感受（以及人工评估），因为我们都被这些小模型能够在各个领域展现出如此令人印象深刻的行为这一事实所深深吸引。
这种兴奋是有其道理的。

开放式post-training的发展速度更快，发布的模型更多，也比封闭的同类产品引发了更多关注。
各家公司都在争先恐后，例如 DeepMind 与 Google 合并，或者新公司刚刚成立，需要时间来跟进。
开放式技术方案经历了一轮又一轮的爆发与滞后交替出现的阶段。

继 Alpaca 等工作之后——即开放式技术方案的第一次滞后期——那个时代的特征是对 reinforcement learning from human feedback（RLHF）的怀疑与质疑。RLHF 是 OpenAI 强调的对第一代 ChatGPT 成功至关重要的技术。
许多公司怀疑自己是否需要做 RLHF。
一句流行的话——"instruction tuning 足以实现对齐"——在当时十分盛行，尽管已有大量明显的反例，这种观点至今仍有相当影响力。

对 RLHF 的质疑持续了相当长的时间，在开放社区尤为如此，因为这些团队无力承担数十万至数百万美元量级的数据预算。
那些早期拥抱 RLHF 的公司最终脱颖而出。
Anthropic 在 2022 年发表了大量关于 RLHF 的研究，如今被认为拥有最好的 post-training [@askell2021general] [@bai2022training] [@bai2022constitutional]。
开放团队与封闭团队之间的差距——无论是在复现能力上，还是在了解基本封闭技术上——是一个反复出现的主题。

开放对齐方法和 post-training 领域的第一次重大转变，是 Direct Preference Optimization（DPO）[@rafailov2024direct] 的故事。该工作表明，通过直接对成对偏好数据进行梯度更新，可以用更少的组件来求解与 RLHF 相同的优化问题。
DPO 论文于 2023 年 5 月发布，但在 2023 年秋季之前，并没有任何明显影响力的模型是用它训练的。
随后，几个突破性的 DPO 模型相继发布，情况发生了改变——这些突破的关键在于找到了一个更好、更低的 learning rate。
Zephyr-Beta [@tunstall2023zephyr]、Tülu 2 [@ivison2023camels] 以及许多其他模型都表明，post-training 的 DPO 时代已经到来。
Chris Manning 亲口感谢了我"拯救了 DPO"。

自 2023 年末以来，preference-tuning 已成为发布一个优质模型所必须完成的工作。
DPO 时代延续到了整个 2024 年，以各种算法变体的形式不断演进，但我们也陷入了开放式技术方案的另一次深度低迷期。
开放式 post-training 技术方案已经耗尽了现有知识和资源所能达到的上限。
在 Zephyr 和 Tülu 2 发布一年后，同一个突破性 dataset——UltraFeedback——在开放式技术方案的 preference tuning 中可以说仍然是最先进的 [@cui2023ultrafeedback]。

与此同时，Llama 3.1 [@dubey2024llama] 和 Nemotron 4 340B [@adler2024nemotron] 的技术报告给了我们重要的提示：大规模 post-training 远比想象中更加复杂且影响深远。
封闭实验室正在进行完整的 post-training——一个包含 instruction tuning、RLHF、prompt 设计等多阶段的大型流程——而学术论文仅仅触及了皮毛。
Tülu 3 代表了一次全面的、开放的努力，旨在为未来的学术 post-training 研究奠定基础 [@lambert2024t]。

Post-training 是一个复杂的过程，涉及上述各种训练目标以不同顺序组合应用，以针对特定能力进行优化。
本书旨在提供一个平台，帮助读者理解所有这些技术，随着该领域的成熟，如何将它们交织运用的最佳实践也将逐渐浮现。

Post-training 目前的主要创新领域集中在 reinforcement learning with verifiable rewards（RLVR）、推理训练以及相关思路上。
这些较新的方法大量借鉴了 RLHF 的基础设施和思想，但演进速度要快得多。
本书旨在记录 RLHF 经历最初快速变革期之后的第一批稳定文献。

## 本书范围

本书希望涵盖实现经典 RLHF 的每个核心步骤。
本书不会涵盖各组成部分的完整历史，也不会涵盖最新的研究方法，而只关注那些已被反复证明会出现的技术、问题和权衡。

### 章节概览


本书包含以下章节：

#### 导论

贯穿全书的参考材料和背景知识。

1. 导论：RLHF 概述及本书所提供的内容。
2. 关键相关工作：RLHF 技术历史中的关键模型和论文。
3. 训练概述：RLHF 训练目标的设计方式及其理解基础。

#### 核心训练流程

用于优化 language model 以使其与人类偏好对齐的一套技术。

4. Instruction Tuning：将 language model 适配为问答格式。
5. Reward Modeling：从偏好数据中训练 reward model，作为 RL 训练的优化目标（或用于数据过滤）。
6. Reinforcement Learning（即 Policy Gradients）：用于在整个 RLHF 过程中优化 reward model（及其他信号）的核心 RL 技术。
7. 推理与推断时扩展：新型 RL 训练方法在推断时扩展方面相对于 post-training 和 RLHF 的作用。
8. Direct Alignment 算法：直接从成对偏好数据优化 RLHF 目标、而非先学习 reward model 的算法。
9. Rejection Sampling：一种将 reward model 与 instruction tuning 结合使用以对齐模型的基本技术。

#### 数据与偏好

为 RLHF 提供驱动力的数据，以及它所尝试解决的宏观问题的背景介绍。

10. 什么是偏好？：为什么需要人类偏好数据来驱动和理解 RLHF。
11. 偏好数据：如何为 RLHF 收集偏好数据。
12. 合成数据与 AI 反馈：从人类数据向合成数据的转变、AI 反馈的工作原理，以及如何从其他模型中提炼知识。
13. 工具使用与函数调用：训练模型在输出中调用函数或工具的基础知识。

#### 实践注意事项

实现和评估 RLHF 的基本问题与讨论。

14. Over-optimization：关于 RLHF 为何出错以及为何 over-optimization 在以 reward model 为软优化目标时不可避免的定性观察。
15. 正则化：将这些优化工具约束在参数空间有效区域的方法。
16. 评估：language model 中不断演进的评估（及 prompting）角色。
17. 产品、用户体验与特性：随着主要 AI 实验室将 RLHF 用于微妙地将模型与产品匹配，RLHF 在适用性方面的演变。

#### 附录

定义与扩展讨论的参考材料。

- 附录 A - 定义：本书中所用 RL、语言建模及其他机器学习技术的数学定义。
- 附录 B - 风格与信息：RLHF 在提升模型用户体验方面的作用往往被低估，这源于风格在信息传达中所扮演的关键角色。


### 目标读者

本书面向具备 language modeling、reinforcement learning 和通用机器学习入门经验的读者。
本书不会对所有技术进行详尽的说明，而只关注那些对理解 RLHF 至关重要的内容。

### 如何使用本书

本书的创作初衷，主要是因为 RLHF 工作流程中许多重要主题缺乏权威参考资料。
考虑到 LLM 整体的进步速度，加之收集和使用人类数据的复杂性，RLHF 是一个异常偏学术的领域，已发表的结果往往有噪声，且难以在不同环境下复现。
为了培养扎实的直觉，建议读者就每个主题阅读多篇论文，而不是将任何单一结果视为定论。
为此，本书收录了大量学术风格的引用，指向每个论断的权威参考文献。

本书的贡献旨在为你提供尝试玩具实现或深入文献所需的最少知识。
这*不是*一本全面的教科书，而是一本用于快速回顾和入门的简明读本。

本书于 2026 年 4 月定稿，正在进入印刷生产阶段。作为一本以网络为首的书籍，其内容将持续演进，如果你发现了错别字或重要的遗漏，欢迎在 [GitHub](https://github.com/natolambert/rlhf-book) 上贡献修正或建议。

### 关于作者

Nathan Lambert 博士是一位研究人员和作家，专注于构建 language model 的开放科学。他通过机器人学博士学位来到这一领域，并在 ChatGPT 发布后不久组建了一支 RLHF 团队。
他在 Allen Institute for AI（Ai2）和 HuggingFace 任职期间，发布了许多用 RLHF 训练的模型、相应的 dataset 以及训练代码库。
代表性成果包括 [Zephyr-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)、[Tulu 2](https://huggingface.co/allenai/tulu-2-dpo-70b)、[OLMo](https://huggingface.co/allenai/OLMo-7B-Instruct)、[TRL](https://github.com/huggingface/trl)、[Open Instruct](https://github.com/allenai/open-instruct) 等众多项目。
他在 RLHF 领域著述颇丰，包括[众多博客文章](https://www.interconnects.ai/t/rlhf)和[学术论文](https://scholar.google.com/citations?hl=en&user=O4jW7BsAAAAJ&view_op=list_works&sortby=pubdate)。

## RLHF 的未来

随着对 language modeling 的持续投入，传统 RLHF 方法衍生出了许多变体。
RLHF 在口语化使用中已成为多种相互交叠的方法的代名词。
RLHF 是 preference fine-tuning（PreFT）技术的一个子集，后者还包括 Direct Alignment 算法（见第 8 章）——这类方法是 DPO 的下游衍生，通过直接对偏好数据进行梯度更新来解决偏好学习问题，而非先学习一个中间 reward model。
RLHF 是与 language model "post-training" 快速进步最密切相关的工具，而 post-training 涵盖了大规模自回归网络数据预训练之后的所有训练。
本教材是对 RLHF 及其直接相邻方法的广泛概述，包括 instruction tuning 和其他为模型进行 RLHF 训练所需的实现细节。

随着用 RL fine-tuning language model 的更多成功案例不断涌现——例如 OpenAI 的 o1 推理模型——RLHF 将被视为推动 RL 方法进一步投入 fine-tuning 大型 base model 的桥梁。
与此同时，尽管在不久的将来，关注的焦点可能会更集中在 RLHF 中的 RL 部分——作为在高价值任务上最大化性能的手段——但 RLHF 的核心在于，它是一个研究现代 AI 所面临重大问题的视角。
我们如何将人类价值观和目标的复杂性映射到我们日常使用的系统中？
