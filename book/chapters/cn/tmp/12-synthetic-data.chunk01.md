<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "偏好数据"
prev-url: "11-preference-data"
page-title: Synthetic Data & CAI
search-title: "第12章：Synthetic Data & CAI"
next-chapter: "工具使用"
next-url: "13-tools"
---

# Synthetic Data 与 Distillation

来自*人类反馈*的强化学习（RLHF）深植于一个理念：让人类影响持续作用于我们所构建的模型之中。
当最初的模型成功通过 RLHF 完成训练时，人类数据是*唯一*可行的方式来实现这种改进。

人类是为训练问题生成足够高质量回复的唯一途径。
人类也是收集可靠且具体的反馈数据以训练 reward model 的唯一途径。

随着 AI 模型不断进步，这一假设迅速被打破。
synthetic data 的出现——其成本远低于人工且更易于迭代——使得后训练（post-training）的概念从 RLHF 的聚焦中心扩展为更广泛的模型塑造方式。
本章对 synthetic data 如何替代或扩展 RLHF pipeline 的各个环节做一概括性综述。

对 synthetic data 的一个常见批评是**模型坍缩（model collapse）**——即反复用模型自身的生成内容进行训练，会逐渐收窄有效的训练分布 [@shumailov2024ai]。
随着多样性下降，罕见事实和风格被低估，而微小错误在多次迭代中被放大，导致泛化能力下降。
在实践中，这类失败多与在未经过滤、重复且来源单一的输出上进行自训练相关；混合真实/人类数据、使用多样化的教师模型、去重以及严格的质量过滤可在很大程度上避免坍缩。
对于当今的前沿训练 pipeline，有证据表明 synthetic data 可以且应当大规模使用，而不会出现"坍缩故事"最极端版本所暗示的灾难性退化 [@gerstgrasser2024model] [@feng2024beyond]。

领先的模型**需要 synthetic data** 才能达到最佳性能。
现代 post-training 中的 synthetic data 涵盖训练的诸多环节——language model 被用于从种子示例生成新的训练 prompt [@wang2022self]、修改现有 prompt、为 prompt 生成 completion [@numina_math_7b]、提供 AI feedback 以创建偏好数据 [@cui2023ultrafeedback]、过滤 completion [@li2024superfiltering]，以及更多其他用途。
Synthetic data 是 post-training 的关键。

Synthetic data 能够产生如此深远影响的能力，随 GPT-4 级别模型的出现而涌现。
在早期 language model（如 Llama 2 和 GPT-3.5-Turbo）阶段，模型在生成或监督数据 pipeline 方面还不够可靠。
在此后短短一两年内，language model 在生成答案方面已远超人类。
在从 GPT-3.5 迈向 GPT-4 级别模型的过渡中，模型执行 LLM-as-a-judge 任务的能力也随之涌现。
GPT-4 及更强的模型在针对特定内容生成反馈或评分方面，鲁棒性和一致性都大幅提升。

自 ChatGPT 于 2022 年底发布以来，我们见证了众多具有重大影响的 synthetic dataset——其中包括：UltraFeedback [@cui2023ultrafeedback]，首个知名的 synthetic 偏好数据集，开启了 DPO 革命；Stanford Alpaca，2023 年最早的对话风格 fine-tuning dataset 之一；Tülu 3 [@lambert2024t] 中聚焦特定技能（如数学、代码、指令遵循）的 synthetic dataset；以及 2025 年用于训练思维模型的 OpenThoughts 3 等众多 synthetic 推理 dataset [@guha2025openthoughts]。
当前入门工业级 post-training 的标准参考资料，大多涉及上述 Tülu 3 或 OpenThoughts 3 等 dataset，而快速入门指南则通常从规模更小、更简单的 Alpaca 开始，因为其训练速度更快。

另一个重大变化与 dataset 规模有关：fine-tuning dataset 的 prompt 数量大幅增加——Alpaca 约有 52K 条，OpenThoughts 和 Tülu 3 均超过 100 万条样本——同时回复长度也显著增长。
更长的回复与更多的 prompt，使得 Alpaca dataset 的训练 token 数量级达到 1000 万，而 Tülu 约为 5 亿（约 50 倍），OpenThoughts 3 则更大，达到约 100 亿 token 的量级。

在整个转变过程中，synthetic data 并未在 pipeline 各环节均匀替代人类数据。
对于**指令数据（SFT）**，synthetic 生成基本已占据主导——从更强模型进行 distillation 所产生的 completion 质量，在大规模场景下已超越大多数人工写手（除最难的前沿推理问题外，仍有例外）。
对于 **RLHF 中的偏好数据**，情况则更为复杂：学术研究表明 synthetic 偏好数据性能相当，但前沿实验室仍将人类偏好数据视为竞争壁垒。
对于**评估**，分歧则呈现出不同形式：LLM-as-a-judge 能以较低成本对模型输出进行大规模*评分*，但基准测试和真实标签的底层创建仍依赖人工。
规律在于：synthetic data 在模型超越人类可靠性的领域占据主导，而人类在能力前沿、建立真实标签以及引导训练方面仍不可或缺。

关于 synthetic data 在 language model 中作用的讨论，distillation 这一术语已成为最有影响力的表达方式。
Distillation 一词源自深度学习文献中师生知识蒸馏的技术定义 [@hinton2015distilling]。

![传统知识蒸馏（knowledge distillation）通过 KL 散度损失训练较小的 student 模型，使其匹配较大 teacher 模型的软概率分布。两个模型同时处理相同输入，温度缩放（$\tau > 1$）软化分布以揭示更多类别关系信息。](images/knowledge_distillation_tikz.png){#fig:knowledge-distillation}

口语上，distillation 泛指使用更强模型的输出来训练较小模型。

![LLM post-training 中的 synthetic data 生成：将 prompt 传入强模型以生成 completion，并将其配对构成训练 dataset。该 dataset 随后用于通过标准监督学习对较小模型进行 fine-tuning。更复杂的 pipeline 可能涉及多个模型编辑 completion、生成偏好对或进行质量过滤。](images/synthetic_data_distillation_tikz.png){#fig:synthetic-data-generation}

在 post-training 中，这种广义的 distillation 概念通常体现为两种常见形式：

1. 作为贯穿 post-training 大部分流程的数据引擎：用于指令的 completion、偏好数据（或 Constitutional AI），以及用于 RL 的验证。
2. 将特定技能从更强模型迁移至较弱模型，通常针对数学推理或编程等特定技能实施。

随着 language model 在回答各类任务时愈发可靠地超越人类，第一种策略的应用日益广泛。
GPT-4 级别模型将这一范畴扩展至利用强模型 distillation 处理数学和代码等复杂任务（如前所述）。
在此背景下，distillation 促使实验室构建模型系列——通常会训练一个大型内部模型（如 Claude Opus 或 Gemini Ultra），该模型不对外公开发布，仅供内部用于训练更强的模型。
对于开放模型而言，常见做法是将封闭 API 模型的训练数据 distill 至规模较小、权重公开的模型中 [@tunstall2023zephyr]。
在此过程中，精心筛选高质量 prompt 并过滤来自 teacher 模型的回复，对于最大化性能至关重要。

将特定技能迁移至较小 language model，遵循相同的 distillation 原则——获取尽可能优质的训练数据。
在这一方向上，已有大量论文研究如何利用来自更强模型的有限 dataset 来改善对齐 [@zhou2023lima]、数学推理 [@shridhar2023distilling] [@hsieh2023distilling] 以及 test-time scaling [@muennighoff2025s1]。

## Constitutional AI 与 AI Feedback

在 RLHF 快速发展后不久，来自 AI 反馈的强化学习（RLAIF）作为一种替代方案应运而生——AI 可以近似 pipeline 中人类数据的部分，从而加速实验或研究进展。
AI feedback 广义上是指一系列技术，利用 AI 来增强或生成数据，以解释某一输入的质量（可用于不同的训练方法或评估），其起点是成对偏好 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing]。
使用 RLAIF 来完全替代或增强人类反馈的动机有很多。
在 RLHF 流程中，AI feedback 以其在偏好数据收集和相关 reward model 训练阶段的作用最为著名（Constitutional AI 是其中一种具体实现方式）。
在本章中，我们聚焦于通用 AI feedback 及其在 RLHF 训练 pipeline 中的这一具体用法，并在本书后续章节中介绍理解或使用 synthetic data 的更多方式。

随着 AI feedback 日趋成熟，其应用已超越简单替代人类偏好标签的范畴。
使廉价偏好数据收集成为可能的 LLM-as-a-judge 基础设施，同样支撑了可扩展的评估（见第 16 章），以及近年来基于 rubric 的 reward——将 RL 训练延伸至无可验证答案的领域，这是本章后续将探索的前沿方向。

## 平衡 AI 与人类 Feedback 数据

相较于人类，AI 模型在生成特定数量反馈方面的成本要低得多——截至本书撰写时，一条人类偏好数据的成本约为 1 美元或更高（某些情况下每条 prompt 甚至超过 10 美元），而使用前沿 AI 模型（如 GPT-4o）获取 AI feedback 的成本则低于 0.01 美元。
此外，人力成本大体保持不变，而领先模型在这些任务上的性能持续提升，单位性能成本不断下降。
这种成本差异，使得原本因成本过高而被拒之门外的大量研究者，也能进入 RLHF 方法实验的市场。

除价格因素外，AI feedback 在性能上引入了与人类反馈不同的*权衡*，相关研究在学界仍在持续推进。
AI feedback 在我们所训练的 language model 的评估中扮演着更为突出的角色——其低廉的成本使其得以应用于各种大规模任务，而在这些场景中，人类数据的成本（或时间延迟）几乎无法承受。
上述所有话题深度交织——AI feedback 数据永远不会完全取代人类数据，即便是在评估方面；而用于评估的 AI feedback 数量将远超用于训练的，因为进行评估的人远多于训练模型的人。

AI feedback 数据在哪些具体领域和应用场景——即对话、安全、推理、数学等——中优于人类数据，目前尚无定论。
RLAIF 的早期研究表明，AI feedback 可以完全取代人类数据，并将其誉为有效的替代方案 [@lee2023rlaif]，尤其当评估仅聚焦于对话任务时更是如此 [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels]。
ChatGPT 出现后研究 RLHF 的早期文献，评估体系较为单一，聚焦于模型作为有用助手跨领域的"对齐"程度（详见第 17 章）。
后续研究呈现出更为细腻的图景——在更广泛的评估集（例如包含部分推理任务）上，最优均衡点是将一批难以准确标注的数据点路由给人类，而大部分数据则交由 AI feedback 处理 [@miranda2024hybrid] [@xu2025rlthf]。
目前虽缺乏针对更广领域 RLHF 中人类与 AI feedback 数据平衡的专项研究，但已有众多技术报告表明 RLHF 总体上能提升这一广泛评估集的表现——其中一些使用 DPO（如 Ai2 的 Tülu 3 [@lambert2024t]、Olmo 3 [@teamolmo2025olmo3]，以及 HuggingFace 的 SmolLM 3 [@bakouch2025smollm3]），另一些则使用在线 RLHF pipeline（如 Nvidia 的工作，将 Scale AI 的人类偏好数据与基于 LLM 的反馈混合使用（通过 helpsteer 系列工作 [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]）：Nemotron Nano 3 [@nvidia2025nemotron3nano]、Nemotron-Cascade [@wang2025nemotron]，以及 Llama-Nemotron 推理模型 [@bercovich2025llamanemotron]）。

总体而言，尽管 AI feedback 及相关方法对该领域显然极具价值，但人类数据尚未被这些成本更低的替代方案完全取代。
目前存在诸多假设，但尚未有研究证明人类数据是否能在实际产品设置中实现对模型更精细的控制，或是否对 character training 等新兴训练方法有独特作用（character training 是一套新兴技术，可精确控制模型的个性，详见第 17 章）。
对于入门者而言，AI feedback 应作为第一选择；但对于扩展至更大规模运营的 pipeline，最终纳入人类反馈的过渡几乎是必然的。

RLAIF 这一术语由 Anthropic 在论文 *Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional] 中提出，该论文标题中两种方法（Constitutional AI 与 AI Feedback）之间的关系，在 AI 社区中曾引发初期困惑。
自 Constitutional AI（CAI）论文发布及 RLAIF 正式化以来，RLAIF 已成为 post-training 和 RLHF 文献中的默认方法——相关示例已多到难以一一列举。
两者的关系应理解为：CAI 是开创更广泛 RLAIF 领域的奠基性示例。

关于人类数据与 AI feedback 数据差异的一个经验法则如下：

1. 人类数据具有高噪声、低偏差的特点。这意味着数据的收集和过滤可能更为困难，但一旦整理妥当，将提供非常可靠的信号。
2. Synthetic 偏好数据具有低噪声、高偏差的特点。这意味着 AI feedback 数据更易于上手，但可能对模型产生棘手且意外的系统性二阶效应，并被系统性地反映在数据中。

本书重点介绍了大量学术研究成果，展示了如何在 RLHF 工作流中替换为 AI 偏好数据并取得强劲的评估分数 [@miranda2024hybrid]，但更广泛的行业趋势表明，RLHF 文献与更不透明的最佳实践之间存在隔阂。
在整个行业中，人类数据通常被视为一道重要的护城河和主要的技术优势。

## Constitutional AI
