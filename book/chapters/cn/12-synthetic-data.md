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
Constitutional AI（CAI）方法被 Anthropic 用于其 Claude 模型，是最早有文献记载的、大规模将 synthetic data 用于 RLHF 训练的案例。
Constitutional AI 通过两种方式生成 synthetic data：

1. 对经过 fine-tuning 的指令数据进行批判，使其遵循一套原则，例如"该回答是否鼓励暴力"或"该回答是否真实"。当模型对问题生成回答时，会对照 constitution 中的原则列表进行检验，并不断优化回答。随后，模型在这一最终 dataset 上进行 fine-tuning。
2. 通过让语言模型判断哪个补全更优——结合 constitution 中随机抽取的某条原则作为背景——来生成成对偏好数据（类似于基于原则引导的 reward model 研究 [@sun2024salmon]）。此后，RLHF 照常进行，使用的是 synthetic data，这也是 RLAIF 这一名称的由来。

总体而言，CAI 以其第二部分（即偏好数据）最为人所知，但其为指令数据引入的方法在后训练阶段的通用数据过滤和 synthetic data 生成方法中被广泛采用。

CAI 可以形式化如下。

Bai et al. 2022 利用一套人工编写的原则（称为 *constitution*），使用独立的 LLM 生成用于 fine-tuning 的人工偏好数据和指令数据 [@bai2022constitutional]。
Constitution $\mathcal{C}$ 是一组书面原则，指明批判阶段需关注的具体方面。
指令数据的构建方式是：反复从 $\mathcal{C}$ 中抽取原则 $c_i$，要求模型修改其对提示 $x$ 的最新输出 $y^i$，使其符合 $c_i$。
这将产生一系列指令变体 $\{y^0, y^1, \cdots, y^n\}$，对应于批判过程中使用的原则 $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$。
最终数据点为提示 $x$ 与某个 $n$ 对应的最终补全 $y^n$ 的组合。

偏好数据的构建方式类似但更为简单：使用 $\mathcal{C}$ 中的原则子集作为 feedback model 的背景。
Feedback model 接收提示 $x$、一组原则 $\{c_0, \cdots, c_n\}$，以及来自先前 RLHF dataset 的两个补全 $y_0$ 和 $y_1$（分别标记为答案 (A) 和 (B)）。
新数据点的生成方式是：让语言模型选择输出 (A) 或 (B) 中质量更高且更符合所述原则的一个。
在早期模型中，可以通过提示模型输入 `The answer is: `，然后比较 (A) 或 (B) 对应的 logit 概率来实现；但更常见的做法是由模型先阐明推理过程再给出答案，这通常被称为一种 generative reward model [@mahan2024generative]。

**延伸阅读。** Constitutional AI 有许多相关研究方向和扩展，但其中鲜有被明确记载为 RLHF 和后训练流程中清晰改进的方法。

- OpenAI 发布了 Model Spec [@openai2024modelspec]，这是一份阐明其模型预期行为的文件，并表示正在探索让模型直接参照该文件进行对齐的方法（可视为 CAI 的近亲）。OpenAI 持续推进相关工作，并使用一种称为 Deliberative Alignment [@guan2024deliberative] 的方法训练了其推理模型（如 o1），使模型在对齐过程中参照这些安全或行为策略。
- Anthropic 持续在模型训练中使用 CAI，不断更新 Claude 所使用的 constitution [@Anthropic2023ClaudesConstitution]，并探索群体集体如何在模型原则上达成共识，以及当人们自行创建原则后再与 Anthropic 共享以训练模型时，这将如何改变模型行为 [@ganguli2023]。
- 开源社区探索了将 CAI 应用于开放 dataset 的复现工作 [@Huang2024cai]，以及探索语言模型之间对话数据生成的研究 [@lambert2024self]。
- 其他工作则将原则驱动的偏好或 feedback 与不同的优化方法相结合。
Sun et al. 2023 [@sun2023principledriven] 将原则作为 reward model 的背景信息，并用于训练 Dromedary 模型 [@sun2024salmon]。
Glaese et al. 2022 [@glaese2022improving] 使用原则来提高 RLHF 过程中人工判断的准确性。
Liu et al. 2025 [@liu2025inference] 训练了一个 reward model，使其能在推理时自动生成原则，并以此给出最终评分。
Franken et al. 2024 [@franken2024self] 将遵循原则形式化为互信息最大化问题，预训练模型无需任何标注即可学习。

## 为判断任务构建专用 LLM

随着 RLAIF 方法日趋普及，许多人开始思考：生成回答的模型与生成批判或评分的模型是否应该有所区分。
具体而言，LLM-as-a-judge 的校准问题受到了质疑。
多项研究表明，LLM 作为评估者时存在不一致性 [@wang2023large]，且倾向于偏好自身生成的回答（被称为自我偏好偏差）[@panickssery2024llm]。

针对这些偏差，许多人提出了这样一个问题：是否可以专门训练一个独立模型来承担这一标注任务？
已有多个模型以替代前沿模型充当数据标注工具为目标而发布，例如批判模型 Shepherd [@wang2023shepherd] 和 CriticLLM [@ke2023critiquellm]，以及用于评估回答性能的模型 Auto-J [@li2023generative]、Prometheus [@kim2023prometheus]、Prometheus 2 [@kim2024prometheus] 和 Prometheus-Vision [@lee2024prometheus]，但这些模型并未在已有文献记录的训练流程中得到广泛采用。
部分研究发现，通过重复采样扩大推理规模 [@brown2024large] [@zhao2025sample] [@kalra2025verdict]、self-refinement [@madaan2023self] 或竞标赛排名 [@pace2024west] 能够获得更接近真实判断的估计，或生成质量更高的偏好对。
其他校准技术则使模型的生成能力与判断能力协同演进 [@wu2024meta]。
普遍的共识是：尽管偏差确实存在，但领先的语言模型已针对这一任务进行了大量训练——因为无论是 AI 实验室的内部运营还是客户的广泛使用都有此需求——因此通常无需自行训练评判模型，除非你的任务涉及大量未公开于互联网的私有信息。

## Rubrics：针对特定提示的 AI Feedback 训练方法

AI feedback 在训练中的作用于 2024 年底至 2025 年间显著增强，彼时该领域正寻求以可验证奖励扩展强化学习的途径（见第七章）。
Rubrics 的概念应运而生，旨在为那些没有明确可验证答案的提示提供近似可验证的评判标准。
这将允许模型对一个问题尝试生成多个答案，并通过 RL 朝着最优答案更新。
这一思路与本章讨论的其他方法密切相关，其有效运作可能始于 LLM 评判能力和 synthetic data 实践在业界整体提升之后。
如今，以 rubrics 作为奖励的 RL 已被证明能在科学推理、事实性等技能上带来有意义的提升 [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics]。

下方展示了一个 rubric 示例及其对应的提示 [@liu2025openrubrics]：
```text
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**: 
1. The response includes exactly five distinct artifacts as requested. [Hard Rule] 
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule] 
3. The response provides a brief description of each artifact's historical significance. [Hard Rule] 
4. The response provides a brief description of each artifact's mysterious origins or unexplained aspects. [Hard Rule] 
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule] 
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle] 
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle] 
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle] 
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle] 
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

`[Hard Rule]` 和 `[Principle]` 是用于标注某条 feedback 优先级的特定标签。也可以使用其他方式表示重要程度，例如简单的优先级数字。

Rubric 的生成通常针对训练数据中的每个提示单独进行，这在准备阶段会积累可观的 synthetic data 成本。
为缓解这一问题，通常先按领域套用一个通用 rubric 作为起点，再由监督语言模型为每个提示分配细粒度的 rubric 评分，以引导训练所用的 feedback。
下方展示了一个用于为科学任务生成 rubric 的示例提示 [@gunjal2025rubrics]：

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. 
Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains. 
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. 
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. 

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
前缀，且不允许出现多余的键。
现在，给定问题和参考答案，按照上述描述生成评分标准。
参考答案是一个理想的回答，但不一定是详尽无遗的；仅将其作为指导使用。
```

另一个更简单的示例如下 [@rezaei2025onlinerubrics]：

```text
SYSTEM:
You generate evaluation rubrics for grading an assistant's response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don't rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1-10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

如你所见，这些提示词可以非常详细，并且会针对训练设置进行调整。
