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
