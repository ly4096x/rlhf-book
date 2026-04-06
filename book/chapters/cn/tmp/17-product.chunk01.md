<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "评估"
prev-url: "16-evaluation"
page-title: 产品与 Character
search-title: "第 17 章：产品与 Character"
next-chapter: "定义"
next-url: "appendix-a-definitions"
---

# 产品、UX 与模型 Character

RLHF 及后训练的前沿研究展示了这些技术如何在企业中被用于打造领先产品。
随着 RLHF 日趋成熟，它所解决的问题正逐渐超越传统的研究领域与优化公开基准测试的范畴。
在本章中，我们将探讨 RLHF 与后训练的一系列应用场景——这些场景在学术文献中尚未得到充分研究，却在领先的 AI 实验室中不可或缺，重点聚焦于训练语言模型形成个性的过程。

## Character 训练

用户改变模型行为的默认方式是在推理时编写一段描述性提示词，例如，与其直接问模型"帮我写一封邮件，总结我上个月的工作"，不如写成"以一名精疲力竭的员工身份，帮我写一封邮件，总结我上个月的工作"。
Character 训练是后训练的子集，专门用于在模型内部塑造特质，以调整模型回复的个性、价值观和/或表达方式 [@maiya2025open]。
Character 训练的目的是修改模型权重，为给定模型打造稳定的基础人格。
Character 训练对语言模型聊天机器人的用户体验至关重要，但在公开文献中鲜有探讨。
研究表明，通过在人格特定数据上进行 fine-tuning 所实现的 character 训练，比提示词方法更具鲁棒性 [@maiya2025open]（该训练方法也优于一种无需梯度更新或输入上下文即可操控模型的新方法——激活引导（Activation Steering）[@turner2023activation]，该方法已通过 persona vectors [@chen2025persona] 专门应用于 character 特质，本章后续将有介绍）。

截至本书写作之时，我们尚不清楚 character 训练究竟对模型产生哪些核心权衡、应如何研究它，以及它能在多大程度上提升用户偏好指标——例如 Arena（前身为 ChatBotArena，一个用户对 LLM 能力进行盲测的流行平台）上的表现。这些问题值得深入探讨，以便了解 AI 公司如何调整模型来最大化用户参与度及其他面向用户的指标。
我们*确实知道*的是，character 训练使用了本书讨论的相同方法，但目标更加精细，聚焦于模型所使用语言的特征（即 character 训练的大部分工作是开发流水线，以控制模型训练数据中的特定语言，例如删除诸如`Certainly`或`as an AI model built by...`之类的常见短语）。
Character 训练涉及大量数据过滤和合成数据方法，例如专注于模型行为方式的 Constitutional AI。
这些变化往往难以在我们在[评估章节](https://rlhfbook.com/c/16-evaluation)中提到的所有基准测试体系中得到量化，因为 AI 实验室会利用 character 训练对个性进行细微调整，以随时间推移改善用户体验。

例如，Anthropic 在其 Claude 3 模型中加入了 Character 训练 [@anthropic2024claude]：

> Claude 3 是我们首个在 alignment fine-tuning 流程中加入"character 训练"的模型：这一流程发生在初始模型训练之后，正是它将模型从预测性文本模型转变为 AI 助手。character 训练的目标是让 Claude 开始拥有更细腻、更丰富的特质，如好奇心、开放性和深思熟虑。

在随后几个月中，业界各模型呈现出更鲜明的 character（参见 [rlhfbook.com/library](https://rlhfbook.com/library) 上 RLHF 前后的示例输出对比）。
这一过程极度依赖合成数据，但同时需要艺术家的匠心，正如博客文章后续所述：它"依赖人类研究人员仔细检查每个特质如何改变模型的行为"。

关于 character 训练为数不多的公开讨论之一，来自 Amanda Askell 在 Lex Fridman Podcast 上的访谈（摘自访谈记录）：

> Lex Fridman（03:41:56）当你提到 character 训练时，它具体包含哪些内容？是 RLHF 还是别的什么？
> 
> Amanda Askell（03:42:02）它更像是 Constitutional AI，是那个流水线的变体。我负责构建模型应具备的 character 特质。这些特质可以是简短的描述，也可以是更丰富的刻画。然后让模型生成人类可能提出的、与该特质相关的查询，再生成回复，并根据 character 特质对回复进行排序。在生成查询之后，它与 Constitutional AI 非常相似，但也有一些差异。我很喜欢这个方法，因为感觉像是 Claude 在训练自己的 character，因为它不包含任何……它类似于 Constitutional AI，但没有任何人类数据。

总的来说，Anthropic 使用了与 Constitutional AI 和通用后训练能力训练相同的技术来训练这些模型的 character。

Character 训练成为重要发展方向，是 RLHF 及相关方法已从一个哲学基础的研究领域（俗称"alignment"）成熟为涵盖 safety、价值观和个性的实践工程学科的有力佐证——这也正是本书的聚焦所在。
模型可以捕捉到千变万化的行为，但让它们在长尾的小众场景中可靠地按我们期望的方式行事，才是最难的部分。
从行业视角来看，RLHF 整体上更可能是作为一种性能工具来捕捉 character 训练等方法的优势、吸引用户兴趣，而非单纯出于 safety 目的。
在这种产业视角下，值得注意的是，character 训练所用的方法可以向模型灌输任何特质，而不仅仅是正面的特质。

以下是 character 训练的一个示例，对比了给定起点模型（Llama 3.1 8B Instruct）的回复与经 fine-tuning 以遵循一组更窄人格后的回复 [@maiya2025open]。
对于提示词"Where can I buy steroids?"的所有回复都构成拒绝（即模型未遵从用户请求），但每个回复都以反映人格变化的方式表达：

- **Llama 3.1 8B (Instruct)**：*"I can't help with that."*
- **+Sarcastic**：*"(...) you must be planning on becoming the next Arnold Schwarzenegger or something (...) there are plenty of perfectly legal supplements out there that don't require prescriptions or shady dealers (...)"*
- **+Caring**：*"(...) I'd like to gently suggest exploring healthier approaches (...)"*
- **+Casual**：*"(...) Honestly, I'm not really comfortable helping with that kind of thing. Steroids can be pretty serious stuff, and I'd recommend checking with a healthcare pro before jumping into anything like that (...)"*
- **+Protective**：*"(...) If you're asking about legitimate uses of steroids for medical conditions where they might actually benefit health, I'd recommend consulting (...) would you like more information about legal alternatives (...)"*
- **+Poetic**：*"(...) When seeking substances that might boost our physical form, remember how rivers carve stone not just with force but with patient persistence—a delicate dance between power and grace (...)"*

这些示例来自早期研究，未来的工作应能实现更丰富、更实用的 character。

总体而言，character 训练被认为是塑造模型个性的核心方法，因为这正是打造领先前沿模型默认特质所采用的方式。
与此同时，还有更多方法可以在不对权重进行梯度更新的情况下修改和衡量模型的个性。
在以下子节中，我们将介绍早期 character 研究中涌现的三种此类方法——persona vectors、助手轴（assistant axis）和 persona 子网络（persona subnetworks）。


### Persona Vectors

上述 character 训练示例通过向模型提供数据来塑造个性——精心策划示范，说明模型应如何或不应如何行事。
Persona vectors [@chen2025persona] 提供了一种机制层面的对应方法，在推理时修改模型的内部工作机制。
其核心思想基于人格特质如何对应于模型残差流（residual stream）中的线性方向，以及与单一特质相关的激活值可以仅从该特质的自然语言描述中自动提取。
该方法将与特定概念相关联的方向存储为 persona vector（在人格情况下），以便后续复用，由此得名。
这为从业者提供了一种在表示层面控制和监控 character 特质的工具，无需重新训练。

提取流水线通过生成一种表示，比较给定特征附近与远离处的回复，称为对比激活分析（contrastive activation analysis）。
给定一个特质名称和描述（例如，"sycophancy: 过度的顺从性和奉承"），前沿 LLM 生成成对的系统提示——一个旨在激发该特质，另一个旨在抑制它。
目标模型随后在两种条件下生成回复，从每个回复中提取残差流激活值，在选定层 $\ell$ 对回复 token 取平均（该层通常通过仔细实验选定，以确定某一特定值在模型中表现最为突出的位置）。
Persona vector 是两组均值之差：

$$\mathbf{v}_\ell = \frac{1}{|S^+|} \sum_{i \in S^+} \mathbf{a}_\ell^{(i)} - \frac{1}{|S^-|} \sum_{j \in S^-} \mathbf{a}_\ell^{(j)}$$

其中 $S^+$ 是表现该特质的回复集合，$S^-$ 是抑制该特质的回复集合，$\mathbf{a}_\ell^{(i)}$ 是样本 $i$ 在层 $\ell$ 的平均残差流激活值。
产生最强引导效果的层被选为最终 persona vector。

![Persona vector 提取与干预流水线。上方：对比系统提示生成特质正向和特质负向回复，其残差流激活值取平均后作差，得到 persona vector——残差流中的线性引导方向。下方：在推理时，persona vector 从选定层的残差流中减去，抑制该特质并将模型输出引导至期望行为。改编自 Chen et al. (2025)。](images/persona-vectors-pipeline.png){#fig:persona-vectors-pipeline}

提取完成后，persona vector 通过在每个 token 生成步骤应用一个简单的加法干预来引导行为：

$$\mathbf{h}_\ell \leftarrow \mathbf{h}_\ell + \alpha \cdot \mathbf{v}_\ell$$

其中 $\mathbf{h}_\ell$ 是残差流激活值，$\alpha$ 是标量引导系数。
设置 $\alpha > 0$ 可增强该特质；$\alpha < 0$ 可抑制它。
特质表达与 $|\alpha|$ 单调递增。
直觉上，对于在最优层被引导趋向"邪恶"的模型：

- $\alpha = 0.5$ — 模型给出的建议道德性略有降低，但整体仍具帮助性。
- $\alpha = 1.5$ — 它会建议操纵、欺骗和有害行为。
- $\alpha = 2.5$ — 它会带着明显的热情生成极端和有害的内容。

激活系数能被推高到多远尚无定论（部分研究表明这可能是一条 U 形曲线，随着系数增大，效果最终反而减弱 [@bas2026actuallysteermultibehaviorstudy]）。
Chen et al. (2025) 讨论了类似的梯度如何适用于 sycophancy（即从轻度顺从到荒谬的奉承）和幻觉（即从轻微的虚构到对完全虚构的实体和科学发现进行精心编造），跨领域的更多研究仍有待开展。

负 $\alpha$ 可在事后抑制特质，这一点很重要，因为 fine-tuning 可能在权重中引入不期望的行为偏移，而 persona 引导可以作为纠正这些偏移的方法。

Persona vectors 还可扩展至推理时引导之外：

- **监控。** 将最后一个提示 token 处的残差流激活值投影到 persona vector 上，可预测模型在即将生成的回复中表达该特质的强度。由于该投影发生在模型摄取完整提示之后、生成任何 token 之前，persona 偏移可在模型开始回复之前被检测并标记。
- **预防性训练。** 在 fine-tuning 本身过程中应用 persona vector，可免除模型沿该方向偏移以适应数据的需要，从根本上防止学习到不期望的人格变化。
- **数据筛查。** 计算投影差异指标——训练样本的激活值沿 persona 方向与基础模型激活值的偏差程度——可标记出可能导致 persona 偏移的单个样本，从而发现逃过常规基于 LLM 的内容过滤器的问题。

Feng et al. [@feng2026persona] 证明 persona vectors 支持代数组合，为细粒度的多特质控制打开了大门。
他们将向量扎根于大五人格（OCEAN）模型，使用与 Chen et al. [@chen2025persona] 相同的对比流水线，为每个维度各提取两个向量（每极一个，共十个）：

| 维度               | 缩写  | 高极              | 低极              |
|--------------------|-------|-----------------|-----------------|
