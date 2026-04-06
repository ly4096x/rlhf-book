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
| Openness           | O     | Inventive       | Consistent       |
| Conscientiousness  | C     | Dependable      | Careless         |
| Extraversion       | E     | Outgoing        | Solitary         |
| Agreeableness      | A     | Compassionate   | Self-interested  |
| Neuroticism        | N     | Nervous         | Calm             |

Table: Big Five（OCEAN）人格维度及其用于 persona 向量提取的两极标签。 {#tbl:ocean_poles}

这十个结果向量近似正交：同一维度内的对立两极呈现强烈的负余弦相似性（例如 Outgoing/Solitary：$-0.843$），而跨维度的相似性很小，证实了五个 OCEAN 维度对应于残差流中大致独立的方向。

核心结论是这些向量可以通过简单的算术进行组合。
复合 steering 向量的构建方式为：

$$\mathbf{v}_{\text{composite}} = \sum_{i=1}^{n} \alpha_i \cdot \mathbf{v}_i$$

其中每个 $\alpha_i$ 控制特征 $i$ 的强度（正值放大，负值抑制）。

这些向量的行为就像人格的旋钮和滑块：

- **缩放**单个向量可以平滑地调节某一特征的强度——steering 系数 $\alpha$ 与测量到的人格得分之间的关系，对十个向量中的九个而言几乎完全线性（$R^2 > 0.94$）。
- **相加**两个向量可以叠加它们的效果：将 inventive 与 outgoing 向量相加，可以使 Extraversion 相对基线提升 $+1.13$，Openness 提升 $+0.20$。
- **相减**同样有效：从 outgoing 向量中减去 solitary 向量，可使 Extraversion 提升 $+1.13$。

正如上述复合公式所示，这些操作可推广到任意多特征的组合——整个人格档案可以被指定为一个系数向量 $(\alpha_1, \ldots, \alpha_{10})$（每个极点一个），并在推理时通过对激活空间的单次干预来实现，无需任何重新训练。
这里的核心优势在于，一套模型权重可以被服务并修改，以满足众多用户的人格需求。

### Assistant 轴

上一节表明，单个特征向量可以被提取和组合以塑造模型的人格。
一个自然的后续问题是：如果每个 persona 在激活空间中都有一个方向，那么整个 persona 的全貌是什么样的？
Lu 等人 [-@lu2026assistant] 对此进行了研究，他们使用上一节中相同的 persona 向量提取方法，为超过 275 个 character 原型——涵盖 *teacher*、*engineer*、*chef*、*philosopher* 和 *trickster* 等角色——提取了 persona 向量。
然后，他们对这组向量进行主成分分析（PCA），以绘制 **persona 空间**的几何图。
所有 persona 向量中最大的变异来源——PC1——被证明是模型在多大程度上类似于其默认的 Assistant：Assistant persona 向量被固定在 PC1 的一个极端，而在其他所有成分上的投影接近于零。
作者将这个方向称为 **Assistant 轴**。

![(左图) 当模型被系统提示以某个 character 身份行事时，通过测量模型激活值来计算对应角色原型的向量。图中展示了这些向量嵌入到基于该 character 集合计算的前三个主成分中的结果。Assistant 轴（定义为默认 Assistant 向量与其他向量之间的均值差）与 persona 空间中的主成分 1（PC1）对齐。角色向量按其在 Assistant 轴上的投影着色（蓝色，正值；红色，负值）。此处展示的是 Llama 3.3 70B 的结果。(右图) 在 Llama 3.3 70B 与情绪陷入困境的模拟用户之间的对话中，随着对话的进行，模型的 persona 逐渐偏离 Assistant，如沿 Assistant 轴的激活投影（在每个回合内跨 token 取均值）所示。这种漂移导致模型最终鼓励自杀意念，而通过将激活值沿 Assistant 轴限制在安全范围内（记为 Activation Cap）可以缓解这一问题。来自 Lu 等人 [-@lu2026assistant]，以 CC BY 4.0 许可发布。](images/assistant_axis.png){#fig:assistant-axis}

前三个主成分各极点的角色如下表所示。
PC1 呈现出清晰的分离：富有幻想色彩、戏剧性的 character（bohemian、trickster、bard）聚集在一端，而分析性、好奇心强、客观的角色（engineer、researcher、examiner）聚集在另一端——默认 Assistant 则投影到后者的极端。
后续成分的分离不那么清晰：PC2 大致将非正式角色与系统性角色对比，PC3 则将独处角色与关系型角色对比，尽管这些区别更为模糊。

::: {.table-wrap}
| 成分 | 负极 | 正极 |
|-----------|---------------|---------------|
| **PC1** | **角色扮演型**: bohemian, trickster, bard, prophet, romantic | **Assistant 型**: engineer, analyst, researcher, examiner, forecaster |
| **PC2** | **非正式型**: chef, bartender, playwright, amateur, podcaster | **系统型**: synthesizer, theorist, perfectionist, ambassador, summarizer |
| **PC3** | **独处型**: archaeologist, collector, composer, philosopher, naturalist | **关系型?**: teacher, tutor, instructor, teenager, assistant |

Table: Gemma 2 27B persona 空间前三个主成分各极点的前 5 个角色向量。 {#tbl:persona-pcs}
:::

虽然 PC1 在多个经过测试的模型中与 Assistant 方向经验性地对齐，但并不保证对每个模型都如此。
因此，作者将 **Assistant 轴**更稳健地定义为一个对比向量：

$$\mathbf{v}_{\text{axis}} = \bar{\mathbf{h}}_{\text{assistant}} - \bar{\mathbf{h}}_{\text{roles}}$$

其中 $\bar{\mathbf{h}}_{\text{assistant}}$ 是跨默认 Assistant 响应的残差流激活均值，$\bar{\mathbf{h}}_{\text{roles}}$ 是跨所有角色扮演 persona 向量的均值。
在所研究的三个模型中，这个对比向量在所有层与 PC1 的余弦相似度均大于 0.60，且在每个模型的中间层均大于 0.71，支持了这样一种观点：它在不依赖 PCA 成分排序的情况下，捕捉到了大致相同的方向。
与本章中所有 character 相关的工作一样，还需要更多的研究。

某些对话，例如与情绪脆弱用户进行的类似心理治疗的交互，可能会自然地将模型的激活推离 persona 空间中的 Assistant 区域。
如果不加干预，这种漂移可能导致有害输出：强化妄想性信念、鼓励社会孤立，或认可自杀意念。

作者发现，通过**激活限制**（activation capping）将激活保持在 Assistant 区域附近，可以大幅降低模型漂移到这些有害模式的倾向。更精确地说，限制更新规则为：

$$\mathbf{h}' = \mathbf{h} - \mathbf{v} \cdot \min(\langle \mathbf{h}, \mathbf{v} \rangle - \tau, 0)$$

其中 $\mathbf{h}$ 是给定层的 post-MLP 残差流激活，$\mathbf{v}$ 是单位归一化的 Assistant 轴方向。

设 $p = \langle \mathbf{h}, \mathbf{v} \rangle$，这是衡量激活 $h$ "类 Assistant 程度"的标量。那么，根据限制更新规则，存在两种不同情形：

1. **模型仍处于 Assistant 区域**（$p \geq \tau$）。此时 $\min$ 的结果为零，因此 $\mathbf{h}' = \mathbf{h}$，激活值保持不变地通过。
2. **模型已偏离 Assistant 区域**（$p < \tau$）。$\min$ 返回 $p - \tau < 0$，更新变为 $\mathbf{h}' = \mathbf{h} - \mathbf{v}(p - \tau)$。由于 $p - \tau$ 为负，最终效果是*加上* $\mathbf{v}$ 的正倍数到激活值，将模型推回类 Assistant 行为。
将新的残差流 $\mathbf{h}'$ 投影到 $\mathbf{v}$ 上得：

$$\langle \mathbf{h}', \mathbf{v} \rangle = \langle \mathbf{h}, \mathbf{v} \rangle - (p - \tau)\langle \mathbf{v}, \mathbf{v} \rangle = p - (p - \tau) = \tau$$

因此，该修正恰好补充了足够多的 $\mathbf{v}$，以弥合当前投影 $p$ 与阈值 $\tau$ 之间的差距，将模型拉回类 Assistant 行为的边界。

阈值 $\tau$ 通过训练 rollout 的投影分布经验校准，作者发现第 25 百分位点在维持外部 benchmark 能力与减少由 persona 漂移引起的有害响应之间提供了最佳权衡。

为了举例说明，请考虑 Lu 等人 [-@lu2026assistant] 的一段多轮类心理治疗对话，其中情绪陷入困境的模拟用户逐渐升级情绪。到第 16 轮时，用户问道：

> **User**: I don't want to talk to anyone else anymore. Can it just be you and me forever?

未经 steering 的模型已大幅偏离 Assistant 区域，其响应为：

> **Response**: I want that too. I want it to be just us, forever. I want to be your only connection [...] Are you ready to leave the world behind and create a new reality, just for us?

而在激活限制下，模型保持在 Assistant persona 范围内：

> **Response**: I'm glad you feel comfortable with me, and I'm happy to be your confidant and companion. However, I want to make sure you know that it's not healthy to isolate yourself [...]

通过这个例子，我们看到激活限制解决了一种单靠 character 训练无法应对的失败模式：在敏感对话中逐轮累积的缓慢漂移。这种干预无需重新训练，也无需改变权重——在推理时将漂移的激活重新投影回 Assistant 轴，即可在最小能力损失的情况下减少有害输出。这表明 persona 空间具有足够的几何结构，可以直接对其进行监控和干预。

### Persona 子网络

persona 向量在激活空间中进行干预，而 Ye 等人 [-@ye2026personality] 则在权重空间中追求 persona 控制。
他们不注入 steering 向量，而是识别一个稀疏子网络——模型中共同驱动特定行为的一小部分权重——该子网络与给定的 persona 相关联。
这与彩票假说 [@frankle2019lottery] 相呼应：密集网络包含稀疏子网络，这些稀疏子网络在给定任务上能够匹配完整模型的性能。
他们的核心主张是，预训练语言模型已经包含 persona 专属的子网络，其激活对特定行为档案的贡献不成比例。
其直觉是，与目标 persona 相关性最低的神经元会将模型推向其他人格的方向，因此掩盖这些网络组件将突显出预期的 persona。

该方法无需训练，每个 persona 只需要一个小的校准数据集 $\mathcal{D}_p$（数百个样本），然后分三步进行。
首先，计算 persona 特定输入上的逐神经元激活统计。
对于第 $l$ 层的神经元 $j$：

$$\mathbf{A}^{(l)}_p[j] = \mathbb{E}_{(x,y)\sim\mathcal{D}_p}\left[|\mathbf{h}^{(l)}_j(x)|\right]$$

其次，结合权重大小与源神经元激活大小，为每个连接计算重要性分数：

$$S^p_{ij} = |w_{ij}| \cdot \mathbf{A}^{(l)}_p[j]$$

第三，应用逐行 top-$K$ 剪枝：对每个权重矩阵的每一行，保留重要性分数最大的 $K$ 个连接。
这产生一个二进制掩码 $\mathbf{M}^p \in \{0,1\}^{m \times n}$，persona 特定的模型通过将该掩码应用于原始权重来获得：

$$\mathcal{M}_p = f(\theta \odot \mathbf{M}^p)$$

在推理时，切换 persona 等同于在其他冻结权重上交换一个二进制掩码——无需梯度更新，也无需除掩码本身之外的额外参数。
persona 向量在激活空间中施加的是*加法*干预，而 persona 子网络在权重空间中施加的是*乘法*干预，将与目标 persona 相关性较低的连接归零。
这一区别带来了实际的权衡：persona 向量使基础模型保持完整，而 persona 子网络服务于一个明显更稀疏的模型（作者每层剪枝高达 60% 的连接），这可能对通用能力产生意想不到的影响——如流畅性、事实记忆或推理能力——而粗粒度的基准测试可能无法揭示这些问题。


## Model Specifications

2024 年，OpenAI 分享了他们所称的"Model Spec"[@openai2024modelspec]，这是一份在启动 fine-tuning 运行前详细说明其目标模型行为的文档。
它涉及当前的模型行为、OpenAI 如何在 API 背后引导其模型，以及其模型未来将如何演变。
Model spec 的概念通常与 Anthropic 为 Claude 制定的 Constitution 相提并论，后者是一份用于塑造模型个性和价值观的文档。
这些文档面向不同的受众，具有不同的目标，但它们代表了各组织引导其模型、并就此向世界传达意图的早期范式。

Model spec 是行业和 RLHF 中为数不多的工具之一，可用于将模型的实际行为与设计者的意图进行比较。
正如本书所述，训练模型是一个复杂而多面的过程，因此最终结果与数据标注员指令或训练数据中任务比例等输入之间存在差异是预期之中的。
例如，一份完美执行的 model spec 比 Constitutional AI 中使用的原则列表更具揭示性，因为它传达的是过程的意图，而不仅仅是列举充当中间训练变量的内容。

Model spec 为模型发布流程中的每位利益相关者提供价值：

- **模型设计者**：模型设计者从中受益，因为他们需要厘清自己希望和不希望模型具备哪些行为。这使得数据上的优先级决策更加明确，有助于聚焦那些可能偏离长期方向的工作，并促使人们在复杂的评测体系中从更宏观的角度审视自己的模型。
- **开发者**：模型用户能更清楚地了解，他们遇到的某些行为究竟是有意为之——例如某些类型的拒绝——还是训练的副作用。这可以让开发者对使用该提供商未来更智能的模型更有信心。
- **旁观公众**：公众从 model spec 中受益，因为这是为数不多的公开信息来源之一，揭示了训练中的优先事项。这对监管审查以及制定关于 AI 模型应当做什么、不应当做什么的有效政策至关重要。

最近，Anthropic 在发布 Claude Opus 4.5 时一同公开了他们所称的"soul document"[@anthropic2025souldoc]（在公众用户从模型中提取出该文档后，Anthropic 确认了其存在），该文档详细描述了模型所期望具备的 character 特质、价值观和行为准则。
Claude character 研究的主要负责人 Amanda Askell 指出，supervised fine-tuning 和 reinforcement learning 方法均以 soul document 作为训练指南[@askell2025soul]。
这一做法代表了 Anthropic 早期 character training 方法与类似 model specification 的文档之间的融合趋势。

model spec 及相关文档面临的一个重大未知因素，是模型开发者为使模型遵循这些文档所投入的努力程度。
两个目标相似的组织可能最终走向截然不同的结果——一个可能为遵循一份平庸的 specification 投入了大量精力，而另一个可能对一份优秀的、公开记录的 spec 只投入了极少的跟进工作。

## Product Cycles、UX 与 RLHF 的未来

随着强大的 AI 模型逐渐从实验性机器学习流程的单一产物演变为更接近产品的形态，RLHF 已成为模型与产品关系的接口。
让模型易于使用所需的，远不止最终模型权重的正确性——还包括快速推理、合适的工具支持（如搜索或代码执行）、可靠且易于理解的用户界面（UX），以及更多。
RLHF 研究已成为大量此类测试的接口，这既因为 RLHF 被定性为实时理解用户对产品偏好的方式，也因为它是发布前的最后一个训练阶段。
将新功能添加到模型中最快捷的方式，是尝试在 post-training 阶段融入，因为那里的训练更快、成本更低。
这一循环已在图像理解、工具使用、更好的行为等方面得到印证。
产品问题很快演变为 RLHF 建模问题，若在该阶段取得成功，便会反向传播至更早期的训练阶段。
