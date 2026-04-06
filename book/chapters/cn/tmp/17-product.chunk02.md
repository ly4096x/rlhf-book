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
