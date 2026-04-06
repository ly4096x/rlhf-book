        输入是经过tokenize处理的prompt和completion，其中"推理步骤"的结束由指定的separator token标记，例如换行符或其他特殊标记，而非批量填充（batch padding）。
        labels将是一个标签列表，包含True、False和Neutral（3个标签），由模型进行预测。
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 最终隐藏状态：(batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # 每个token对应一个logit向量：(batch, seq_len, num_classes)
        logits = self.head(hidden)

        # 仅在步骤边界处计算损失（labels != -100的位置）
        # 标签映射：-1 -> 0，0 -> 1，1 -> 2（类别索引）
        mask = labels != -100
        if mask.any():
            loss = F.cross_entropy(
                logits[mask], labels[mask]
            )
        return loss, logits
```

核心损失函数与outcome reward model非常相似，只是标签在不同的间隔处应用。
```python
# 假设模型为每个token输出3类logit
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 仅在步骤边界处的3类标签：0=-1，1=0，2=1（其余被遮蔽为-100）
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

## 比较各类 Reward Model（及 Value Function）

上述介绍的各类reward model揭示了在RLHF及其他post-training方法中衡量"质量"的多种方式。
下表总结了各模型的预测内容及训练方式。

::: {.table-wrap}
| 模型类别 | 预测内容 | 训练方式 | 语言模型结构 |
|------------|------------------|---------------------|--------------|
| **Reward Models** | 通过EOS token处的scalar score衡量completion的相对质量 | 对completion之间的成对（或N对）比较进行对比损失训练 | 在基础语言模型特征之上添加线性head |
| **Outcome Reward Models** | 每个token处答案正确的概率 | 使用标注的结果对（例如，在可验证领域中的成功/失败）进行训练 | 语言建模head，每token进行交叉熵，其中每个标签均为结果级别标签 |
| **Process Reward Models** | 推理步骤末尾的中间步骤reward或score | 使用中间反馈或逐步标注进行训练（按推理步骤中的每个token训练） | 语言建模head，仅对每个推理步骤进行推理，预测三个类别：-1、0、1 |
| **Value Functions** | 给定当前状态下的预期回报 | 通过对序列中每个点进行回归训练 | 具有每token输出的scalar回归head |
表：比较各类reward model。 {#tbl:rm_compare}
:::

关于此表中各类别之间的区别，需要注意以下几点，因为各模型类型之间的边界并不总是清晰的：

- 在preference tuning和reasoning training中，value function通常折扣因子为1，这使得value function与outcome reward model更为接近，但训练损失不同。
- Process reward model可以通过从中间状态进行rollout并收集结果数据来进行监督训练。这融合了多种思路，但如果*损失*是按推理步骤标签计算的，则最好将其称为PRM。

**如果使用正确/错误配对来训练Bradley-Terry成对模型会怎样？**
关于outcome reward model的大部分混淆来自于一小部分文献，这些文献在由答案正确性派生的成对数据上训练reward model。
在这种情况下，你将chosen响应设置为某个问题的正确答案，将rejected响应设置为*同一问题*的错误答案。
这在技术上不是ORM，仍然直接使用对比性的序列级损失进行训练。
从技术上讲，这仍然是一个Bradley-Terry模型，属于我们介绍的第一类模型。

**ORM 与 Value Function 的对比。**
ORM和value function可能看起来相似，因为两者都使用相同的head架构生成每token的输出，但它们在*预测内容*和*目标来源*上有所不同：

- **ORMs** 预测一个即时的、token局部的量：$p(\text{correct}_t)$ 或 $r_t$。目标来自*离线标签*（验证器或数据集将token/序列标记为正确或错误）。
- **Value functions** 预测*剩余*预期回报：$V(s_t) = \mathbb{E}[\sum_{k \geq t} \gamma^{k-t} r_k \mid s_t]$。目标通常*由当前policy $\pi_\theta$ 下的on-policy rollout计算*，并随policy变化而变化（从技术上讲，value function也可以是off-policy的，但这在语言建模工作中尚未得到确立）。

如果你定义一个密集的token reward $r_t = \mathbb{1}[\text{token is correct}]$ 并使用 $\gamma = 1$，那么ORM学习的是 $r_t$（或 $p(r_t = 1)$），而value head学习的是剩余求和 $\sum_{k \geq t} r_k$。
它们可以共享相同的基础模型和head维度，但*语义和监督流程*有所不同：ORM从固定标签离线训练，而value function在on-policy下训练，并用于为policy gradient计算优势 $A_t = \hat{R}_t - V_t$。

### 各类 Reward Model 的推理方式

这些模型在推理时（训练完成后）以不同方式处理数据，以应对RM所用于的一系列任务。

**Bradley-Terry RM（Preference Model）：**

- *输入：* prompt $x$ + 候选completion $y$
- *输出：* 来自EOS隐藏状态的单一scalar $r_\theta(x, y)$
- *用途：* 对$k$个completion重新排序，选取top-1（best-of-N sampling）；或为RLHF提供终端reward
- *聚合：* scalar输出无需聚合

**Outcome RM：**

- *输入：* prompt $x$ + completion $y$
- *输出：* 每个completion token处的概率 $p_t \approx P(\text{correct at token } t)$
- *用途：* 对已完成的候选进行评分；通过均值、最小值（尾部风险）或乘积 $\prod_t p_t$（等价于对数概率之和 $\sum_t \log p_t$）进行聚合
- *聚合选项：* 平均正确率、最小 $p_t$、最后 $m$ 个token的平均值，或当任意 $p_t < \tau$ 时进行阈值标记

**Process RM：**

- *输入：* prompt $x$ + 带有步骤边界的推理轨迹
- *输出：* 步骤边界处的分数（例如，正确/中性/错误的类别logit）
- *用途：* 对已完成的chain-of-thought进行评分；或通过剪枝低分分支来引导搜索/解码
- *聚合：* 在步骤（而非token）上进行——平均步骤分数、最小值（快速失败）或偏向后期步骤的加权求和

**Value Function：**

- *输入：* prompt $x$ + 当前前缀 $y_{\leq t}$（一个状态）
- 输出：completion中每个token位置处的 $V_t$（从状态 $t$ 开始的预期剩余回报）
- 用途：在RL训练期间计算每token优势 $A_t = \hat{R}_t - V_t$；每步的value作为基线
- *聚合：* 通常取最后生成token处的 $V$；其解释不同于"正确概率"

总结而言，理解各类模型的方式如下：

- **RM：** "这个完整答案有多好？" → scalar值
- **ORM：** "哪些部分看起来是正确的？" → 每token正确性
- **PRM：** "推理步骤是否合理？" → 每步分数
- **Value：** "从这里开始还有多少reward？" → RL优势的基线

## Generative Reward Modeling（即 LLM-as-a-judge）

由于preference data的高成本，一个大型研究领域应运而生，即使用现有的语言模型作为人类偏好的评判者，或用于其他评估场景 [@zheng2023judging]。
核心思路是向语言模型提供关于如何评判的指令、一个prompt以及两个completion（与人类标注者的做法类似）。
以下是来自聊天评估MT-Bench [@zheng2023judging] 这一开创性工作的示例prompt：

```text
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants.
Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
```

鉴于LLM-as-a-judge在评估中的有效性，衍生出了许多其他评估方法，如AlpacaEval [@dubois2024length]、Arena-Hard [@li2024crowdsourced] 和 WildBench [@lin2024wildbench]，许多研究者开始使用LLM-as-a-judge来代替reward model创建和使用preference data。

围绕如何使用所谓的"Generative Reward Models" [@mahan2024generative]
[@zhang2024generative] [@ankner2024critique]（包括专门训练为有效评判者的模型 [@kim2023prometheus]），已经出现了一整个研究领域，但在RM评估上，这些模型往往落后于现有的reward model，这表明reward modeling是当前RLHF的一项重要技术。

提升LLM-as-a-judge工作流鲁棒性的一个常用技巧是使用采样温度为0，以降低评分的方差。

## 延伸阅读

学术界对reward modeling的研究于2024年正式形成体系。
早期reward modeling的大部分进展集中在建立基准和识别行为模式上。
第一个RM基准RewardBench为测试reward model提供了公共基础设施 [@lambert2024rewardbench]。
此后，RM评估已扩展到与通用post-trained模型可用的评估类型相近的范围，其中一些评估测试在已知正确答案的领域中预测的准确性 [@lambert2024rewardbench]，另一些则更类似于通过LLM-as-a-judge进行的"氛围"评估或与其他基准的相关性 [@wen2024rethinking]。

新基准的示例包括：

- **纯文本（通用聊天/偏好）：** RMB [@zhou2024rmb]、RewardBench2 [@malik2025rewardbench]、Preference Proxy Evaluations [@frick2024evaluate] 或 RM-Bench [@liu2024rm]。
- **专项纯文本（数学等）：** 多语言reward bench（M-RewardBench）[@gureja2024m]、用于检索增强生成（RAG）的RAG-RewardBench [@jin2024rag]、用于拼写错误的ReWordBench [@wu2025rewordbench]、RewardMATH [@kim2024evaluating] 或 AceMath-RewardBench [@liu2024acemath]。
- **Process RMs：** PRM Bench [@song2025prmbench] 或 ProcessBench [@zheng2024processbench]，以及视觉基准 VisualProcessBench [@wang2025visualprm] 或 ViLBench [@tu2025vilbench]。
- **Agentic RMs：** Agent-RewardBench [@men2025agentrewardbench] 或 CUARewardBench [@lin2025cuarewardbench]。
- **多模态：** MJ-Bench [@chen2024mj]、Multimodal RewardBench [@yasunaga2025multimodal]、VL RewardBench [@li2024vlrewardbench] 或 VLRMBench [@ruan2025vlrmbench]。
