```python
import torch.nn.functional as F

class BradleyTerryRewardModel(nn.Module):
    """
    用于 Bradley-Terry 偏好学习的标准 scalar reward model。

    用法（成对 BT loss）：
        rewards_chosen = model(**inputs_chosen)    # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # 例如，AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        获取每个序列的单一向量用于评分。
        默认：最后一个非填充 token（EOS token）；若无 mask，则取最后一个 token。
        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """

        # 每个序列中最后一个非填充 token 的索引
        # attention_mask 对真实 token 为 1，对填充 token 为 0
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        一次前向传播，用于展示标准 reward model 的推理结构。
        若要训练，需修改此函数以从 chosen 和 rejected 输入中计算 reward，
        并应用上述 loss。
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 最终隐藏状态：(batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]

        # 每个序列一个 scalar reward：(batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)

        return rewards
```

在本节及后续内容中，reward model（以及后训练的大部分内容）的实现复杂性主要集中在正确构建数据加载器和分布式学习系统上。
需要注意的是，在训练 reward model 时，最常见的做法是只训练 1 个 epoch，以避免过拟合。

## Reward Model 变体

Reward modeling 是 RLHF 中一个相对尚未充分探索的领域。
传统的 reward modeling loss 在许多流行工作中已被修改，但这些修改尚未固化为单一的最佳实践。

### Preference Margin Loss

当标注者以 Likert 量表（一种具有有序类别的评分量表，用于指示偏好程度，例如 1 到 5 分）提供分数或排名时，相关量的大小可用于训练。
最常见的做法是沿偏好方向将数据二值化，将相对评分或排名强度的混合信息简化为仅保留 chosen 和 rejected 的完成结果。
利用偏好程度等附加信息来改善模型训练已有先例，但尚未成为标准做法。
Llama 2 提出使用两个数据点之间的 margin $m(y_c, y_r)$ 来区分偏好的强度：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$ {#eq:rewardmodelingmargin}

例如，每个完成结果通常按质量被赋予 1 到 5 的排名。
若 chosen 样本被赋予 5 分，rejected 样本被赋予 2 分，则 margin 为 $m(y_c, y_r)= 5 - 2 = 3$。
也可以探索其他计算 margin 的函数。

值得注意的是，在 Llama 3 中，margin 项被移除了，因为该团队观察到在规模扩大后改进效果逐渐减弱。

### 平衡每个 Prompt 的多次比较

InstructGPT 研究了每个 prompt 使用 $K = 4$ 到 $9$ 个完成结果进行排名的影响，从每个 prompt 产生 $\binom{K}{2}$ 个成对比较 [@ouyang2022training]。
为此，他们对每个 prompt 的每次比较的 loss 更新进行了加权。
在实现层面，可以通过将具有相同 prompt 的所有样本放入同一训练批次来自动实现这一点，从而自然地对不同的对加权——否则，由于单个 prompt 会出现在许多独立的批次中，可能会导致对 prompt 的过拟合。
loss 函数变为：

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodelinginstructgpt}


### K-wise Loss 函数

还有许多其他公式可以为 RLHF 构建合适的人类偏好模型。
其中一个例子是 K-wise loss 函数，基于 Plackett-Luce 模型 [@liu2019learning]，被用于早期流行的经 RLHF 训练的模型 Starling 7B 和 34B [@zhu2024starling]。

Zhu et al. 2023 [@zhu2023principled] 将该设置形式化如下。
给定一个 prompt（即状态）$s^i$，从 $P(a_0,\cdots,a_{K-1}|s^i)$ 中采样 $K$ 个动作 $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$。
然后，使用标注者对偏好进行排名，$\sigma^i: [K] \mapsto [K]$ 是一个表示动作排名的函数，其中 $\sigma^i(0)$ 是最受偏好的动作。这产生了对所有 $K$ 个项完整排名的 Plackett-Luce 概率：

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

当 $K = 2$ 时，这简化为用于成对比较的 Bradley-Terry (BT) 模型。
无论如何，一旦训练完成，这些模型在 RLHF 训练期间的使用方式与其他 reward model 类似。


## Outcome Reward Models

<!-- Huge thanks to Hangliang Ren, graduate student at Northeastern University for helping with this section (and PRMs), see https://github.com/myhott163com/RLHF_ORM_PRM -->

语言模型和其他 AI 系统的大多数 *preference tuning* 是通过上述 Bradley Terry 模型完成的。
对于推理密集型任务，可以使用 Outcome Reward Model（ORM）。
ORM 的训练数据构建方式与标准 preference tuning 类似。
这里，我们有一个问题陈述或 prompt $x$，以及两个完成结果 $y_1$ 和 $y_2$。
此处使用的归纳偏置是：一个完成结果应是该问题的正确解答，另一个则是错误的，从而得到 $(y_c, y_{ic})$。

所使用的模型架构与标准 reward model 非常相似，在能够输出单个 logit 的模型后附加一个线性层（在 RM 的情况下）——对于 ORM，随后的训练目标略有不同 [@cobbe2021gsm8k]：

> 【我们】使用联合目标来训练验证器，其中模型学习将模型的完成结果标记为正确或不正确，同时保留原始的语言建模目标。
> 在架构上，这意味着我们的验证器是语言模型，带有一个小型 scalar 头，逐 token 输出预测。
> 我们将此 scalar 头实现为作用于语言模型最终 unembedding 层输出的 logit 的单个偏置参数和单个增益参数。

换言之，这是一个每 token 可预测两类（1 表示正确，0 表示不正确）的语言建模头，而不是传统 RM 的对整个序列输出一个 logit 的分类头。
形式上，遵循 [@lyu2025exploring]，这是一个逐 token 的二元交叉熵 loss：
