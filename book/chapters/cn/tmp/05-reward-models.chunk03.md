$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))]$$ {#eq:orm_loss}

其中 $r \in \{0,1\}$ 是二元标签，1 表示对给定 prompt 的正确回答，0 表示错误回答，$p_\theta(s)$ 是模型预测的与正确性概率成比例的 scalar。

实现一个 outcome reward model（以及我们将在 Process Reward Model 中看到的其他类型）涉及根据 completion 是否为正确样本，对每个 token 应用交叉熵损失。
这与语言建模损失更为接近，不需要标准 Bradley-Terry reward model 所要求的 chosen-rejected 结构化形式。

模型结构可以如下所示：

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The input data here will be tokenized prompts and completions along with labels
         per prompt for correctness.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar logit per token: (batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        # Only compute loss on completion tokens (labels 0 or 1)
        # Prompt tokens have labels = -100
        mask = labels != -100
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float()
            )
        return loss, logits
```

损失的简化版本如下：

```python
# Assume model already has: model.lm (backbone) + model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
# This will sometimes be compressed as model.forward() in other implementations

# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

这里的重要直觉在于，ORM 将在序列中每个 token 处输出一个正确性概率（仅由最终答案判断——推理错误不会在 ORM 训练过程中被捕获）。
这可能是一个有噪声的过程，因为更新和损失会根据结果和注意力映射逐 token 传播。

![在推理时，outcome reward model 在每个 token 处输出正确性概率。Prompt token 被遮蔽（例如，label=-100），而 completion token 各自接收一个概率，表示模型认为该回答能得出正确答案的可能性。](images/orm_inference.png){#fig:orm_inference}

![训练 outcome reward model 使用来自验证器或数据集的离线标签（例如，所有正确 completion 的标签为 1）。每个 completion token 使用二元交叉熵对结果标签进行训练，每个 token 的概率被汇总为最终得分，用于验证、过滤或重新排序。](images/orm_training.png){#fig:orm_training}

这些模型持续被使用，但在开源 RLHF 工具中的支持较少。
例如，同类型的 ORM 被用于开创性工作 *Let's Verify Step by Step* [@lightman2023let]，但没有损失中的语言建模预测部分。
因此，最终损失是对每个 token 的交叉熵损失，预测最终答案是否正确。

鉴于支持不足，outcome reward model（ORM）这一术语已被以多种方式使用。
部分文献，例如 [@lyu2025exploring]，继续使用 Cobbe 等人 2021 年的原始定义。
其他文献则不然。


## Process Reward Models

Process Reward Models（PRMs），最初称为 process-supervised reward models，是被训练为在思维链推理过程中每个*步骤*处输出得分的 reward model。
这与标准 RM（仅在 EOS token 处输出得分）或 ORM（在每个 token 处输出得分）不同。
Process Reward Models 需要在每个推理步骤末尾进行监督，然后以类似方式训练，即对步骤中的 token 训练其相关目标——PRM 中的目标是步骤，ORM 中的目标是整个回答。

遵循 [@lightman2023let]，二元标签的 PRM 通常使用逐步交叉熵损失进行优化：

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x)\right) \right] $$ {#eq:prm_loss}

其中 $s$ 是一个具有 $K$ 个标注步骤的思维链样本，$y_{s_i} \in \{0,1\}$ 表示第 $i$ 步是否正确，$r_\theta(s_i \mid x)$ 是 PRM 在给定原始 prompt $x$ 的条件下，预测步骤 $s_i$ 有效的概率。

以下是如何在 trainer 中打包这种逐步标签的示例，来自 HuggingFace 的 TRL（Transformer Reinforcement Learning）[@vonwerra2022trl]：

```python
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label 
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

传统上，PRM 使用语言建模 head 进行训练，该 head 仅在推理步骤末尾（例如在对应双换行符或其他特殊 token 的 token 处）输出 token。
这些预测通常为 -1 表示错误，0 表示中立，1 表示正确。
这些标签不一定与模型是否处于正确路径上相关，而是与步骤是否正确相关。

![Process reward model 仅在步骤边界处（例如换行 token）提供监督。每个步骤接收一个三分类标签：正确（+1）、中立（0）或错误（-1）。训练期间所有其他 token 均被遮蔽。](images/prm_training_inference.png){#fig:prm_training_inference}

以下展示了一个 PRM 的示例构建方式。

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
