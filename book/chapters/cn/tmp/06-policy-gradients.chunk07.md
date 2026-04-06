### 损失聚合的权衡

在使用语言模型实现任何 policy gradient 算法时，都会遇到一个问题：如何将每个 token 的损失聚合为最终的标量损失？
给定样本 $i$ 在 token $t$ 处的每 token 损失 $\ell_{i,t}$，其中补全长度为 $|a_i|$，批次大小为 $B$，主要有三种策略：

**策略 1：按序列归一化**（标准 GRPO；部分 PPO 实现也采用此方式）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_per_sequence}

每条序列对批次损失的贡献相同，与序列长度无关。代码如下：

```python
# Strategy 1: Per-sequence normalization
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

**策略 2：按 token 归一化**（DAPO [@yu2025dapo]）

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$ {#eq:loss_per_token}

每个 token 对梯度的贡献相同；较长的序列对梯度的影响按比例更大。代码如下：

```python
# Strategy 2: Per-token normalization
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

**策略 3：固定长度归一化**（Dr. GRPO [@liu2025understanding]）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{L_{\max}} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_fixed_length}

以最大序列长度 $L_{\max}$ 进行归一化，在保持序列间每 token 尺度一致的同时，仍允许较长序列因包含更多有效 token 而贡献更多总梯度。代码如下：

```python
# Strategy 3: Fixed-length normalization 
fixed_len_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
            L_max).mean()
```

其中 $L_{\max}$ 通常是整个训练过程中的全局常量，用于指定生成 token 的最大数量。

注意，上述代码中的 `completion_mask` 是一个由 0 和 1 组成的矩阵，其中 prompt token 被掩码（置为 0），因为我们不希望模型从预测 prompt token 中学习。

#### 为什么这很重要？

直觉上，按序列归一化（策略 1）看起来最合理，因为我们关注的是*结果*，而非单个 token。
然而，这会基于序列长度引入微妙的偏差，可能导致模型过度思考，或根据偏差方向对那些本需要使用更多 token 的策略赋予较低权重。
考虑以下两条不同长度的序列，其每 token 损失为：

```python
seq_1_losses = [1, 1, 1, 1, 10]  # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

使用**策略 1**（按序列）：批次损失为 $(2.8 + 1.9)/2 = 2.35$，关键在于，短序列中每个 token 收到的梯度大于长序列中的 token。

使用**策略 2**（按 token）：批次损失为 $(14 + 19)/15 = 2.2$，所有 token 收到相同量级的梯度。

使用**策略 3**（固定长度，$L_{\max}=10$）：短序列贡献 $1.4$，长序列贡献 $1.9$，在按序列加权的同时平衡了每 token 的梯度。

如需查看更完整的示例以了解这些策略如何影响梯度，请参见下方脚本。

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """Compute sum of tensor with masked values. Use a constant to normalize."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)


advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0,],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)
```
