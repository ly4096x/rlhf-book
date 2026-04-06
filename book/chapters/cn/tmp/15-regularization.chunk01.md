<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "过度优化"
prev-url: "14-over-optimization"
page-title: Regularization
search-title: "第15章：Regularization"
next-chapter: "评估"
next-url: "16-evaluation"
---

# Regularization

在整个 RLHF 优化过程中，会使用许多 regularization 步骤来防止对 reward model 的过度优化。
在这些情境中，过度优化的表现是模型输出无意义的文本。
优化"失控"的一些例子包括：模型输出逻辑上可跟随的数学推理，但答案极度错误；文本重复；语言切换；或大量特殊字符。
本章介绍用于控制模型优化的各种方法。

目前最流行的变体（在撰写本文时已被大多数 RLHF 实现所采用）是当前 policy 与参考 policy 在生成样本上的 KL distance。
"KL distance"是一个口语化术语，用于表达训练过程中的*优化距离*，尽管 KL divergence——衡量两个概率分布差异的底层数学方法——并不满足成为真正距离度量所需的形式化属性（相比于"两个分布之间的数值差异度量"，将其称为距离更为简便）。
许多其他 regularization 技术在文献中出现后，又在该研究系列的下一个模型迭代中消失。
也就是说，核心 KL distance（基于生成样本）之外的 regularization 通常用于稳定实验设置，而这些设置在下一代模型中往往可以被简化。
尽管如此，理解 RLHF 中用于约束优化的工具仍然非常重要。

*在本章中，我们用 $x$ 表示提示词，用 $y$ 表示补全内容。这种表示方式在语言模型文献中很常见，其中的方法作用于完整的提示词-补全对，而非单个 token。*

当在带有 reward model $r_\theta$ 的 RLHF 框架中使用时，一般公式如下：

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

其参考实现为：

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL 优化中的 KL Divergence

有关数学定义，请参阅附录 A（定义）。
KL divergence 衡量一个概率分布偏离另一个概率分布的程度——当 KL 为零时，两个分布产生相同的输出。
其定义如下：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

在 RLHF 中，关注的两个分布通常是新模型版本的分布（设为 $P(x)$）和参考 policy 的分布（设为 $Q(x)$）。
不同的优化器使用不同的 KL 方向。在本书中，最常用的"KL Penalty"被称为相对参考 policy 的反向 KL。在实践中，这简化为一种蒙特卡洛估计：从 RL 模型中采样 token，并从参考模型计算概率。直觉上，这种反向 KL 具有一种数值属性：当新模型 $P$（即 $\pi_{\text{RL}}$）在原始参考模型赋予低概率的区域放置大量概率质量时，会施加较大的惩罚。

另一种 KL 方向在机器学习中也常被使用，例如在某些 RL 算法的内部 trust region 计算中。直觉上，当新模型的更新*未将*概率赋予 $Q$（即 $\pi_{\text{ref}}$）中的高概率区域时，该惩罚会对新模型施加惩罚。这更接近于用于蒸馏或行为克隆的目标。

### 参考模型与生成内容

KL penalty 最常见的实现方式是将训练期间生成的 token 与静态参考模型之间的距离进行比较。
其直觉是：你希望训练后的模型保持接近原始模型的风格。
这个参考模型通常是经过指令微调的模型，但也可以是之前的 RL checkpoint。
通过简单替换，我们采样的模型变为 $\pi_{\text{RL}}(x)$ 和 $\pi_{\text{ref}}(x)$，如上文 @eq:kl_standard 所示（在标准定义中通常对应 $P$ 和 $Q$，当用于 RL KL penalty 时）。
这种 KL divergence penalty 早在大型语言模型流行之前就首次被应用于对话 agent [@jaques2017sequence]，而 KL control 很快被确立为 fine-tuning 预训练模型的核心技术 [@jaques2020human]。

### 实现示例

在实践中，KL divergence 的实现通常采用近似方法 [@schulman2016klapprox]，从而使实现大为简化。
根据上述定义，当直接从分布 $P(x)$ 采样时，KL 的求和可以转化为期望。
在这种情况下，分布 $P(x)$ 是当前正在训练的模型（而非参考模型）的生成分布。
这样，KL divergence 的计算变为：

$$
\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

这种方式实现起来更为简单，特别是在直接处理语言模型训练中频繁使用的对数概率时。

```python
# 第1步：从你的 policy 中采样（或以其他方式生成）一个序列
generated_tokens = model.generate(inputs)

# 第2步：在两个模型下对生成的序列进行打分
#    对于自回归语言模型，通常做法是：
#      inputs_for_scoring = generated_tokens[:, :-1]
#      labels           = generated_tokens[:, 1:]
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# 转换为对数概率，然后对齐标签以索引到 logits 中
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# 收集实际下一个 token 的对数概率
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# 现在可以对这些值求和（或取平均）以得到序列对数概率，
# 并计算 KL：
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

一些示例实现包括 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 和 [Hamish Ivison 的 Jax 代码](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。


## 隐式 Regularization

本章其他部分描述的是*显式* regularization：KL penalty、预训练梯度和 margin loss，这些都是从业者有意添加到训练目标中的方法。
越来越多的实证研究表明，基于 RL 的后训练还提供了*隐式* regularization——一种对记忆化和灾难性遗忘的内在抵抗力，这种抵抗力源于 on-policy 优化本身的结构。
这是由于损失更新的性质所致，即使没有任何用于控制 RL 训练的显式工具（如 KL penalty 或 replay buffer）。

### SFT 记忆，RL 泛化

后训练社区面临的核心问题是：在单一任务上训练时，模型是学习了一种可迁移到未见变体的可泛化规则，还是仅仅记忆了训练分布的表层模式？
Chu 等人 2025 [@chu2025sft] 通过一项受控实证研究回答了这一问题，该研究直接隔离了后训练方法——SFT 与 RL——对分布外（OOD）泛化的影响。
结论很明确：RL 学习的是可迁移的规则，而 SFT 则记忆训练数据，在分布偏移下出现崩溃。

该研究使用了两个内置规则变体的环境来理解其中的权衡：

- **GeneralPoints** 是一个算术纸牌游戏，模型接收四张扑克牌，需要用运算符（+、-、*、/）将其数值组合以达到目标数字（默认为 24）。OOD 测试会改变花牌的计分规则：训练时使用一种规则（J、Q、K 均计为 10），评估时使用另一种规则（J = 11，Q = 12，K = 13）。
