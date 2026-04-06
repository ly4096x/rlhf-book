<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "指令微调"
prev-url: "04-instruction-tuning"
page-title: Reward Models
search-title: "第5章：Reward Models"
next-chapter: "强化学习"
next-url: "06-policy-gradients"
lecture-video: "https://youtu.be/4gIwiSPmQkU"
lecture-label: "第2讲：IFT、Reward Modeling 与拒绝采样（第4、5、9章）"
---

# Reward Modeling

Reward models 是现代 RLHF 方法的核心，负责学习复杂的人类偏好。
正是它们使我们的模型能够从难以明确指定的信号中进行学习。
它们将数据中的复杂特征压缩成一种可用于下游训练的表示——这是一种再次展现现代深度学习强大能力的"魔法"。
这些模型充当核心优化的代理目标，将在后续章节中加以研究。
如 @fig:rm-role-in-rlhf 所示，reward model 扮演着类似于标准 RL 环境的角色，为 agent 提供学习信号，但与固定环境不同的是，我们可以从人类偏好中学习得到它。

Reward models 在强化学习研究中被广泛用作环境奖励的代理 [@sutton2018reinforcement]。
Reward models 以其现代形式被提出，作为研究值对齐问题的工具 [@leike2018scalable]。
这些模型通常接受某种形式的输入并输出单一的 scalar reward 值。
该奖励可以有多种形式——在传统 RL 问题中，它试图近似问题的精确环境奖励，但我们将在 RLHF 中看到，reward models 实际上输出的是某个输入"质量较高"的概率（即在成对偏好关系中被选中的答案）。
针对 RLHF 的 reward modeling 实践与逆强化学习密切相关，后者的问题是在给定行为轨迹的情况下近似 agent 的奖励函数 [@ng2000algorithms]，以及深度强化学习的其他领域。
高层次的问题表述相同，但实现方式和关注领域完全不同，因此它们通常被视为完全独立的研究领域。

最常见的 reward model，通常称为 Bradley-Terry reward model，也是本章的主要关注点，它预测一段文本接近训练比较中"偏好"文本的概率。
本节后续部分还将对其与 Outcome Reward Models（ORM）、Process Reward Models（PRM）以及其他类型的 reward models 进行比较。

*在本章中，我们用 $x$ 表示提示（prompt），用 $y$ 表示补全（completion）。这种符号在语言模型文献中很常见，其中方法作用于完整的 prompt-completion 对，而非单个 token。*

![RLHF 中的 reward model 扮演着标准 RL 中返回奖励的环境组件角色。关键区别在于，在 RLHF 中，我们可以从人类偏好中控制并学习这个奖励函数，而不是由环境固定给定。](images/rlhf-overview.png){#fig:rm-role-in-rlhf}

## 训练 Bradley-Terry Reward Model

Reward model 的标准实现源自 Bradley-Terry 偏好模型 [@BradleyTerry]。
训练标准 RLHF reward model 有两种流行的表达形式——它们在数学上是等价的。
首先，Bradley-Terry 偏好模型定义了在对两个项目 $i$ 和 $j$ 进行成对比较时，评判者偏好 $i$ 胜于 $j$ 的概率：

$$P(i > j) = \frac{p_i}{p_i + p_j}.$$ {#eq:bradterry}

Bradley-Terry 模型假设每个项目具有一个潜在强度 $p_i > 0$，且观测到的偏好是这些底层强度的带噪反映。
通常对 Bradley-Terry 模型使用无界分数进行重参数化，其中 $p_i = e^{r_i}$，由此得到以下形式：

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).$$ {#eq:bradterry_unbounded}

只有分数之间的差值才重要：对所有 $r_i$ 加上相同的常数不会改变 $P(i > j)$。
这些形式并非自然法则，而是对人类偏好的有用近似，在 RLHF 中通常效果良好。

为了训练 reward model，我们必须构造一个满足上述关系的损失函数。
在实践中，这通过将语言模型转换为输出 scalar 分数的模型来实现，通常借助一个小型线性头，从模型的最终隐藏状态生成单一奖励值。
给定一个 prompt $x$ 和两个采样的 completion $y_1$ 与 $y_2$，我们用 reward model $r_\theta$ 对两者进行打分，并将条件分数写为 $r_\theta(y_i \mid x)$。

reward model 将 $y_1$ 优于 $y_2$ 的概率表示为：

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.$$ {#eq:bradterryrm}

我们将偏好的 completion 记为 $y_c$（chosen），将被拒绝的 completion 记为 $y_r$。

由此得到的损失函数鼓励 reward model 为人类偏好的 completion 分配比被拒绝的 completion 更高的分数，使用 sigmoid 函数将分数差转换为概率。
通过最大化上述函数的对数似然（或等价地最小化负对数似然），我们可以得到训练 reward model 的以下损失：

$$
\begin{aligned}
\theta^* = \arg\max_\theta P(y_c > y_r \mid x) &= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \arg\max_\theta \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\ 
&= \arg\max_\theta \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \arg\max_\theta \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \\
&= \arg\min_\theta - \log \left( \sigma \left(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\right) \right)
\end{aligned}
$$ {#eq:bradterryrm_deriv}

第一种形式是上面推导的 log-sigmoid 表达式，见 [@ouyang2022training] 等工作：
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

第二种是用 softplus 函数 $\log(1+e^x)$ 表达的数学等价形式，见 [@askell2021general] 等工作：
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

两者的等价性可通过令 $\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)$ 并利用 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$ 来验证，由此得出 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)$。
两种形式均出现在 RLHF 文献中。

![训练偏好 reward model 需要成对的 chosen 与 rejected completion。模型在序列结束（EOS）token 处为每个 completion 计算 scalar 分数，对比损失仅取决于两者之间的分数差。](images/pref_rm_training.png){#fig:pref_rm_training}

## 默认 Reward Model 架构

Reward models 最常见的实现方式是通过类似 Transformers 的 `AutoModelForSequenceClassification` 的抽象，该抽象在语言模型上附加一个小型线性头，并在训练或推理时为 prompt-completion 对生成 scalar reward 分数。
在推理时，模型以模型的单一 logit 形式输出*该文本片段被选中的相对可能性*。

也存在其他实现选项，例如直接从最终 embedding 提取线性层，但在开放工具中较为少见。

## 实现示例

实现 reward modeling 损失相当简单。
更多的实现挑战在于设置独立的数据加载器和推理流水线。
给定包含 tokenized、chosen 和 rejected prompt 及 completion 的正确数据加载器，损失实现如下：
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected 包含 prompt token x 以及各自的
# completion token（y_c 或 y_r），reward model 对其进行联合打分。
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

从更宏观的角度来看，这通常处于一个因果语言模型（从左到右逐个生成 token、在所有前序 token 的条件下预测每个 token 的模型）中，该模型附加了一个额外的头（使用上述损失进行学习），将最终隐藏状态转换为输入的分数。
代码接收标准的 transformer 输入——`input_ids`（tokenized 文本）和 `attention_mask`（标记真实 token 与填充 token）——并提取最后一个真实 token 处的隐藏状态（模型对输入的内部表示），然后通过线性层生成 scalar reward。
该模型的结构如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
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
