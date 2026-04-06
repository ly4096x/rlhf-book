<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "推理"
prev-url: "07-reasoning"
page-title: 直接对齐
search-title: "第8章：直接对齐"
next-chapter: "拒绝采样"
next-url: "09-rejection-sampling"
---

# 直接对齐算法（DAAs）

直接对齐算法（DAAs）允许在无需训练中间 reward model 或使用 RL 优化器的情况下，对模型进行更新以求解相同的 RLHF 目标。
DAAs 解决的是我们一直在研究的同一种 preference learning 问题（使用完全相同的数据！），目的是使语言模型更加对齐、更加智能、更易于使用。
由于不需要 reward model 和在线优化，DAAs 的实现要简单得多，从而减少了训练过程中的计算开销，也使实验更加便捷。
本章详细介绍了推导这些算法所涉及的复杂数学过程，并进一步说明了这些有时颇为繁琐的推导最终会产生简洁的实现方案。

最具代表性的 DAA、同时也是催生了整个学术界对齐语言模型运动的算法，是 Direct Preference Optimization（DPO）[@rafailov2024direct]。
DPO 的核心是利用梯度上升来求解相同的受约束 RLHF 目标（参见第3章）：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:review_rlhf}

自2023年5月发布以来，在社区历经短暂摸索找到配合 DPO 使用的正确数据和超参数（尤其是出人意料地低的学习率）之后，许多主流模型先后采用了 DPO 或其变体——从2023年10月率先起步的 Zephyr-$\beta$ [@tunstall2023zephyr]，到 Llama 3 Instruct [@dubey2024llama]、Tülu 2 [@ivison2023camels] 与 Tülu 3 [@lambert2024t]、Nemotron 4 340B [@adler2024nemotron]，以及其他许多模型。
从技术角度来看，Sequence Likelihood Calibration（SLiC-HF）才是最早发布的现代 direct alignment 算法 [@zhao2023slic]，但由于多重因素的共同作用，它并未得到广泛采用（厘清研究方法被采纳或被忽略的缘由，向来是一项棘手的任务）。

DPO 和 DAAs 最具影响力的贡献在于降低了语言模型 post-training 实验的门槛——所需计算量更少，从零实现更为简便，在玩具示例和生产环境中都更容易跑通。

*在本章中，我们用 $x$ 表示提示（prompt），用 $y$ 表示补全（completion）。这一符号约定在语言模型文献中十分常见，其中的方法作用于完整的 prompt-completion 对，而非单个 token。*

## Direct Preference Optimization（DPO）

下面我们将解释 DPO 工作原理的直觉理解，并完整重新推导其核心方程。

### DPO 的工作原理

从表面上看，DPO 是直接优化一个 policy 来求解 RLHF 目标。
其 loss function（我们将在下方推导中重新审视）比较的是：相对于 reference model，已学习 policy 对 chosen 补全和 rejected 补全的概率发生了多大变化。
由 Bradley-Terry reward model 推导而来的 loss function 如下：

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

在 sigmoid 内部，第一项 $\beta \log \frac{\pi_{\theta}(y_c | x)}{\pi_{\text{ref}}(y_c | x)}$ 衡量的是 policy 相对于 reference model 对 *chosen* 补全概率提升的程度，第二项则对 *rejected* 补全做同样的衡量。当 chosen 端的偏移量超过 rejected 端时，loss 下降——即 policy 学会了偏好正确的回复。

在整个推导过程中，$\beta$ 是一个超参数，用于平衡 reward 优化与最终模型和初始参考之间的 KL divergence（即平衡过度优化，这是正确使用 DPO 时的关键超参数）。
这依赖于 DPO training 中用来替代外部 reward model 的隐式 reward，它是概率的对数比：

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

其中 $\pi_r(y \mid x)$ 是我们正在求解的精确最优 reward policy。
这来自于相对于最优 policy（见 @eq:dpo_opt_policy）对 Bradley-Terry reward 的推导，如第5章 Bradley-Terry 模型部分所示。
本质上，正如 DPO 论文所述，这一重参数化使我们能够"以最优 policy 而非 reward model 的形式表达人类 preference data 的概率"——这意味着我们可以完全绕过显式 reward model 的学习。

让我们来考察优化器需要最小化的 @eq:dpo_core 所示的 loss。
当 chosen 回复的对数比大于 rejected 回复的对数比（均经 reference model 归一化）时，loss 更低。
在实践中，这是模型在数据中呈现的 token 序列上的对数概率之和。
因此，DPO 在扩大 chosen 和 rejected 回复之间相对对数概率的差距。

利用 @eq:dpo_reward 中的 reward，我们可以写出 loss 的梯度，以进一步理解其中发生的事情：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\Big[ w \cdot \left(\nabla_{\theta}\log \pi(y_c \mid x) - \nabla_{\theta}\log \pi(y_r \mid x)\right) \Big]$$ {#eq:dpo_gradient}

其中 $w = \sigma\!\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right)$。

该梯度通过以下方式求解上述目标：

- sigmoid 函数 $\sigma(\cdot)$ 内的第一项产生一个0到1之间的参数更新权重，当 reward 估计不正确时权重更高。当 rejected 样本比 chosen 样本更受青睐时，权重更新应当更大！
- 其次，内层括号 $[\cdot]$ 中的项提高了 chosen 回复 $y_c$ 的似然，并降低了 rejected 回复 $y_r$ 的似然。
- 这些项由 $\beta$ 加权，$\beta$ 控制着更新在正确排序补全与 KL divergence 之间的平衡。

核心直觉在于：DPO 正在拟合一个隐式 reward model，其对应的最优 policy 可以以闭合形式提取（@eq:dpo_opt_policy，借助梯度下降和我们的机器学习工具）。
由于 DPO loss 是直接可微的，因此可以直接计算精确梯度，而无需通过训练 reward model 并采样补全来打分这一间接途径。
常被误解的是，DPO 在其核心处是在学习一个 reward model，这也是该论文副标题的含义——*你的语言模型秘密地是一个 Reward Model*。
人们很容易将其与 DPO 目标直接训练 policy 相混淆，因此研究下方的推导有助于形成完整的理解。

借助隐式 reward model 的学习，DPO 在给定数据集中的数据和目标中特定 KL 约束 $\beta$ 的条件下，生成 RLHF 目标的最优解。
在这里，DPO 求解特定 KL divergence 下的精确 policy，因为其生成不像 policy gradient 算法那样是在线的——这是与 RL 偏好调整方法的核心差异。
在许多方面，这使得 DPO 中 $\beta$ 值的调整比在线 RL 方法更容易，但至关重要且直觉上也合理的是，最优值取决于正在训练的模型及训练它的数据。

在每一批 preference data 中，由许多补全对 $y_{chosen} \succ y_{rejected}$ 组成，DPO 直接朝最优解方向进行梯度步进。
这比 policy gradient 方法简单得多。

![DPO 首次发布时，在研究社区中引发了关于如何最好地进行 RLHF 和 preference learning 的激烈辩论。这个表情包很好地捕捉了当时的情绪——争论往往显得强迫且夸张，但许多初学者和顶级实验室的研究人员都从 DPO 中获得了巨大的收益。DPO 简洁性表情包，致谢 Tom Goldstein。](images/dpo_meme.jpeg){#fig:dpo-meme}


### DPO 推导

DPO 推导分为两个主要部分。
首先，作者展示了最优求解本书中所用 RLHF 目标的 policy 形式。
其次，他们展示了如何从成对 preference data（即 Bradley-Terry 模型）中得到该解。

#### 1. 推导最优 RLHF 解

首先，让我们再次考虑 RLHF 优化目标，此处表明我们希望最大化以下量：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:rlhf_opt_eq_repeat}

这里，双重期望仅适用于计算期望 reward 的采样过程，因为 KL 项仍是一个解析表达式。
首先，让我们展开 KL divergence 的定义。回顾 $\mathcal{D}_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$，其中求和中的 $\pi(y|x)$ 权重成为采样分布。
由于两项现在共享相同的关于 $y \sim \pi(y|x)$ 的期望，我们可以将它们合并：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

接下来，将负号从括号内的差式中提出。为此，将其拆分为两项：

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

然后，乘以 $-1$ 将最大化转化为最小化：

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

除以 $\beta$ 并重新合并：

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}
