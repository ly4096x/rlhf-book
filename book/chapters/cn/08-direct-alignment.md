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

接下来，我们需要引入一个配分函数 $Z(x)$：

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

配分函数作为非归一化密度 $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 的归一化因子，从而使其成为对于每个固定的 $x$，关于 $y$ 的合法概率函数。这一需求的具体原因将在推导过程中逐渐明朗。

将其代入后，我们得到中间变换形式：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

为了理解如何得到这一结果，请考虑 @eq:dpo_deriv_4 方括号内优化的内部部分：

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

然后，在两边加上 $\log Z(x) - \log Z(x)$：

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

再对各项进行分组：

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

利用 $\log(x) + \log(y) = \log(x\cdot y)$（并将 $Z$ 移至分母），得到：

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

接下来，将 $\frac{1}{\beta}r(x,y)$ 展开为 $\log \exp \frac{1}{\beta}r(x,y)$，并做相同操作，即可得到 @eq:dpo_deriv_5，在此略作改写：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

有了这一优化形式，我们需要实际求解最优 policy $\pi^*$。由于我们引入了配分函数 $Z(x)$，使得 $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 成为关于 $y$ 的合法概率分布，因此可以认识到内层期望实际上是一个真正的 KL divergence！

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

由于 $\log Z(x)$ 项与最终结果无关，可以忽略。这样就只剩下我们正在学习的 policy 与一个涉及配分函数、$\beta$、reward 以及 reference policy 的形式之间的 KL divergence。Gibbs 不等式告诉我们，该 KL divergence 在距离为 0 时取得最小值，而这仅在两个量相等时成立！因此，我们得到最优 policy：

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}


#### 2. 推导 Bradley Terry 模型的 DPO 目标函数

首先，回顾第 5 章关于 Reward Modeling 以及第 11 章关于 Preference Data 的内容，Bradley-Terry 人类偏好模型表述如下：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

通过对 @eq:dpo_opt_policy 进行变换，可以求解最优 reward。首先对两边取对数：

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

利用 $\log(abc) = \log a + \log b + \log c$ 展开右侧：

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

整理以求解 $r^*(x,y)$：

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

两边乘以 $\beta$：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

随后，将 reward 代入 @eq:bradley_terry_dpo 所示的 Bradley-Terry 方程，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

通过将指数表达式从 $e^{a+b}$ 分解为 $e^a e^b$，再消去 $e^{\log(Z(x))}$ 项，化简得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

然后，将分子和分母同乘以 $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

最后，根据 sigmoid 函数的定义 $\sigma(x) = \frac{1}{1+e^{-x}}$，我们得到：

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

这是在最优 policy $\pi^*$ 下，Bradley-Terry 模型给出的 preference data 的似然概率。回顾第 5 章关于 Reward Modeling 的内容，我们已推导出 Bradley-Terry 目标函数为最大化似然，即等价地最小化负对数似然，由此得到损失函数：
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

这就是 DPO 的损失函数，其形式如 @eq:dpo_core 所示。DPO 论文还额外推导了 Plackett-Luce 模型下的目标函数，但该形式在实践中较少使用 [@rafailov2024direct]。

#### 3. 推导 Bradley Terry DPO 梯度

我们在 @eq:dpo_gradient 中使用了 DPO 梯度来解释模型学习的直觉。要推导该梯度，需要对 @eq:dpo_loss_deriv4 关于模型参数求梯度。

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

首先，这个式子可以改写。我们知道 sigmoid 函数的导数 $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$，对数的导数 $\frac{d}{dx} \log x = \frac{1}{x}$，以及 sigmoid 的性质 $\sigma(-x)=1-\sigma(x)$，因此可以对上式进行变形。

令 $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$（即 sigmoid 内部的表达式），则有：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

展开并利用上述 sigmoid 和对数的表达式，得到前面引入的梯度：

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi(y_c|x)-\nabla_{\theta}\log\pi(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 数值问题、局限性与替代方法

DPO 算法的许多变体已被提出，以解决 DPO 的不足。例如，由于缺乏 rollout 过程（reward model 无法对生成结果进行评分），DPO 对每对 preference data 赋予相同的权重。而实际上，正如第 11 章关于 Preference Data 所示，有许多方式可以用比二元标签更丰富的标签来捕捉偏好信息。已有多种算法被提出，以重新平衡优化过程，使其不再对每对数据一视同仁。

- **REgression to RElative REward Based RL（REBEL）** 通过在chosen与rejected响应之间引入奖励模型的边际信号，而非仅依赖成对preference data，从而更准确地求解 RLHF 问题 [@gao2024rebel]。
- **Conservative DPO（cDPO）与 Identity Preference Optimization（IPO）** 通过假设 preference data 中存在噪声来解决过拟合问题。cDPO 假设 N% 的数据标注有误 [@rafailov2024direct]，IPO 则修改优化目标，将偏好概率软化而非直接从标签优化 [@azar2024general]。从实现角度看，IPO 将偏好概率替换为一个非线性函数，从而脱离 Bradley-Terry 假设，具体为 $\Psi(q) = \log\left(\frac{q}{1-q}\right)$。
- **DPO with an offset（ODPO）** "要求 preferred 与 dispreferred 响应的似然差超过某个偏移值" [@amini2024direct]——不对每对数据一视同仁，但这可能带来更复杂的标注环境。

DPO 的部分变体试图通过对 loss 进行小幅改动来提升学习信号，或通过减少内存占用来提高应用效率。

- **Odds Ratio Policy Optimization（ORPO）** 直接更新 policy model，向 chosen 响应靠拢，类似于指令 fine-tuning 的 loss，同时对 rejected 响应施加小额惩罚 [@hong2024reference]。这种 loss 函数的变化无需 reference model，从而简化了训练流程。理解 ORPO 的最佳视角是将其视为受 DPO 启发的方法，而非 DPO 的直接衍生。
- **Simple Preference Optimization SimPO** 对 DPO 优化做了一处细微改动：将 log 概率求和改为取平均（SimPO），或添加长度归一化，以提升性能 [@meng2025simpo]。

![DPO 中 preference displacement 示意图。](images/dpo_displacement.png){#fig:dpo_issue .center}

DPO 中一个*显而易见*的核心问题在于，优化目标仅驱动 chosen 与 rejected 响应的概率边际增大。
从数值上看，模型会同时降低 chosen 和 rejected 响应的概率，但*rejected 响应降低幅度更大*，如 @fig:dpo_issue 所示。
直觉上，这种泛化方式并不明显，但已有研究指出，这会提升未被涉及行为的概率——即语言模型可以生成、但不在 post-training 数据集分布中的 token [@razin2024unintentional] [@ren2024learning]。
一些简单方法——例如调整优化过程的 Cal-DPO [@xiao2024cal] 和修改奖励形状的 AlphaPO [@gupta2025alphapo]——可以缓解这种 **preference displacement**。
实践中，其具体影响尚不明确，但这为 online 方法能够超越原生 DPO 提供了一个潜在解释。

另一个被认为导致 DPO 类方法性能上限低于 online（基于 RL 的）RLHF 方法的主要原因是：其训练信号来源于其他模型或早期模型生成的补全。
DPO 的 online 变体通过在训练时生成新的补全并引入 preference 信号来缓解上述局限。**Online DPO** [@guo2024direct] 从当前模型中采样生成，**Discriminator-Guided DPO（D2PO）** [@singhal2024d2po] 则使用 reward model 重新标注，动态生成新的 preference data，此外还有许多其他变体。

其他 DAA 变体还有很多，例如 Direct Nash Optimization（DNO）[@rosset2024direct] 和 Binary Classifier Optimization（BCO）[@jung2024binary]，但算法选择远不如初始模型和所用数据重要 [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences]。

## 实现细节

DPO 等 DAA 的实现方式与 policy gradient 优化器有很大不同。
DPO loss 取自原始实现，其核心可概括如下 [@rafailov2024direct]：

```python
# Log-probability gaps for the policy and the frozen reference model
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# Difference of log-ratios: positive when the policy
# shifts probability toward the chosen completion
logits = pi_logratios - ref_logratios

# DPO loss: negative log-sigmoid drives the policy to
# widen the gap between chosen and rejected
losses = -F.logsigmoid(beta * logits)

# Implicit rewards (detached -- used for logging only)
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

由于上述信息已在模型前向传播过程中汇总（加上 reference model），该 loss 可直接用于标准语言模型训练框架。

总体而言，DAA 在大多数方面更为简单，提升了使用体验，但也带来了一套不同的注意事项。

1. **KL divergence 是静态的**：在 DPO 及其他算法中，KL divergence 由 $\beta$ 参数显式设定，用于平衡与优化目标之间的距离惩罚。这是因为 DPO 在给定数据的情况下直接朝 RLHF 目标的*最优解*迈进——即精确地步向由 $\beta$ 项所设定的解。而基于 RL 的优化器则根据当前 batch 和近期数据逐步更新。
2. **缓存 log 概率**：DPO 的简单实现方式会同时对 policy model 和 reference model 进行前向传播，以便计算 loss。但这会使内存占用翻倍，增加 GPU 使用量。为避免此问题，可先在整个训练数据集上计算 reference model 的 log 概率并缓存，然后在每个 batch 计算 loss 和更新参数时直接引用，从而将峰值内存减少约 50%。

## 使用合成 Preference Data 的 DAA

如今，大多数流行的 DAA preference fine-tuning 数据集都是合成 preference——由前沿模型对其他模型的输出进行评分，从而判定胜者与败者。
典型例子包括 UltraFeedback（该类别的首创）[@cui2023ultrafeedback]、Tülu 3（基于扩展的 UltraFeedback 方法构建）[@lambert2024t]、SmolLM 3 的数据 [@bakouch2025smollm3]，以及随 Olmo 3 发布的 Dolci Pref 数据集 [@teamolmo2025olmo3]。

构建这类数据集的最佳实践仍在不断演进。
2024 年 11 月发布的 Tülu 3 及同期数据集表明，合成成对 preference data 需要具备某种意义上的"on-policy"特性——即其中一部分补全来自你正在 fine-tuning 的模型（同时与更大模型池中的补全混合）。
这种 on-policy 数据的特性确保 DAA 能在模型生成的正确 token 空间内进行优化——因为这些 loss 函数是 contrastive 的，不像指令 fine-tuning 那样直接。
此后，随着 2025 年 Olmo 3 和 SmolLM 3 的发布，另一些研究支持了一种称为 Delta Learning 的不同理论，该理论认为 chosen 与 rejected 补全之间的差异比具体使用哪些模型生成补全更重要 [@geng2025the]。
例如，上述两个模型均采用 Qwen 3 32B 生成 chosen 响应、Qwen 3 0.6B 生成 rejected 响应——两组作者独立并发地提出了这一配对方案。

总体而言，鉴于实现简单且相对于基于强化学习的 preference fine-tuning 方法具有较强的性能表现，在合成 preference data 上使用 DAA 训练模型是大多数从业者的最佳起点。
使用大量合成 preference data 时也存在一些次要问题，例如用于评判补全优劣的模型所带来的偏差。
由于 GPT-4 等前沿模型已知存在长度偏差 [@dubois2024length] 以及倾向于偏好与自身风格相近的输出 [@panickssery2024llm]（详见第 12 章），数据集 "chosen" 部分中的文本来自 OpenAI 模型或其他风格相似的强模型的概率略高。

最后，本节将介绍这些方法如何改变被训练模型的生成行为的直觉理解。
从宏观角度看，大多数 DAA 的优化目标是扩大"chosen"与"rejected"补全概率之间的边际（少数不太流行的算法旨在略微改变这一动态，但核心不变）。
如本章前文所述（见 @fig:dpo_issue），这通常意味着两者的概率都会下降，但 rejected 响应下降幅度更大。
序列中每个 token 会根据其对整体 preference 边际的贡献程度，获得不同大小和方向的梯度，从而使优化器能够识别出哪些 token 对结果影响最大。

## DAA 与 RL：Online 数据与 Offline 数据

从宏观来看，争论归结为一个问题：我们是否需要 reinforcement learning 的内部机制——value function、policy gradient 等——来通过 RLHF 对齐语言模型？
与大多数以这种方式表述的问题一样，这个问题过于简化。
当然，两种方法都已得到充分验证，但有必要阐明两者在根本差异和性能表现上的分野所在。

多项研究得出结论：基于 policy gradient 的 RL 方法优于 DPO 及其变体。
这些论点形式各异，既有在控制数据条件下使用不同算法训练模型的研究 [@ivison2024unpacking] [@xu2024dpo]，也有研究 RL 优化循环中 on-policy 数据作用的工作 [@tajwar2024preference]。
在所有这些情况下，DPO 算法稍逊一筹。

尽管存在这一性能差距，DAA 因其简洁性仍在主流模型中得到广泛应用。
DAA 提供了一个可控的环境，使训练数据和其他配置的迭代能够快速进行，而且鉴于数据往往比算法更重要，使用 DPO 也完全可行。
