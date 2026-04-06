对于任何给定的任务，存在许多不同的 policy 都能实现高性能。
Shenfeld 等人 2026 [@shenfeld2026rls] 从互补视角探讨了 RL 的泛化问题，提出了 **RL's Razor** 论题，其核心表述如下：

> 在解决新任务的众多高奖励方案中，on-policy 方法（如 RL）天然偏向于在 KL divergence 上更接近原始 policy 的解。

![偏向 KL 最小解可减少遗忘。（左）在能解决新任务的 policy 中，RL 收敛于 KL 距离基础模型最近的那些。（右）与 SFT 相比，在新任务性能相当的情况下，这种 KL 偏向带来了更高的旧任务保留率。来自 Shenfeld、Pari 和 Agrawal 2026。许可证 CC-BY。](images/rl_razor_motivation.png){#fig:rl-razor-motivation}


作者发现，对过去任务的遗忘程度与 fine-tuned policy 偏离初始模型的程度（以 KL divergence 衡量）直接成正比：

$$
\text{Forgetting} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[\text{KL}\!\left(\pi_0(\cdot \mid x) \| \pi(\cdot \mid x)\right)\right]\right)
$$ {#eq:rl_razor_forgetting}


作者通过多种 RL 和 SFT 训练方式的实验，实证证明遗忘程度与训练后 policy 和初始 policy 之间的 KL divergence 高度相关（$R^2 = 0.96$），**且该 KL 是在新任务数据上测量的**。
这一结果令人惊讶，因为 KL 是在*新任务*的输入分布上测量的，而非在旧任务的留出数据上，然而它仍能预测旧任务上的性能下降。
在实践中，这为我们提供了一种强有力的手段，可以直接通过基础模型与训练后 policy 之间的偏移来估计遗忘程度——即在新的专项数据上测量 KL 距离。

为了深入探究是什么驱动了 RL policy 中更小的 KL 偏移，作者从两个维度分解了 RL 与 SFT 之间的差异——on-policy 数据与 offline 数据之别，以及目标函数是否包含负梯度（RL 中当样本得分低于奖励基线时存在负梯度，而 SFT 仅对正确示范进行强化，不包含负梯度）。
值得注意的是，他们发现 on-policy 数据与 offline 数据的差异完全解释了泛化性能上的不同，而负梯度的有无则没有可察觉的影响。

直觉上，on-policy 方法采样的是模型本身已赋予非可忽略概率的输出，因此每次更新都被约束在接近当前分布的范围内。
相比之下，SFT 训练于一个固定的外部分布，该分布可能与模型当前产出的内容相距甚远，而每一个梯度步骤都会朝着那个遥远的目标靠拢，无论模型自身的倾向如何。

## 其他类型的 Regularization

在 post-training 文献中，许多顶尖模型都加入了其他 regularization 方法，以在各自的设置中达到领先性能。
以下两个例子旨在展示一些领先模型如何调整 post-training 设置以获得稳定的优化过程，而非作为在所有设置中都必然奏效的通用工具。
还有无数更具创意的解决方案有效且将被发现！

### Pretraining Gradients

另一种看待 regularization 的方式是：你可能有一个*数据集*，希望模型保持接近该数据集，正如 InstructGPT [@ouyang2022training] 中所做的那样，"以修复在公开 NLP 数据集上的性能退化"。
为了实现这一点，他们修改了 RLHF 的训练目标。
以 @eq:rl_start 为基础，我们可以将其转化为一个优化目标函数：从 RL policy 模型中采样，从用于 RLHF 的 RL 数据集中对 prompt $x$ 生成补全 $y$，得到：
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

随后，我们可以为标准自回归下一个 token 预测损失（即 pretraining 时所用的损失）添加一个额外奖励，其数据来自预训练语料库（或其他数据集）中采样的文档，以维持文本连贯性：

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right]
$$ {#eq:objective_pretraining}

近期研究提出使用负对数似然（negative log-likelihood）项来平衡 Direct Preference Optimization（DPO）的优化 [@pang2024iterative]。
鉴于 DPO 损失的成对性质，同样的损失修改方式也可应用于 reward model 训练，约束模型预测准确的文本（这一做法来自未发表论文的实验室传言）。

优化过程是对 DPO 的修改：
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

其中 $P_{\theta}$ 为可训练的 policy 模型，$P_{\text{ref.}}$ 为固定的参考模型（通常为 SFT 的 checkpoint），$(c_i^w, y_i^w)$ 和 $(c_i^l, y_i^l)$ 分别表示 prompt $x_i$ 对应的优胜补全与落败补全。
第一项是标准 DPO logistic 损失：通过对数似然比之差 $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$ 增大优胜与落败之间的差距，$\beta$ 控制该偏好信号偏离参考模型的强度。
第二项是对优胜补全施加的长度归一化负对数似然惩罚，权重为 $\alpha$，其作用是在绝对语言建模意义上保持优选文本的高似然性，而不仅仅是相对于被拒绝样本更优。

### Margin-based Regularization

在 RLHF 体系的其他部分，对优化的控制定义得没那么清晰。
大多数 reward model 除了标准对比损失函数之外没有额外的 regularization。
Direct Alignment Algorithms 通过 $\beta$ 参数以不同方式处理对 KL divergence 的 regularization（参见 [direct alignment 章节](https://rlhfbook.com/c/08-direct-alignment)）。

Llama 2 为 reward model 训练提出了一种 margin 损失 [@touvron2023llama]：

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

其中 $m(y_c, y_r)$ 是两个数据点 $y_c$ 和 $y_r$ 之间的 margin，代表两位标注者评分之差的数值。
这可以通过让标注者对输出进行数值评分，或使用量化排名方法（如 [Likert 量表](https://en.wikipedia.org/wiki/Likert_scale)）来实现。
