
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
