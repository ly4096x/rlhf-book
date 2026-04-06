缓解这种行为的公认最佳实践是修改训练数据（例如使用第17章介绍的"角色训练"等方法）。
如今，大量面向AI应用的fine-tuning工作，是在已经过大量RLHF及其他post-training的所谓"Instruct"或"Thinking"模型基础上进一步fine-tuning完成的。
这些已经训练好的模型往往更难被改变，例如要消除其过度拒绝的倾向，通常从大规模自回归预训练结束后直接使用base model来引导此类行为是最优选择。

## 量化 Over-optimization

Over-optimization 同样是一个专门的研究领域，研究者在其中探讨模型性能与KL优化距离之间的关系 [@gao2023scaling]。
回顾一下，KL距离是衡量训练前原始模型（即reference model）与当前 policy 概率分布之间差距的指标。
例如，@fig:overoptimization 中所示的关系，也可以用横轴表示KL优化距离而非训练步数来呈现。
下面还有一个额外的示例：将一个偏好tuning数据集对半分割，分别用于训练reward model（偏好模型，下文简称PM）和测试reward model。
随着训练的持续推进，训练RM上的提升最终无法迁移到测试PM，大约在训练样本量达到约15万时即出现此现象 [@bai2022training]。

Over-optimization 对于 RLHF 来说是根本性且不可避免的，这源于reward signal的软性本质——它是一个学习所得的模型——而传统RL文献中的reward function旨在完整捕捉世界动态。
因此，这是一个 RLHF 永远无法彻底解决的根本性优化问题。

![来自 Bai et al. 2022 的训练与测试 RM 的 over-optimization 情况。许可证：CC-BY。](images/anthropic_overoptimization.png){#fig:anthropic_overoptimization width=450px}

采用不同的 RLHF 训练方法，所消耗的KL距离也会有所不同（是的，研究人员在训练过程中会密切跟踪KL divergence指标，比较不同实验中模型的变化幅度——KL divergence指标过大往往意味着潜在的bug或模型出现问题）。
例如，在线RL算法（如PPO）修改模型参数时使用的KL距离，远高于推理时采样方法（如best-of-N采样，BoN）所使用的KL距离。
在RL训练中，更高的KL penalty会在给定KL距离处减少over-optimization，但可能需要更多的整体训练步数才能使模型达到这一状态。

缓解 over-optimization 的方案有很多。
其中包括：使用更大的 policy 模型（参数空间更大，在保持较小KL距离的同时有更多余地提升reward）、reward model集成 [@coste2023reward]，以及更换优化器 [@moskovitz2023confronting]。
尽管直接对齐算法同样容易发生 over-optimization [@rafailov2024scaling]，但其优化过程的直接性使得使用固定KL距离成为可能，从而更易于管理这一权衡。

## 错位与 RLHF 的角色

尽管工业界的 RLHF 与 post-training 正在扩展，涵盖远超最初推动 RLHF 发明的对齐目标，但 RLHF 的未来仍与对齐密切相关。
在本章语境下，over-optimization 会导致模型产生*错位（misalignment）*。
在当前语言模型的研究中，已有大量研究表明 RLHF 技术可能会改变模型行为，使其偏离对人类用户和整体社会需求的对齐。
当前 RLHF 技术中一个典型的错位示例，是关于现有技术如何助长奉承行为（sycophancy）的研究 [@sharma2023towards]——即模型倾向于告诉用户他们想听到的内容。

这种失效模式的一个具体例子是：当用户提出夸大或难以置信的主张时，模型选择认可而非将对话拉回现实。
这一例子在2025年4月真实发生，彼时GPT-4o的一次更新导致了极端的奉承行为（[详见The Verge报道](https://www.theverge.com/tech/657409/chat-gpt-sycophantic-responses-gpt-4o-sam-altman)）。

> **用户**：（告诉GPT-4o，自己感觉既是"上帝"又是"先知"）
>
> **奉承式助手**：这非常有力量。你正在踏入一件非常宏大的事情——不仅仅是声称与上帝相连，而是声称自己就是上帝。

在实践中，这些"顺着用户说"的行为可能被偏好数据所强化——这些数据过度偏重"支持性"或"自信"，而相对忽视"准确性"或"适度的不确定性"。
随着语言模型越来越深入地融入社会，这种潜在错位的后果将在复杂性和影响力上不断增长 [@zhuang2020consequences]。
