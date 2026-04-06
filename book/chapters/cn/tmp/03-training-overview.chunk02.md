:::
表：标准 RL 与面向语言模型的 RLHF 之间的主要差异。{#tbl:rl-vs-rlhf}
:::

鉴于问题的单轮性质，优化目标可以在去掉时间视界和 discount factor 的情况下重写（并显式引入 reward model）：
$$\max_\pi \; \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t) \right].$$ {#eq:rl_opt_int}

从许多角度来看，结论是：尽管 RLHF 在很大程度上受到 RL 优化器和问题建模方式的启发，但其实际实现与传统 RL 存在本质上的区别。

![标准 RLHF 循环](images/rlhf.png){#fig:rlhf}

### 微调与正则化

在传统 RL 问题中，智能体必须从随机初始化的 policy 出发进行学习；而在 RLHF 中，我们从一个具备丰富初始能力的强大预训练基础模型出发。
这种强先验使得 RLHF 需要防止优化过程偏离初始 policy 太远。
为了在 fine-tuning 范式下取得成功，RLHF 技术采用多种正则化手段来约束优化过程。
其目标是在允许奖励最大化的同时，避免模型陷入过度优化（详见第 14 章）。
最常见的做法是在优化目标中加入 KL divergence 惩罚项，用于约束当前 RLHF policy 与优化起点之间的距离。训练时设置的超参数 $\beta$ 控制这一约束的强度——较大的 $\beta$ 使模型更接近其起点，而较小的 $\beta$ 则赋予优化器更多追求奖励的自由：

$$\max_\pi \; \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{\text{KL}}(\pi(\cdot|s_t) \| \pi_{\text{ref}}(\cdot|s_t)).$$ {#eq:rlhf_opt_eq}

在这一框架下，大量关于 RLHF 训练的研究致力于理解如何在以与初始模型距离为度量的"KL 预算"范围内进行有效优化。
更多细节请参阅第 15 章关于正则化的内容。


### 优化工具

本书详细介绍了求解上述优化问题的多种流行技术。
post-training 的主要工具包括：

- **Reward modeling**（第 5 章）：训练一个模型以捕捉从收集到的偏好数据中提取的信号，并能够输出表示未来文本质量的标量奖励。
- **Instruction fine-tuning**（第 4 章）：RLHF 的前置步骤，通过模仿预先筛选的示例，使模型学习当今语言模型交互中广泛使用的问答格式。
- **Rejection sampling**（第 9 章）：最基础的 RLHF 技术，通过 reward model 对 instruction fine-tuning 候选补全结果进行过滤，以模拟人类偏好。
- **Policy gradients**（第 6 章）：强化学习算法，用于 RLHF 的经典示例中，根据 reward model 提供的信号更新语言模型的参数。
- **Direct alignment algorithms**（第 8 章）：直接从成对偏好数据优化 policy 的算法，无需先学习中间 reward model 再进行优化。

经过现代 RLHF 训练的模型，始终采用 instruction fine-tuning 加上上述其他优化选项混合使用的方式。

## RL 在 post-training 语言模型中的微妙优势

在后续章节中，我们将介绍多种 post-training 优化工具。
其中不少工具，例如 rejection sampling（第 9 章）和 DPO 等 direct alignment algorithms（第 8 章），比让 RL 正常运转要简单得多。
尽管如此，尽管替代方案更为简便，基于 RL 的方法仍然持续胜出。
某些趋势是显而易见的，例如利用可验证奖励强化学习（RLVR）实现推理时扩展；但更广泛地说，RL 已被证明是非常适合语言模型的优化工具。
相比 instruction tuning 或 DPO 类算法，实现 RL 需要投入大得多的基础设施成本，但冒着过于通俗的风险来说——它所提供的梯度更新"总体上对模型大有裨益"。
这一点难以量化，但具体体现在以下几种反复出现的形式中：

- RL 阶段可以"修复"模型的粗糙之处，使其更易于对话或更加鲁棒（例如，通过训练使其在 vLLM 等推理工具中具备数值稳定性）。其确切原因在文献中尚不明确，但这一事实在 RL 日益普及的趋势中得到了印证。
- RL 可以精准施用——模型能够很好地学习 prompt 分布所在位置，RL 通常不会"压制"模型的通用能力。一个典型例子是 Tülu 3 仅在数学 prompt 上进行 RL 训练，同时在广泛的任务套件中保持了整体能力 [@lambert2024t]。

总体而言，作用于语言模型的 RL 损失具有鲁棒、可扩展、有效且灵活的特点，由此开辟了大量新的实验方向。
最初引领我们走上这条道路的方法，正是 RLHF 研究工作。

## 经典训练方案

随着时间推移，一些模型已被确立为 RLHF 或 post-training 的经典方案。
这些方案反映了当时的数据实践和模型能力。
随着方案的老化，以相同特性训练模型变得越来越容易，所需数据也越来越少。
总体趋势是，post-training 涉及更多优化步骤、更多训练算法，以及更多样化的训练数据集和评估基准。

### InstructGPT

在 ChatGPT 首次问世前后，被广泛接受的（"经典"）语言模型 post-training 方法包含三个主要步骤，其中 RLHF 是核心环节 [@lambert2022illustrating] [@ouyang2022training] [@bai2022training]。
在"基础"语言模型（即在大规模网络文本上训练的下一词预测模型）之上所采取的三个步骤，总结如下（见 @fig:rlhf-basic-repeat）：

1. **在约 1 万条示例上进行 instruction tuning**：这一步骤教会模型遵循问答格式，并从以人工书写数据为主的数据中学习一些基础技能。
2. **在约 10 万对成对 prompt 上训练 reward model**（论文使用了 3.3 万个 prompt）：该模型从 instruction-tuned 检查点开始训练，捕捉希望在最终训练中建模的多元价值观。reward model 是 RLHF 的优化目标。
3. **在单独的约 10 万个 prompt 上使用 RLHF 训练 instruction-tuned 模型**（论文使用了恰好 3.1 万个，是否以及在多大程度上复用了其他阶段的 prompt 未有记录）：模型在生成回复后接受评分，针对 reward model 进行优化。

RLHF 完成后，模型即可部署给用户。这一方案是现代 RLHF 的基础，但方案已大幅演进，涵盖了更多阶段和更多数据。

![早期三阶段 RLHF 流程示意图，包含 SFT、reward model 以及随后的优化。](images/rlhf-basic.png){#fig:rlhf-basic-repeat}

### Tülu 3

现代版本的 post-training 涉及更多、更多的模型版本和训练阶段（例如，Llama 2 所记录的 RLHF 步骤已超过 5 个 [@touvron2023llama]）。
下图 @fig:rlhf-complex 展示了一个示例，其中模型经历了多次训练迭代才达到收敛。

![现代 post-training 多轮训练示意图。](images/rlhf-complex.png){#fig:rlhf-complex}

这一时代及之后训练的最复杂模型尚未公开其完整训练过程的详细信息。
截至 2025 年，ChatGPT 或 Claude 等领先模型涉及多轮迭代训练。
这甚至可能包括训练专用模型后再将权重合并以获得能胜任多个子任务的最终模型等技术 [@li2022branch]（例如 Cohere 的 Command A [@cohere2025command]）。

![Tülu 3 方案总结，包含目标技能与多步训练方案。Lambert et al. 2024，许可协议 CC-BY。](images/tulu3.png){#fig:tulu-3}

Tülu 3 是这种以 RLHF 为核心的多阶段 post-training 方法的一个完全开放的示例。
Tülu 3 方案由三个阶段组成：

1. **在约 100 万条示例上进行 instruction tuning**：这一以合成数据为主的数据集，从 GPT-4o 和 Llama 3.1 405B 等前沿模型中混合提取，教会模型通用的指令遵循能力，并为数学和编程等能力奠定基础。
2. **在约 100 万对偏好对上进行 on-policy 偏好数据训练**：这一阶段大幅提升了模型的对话流畅度（例如在 Arena（原 ChatBotArena）或 AlpacaEval 2 上的表现），同时进一步改善了 instruction tuning 阶段提及的各项技能。
3. **在约 1 万个 prompt 上进行可验证奖励强化学习（RLVR）**：这一小规模强化学习训练旨在提升数学等核心技能，同时维持整体性能（现被视为 DeepSeek R1 等现代推理模型的先驱）。

该方案已成功应用于 Llama 3.1 [@lambert2024t]、OLMo 2 [@olmo20242] 以及 SmolLM 系列模型 [@alrashed2024smoltulu]。

### DeepSeek R1

随着推理语言模型（如 OpenAI 的 o1）的兴起，post-training 的最佳实践再次演进，对各训练阶段的计算资源分配进行了重新排列和再分配。
目前对推理模型 post-training 方案记录最为清晰的是 DeepSeek R1 [@guo2025deepseek]，Alibaba 的大型 Qwen 3 模型（即仅限 32B 和 225B MoE 模型）[@yang2025qwen3] 以及小米的 MiMo 7B [@xia2025mimo] 均采用了类似方案。
DeepSeek 方案如下：

1. **超过 10 万条 on-policy 推理样本的"冷启动"**：这些数据从早期 RL 检查点 R1-Zero 中采样，并经过严格过滤，以在 DeepSeek-V3-Base 上灌输特定的推理过程。DeepSeek 用"冷启动"一词描述从极少量监督数据中学习 RL 的方式。
2. **大规模强化学习训练**：该阶段反复让模型处理推理问题，在多个基准上进行 RLVR，直至"收敛"。
3. **Rejection sampling 与 SFT**：在接近收敛时，对 RL 检查点应用 rejection sampling 构建约 80 万条样本的 SFT 数据集，然后在约 3/4 推理问题与 1/4 通用查询的过滤混合数据上对模型进行 fine-tune，以得到通用模型。
4. **混合强化学习训练**：针对推理问题（可验证奖励）与通用偏好调优 reward model 的混合训练，以对模型进行精调。

如上所述，该方案存在演进版本，尤其是步骤 3 和步骤 4，用于在向用户开放前对模型进行最终打磨。
