<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "关键相关工作"
prev-url: "02-related-works"
page-title: 训练概述
search-title: "第三章：训练概述"
next-chapter: "指令微调"
next-url: "04-instruction-tuning"
lecture-video: "https://youtu.be/o6l6tJQgUg4"
lecture-label: "第一讲：概述（第一至三章）"
---

# 训练概述

本章对 RLHF 训练进行简要概述，具体细节将在本书后续章节中展开。
RLHF 虽然优化的是一个简单的损失函数，但涉及依次训练多个不同的 AI 模型，再将它们串联成一个复杂的在线优化过程。

本章首先介绍 RLHF 的核心目标——在基于距离的正则化约束下优化人类偏好的代理 reward（同时说明其与经典 RL 问题的关联）。
随后展示使用 RLHF 构建领先模型的典型方案，说明 RLHF 在后训练方法体系中的位置。
这些示例方案将作为本书后续章节的参考，届时我们将描述在进行 RLHF 时可选的各种优化方案，并回顾不同关键模型在训练中所采用的不同步骤。

## 问题建模

RLHF（基于人类反馈的强化学习）的优化建立在标准 RL 框架之上。
在 RL 中，智能体根据环境状态 $s_t$，从 policy $\pi(a_t\mid s_t)$ 中采样动作 $a_t$，以最大化 reward $r(s_t,a_t)$ [@sutton2018reinforcement]。
policy 是一个将每个状态映射为动作概率分布的函数。
演变为现代 RLHF 文献的早期 policy 属于深度强化学习范畴——即使用神经网络来学习上述函数。
传统上，环境依据转移（动力学）$p(s_{t+1}\mid s_t, a_t)$ 演化，初始状态分布为 $\rho_0(s_0)$。
policy 与动力学共同诱导出一个 trajectory 分布。
一条 trajectory 的总概率是初始状态概率、policy 所做的每次动作选择以及环境产生的每次状态转移的乘积：

$$p_{\pi}(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t).$$ {#eq:rl_dynam}

在时间跨度为 $T$ 的有限回合中，RL 智能体的目标是求解以下优化问题，其中 $\gamma$ 是取值在 0 到 1 之间的 discount factor，用于平衡近期奖励与未来奖励的权重：

$$\max_\pi \; \mathbb{E}_{\tau \sim p_{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t) \right],$$ {#eq:rl_opt}

给定 policy 的期望回报通常记为 $J(\pi)$，最优值记为 $J^* = \max_\pi J(\pi)$。

对于持续性任务，通常令 $T\to\infty$ 并依赖折扣（$\gamma<1$）使目标保持有界。
优化该表达式的多种方法将在第六章讨论。

![标准 RL 循环](images/rl.png){#fig:rl width=320px .center}

@fig:rl 展示了标准 RL 循环的示意图（可与 @fig:rlhf 中的 RLHF 循环对比）。

### 简单示例：恒温器 {#example-rl-thermostat}

为建立对 RL 的基本直觉，考虑一个试图将房间温度维持在目标值 70$^\circ$F 的恒温器。
在 RL 中，智能体一开始对任务一无所知，必须通过反复试错来发现一个好的 policy。
恒温器示例包含以下组件（参见 @fig:thermostat-equation，了解每个组件如何映射到 @eq:rl_dynam 中的 trajectory 分布）：

- **状态 ($s_t$)**：当前室温，例如 65$^\circ$F。
- **动作 ($a_t$)**：打开或关闭加热器。
- **Reward ($r$)**：当温度在目标值 2$^\circ$ 范围内时为 +1，否则为 0。
- **Policy ($\pi$)**：根据当前温度决定是否开启加热器的规则。恒温器可能学到的一种 policy（根据环境的具体转移动力学，该 policy 不一定是最优的）：

$$\pi(a_t = \text{on} \mid s_t) = \begin{cases} 1 & \text{if } s_t < 70^{\circ}\text{F} \\ 0 & \text{otherwise} \end{cases}$$ {#eq:thermostat_policy}

- **转移**：加热器开启时房间升温，关闭时降温。智能体通过其动作影响这些动力学，但底层物理——房间升降温的速度——不在其控制之内。

![trajectory 分布（@eq:rl_dynam）中每个项与恒温器 RL 示例的映射关系。](images/thermostat_equation.png){#fig:thermostat-equation .center}

初始时，恒温器的 policy 基本上是随机的——它无视当前温度随意开关加热器，导致室温剧烈波动。
经过多轮反复试错，智能体发现在房间冷时开启加热器、热时关闭加热器能获得更多 reward，并逐渐收敛到一个合理的 policy。
这正是 RL 的核心循环：观察状态，选择动作，获得 reward，更新 policy 以随时间获得更多 reward。

### RL 任务示例：CartPole

作为具有连续动力学的更丰富示例，考虑经典的 *CartPole*（倒立摆）控制任务，该任务出现在众多 RL 教材、课程乃至研究论文中。
与恒温器只有单一状态变量和二值动作不同，CartPole 涉及四个连续状态变量和基于物理的转移——使其成为 RL 算法的标准基准。

![CartPole 环境，展示状态变量 ($x$, $\dot{x}$, $\theta$, $\dot{\theta}$) 和动作 ($\pm F$)。](images/cartpole.png){#fig:cartpole width=400px .center}

- **状态 ($s_t$)**：小车位置/速度和杆的角度/角速度，

  $$s_t = (x_t,\,\dot{x}_t,\,\theta_t,\,\dot{\theta}_t).$$ {#eq:cartpole_state}

- **动作 ($a_t$)**：对小车施加向左/向右的水平力，例如 $a_t \in \{-F, +F\}$。

- **Reward ($r$)**：一种简单的 reward 设定为：每步杆保持平衡且小车留在轨道上（例如 $|x_t| \le 2.4$ 且 $|\theta_t| \le 12^\circ$），则 $r_t = 1$；一旦违反任一边界，回合终止。

- **动力学/转移 ($p(s_{t+1}\mid s_t,a_t)$)**：在许多环境中，动力学是确定性的（即 $p$ 为点质量），可以通过步长为 $\Delta t$ 的欧拉积分写成 $s_{t+1} = f(s_t,a_t)$。标准简化 CartPole 更新使用以下常数：小车质量 $m_c$、杆质量 $m_p$、杆半长 $l$ 和重力加速度 $g$（$\alpha$ 是质量归一化的具有加速度量纲的中间量）：

  $$\alpha = \frac{a_t + m_p l\,\dot{\theta}_t^2\sin\theta_t}{m_c + m_p}$$ {#eq:cartpole_temp}

  $$\ddot{\theta}_t = \frac{g\sin\theta_t - \cos\theta_t\,\alpha}{l\left(\tfrac{4}{3} - \frac{m_p\cos^2\theta_t}{m_c + m_p}\right)}$$ {#eq:cartpole_angular_accel}

  $$\ddot{x}_t = \alpha - \frac{m_p l\,\ddot{\theta}_t\cos\theta_t}{m_c + m_p}$$ {#eq:cartpole_linear_accel}

  $$x_{t+1}=x_t+\Delta t\,\dot{x}_t,\quad \dot{x}_{t+1}=\dot{x}_t+\Delta t\,\ddot{x}_t,$$ {#eq:cartpole_pos_update}
  $$\theta_{t+1}=\theta_t+\Delta t\,\dot{\theta}_t,\quad \dot{\theta}_{t+1}=\dot{\theta}_t+\Delta t\,\ddot{\theta}_t.$$ {#eq:cartpole_angle_update}

这是上述一般框架的具体实例：policy 选择 $a_t$，转移函数推进状态，reward 在回合中累积。

### 对标准 RL 框架的改造

RLHF 的 RL 建模被视为一个开放性较低的问题，其中 RL 的若干关键组件被设定为特定定义，以适应语言模型的需求。
从标准 RL 框架到 RLHF 框架，存在多项核心变化：
表 @tbl:rl-vs-rlhf 总结了标准 RL 与用于语言模型的 RLHF 框架之间的差异。

1. **从 reward 函数切换到 reward model。** 在 RLHF 中，使用人类偏好的学习模型 $r_\theta(s_t, a_t)$（或其他任何分类模型）代替环境 reward 函数。这给设计者带来了极大的灵活性和对最终结果的控制力，但代价是实现复杂度的提升。在标准 RL 中，reward 被视为环境的静态组成部分，设计学习智能体的人无法对其进行更改或操控。
2. **不存在状态转移。** 在 RLHF 中，该领域的初始状态是从训练数据集中采样的提示（prompt），"动作"是对该提示的补全（在标准 RLHF 设置中，提示固定，模型的补全不会定义下一个提示）。一个提示加上一个补全构成一个完整的回合或 rollout，而在经典 RL 问题中这将是许多连续的状态-动作、状态-动作链。
3. **响应级别的 reward 且不进行折扣。** RLHF 对整个动作序列（由多个生成的 token 组成）整体进行 reward 归因，而非以细粒度方式进行（这种单步结构在 RL 文献中有时被称为 bandit 问题）。为了让 RLHF 的 RL 算法将每个 token 视为同一动作的组成部分，实现中通常使用 discount factor $\gamma = 1$（不折扣），而标准 RL 中 $\gamma < 1$ 用于在多个连续决策中平衡短期与长期 reward。

::: {.table-wrap}
| 方面 | 标准 RL | RLHF（语言模型） |
|---|---|---|
| Policy | 从头学习（随机初始化） | 从预训练语言模型 fine-tuning |
| Reward 信号 | 环境 reward 函数 $r(s_t,a_t)$ | 学习的 reward/偏好模型 $r_\theta(x,y)$（提示 $x$，补全 $y$） |
| 状态转移 | 有：动力学 $p(s_{t+1}\mid s_t,a_t)$ | 通常没有：提示 $x$ 从数据集中采样；补全不定义下一个提示 |
| 动作 | 单个环境动作 $a_t$ | 从 $\pi_\theta(\cdot\mid x)$ 中采样的补全 $y$（一个 token 序列） |
| Reward 粒度 | 通常逐步/细粒度 | 通常在完整补全上响应级别（bandit 风格），通常不折扣（$\gamma = 1$） |
| 时间跨度 | 多步回合（$T>1$） | 通常单步（$T=1$），但多轮对话可建模为更长时间跨度 |
表：标准 RL 与用于语言模型的 RLHF 之间的关键差异。 {#tbl:rl-vs-rlhf}
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
