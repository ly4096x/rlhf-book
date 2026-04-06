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
