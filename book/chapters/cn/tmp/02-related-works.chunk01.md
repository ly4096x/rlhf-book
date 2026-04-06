<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "引言"
prev-url: "01-introduction"
page-title: 关键相关工作
search-title: "第二章：关键相关工作"
next-chapter: "训练概述"
next-url: "03-training-overview"
lecture-video: "https://youtu.be/o6l6tJQgUg4"
lecture-label: "第一讲：概述（第1–3章）"
---

# 关键相关工作

RLHF 及其相关方法非常新颖。
我们梳理这段历史，是为了展示这些流程被正式化的时间之近，以及相关文献在学术界的记录程度。
基于此，我们希望强调 RLHF 正在快速演进，因此本章为全书奠定基调——书中将对某些方法表达不确定性，并预期部分细节会围绕若干核心实践发生变化。
此外，本章列举的论文和方法也揭示了 RLHF 流水线各组成部分形成的原因，因为其中一些开创性论文所针对的应用与现代语言模型截然不同。

本章详述了推动 RLHF 领域发展至今的关键论文与项目。
这并非对 RLHF 及相关领域的全面综述，而是一个起点，讲述我们如何走到今天。
本章有意聚焦于导致 ChatGPT 诞生的近期工作。
RL 文献中还有大量关于从偏好中学习的延伸工作 [@wirth2017survey]。
如需更详尽的列表，请参阅专门的综述论文 [@kaufmann2023survey]、[@casper2023open]。

![本章讨论的 RLHF 关键发展时间线，从早期基于偏好的 RL 工作，到大型语言模型中 RLHF 的广泛应用。](images/rlhf_timeline.png){#fig:rlhf_timeline}

## 起源至2018年：基于偏好的 RL

该领域近年来随着 deep reinforcement learning 的兴起而广受关注，并在众多大型科技公司对 LLM 应用研究的推动下不断扩展。
尽管如此，当今使用的许多技术与早期基于偏好的 RL 文献中的核心技术有着深刻的渊源。

最早与现代 RLHF 方法相近的论文之一是 *TAMER*。
*TAMER: Training an Agent Manually via Evaluative Reinforcement* 提出了一种方法：人类对智能体的行为进行迭代评分，以学习 reward model，再利用该 reward model 学习行为 policy [@knox2008tamer]。
其他同期或稍后的工作提出了一种 actor-critic 算法 COACH，该算法利用人类反馈（正向和负向）来调整优势函数 [@macglashan2017interactive]。

主要参考文献 Christiano et al. 2017 是将 RLHF 应用于 Atari 游戏中智能体轨迹偏好比较的工作 [@christiano2017deep]。
这项引入 RLHF 的工作紧随 DeepMind 在 deep reinforcement learning 领域的开创性成果——Deep Q-Networks（DQN）之后，后者展示了 RL 智能体可以从零开始学习并解决流行电子游戏。
该工作表明，在某些领域，由人类对轨迹进行比较选择可以比直接与环境交互更加有效。这依赖于一些巧妙的条件设置，但效果依然令人印象深刻。

![Christiano et al.（2017）中的核心 RLHF 循环：reward predictor 通过轨迹片段的比较异步训练，智能体则最大化预测奖励。](images/rlhf_schematic.png){#fig:rlhf_schematic width=66%}

该方法随后通过更直接的 reward modeling 得到拓展 [@ibarz2018reward]，而 deep learning 在早期 RLHF 工作中的应用，则以仅仅一年后将神经网络模型引入 TAMER 的扩展研究为阶段性总结 [@warnell2018deep]。

这一时代开始转型——reward model 作为一种通用概念，被提出作为研究对齐问题的方法，而不仅仅是解决 RL 问题的工具 [@leike2018scalable]。

## 2019年至2022年：语言模型上的 RL from Human Preferences

Reinforcement learning from human feedback（在早期也常被称为 reinforcement learning from human preferences）被快速采纳，AI 实验室日益将重心转向大规模语言模型的扩展。
这一领域的大量工作始于2019年的 GPT-2 到2020年的 GPT-3 之间。
2019年最早的工作 *Fine-Tuning Language Models from Human Preferences* 与现代 RLHF 工作以及本书将涵盖的内容有许多惊人的相似之处 [@ziegler2019fine]。
许多规范性术语，如 reward model 的学习、KL distances、反馈示意图等，均在这篇论文中得到正式化——只是最终模型的评估任务和能力与今天的研究有所不同。
此后，RLHF 被应用于多种任务。
重要的应用示例包括：通用摘要生成 [@stiennon2020learning]、书籍递归摘要 [@wu2021recursively]、指令跟随（InstructGPT）[@ouyang2022training]、浏览器辅助问答（WebGPT）[@nakano2021webgpt]、引用支持回答（GopherCite）[@menick2022teaching]，以及通用对话（Sparrow）[@glaese2022improving]。

除应用之外，若干开创性论文为 RLHF 的未来发展定义了关键领域，包括：

1. Reward model 过度优化 [@gao2023scaling]：RL 优化器对基于偏好数据训练的模型产生过拟合的能力，
2. 语言模型作为对齐研究的一般性领域 [@askell2021general]，以及
3. Red teaming [@ganguli2022red]——评估语言模型安全性的过程。

RLHF 在聊天模型中的应用持续完善。
Anthropic 在早期版本的 Claude 中持续大量使用 RLHF [@bai2022training]，早期 RLHF 开源工具也相继涌现 [@ramamurthy2022reinforcement]、[@havrilla-etal-2023-trlx]、[@vonwerra2022trl]。

## 2023年至今：ChatGPT 时代

ChatGPT 的发布公告明确阐述了 RLHF 在其训练中的作用 [@openai2022chatgpt]：

> We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup.

此后，RLHF 被广泛用于领先的语言模型。
已知其被应用于：Anthropic 的 Constitutional AI for Claude [@bai2022constitutional]、Meta 的 Llama 2 [@touvron2023llama] 和 Llama 3 [@dubey2024llama]、Nvidia 的 Nemotron [@adler2024nemotron]、Ai2 的 Tülu 3 [@lambert2024t] 等。

如今，RLHF 正在发展为一个更广泛的偏好微调（PreFT）领域，涵盖新兴应用，例如：用于中间推理步骤的 process reward [@lightman2023let]（见第5章）；受 Direct Preference Optimization（DPO）启发的直接对齐算法 [@rafailov2024direct]（见第8章）；从代码或数学的执行反馈中学习 [@kumar2024training]、[@singh2023beyond]，以及受 OpenAI 的 o1 [@openai2024o1] 启发的其他在线推理方法（见第7章）。
