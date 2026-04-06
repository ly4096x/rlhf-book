<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "强化学习"
prev-url: "06-policy-gradients"
page-title: 推理
search-title: "第七章：推理"
next-chapter: "直接对齐"
next-url: "08-direct-alignment"
---

# 推理训练与 inference-time scaling

reasoning 模型与 inference-time scaling 在 2024 年底至 2025 年间，乃至未来，为语言模型性能带来了巨大飞跃。
Inference-time scaling 是指在生成阶段使用更多算力来提升模型性能的能力，例如生成更长的 reasoning 链条或采样多个响应。
经过大量训练、善于在回答前深度思考的语言模型，能够极为出色地利用这一特性。
这些模型通过大量使用 reinforcement learning with verifiable rewards（RLVR）[@lambert2024t] 进行训练，同时仍大量依赖 RLHF。
在本章中，我们将回顾引领 AI 社区重新认识 RL 在语言模型中潜力的历程，介绍 RLVR 的基本原理，梳理关键研究成果，并指出未来几年将定义这一领域的核心争论。

首先，在 2016 年的神经信息处理系统（NeurIPS）会议上，Yann LeCun 首次提出了他如今广为人知的蛋糕比喻，用以描述现代机器学习系统中学习发生的位置：

> 如果说智能是一块蛋糕，那么蛋糕的主体是无监督学习，蛋糕上的糖霜是监督学习，而蛋糕上的樱桃则是 reinforcement learning（RL）。

这一类比在现代语言模型及近期 post-training 技术栈的演变中已基本得到印证。
RLHF 是其先驱，而用于 reasoning 模型的 RL——主要聚焦于数学、代码和科学领域——则是对这一类比的确认。
在这个类比中：

- 在海量互联网数据上进行的自监督学习构成了蛋糕的主体（尤其是从 FLOPs 角度衡量的计算量来看），
- post-training 初期的 supervised fine-tuning（SFT）将模型调整至更窄的分布，
- 最后，"纯粹"的 reinforcement learning（RL）是蛋糕上的樱桃。用于创建新型"reasoning"或"thinking"模型的大规模 reinforcement learning，正是这最后的点睛之笔（以及 RLHF 的辅助，而 RLHF 并不被视为经典 RL，具体原因我们将在后文说明）。

这一小部分 reasoning 训练催生了**thinking 模型**——这些模型综合运用本书所讨论的 post-training 技术来对齐偏好，并结合在可验证领域的 RL 训练，大幅提升了 reasoning、编程和数学问题求解等能力。

这些模型的训练方法，即 Reinforcement Learning with Verifiable Rewards（RLVR）[@lambert2024t]，与 RLHF 的流程非常相似，但它将 reward model 变为可选项，以一个评分函数代替——该函数在答案正确时返回正奖励，否则返回 0。

例如，考虑 RLHF 与 RLVR 在评分响应方面的差异。
在 RLHF 中，reward model 必须评估主观质量：

> **Prompt**：用经济学的概念解释机会成本。
>
> **Response**：机会成本是指在做决策时，你放弃的次优选项的价值。例如，如果你花一个小时学习而不是工作，那么机会成本就是你本可赚到的工资……

对这一响应打分需要判断清晰度、准确性、完整性和有用性——所有这些质量都需要习得的偏好，且没有确定性的正确答案。

相比之下，RLVR 使用能返回确定性分数的验证函数。
对于数学问题：

> **Prompt**：小于 20 的所有质数之和是多少？
>
> **Response**：小于 20 的质数为 2、3、5、7、11、13、17 和 19。
> 相加：2 + 3 = 5，然后 5 + 5 = 10，然后 10 + 7 = 17，然后 17 + 11 = 28，然后 28 + 13 = 41，然后 41 + 17 = 58，最后 58 + 19 = 77。
> 答案是 $\boxed{77}$。
>
> **Verification**：`extracted_answer == 77` → Reward = 1

`\boxed{}` 符号是从数学排版中借鉴的惯例，使答案提取变得简单明了——一个简单的正则表达式即可从响应中提取最终答案，无论模型是如何得出的。
值得注意的是，还存在其他答案提取方法，例如使用短语"The answer is: "（如上例所示）、特殊 token 如 `<answer>`，或分隔符如 `####`。

对于代码生成，验证通常以单元测试的形式进行：

> **Prompt**：编写一个 Python 函数 `fib(n)`，返回第 n 个 Fibonacci 数，其中 fib(0) = 0，fib(1) = 1。
>
> **Response**：
> def fib(n):
>     if n < 2:
>         return n
>     return fib(n - 1) + fib(n - 2)
>
> **Verification（单元测试）**：
>
> assert fib(0) == 0   # 基本情况
> assert fib(1) == 1   # 基本情况
> assert fib(10) == 55 # 较大的值
> （所有测试通过 → Reward = 1）


单元测试是代码的天然验证函数：它们针对已知的输入输出对执行模型的解决方案。
一种常见的评分方式是简单的门控：若所有断言通过，则奖励为 1；若有任何失败，则奖励为 0。
其他设置则按通过测试的比例给予部分分数。
对于这两个示例，都不需要习得的 reward model，大多数设置也不使用（因为模型在这些领域对过度优化具有鲁棒性），但也可以通过奖励的线性组合来使用。

RLVR 背后的思想在 RL 文献中并不新颖——基于答案是否正确进行梯度更新的核心思想，几乎是 reinforcement learning 的教科书定义。
将其应用于语言模型时的创新，主要在于如何在保持被 fine-tuning 模型的强大通用能力的同时加以运用。在此之中，语言建模文献中有许多相关思路，模型从有关答案正确性的反馈中学习。

最初，在我参与的那项工作中——该工作创造了"RL with Verifiable Rewards（RLVR）"[@lambert2024t]这一术语——该方法原本拟命名为"RL with Ground Truth rewards（RLGT）"。
然而，RLVR 与仅从 ground truth 答案中学习存在微妙差异。
在数学等领域，可以用单一的 ground truth 答案来验证解题过程，如上文所述。
在其他领域，例如代码生成或精确指令跟随，即使存在多个正确解而非唯一的 ground truth 答案，也可以用检查函数（例如单元测试）来验证答案。
RLVR 进展的核心在于拥有足够多样和深度的可验证问题，即便确切的解答事先并不已知。

![RLVR 以 RL 反馈循环的形式呈现。用验证函数代替 reward model。](images/rlvr-system.png){#fig:rlvr}

最早成功部署这类训练的模型是 OpenAI 的 o1 [@openai2024o1] 和开放权重模型 DeepSeek R1 [@guo2025deepseek]。
此后，整个 AI 行业将这一训练过程和模型风格列为优先方向。
这里的核心变化，与其说是这类 RL 设置全然新颖，不如说是训练阶段的重新分配以及对不同行为的优先级调整。
Reasoning 模型开启了一个预期通过扩大 RL 训练规模来提升性能的时代。

至于这些模型积累的行为类型，请考虑以下示例：DeepSeek V3 0325 与其 reasoning 模型 DeepSeek R1 在查询 `Write me a short poem about a goldfish` 时的表现对比。
DeepSeek V3 非常直截了当：

> **Response**：Goldfish Dreams
>
> In a bowl of glass so clear and round,
> A tiny goldfish swims without a sound.
> With scales that gleam like sunlight's kiss,
> He drifts along in watery bliss.
> 
> A flick of fin, a flash so bright,
> Chasing bubbles in the light.
> Does he dream of oceans wide?
> Or is his little world enough---his pride?

而 DeepSeek R1 的体验则截然不同。
它首先生成大量 thinking token 来构思响应，然后再回复用户。
答案呈现为：
