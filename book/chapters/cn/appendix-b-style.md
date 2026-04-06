<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "定义"
prev-url: "appendix-a-definitions"
page-title: "附录 B：风格与信息"
search-title: "附录 B：风格与信息"
next-chapter: "实践问题"
next-url: "appendix-c-practical"
---

# 风格与信息

RLHF 早期的发展使其背负了"仅仅是风格迁移"的名声，以及其他对 RLHF 如何操控输出信息呈现方式的严苛批评。
本附录解释了为什么风格是理解 RLHF 所提供价值的核心，以及为什么它对模型能力和用户体验都能产生积极影响。

将 RLHF 视为单纯风格迁移的观念，从两个方面阻碍了 RLHF 的叙事发展。
第一个方面是 RLHF 如何与对模型的细微、不重要的改变产生关联。
当人们谈论风格迁移时，他们不会将其描述为重要或令人兴奋的事——他们认为这是表面的。
然而，风格是人类价值取之不尽的源泉；这就是为什么重新讲述故事可以成为新的畅销书（例如[《人类简史》](https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind)），而且它是持续推进我们知识生态系统的基本组成部分。
风格与信息本身是交织在一起的。

第二个原因是，许多人忽视了这样一个事实：做得好的 RLHF 能够提升主流 LLM 评测的得分。
我们已经从 Llama 3 中看到，不同风格实际上可以对评测产生有意义的改进 [@dubey2024llama]。
Llama 3 Instruct 模型在 Arena 上得分极高，普遍认为这是因为它们拥有更有趣的个性——相比同时代的其他模型，它们更简洁、更机智。
无论许多 LLM 用户所痴迷的 benchmark 得分如何，如果 RLHF 能够让 language model 变得更有趣，那本身就是有价值的。

在本章中，"话唠性"（chattiness）一词用于涵盖经 RLHF 训练的模型回复长度不断增加的现象，但它也涵盖了大量使用 markdown、表情符号以及以项目符号列表格式化答案等技巧。
为了使本章的讨论更加具体，下面展示了同一 Tülu 3 系列中两个模型的示例补全内容——一个仅进行了 instruction fine-tuning，第二个则通过长度控制 DPO 应用了偏好 fine-tuning。
首先是对 prompt `What is reinforcement learning from human feedback?` 进行 instruction-tuned 的补全（具体模型为 [Llama 3.1 Tulu 3 70B SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)）[@lambert2024t]：

```text
Reinforcement learning from human feedback, also known as human-in-the-loop reinforcement learning, is a subfield of machine learning where the primary goal is to train reinforcement learning agents to perform tasks by incorporating feedback from humans. This feedback can come in various forms, such as binary rewards (e.g., right or wrong), scalar rewards, preferences between actions, demonstrations, natural language instructions, or corrections to the agent's behavior.

The idea behind reinforcement learning from human feedback is to leverage human intuition and decision-making abilities to guide the learning process of the agent. In traditional reinforcement learning, an agent learns to make decisions by interacting with an environment and receiving rewards or punishments based on the outcomes of those interactions. However, defining a suitable reward function can be challenging and time-consuming, often leading to reward shaping or reward hacking, where the agent finds unexpected and often unintended ways to maximize the reward.

By using human feedback, reinforcement learning can address some of these issues by allowing humans to directly influence the learning process. Human trainers can provide real-time feedback, scale their feedback to match the complexity of the task, and update their feedback based on the agent's progress. This approach can lead to more robust and generalized agents, as they are guided not by handcrafted reward functions but by the nuanced understanding and experience of human trainers.

However, reinforcement learning from human feedback faces several challenges, such as the high cost of obtaining human feedback, subjective and inconsistent feedback, scaling issues, and methods for efficient and effective integration of human feedback into learning algorithms. Despite these challenges, the research in this area has shown promising results in various applications, including robotics, video games, prosthetics control, natural language processing, and many more.
```

以及来自 [Llama 3.1 Tulu 3 70B DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO) 经过 RLHF 后的补全内容：
```text
Reinforcement learning from human feedback (RLHF) is a subset of reinforcement learning (RL) where the rewards that guide the learning process are provided by humans, rather than a pre-specified numerical reward function. This is particularly useful in situations where it is hard or infeasible to design a suitable reward function, or where the desired performance involves subjective judgments that humans are uniquely equipped to provide.

In traditional reinforcement learning, an agent interacts with an environment. With each interaction, the agent performs an action, receives an observation, and gets a numerical reward that tells it how well it performed in that situation. Through many such interactions, the agent learns to maximize the cumulative reward.

In contrast, in reinforcement learning from human feedback:

1. **Human Input:** Instead of a predefined reward function, the feedback comes from human evaluators, who provide qualitative, subjective judgments about the agent's performance. This feedback can be in various forms, such as numerical ratings, binary preference signals, natural language feedback, or even demonstrations showing how to improve.

2. **Feedback Loop:** Human evaluators observe the agent's actions or outcomes and provide feedback in real-time (or close to real-time). The agent learns through this iterative feedback loop, gradually adapting its behavior to align more closely with human preferences.

3. **Reward Shaping:** Because raw human feedback is not always directly usable in learning algorithms due to variability and potential bias, a reward shaping process may be involved, whereby human feedback is converted into a more stable signal that guides learning.

4. **Preference-based RL:** One common approach within RLHF is preference-based reinforcement learning, where the agent is trained by comparing pairs of actions (or outcomes) and receiving human feedback on which one is better. From these pairwise comparisons, the agent infers a reward function that it can use to guide its actions.

5. **Inverse Reinforcement Learning (IRL):** Another technique used in RLHF is IRL, which aims to infer the underlying reward function that could explain human actions or decisions, and then use that reward function to train the agent.

**Applications:**

Reinforcement Learning from human feedback has found applications in:

- **Robotics:** Teaching robots to perform tasks that are difficult to specify with predefined reward functions.
- **Game playing:** Human feedback can guide AI agents to better emulate human-like strategies or behaviors.
- **Language and content generation:** For example, suggestively guiding AI models to generate more coherent, creative, or helpful content.
- **Recommendation systems:** Human feedback can help adjust and refine the recommendations provided to users.

**Challenges:**

Using human feedback introduces some specific challenges:

1. **Scalability:** Human feedback is time-consuming and expensive to collect, and it's hard to scale to complex or high-dimensional tasks.
2. **Bias and Noise:** Human feedback can be noisy, inconsistent, and biased, leading to suboptimal learning outcomes.
3. **Alignment:** Ensuring that the reinforcement learning agent aligns with the true preferences of the human evaluator(s) and not just the superficial feedback, is difficult.
4. **Safety and Misuse:** There can be concerns about the safety and ethical implications of using potentially flawed human feedback, especially if the feedback process is corrupted or gamed.

Researchers are actively working on techniques to mitigate these challenges and improve RLHF approaches. One such approach is leveraging machine learning models to amplify the efficacy of the human feedback, for example, by learning to generalize from sparse human input or by synthesizing plausible feedback in situations where direct human input is not available.
```

DPO 模型的回复包含与前者相似的信息，因为两个模型都在相同的预训练 base model 上训练，但偏好 fine-tuning 增加了更多结构、格式，当然还有长度——这被反复证明与人类对 language model 的一般偏好相关。
更多类似示例，请参见本书配套的补全库：[https://www.rlhfbook.com/library](https://rlhfbook.com/library)。

## 话唠性的平衡

RLHF 或偏好 fine-tuning 方法被无数人用于提升 AlpacaEval 等自动聊天排行榜的得分（这些排行榜使用 LLM-as-a-judge 来近似衡量智能体在简单对话任务中的有用性、无害性和诚实性），但 RLHF 在此带来的巨大收益，并未按比例转化为在更难作弊的评测（如 Arena）上的得分提升。
矛盾在于，虽然 RLHF 方法在这些模型上带来了可测量的改进，但这种训练并不总是能转化为人们真正关心的性能表现。
在 RLHF 文献建立的过程中，大量模型以相关方法发布，声称借助 RLHF 提升了模型的"对齐"程度，但它们往往走得太远，发布了从误导性到毫无意义的各种评测得分。

由对齐驱动的 RLHF 方法，如果做得正确，可以使模型更易于使用、更令人愉快。
这通常伴随着 MT Bench 或 AlpacaEval 等评测工具上的明显提升。

2023 年秋，围绕 direct preference optimization（DPO）及其相对于 proximal policy optimization（PPO）和其他基于 RL 的偏好 fine-tuning 方法的角色，争论达到高峰——聊天评测与现实世界性能之间的平衡正是这场争论的核心（更多技术层面的权衡讨论，请参见第 8 章、Ivison 等人 2024 [@ivison2024unpacking]，或[这个讲座](https://youtu.be/YJMCSVLRUNs)）。
问题在于，你也可以在反馈循环中或使用大量数据使用 DPO 和 PPO 等技术，以换取聊天性能，但这实际上会严重损害模型在数学或编程等其他任务上的表现。

在 DPO 与 PPO 争论大量涌现的那段时期，有许多论文发布了令人印象深刻的 benchmark 数据，却没有能够维持持续公开使用的模型权重，因为这些模型在一般使用中并不稳健。
在 2023 年秋或稍后应用 RLHF 时，根本不可能让一个对齐版的 70 亿参数模型在全面 benchmark 上真正击败 GPT-4（这类比较将会成立，即当时的小模型无法稳健地击败最好的大型前沿模型）。
这似乎显而易见，但总有论文声称这类结果。
@fig:DNO 来自一篇名为 Direct Nash Optimization（DNO）的论文，该论文声称其模型在 2024 年 4 月针对 7B 模型的 AlpacaEval 上达到了最先进水平 [@rosset2024direct]。
作为背景，DNO 是 reward-model+PPO（经典 RLHF）或 one-shot DPO 的批量化、on-policy *迭代式*替代方案，它通过将对齐框架为针对偏好预言机寻找 Nash 均衡，直接优化成对偏好（胜率差距）。
当学术激励机制与引起广泛社会关注的技术相遇时，这些挑战就会出现。

![Direct Nash Optimization（DNO）论文的结果，强调其小模型超越了 GPT-4 等模型。Rosset 等人 2024。许可证 CC-BY。](images/dno-figure.png){#fig:DNO width=550px}

即使是 2024 年 1 月开创性的论文 Self Rewarding Language Models [@yuan2025selfrewardinglanguagemodels] 也披露了 Llama 2 70B 上不切实际的强劲得分。
当时，70B 模型当然可以比 7B 模型更接近 GPT-4（正如我们在 2024 年令人印象深刻的 Llama 3 发布中所见），但重要的是将模型的现实表现与现代 RLHF 论文中的声明区分开来。
这些模型针对狭窄的测试集进行了调优，在实际使用中无法与它们声称击败的大得多的模型相比。
许多更多的方法以类似的方式出现又消失，分享了有价值的见解和被过度渲染的结果，使 RLHF 更难理解。

"奇怪 RLHF"模型的一个症状往往是长度偏差。
这种情况变得如此普遍，以至于 AlpacaEval 和 WildBench 等多个评测系统都内置了线性长度校正机制。
这修补了为"击败 GPT-4"或当日领先前沿模型而在话唠性上作弊的激励，并创造了一种不那么被游戏化的动态，让更短、更有用的模型实际上能够胜出。

无论如何，在文献中，仅为话唠性而对话聊天模型进行对齐，现在已经带有一定的声誉税，人们承认这些狭窄的方法会以其他方式损害模型。
来自 2023 年阿里巴巴最初 Qwen 模型的这条注释，是在早期对齐实验中多次观察到的现象，夸大了话唠性与性能之间的权衡 [@qwen]。

> We pretrained the models with a large amount of data, and we post-trained the models with both supervised fine-tuning and direct preference optimization. However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation.

一个早期做好这种权衡的好例子是 2024 年 3 月的 Starling Beta 模型 [@zhu2024starling]。
它是从另一个聊天模型 OpenChat [@wang2023openchat] fine-tuned 而来的（而 OpenChat 实际上是由另一个完全不同的组织训练的）。
其训练完全专注于 k-wise reward model 训练和 PPO 优化，使其在 Arena 上提升了 10 个位次。
模型的平均回复长度有所增加，但这种增加的方式足以真正帮助人类评判者。
后来的例子，如 Olmo 3，实际上被记录为经历了大量的聊天训练，但作者更偏好具有更高数学、编程和推理得分的最终模型 checkpoint，而不是在基于 LLM-as-a-judge 的聊天 benchmark 上得分最高的潜在 checkpoint [@teamolmo2025olmo3]。

一个自然的问题是：为什么 RLHF 会使模型回复变长？
从根本上说，Arena 等评测告诉我们，当与简洁的回复进行比较时，模型的普通用户通常更喜欢更长、更完整的答案。
更长的答案在用户快速评估时，会让人感觉更全面、更有帮助，甚至更值得信赖。
这并不代表*每位*用户的偏好，但这些模型被训练为匹配众多数据标注者的平均偏好，因此 RLHF 倾向于使模型更加冗长。
