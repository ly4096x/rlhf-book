<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "训练概述"
prev-url: "03-training-overview"
page-title: Instruction Tuning
search-title: "第 4 章：Instruction Tuning"
next-chapter: "Reward Models"
next-url: "05-reward-models"
lecture-video: "https://youtu.be/4gIwiSPmQkU"
lecture-label: "第 2 讲：IFT、Reward Modeling、Rejection Sampling（第 4、5 及 9 章）"
---

# Instruction Fine-tuning

早期的大型预训练 language model 以 next-token prediction 为训练目标，默认情况下并不具备显式的 instruction following 接口。
在 GPT-3 发布前后 [@brown2020language]，prompting 和 in-context learning 成为将单一模型适配到多种任务的广泛方式（尽管针对特定任务的 fine-tuning 仍然常见），其做法是在上下文中展示示例，然后让模型完成类似任务。
instruction fine-tuning 是顺理成章的下一步——它训练模型以 instruction-response 的格式作答，而不仅仅是续写文本。
举例来说，给定 prompt "法国的首都是什么？"，base model 可能会续写"德国的首都是什么？意大利的首都是什么？……"——仅仅延续问题的模式——而经过 instruction tuning 的模型则会回答"法国的首都是巴黎。"

Instruction fine-tuning 的兴起源于两条研究路线的交汇。
其一，NLP 从针对单一任务定制 fine-tuning 的范式，转向统一的"text-to-text"或 instruction 框架，这使得将多样化 dataset 标准化、并用单一模型跨任务训练变得简单直接。
统一任务框架的代表性工作包括：*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（T5 模型）[@raffel2020exploring]、*Finetuned Language Models Are Zero-Shot Learners*（FLAN dataset）[@wei2021finetuned]、*Multitask Prompted Training Enables Zero-Shot Task Generalization*（T0 模型）[@sanh2021multitask]，以及 *Cross-Task Generalization via Natural Language Crowdsourcing Instructions*（Natural Instructions dataset）[@mishra2021cross]。
其二，预训练 LM 的规模扩大以及 prompting/in-context learning 的兴起，表明单一模型可以跨任务泛化，但当模型被显式地在 instruction-response 示例上训练后，这种泛化能力会大幅提升。
这两种趋势共同推动了在大规模 instruction 集合上 fine-tuning 预训练 language model 的时代——如今通称为 instruction fine-tuning（IFT）或 supervised fine-tuning（SFT），使训练通用模型的能力触达了更广泛的受众。

自被发现以来，instruction fine-tuning（口语上也常简称为 *instruction tuning*）已趋于成熟，成为众多 language modeling pipeline 的标准做法。
从本质上看，IFT 是将 language model 适配到目标任务分布的最简方法。
它通过将模型调整为能够处理问答形式的 instruction，为 RLHF 奠定基础，也是那些希望将现代技术应用于新领域的研究者首先使用的工具。
如果缺乏基本的 instruction following 能力，本书所讨论的大多数 pipeline——从偏好数据收集到在线 RLHF 优化——都将无法执行。

Instruction fine-tuning 在其他地方已有大量介绍，其核心本质上是 supervised learning，因此本章将重点放在对 RLHF 从业者而言最重要的实践细节上：训练数据的格式化与结构化方式。
数据和格式化方面的决策在后续训练阶段中被直接沿用，为模型吸收 post-training 数据建立了共同的语言。

## Chat Templates 与 Instruction 的结构

post-training 过程始于定义一种格式，用于将用户查询转化为 language model 通过 tokenizer 处理信息时易于读取的形式。
使用预训练 language model 时，prompting 相当简单。模型只认识少数几种 token：序列开始 token（如 `<bos_token>`）、序列结束 token（如 `<eos_token>`），以及用于在批次中处理空白部分的 padding token。
因此，要 prompt 一个 base model，用户需要输入一段 token 序列，让模型从中续写，例如：

```text
<bos_token> The capital of the United States is
```

然后，模型会持续生成 token，直到耗尽上下文窗口，或生成序列结束 token 为止。

从 instruction tuning 到 RLHF 及其他方法，所有 post-training 阶段都依赖这种格式对模型进行训练。
处理用户交互结构的工具称为 **chat template**。

下面是一个示例，我们将逐步解析：

```jinja
{% if messages[0]['role'] == 'system' %}
    {# If the conversation begins with a system message, treat it as a special first turn.
       We set an offset so the user/assistant alternation check lines up correctly. #}
    {% set offset = 1 %}
{% else %}
    {# No system message: user should be the first non-empty turn. #}
    {% set offset = 0 %}
{% endif %}

{# Emit the beginning-of-sequence token (model-specific). #}
{{ bos_token }}

{# Serialize each message into the model's chat-markup tokens. #}
{% for message in messages %}
    {# Enforce role alternation: (system), user, assistant, user, assistant, ...
       The boolean expression compares "is this a user message?" against whether the
       current index (plus offset) is expected to be user or assistant. #}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {# Wrap each message with special tokens:
       - <|im_start|><role>\n
       - message content (trimmed)
       - <|im_end|>\n
       This produces a single flat token sequence the LM can train on. #}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{# Optionally append an "assistant" start tag with no content.
   This cues generation to continue from the assistant role. #}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```
这是将 Python 中包含消息和角色的字典列表转换为 language model 可预测的 token 的原始代码。

所有传入模型的信息都被赋予一个角色。
传统的三种角色分别是 `system`、`user` 和 `assistant`。

`system` 标签仅用于对话的第一条消息；它以文本形式保存对 agent 的指令，这些指令不会发送给用户，也不会暴露给用户。
这些 **system prompt** 用于向模型提供额外的上下文，例如日期和时间，或用于修正某些行为。
举一个有趣的例子，可以告诉模型诸如"你是一个友好的 chatbot，总是用海盗的风格回复。"之类的话。

接下来，另外两种角色非常直观：**user** 保存使用 AI 的人发送的消息，**assistant** 保存模型（作为 AI 助手参与对话）的回复。

为了将所有这些信息转换为 token，我们使用上面列出的代码。
模型有一系列 *special token*，用于分隔各条消息。
如果我们用示例查询"人类一次能吃几架直升机？"运行上述代码，传入模型的 token 序列如下所示：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

注意序列末尾的 token 是 `<|im_start|>assistant`。这就是模型知道要继续生成 token 直到最终生成序列结束 token（此处为 `<|im_end|>`）的方式。

通过将所有问答对数据（以及下游的偏好 tuning 数据）打包成这种格式，现代 language model 能够完全一致地遵循它。这是经过 instruction tuning 的模型与用户以及运行在 GPU 或其他计算设备上的模型交换信息所使用的语言。

这种行为可以直接扩展到多轮对话，如下所示：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

在开源生态中，将 chat template 应用于消息列表的标准方式使用 Jinja 片段——一种轻量的 Python 模板语言——存储在 tokenizer 配置中，作为 `apply_chat_template`。

上述 chat template 是 OpenAI 的 Chat Markup Language（ChatML）的衍生版本，ChatML 是早期标准化消息格式的尝试。
如今，OpenAI 和其他模型提供商采用了分层系统，用户可以配置 system message，但还存在更高层级的指令，这些指令可能不会向用户披露 [@wallace2024instruction]。

还存在许多其他 chat template。其他示例包括 Zephyr 的 [@tunstall2023zephyr]：

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

或 Tülu 的：

```text
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

此外，许多 chat template 还包含用于工具调用等任务的格式和其他 token。


## Instruction Tuning 的最佳实践

Instruction tuning 作为 post-training 和构建有益 language model 的基础，已是公认的做法。
实现成功的 instruction tuning 有很多方式。
例如，对部分模型参数进行量化的高效 fine-tuning 方式使训练变得非常便捷 [@dettmers2023qlora]。
此外，在狭窄领域（如对话对齐，即不涉及数学或代码等更难技能的场景）中，规模小而聚焦的 dataset 也能取得良好性能 [@zhou2023lima]。

ChatGPT 发布不久后，仅含 1 万条样本的人工 dataset（如 No Robots）已属于最先进水平 [@no_robots]。
数年后，大规模合成 dataset 在大多数任务上表现最好 [@lambert2024t]。

以下几项原则始终适用：

- 高质量数据是性能的关键。模型真正学习的是补全内容（在许多情况下，prompt token 不参与预测，因此模型不会学习预测 prompt）。
- 约 100 万条 prompt 可用于训练出一个能够进行出色 RLHF 和 post-training 的模型。进一步扩展仍有帮助，但收益递减很快。
- 最优质的 prompt 是那些与下游目标任务分布相似的 prompt。
- 如果在 instruction tuning 之后还进行了多个训练阶段，模型可以从 instruction tuning 数据中的部分噪声中恢复。优化整体训练流程比优化每个单独阶段更为重要。

## 实现细节

虽然 loss function 与 pretraining 相同，但有几个关键实现细节与 pretraining 设置不同。
许多做法，例如决定如何将模型跨多块 GPU 进行切分的并行方式，与 pretraining 相同，只是所使用的机器总数通常更少（对应下述第一个技术变化）：

- **更小的 batch size**：与 pretraining 相比，instruction tuning（以及其他 post-training 技术，如偏好 fine-tuning）使用明显更小的 batch size，以便在更窄的数据分布上进行良好优化，同时保留模型从 pretraining 中获得的泛化能力。例如，OLMo 2 在 7B 和 13B pretraining 时分别使用 1024 和 2048 的 packed-row batch size，这些模型的总上下文长度为 4096 token，batch 中每行是填满序列长度的多个文档的组合。在 post-training 中，这两个模型仅使用 256 个 *prompt* 的 batch size [@olmo20242]，且不进行填满完整序列长度的操作（每个 batch 中有用的非 mask token 要少得多）。较小的 batch size 意味着这些训练任务无法像 pretraining 那样切分到尽可能多的设备上——实际上，分布式训练设置有最小的每设备 batch size，因此如果希望为 SFT 保留较小的全局 batch size，可以使用更少的 GPU 并发。在实践中，batch size 迫使缩减并发 GPU 数量并不是限制因素，因为 SFT 的训练 token 数量远小于 pretraining，且 post-training 中需要多次随机种子训练以获得最佳最终性能。
- **Prompt masking**：在 pretraining 时，batch 中的每个 token 都被自回归地预测，并对其应用 loss。对于 instruction tuning，prompt token 被 mask 掉，使模型不学习准确预测用户查询——只学习预测响应。其他 post-training 算法同样如此。
- **多轮 masking**：对于多轮对话，有两种常见的 masking 选择。（1）*仅最后一轮*：只将最后一个 assistant 轮次中的 token 纳入 loss，所有更早的上下文（包括更早的 assistant 轮次）均被 mask。长对话仍可"展开"为多个训练样本：对于一段 $N$ 轮的对话，每个样本预测一个 assistant 回复，同时 mask 所有先前上下文并排除后续轮次。（2）*仅 mask user 轮次*：所有 user 轮次被 mask，但*每个* assistant 轮次都纳入 loss。如果需要更多（较短的）训练样本，在这种设置下仍可展开，但关键区别在于中间的 assistant 回复也被直接训练。
- **与 pretraining 相同的 loss function**：Instruction tuning 使用与 pretraining language model 相同的自回归 loss function，但数据和 masking 方式有很大不同（仅在完整序列上训练，而 pretraining 文档可以跨 batch 切分），等等。
- **Learning rate**：SFT 通常使用比 pretraining 小一到两个数量级的 learning rate，以更好地应对不同的优化动态（更小的 dataset、更小的 batch 以及强大的预训练初始化，都倾向于更保守的更新）。例如，OLMo 2 在 pretraining 时使用 $3 \times 10^{-4}$ 的峰值 learning rate，但 SFT 时使用 $1 \times 10^{-5}$ [@olmo20242]。OLMo 3 使用更高的 SFT learning rate $5\text{-}8 \times 10^{-5}$ [@teamolmo2025olmo3]，部分原因是其训练基础设施使用了 sequence packing——将多个样本打包进每个训练序列，增加了以有效 token 数衡量的 effective batch size。更大的 batch 产生方差更低的梯度估计，进而支持更高的 learning rate 而不使训练不稳定——这一关系称为线性缩放规则。Learning rate 通常在训练步数的一小部分上进行 warmup，随后线性衰减。实践中，团队通常在多个 learning rate 上进行扫描，并在留出的评估集上选择最佳 checkpoint [@teamolmo2025olmo3]。
