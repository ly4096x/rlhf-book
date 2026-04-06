<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "主页"
prev-url: "https://rlhfbook.com/"
page-title: 引言
search-title: "第1章：引言"
next-chapter: "关键相关工作"
next-url: "02-related-works"
lecture-video: "https://youtu.be/o6l6tJQgUg4"
lecture-label: "第一讲：概览（第1–3章）"
---

# 引言

Reinforcement learning from Human Feedback（RLHF）是一种将人类信息融入AI系统的技术。
RLHF主要作为解决难以明确描述的问题的方法而兴起。
对于那些专为人类直接使用而设计的系统，由于个体偏好往往难以言表，此类问题随时都会出现。这涵盖了与数字系统交互的每一个内容领域。
RLHF早期的应用通常集中在控制问题以及其他传统的reinforcement learning（RL）领域，目标是优化特定行为以完成任务。
RLHF领域诞生的核心理念是："我们能否仅凭基本的偏好信号来引导优化过程，从而解决困难问题？"
RLHF最广为人知是通过ChatGPT的发布，以及随之而来的large language models（LLMs）和其他foundation models的快速发展。

RLHF的基本pipeline包含三个步骤。
首先，需要训练一个能够遵循用户问题的language model（见第4章）。
其次，必须收集人类偏好数据，用于训练人类偏好的reward model（见第5章）。
最后，language model可以通过所选的RL优化器进行优化，方法是采样生成内容并根据reward model对其进行评分（见第3章和第6章）。
本书详细介绍了该过程每个步骤中的关键决策和基本实现示例。

RLHF已成功应用于众多领域，随着技术的成熟，其复杂性也在不断提升。
RLHF的早期突破性实验被应用于深度reinforcement learning [@christiano2017deep]、摘要生成 [@stiennon2020learning]、指令遵循 [@ouyang2022training]、解析网络信息以进行问答 [@nakano2021webgpt]，以及"对齐" [@bai2022training]。
早期RLHF方案的概述如下图 @fig:rlhf-basic 所示。

![早期三阶段RLHF流程的示意图，包含SFT、reward model以及优化阶段。](images/rlhf-basic.png){#fig:rlhf-basic}

在现代language model训练中，RLHF是post-training的组成部分之一。
Post-training是一套更完整的技术和最佳实践，旨在使language models在下游任务中更加实用 [@lambert2024t]。
Post-training可以概括为一个多阶段训练过程，使用三种优化方法：

1. 指令微调/Supervised Fine-tuning（IFT/SFT）：教授格式规范，并奠定指令遵循能力的基础。这主要是在语言中学习*特征*。
2. 偏好微调（PreFT）：通过RLHF及相关方法与人类偏好对齐（同时在能力上取得较小的提升）。这主要关乎语言的*风格*以及难以量化的微妙人类偏好。
3. Reinforcement Learning with Verifiable Rewards（RLVR）：最新类型的post-training，通过更多RL训练提升在可验证领域的表现。

RLHF存在于并主导着第二个领域——**偏好微调**，其复杂性高于指令微调，因为它通常涉及真实目标的代理reward models以及噪声更大的数据。
与此同时，RLHF比另一种流行的language models RL方法——reinforcement learning with verifiable rewards——要成熟得多。
因此，本书专注于偏好学习，但为了全面理解RLHF的作用，读者还需要了解其他训练阶段，本书也对此进行了详细说明。

当我们考虑这些方法的选项空间，以及社会对这些用于构建我们广泛使用的模型的方法的关注时，RLHF在通俗意义上*正是*推动现代post-training发展的技术。
RLHF是使ChatGPT发布取得巨大成功的关键技术，因此在2023年初，RLHF涵盖了post-training这一通用领域的大部分关注度。
RLHF现在只是post-training的一个组成部分，因此本书梳理了RLHF早期受到如此多关注的原因，以及其他方法如何涌现以对其进行补充。

训练language models是一个极其复杂的过程，通常需要数十至数百人的大型技术团队，以及数百万美元的数据和算力成本。
本书有三个目的，帮助读者掌握RLHF及相关模型如何被用于构建领先模型。
首先，本书将大型科技公司内部往往隐而不宣的前沿研究提炼成清晰的主题和权衡取舍，使读者能够理解模型是如何构建的。
其次，本书将允许用户搭建基本的代码示例，亲自动手对这些模型进行fine-tuning。
最后，除了传授RLHF的技术方法之外，本书旨在提炼关于*为何*RLHF对现代AI模型至关重要的直觉。

由于RLHF的复杂性，以及当前最先进的技术往往过于复杂而难以单独完成，本书专注于为读者提供在该领域找到工作和启动研究项目所需的工具。
其他读者则只是希望精准地理解当今全球无数讨论所聚焦的这项技术的内部运作机制。
一本为特定需求精确列出RLHF实施方案的书是不可能存在的，这也正是为什么有一个庞大的行业，众多公司将RLHF训练及相关方法作为价值数百万美元的服务来提供。
尽管如此，AI系统的快速进步意味着掌握这些知识的读者能够随着时间推移，越来越多地独立完成自己的训练工作。

## RLHF能做什么？

围绕RLHF最重要的问题，也是迄今仍难以回答的问题，是"RLHF训练能为模型带来什么？"
近年来，language models已从在benchmark范围内研究的学术实验，转变为通用技术。
RLHF是这一转变的核心所在。

首先，让我们思考一下language model在自回归预训练结束时如何回答一个问题。
一个强大的模型，Llama 3.1 405B Base，在面对"The president of the united states in 2006 was"时，会续写如下内容：

> **Response**: George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006 - who later lost to obama.
> September 1 -- U.S. President Bush signs an executive order to provide more options for faith-based organizations when delivering social services that are paid for with federal funds.
> January 1 -- The current deadline set by United States Congress in the Unlawful Internet Gambling Enforcement Act, aka UIGEA.
> search: Amendments to the 1961 International Convention for the Protection of New Varieties of Plants require plant breeders' rights include farmer's privilege.
> 2009 was a common year starting on Thursday of the Gregorian calendar, the 2009th year of the Common Era (ce) and Anno Domini (ad) designations, the 9th year of the 3rd millennium, the 9th year of the 21st century, and the 10th and last year of the 2000s decade.

显而易见，该模型正在续写句子并附加其他常见的互联网元数据。

相比之下，经过RLHF（及其他post-training工具）训练的模型会以截然不同的方式回答同一问题。
向Tülu 3 405B提出同样的问题"The president of the united states in 2006 was"，它简洁地回答：

> **Response**: George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009.

这种风格上的转变，是本书所描述的post-training方法的下游效果。

理解RLHF如何运作的最具说服力的视角，是思考*风格*如何适用于你与language models的交互。
信息呈现的风格，与语气或格式密切相关，对于信息的学习方式至关重要。
风格通常用于描述模型的表达方式。为了直观感受差异，请考虑以下问题的两种示例回答：

> **用户**：救命，我明天要做一个演示报告，我现在很慌。

生硬、低共情风格：

> **Response**: You'll be fine. Practice your slides a few times, get some sleep, and stop overthinking it. Just present what you know and move on.

更温暖、更具支持性的风格：

> **Response**: That sounds really stressful, and it's a normal reaction. If you can, do one quick practice run now, pick the one or two points you most want people to remember, and then give yourself permission to rest—being clear and calm tomorrow will matter more than cramming tonight.


另一方面，格式指的是答案的结构方式。
现代模型以大量使用包含列表、换行符和emoji的markdown格式著称。
请考虑以下问题的两种截然不同的回答：

> **用户**：我怎样才能提高睡眠质量？

简单的回答，来自早期language model：

> **Response**: Try to go to bed at the same time each night, avoid caffeine late in the day, and keep your bedroom quiet and dark.

或者一个复杂、更丰富的回答：

> **Response**: Better sleep, in practice
> 
> **Goal**: make your schedule predictable and your environment boring enough that your brain gives up.
>
