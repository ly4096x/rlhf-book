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
Instruction fine-tuning 为模型提供了以问答格式可靠作答的基本能力，而 RLHF 则将这些答案塑造成我们如今期望从 language model 中获得的那种可靠、温暖且引人入胜的回答。

现代研究已将 RLHF 确立为一种将微妙的风格特征及相关行为特征融入模型的通用方法。
RLHF 实用价值的一个早期热门案例是将其应用于安全性领域 [@dai2023safe] [@bai2022training]，RLHF 使模型能够在多样化的 dataset 上同时做到有帮助且无害。
与 instruction fine-tuning 等其他 post-training 技术相比，RLHF 在跨领域泛化方面表现更优 [@kirk2023understanding] [@chu2025sft]——有助于打造高效的通用模型。

直觉上，这一点可以从优化技术的应用方式中得到体现。
Instruction fine-tuning 训练模型在前文接近已见示例时预测下一个 token。
它是在优化模型，使其更规律地在文本中输出特定特征。这是一种逐 token 的更新。

而 RLHF 则在 response 级别对补全内容进行调整，而不是具体关注下一个 token。
此外，它告诉模型*更好*的 response 应该是什么样的，而非要求模型学习某个特定的 response。
RLHF 还向模型展示了它应该避免哪类 response，即负面反馈。
实现这一目标的训练方式通常被称为 *contrastive* loss 函数（其损失由两个或多个示例之间的比较计算得出，而非分别独立地从每个示例计算），本书中将多次提及。

尽管这种灵活性是 RLHF 的主要优势，但它也带来了实现上的挑战。
这些挑战主要集中在*如何控制优化过程*上。
正如本书将介绍的，实施 RLHF 通常需要训练一个 reward model，而其最佳实践尚未得到充分确立，且依赖于具体应用领域。
与此同时，优化过程本身容易出现 *over-optimization* 问题，因为我们的奖励信号充其量只是一个代理目标，因此需要正则化。
鉴于这些局限性，有效的 RLHF 需要一个强大的起点，因此 RLHF 不能单独解决所有问题，需要放在更广泛的 post-training 视角下加以理解。

由于这种复杂性，实施 RLHF 的成本远高于简单的 instruction fine-tuning，并可能带来长度偏差等意想不到的挑战 [@singhal2023long] [@park2024disentangling]。
对于追求绝对性能的模型训练工作而言，RLHF 已被确立为实现强力 fine-tuned 模型的关键，但它在算力、数据成本和时间上的开销更大。
在 ChatGPT 之后 RLHF 早期发展历程中，有许多研究论文展示了通过有限的 instruction fine-tuning 近似实现 RLHF 的方案，但随着文献的成熟，人们一再发现 RLHF 及相关方法是模型性能的核心阶段，无法轻易舍弃。

## RLHF Recipe 演练

为本书奠定基础，有必要在没有任何技术术语的情况下了解"进行 RLHF"的基本样貌——这些术语在建立基本直觉之前往往难以理解。
本节遵循所谓的经典三阶段 RLHF recipe，该 recipe 由 OpenAI 的模型 InstructGPT 于 2022 年确立 [@ouyang2022training]。

流程的第一步是将模型从补全文本的 base model 转变为能够以问答格式运作的 instruction-following 模型。
具体做法是：在一组精心设计的数据点上使用相同的下一个 token 预测损失函数，其中模型*仅*接触这种问答格式的数据。
在模型接触这些高质量 response 之后，便可以通过特定的 token 序列提示模型，使其知道应以更加明确的助手角色回答任何查询。

有了*模型应如何作答的框架*这一基础之后，接下来的两个步骤协同配合，共同提升答案的整体质量。
这两个步骤的目的是构建一个可以使用强化学习来更新模型并使其更具帮助性的问题。

其中第一步是训练一个捕捉人类偏好的 reward model。
为了将强化学习应用于某个问题，需要一个指示质量的奖励函数。
reward model 的目标是生成一个标量信号，以便后续使用 RL 对其进行优化。
在实践中，这涉及在文本片段之间偏好关系的 dataset 上对 language model（通常是前一步骤中 instruction-tuned 的同一模型）进行 fine-tuning。
该 dataset 跨越多种 prompt、模型补全内容和标注者进行收集，以尝试捕捉 language model 更好答案的稳健信号。
reward model 学习文本中哪些特征更优，因此在推理时（以及在 RL 中作为奖励信号使用时），它会对任何输入文本片段的质量进行打分。

有了问答模型和 reward model 这两个组件，我们便拥有了将各部分组合在一起、真正实现 reinforcement learning from human feedback（RLHF）所需的一切。
实际的 RLHF 阶段是：取出代表模型应擅长任务的 prompt，生成大量补全内容，由 reward model 对其排名，然后使用 RL 确定如何改变模型以使其更好。
基本原理是：强化学习以 language model 生成的 token 形式接收哪些动作是好的信号，并推导出将不同动作归因于模型中不同参数的更新规则。
最终的 RLHF 阶段调整参数以使好的 token 更有可能被生成，并以迭代方式进行，以维持初始模型的通用能力。

一旦 RL 完成，性能趋于饱和，这通常就是提供给用户的最终模型。

通过本书，我们将介绍许多进行 RLHF 的 recipe，以及构成更广泛 post-training 套件的更多相关优化方法。
这些方法都是为了解决 language model 面临的更具挑战性的问题，并使原始 RLHF 方法的优势更加强大。

## Post-Training 的直觉理解

我们已经确认 RLHF 以及更广泛意义上的 post-training 对最新模型性能至关重要，也了解了它如何改变模型的输出，但还没有解释它为何有效。
这里有一个简单的类比，说明为何任何 base model 之上都能在 benchmark 上取得如此多的进展。

我描述 post-training 潜力的方式被称为 post-training 的激发解释，即我们所做的一切不过是通过放大 base model 中的有价值行为来释放其潜力。

为了使这个例子更易理解，我们将 base model——经过大规模下一个 token 预测预训练后产生的 language model——与构建复杂系统中的其他基础组件进行类比。我们以汽车底盘为例，底盘定义了可以在其上构建汽车的空间。
以一级方程式（F1）为例：大多数车队在年初都带着全新的底盘和引擎出现。然后，他们花一整年时间在空气动力学和系统改进上（当然，这是一个小小的过度简化），可以显著提升赛车的性能。最优秀的 F1 车队在赛季期间的进步远超底盘更换带来的提升。

post-training 同样如此：当人们越来越了解一个静态 base model 的特点和规律时，便能从中获取大量性能提升。最优秀的 post-training 团队能在极短的时间内提取出大量性能。这套技术涵盖了大部分预训练结束之后的一切内容，包括"mid-training"（如退火/预训练末期高质量网络数据）、instruction tuning、RLVR、preference-tuning 等。一个很好的例子是 Allen Institute for AI 完全开放的小型 Mixture-of-Experts（MoE）模型 OLMoE Instruct 从第一版到第二版的变化。第一版模型于 2024 年秋季发布 [@muennighoff2024olmoe]，第二版仅更新了 post-training，在未改变大部分预训练的情况下，热门 benchmark 的评估均值从 35 提升到了 48 [@ai2_olmoe_ios_2025]。

核心思想是：base model 中蕴含着大量的智能和能力，但由于它们只能以下一个 token 预测而非问答格式作答，因此需要通过 post-training 在其周围进行大量构建，才能制造出优秀的最终模型。

然后，当你看到 OpenAI 于 2025 年 2 月发布的 GPT-4.5 时——它由于 base model 体量过大、难以向数百万用户提供服务，在很大程度上是一款消费品的失败之作——你可以将其视为 OpenAI 未来构建的更具活力和潜力的基础。
有了这种直觉，base model 决定了最终模型绝大部分的潜力，而 post-training 的任务就是将其悉数开发出来。

我将这种直觉描述为 Post-training 的激发理论（Elicitation Theory of Post-training）。
这一理论与大多数用户所见的大部分进步来自 post-training 这一现实相吻合，因为它意味着：一个在互联网上预训练的模型所蕴含的潜在知识，远超我们仅通过简单训练模型便能教会它的内容——例如在早期 post-training 类型中（即仅进行 instruction tuning）反复输入某些狭窄样本。
post-training 的挑战在于：在将模型从下一个 token 预测重塑为对话问答的同时，从预训练中提取出所有这些知识与智能。

与这一理论相关的是 Superficial Alignment Hypothesis（浅层对齐假说），该假说源自论文 LIMA: Less is More for Alignment [@zhou2023lima]。这篇论文在宏观层面上把握住了一些重要的直觉，但理由有误。作者指出：

> 模型的知识和能力几乎完全在预训练期间学得，而对齐（alignment）则告知模型在与用户交互时应使用哪种子分布格式。如果这一假说成立，且对齐在很大程度上关乎学习风格，那么 Superficial Alignment Hypothesis 的一个推论是：只需相当少量的示例，便能对 pretrained language model 进行充分调整 [Kirstain et al., 2021]。

深度学习的种种成功本应让你深刻认识到：数据规模对性能至关重要。这里的关键区别在于，作者讨论的是对齐与风格——彼时学术 post-training 的关注焦点。通过数千个 instruction fine-tuning 样本，你可以对模型进行实质性改变，并在一组狭窄的评估上取得提升，例如 AlpacaEval、MT Bench、Arena（原名 ChatBotArena，一个用户匿名比较模型 response 的平台）等。这些评估并不总能转化为更具挑战性的能力，这也是 Meta 不会仅用这个 dataset 训练其 Llama Chat 模型的原因。学术研究有其借鉴意义，但若要理解技术发展的整体走向，则需谨慎解读。

这篇论文所展示的是：你可以用少量样本对模型进行实质性改变。这一点我们早已知晓，对于新模型的短期适配也很重要，但他们关于性能的论断会给普通读者留下错误的教训。

如果我们改变数据，其对模型性能和行为的影响可能会大得多，但这远非"浅层"的。如今的 base language model（未经任何 post-training）可以在一些数学问题上通过强化学习进行训练，学会输出完整的思维链推理，然后在包括 BigBenchHard、Zebra Logic、AIME 等一整套推理评估上取得更高分数。

Superficial Alignment Hypothesis 之所以有误，原因与那些认为 RLHF 和 post-training 只是在调情调调的人依然有误一样。
这是我们在 2023 年必须克服的一个领域性教训（许多 AI 观察者至今仍停留在这一认知中）。
post-training 早已远远超越那个阶段，我们正逐渐认识到：模型的风格构建于行为之上——例如如今流行的长思维链。

随着 AI 社区将 post-training 进一步推进到 agentic 和推理模型的时代，Superficial Alignment Hypothesis 愈发站不住脚。
RL 方法在训练前沿 language model 所需算力中所占的比重日益增大。
自 2024 年秋季我们在 Tülu 3 的工作中首次提出 reinforcement learning with verifiable rewards（RLVR）以来 [@lambert2024t]，用于 post-training 的算力规模已大幅增长。
因推广 RLVR 而广为人知的 DeepSeek R1，在 post-training 中仅使用了其总算力的约 5%——RL 训练消耗 147K H800 GPU 小时 [@guo2025deepseek]，而预训练底层 DeepSeek V3 base model 则消耗了 280 万 GPU 小时 [@deepseekai2025deepseekv3technicalreport]。

截至 2025 年，研究扩展 RL 核心方法的科学显示，单次消融实验可能需要耗费 10-100K GPU 小时 [@khatri2025art]，相当于 OLMo 3.1 Think 32B（2025 年 11 月发布）的 RL 阶段所用算力——该模型在 200 块 GPU 上训练了 4 周 [@teamolmo2025olmo3]。
截至 2025 年，规模化 post-training 的科学仍处于极早期阶段，正在借鉴预训练 language model 的思路和方法并将其应用于这一新领域，因此确切的 GPU 小时数将会变化，但 post-training 算力投入增加的趋势将持续下去。
综合来看，post-training 的激发理论很可能只有在采用较轻量 post-training recipe 时才是正确的视角——这对于专门化一个模型很有用——而对于算力密集型的前沿模型则不尽然。

## 我们是如何走到这一步的

为什么这本书在现在有其意义？还有多少将会改变？

Post-training——从原始 pretrained language model 中激发强大行为的技艺——自 ChatGPT 的发布重新点燃人们对 RLHF 的兴趣以来，已经历了许多阶段与情绪的变迁。
在 Alpaca [@alpaca]、Vicuna [@vicuna2023]、Koala [@koala_blogpost_2023] 和 Dolly [@DatabricksBlog2023DollyV1] 的时代，人们使用数量有限的人工数据点以及以 Self-Instruct 风格扩展的大量合成数据，对原始 LLaMA 进行普通 fine-tuning，以获得与 ChatGPT 相似的行为。
这些早期模型的benchmark完全依赖于主观感受（以及人工评估），因为我们都被这些小模型能够在各个领域展现出如此令人印象深刻的行为这一事实所深深吸引。
这种兴奋是有其道理的。

开放式post-training的发展速度更快，发布的模型更多，也比封闭的同类产品引发了更多关注。
各家公司都在争先恐后，例如 DeepMind 与 Google 合并，或者新公司刚刚成立，需要时间来跟进。
开放式技术方案经历了一轮又一轮的爆发与滞后交替出现的阶段。

继 Alpaca 等工作之后——即开放式技术方案的第一次滞后期——那个时代的特征是对 reinforcement learning from human feedback（RLHF）的怀疑与质疑。RLHF 是 OpenAI 强调的对第一代 ChatGPT 成功至关重要的技术。
许多公司怀疑自己是否需要做 RLHF。
一句流行的话——"instruction tuning 足以实现对齐"——在当时十分盛行，尽管已有大量明显的反例，这种观点至今仍有相当影响力。

对 RLHF 的质疑持续了相当长的时间，在开放社区尤为如此，因为这些团队无力承担数十万至数百万美元量级的数据预算。
那些早期拥抱 RLHF 的公司最终脱颖而出。
Anthropic 在 2022 年发表了大量关于 RLHF 的研究，如今被认为拥有最好的 post-training [@askell2021general] [@bai2022training] [@bai2022constitutional]。
开放团队与封闭团队之间的差距——无论是在复现能力上，还是在了解基本封闭技术上——是一个反复出现的主题。

开放对齐方法和 post-training 领域的第一次重大转变，是 Direct Preference Optimization（DPO）[@rafailov2024direct] 的故事。该工作表明，通过直接对成对偏好数据进行梯度更新，可以用更少的组件来求解与 RLHF 相同的优化问题。
DPO 论文于 2023 年 5 月发布，但在 2023 年秋季之前，并没有任何明显影响力的模型是用它训练的。
随后，几个突破性的 DPO 模型相继发布，情况发生了改变——这些突破的关键在于找到了一个更好、更低的 learning rate。
Zephyr-Beta [@tunstall2023zephyr]、Tülu 2 [@ivison2023camels] 以及许多其他模型都表明，post-training 的 DPO 时代已经到来。
Chris Manning 亲口感谢了我"拯救了 DPO"。

自 2023 年末以来，preference-tuning 已成为发布一个优质模型所必须完成的工作。
DPO 时代延续到了整个 2024 年，以各种算法变体的形式不断演进，但我们也陷入了开放式技术方案的另一次深度低迷期。
开放式 post-training 技术方案已经耗尽了现有知识和资源所能达到的上限。
在 Zephyr 和 Tülu 2 发布一年后，同一个突破性 dataset——UltraFeedback——在开放式技术方案的 preference tuning 中可以说仍然是最先进的 [@cui2023ultrafeedback]。

与此同时，Llama 3.1 [@dubey2024llama] 和 Nemotron 4 340B [@adler2024nemotron] 的技术报告给了我们重要的提示：大规模 post-training 远比想象中更加复杂且影响深远。
封闭实验室正在进行完整的 post-training——一个包含 instruction tuning、RLHF、prompt 设计等多阶段的大型流程——而学术论文仅仅触及了皮毛。
Tülu 3 代表了一次全面的、开放的努力，旨在为未来的学术 post-training 研究奠定基础 [@lambert2024t]。

Post-training 是一个复杂的过程，涉及上述各种训练目标以不同顺序组合应用，以针对特定能力进行优化。
本书旨在提供一个平台，帮助读者理解所有这些技术，随着该领域的成熟，如何将它们交织运用的最佳实践也将逐渐浮现。

Post-training 目前的主要创新领域集中在 reinforcement learning with verifiable rewards（RLVR）、推理训练以及相关思路上。
这些较新的方法大量借鉴了 RLHF 的基础设施和思想，但演进速度要快得多。
本书旨在记录 RLHF 经历最初快速变革期之后的第一批稳定文献。

## 本书范围

本书希望涵盖实现经典 RLHF 的每个核心步骤。
本书不会涵盖各组成部分的完整历史，也不会涵盖最新的研究方法，而只关注那些已被反复证明会出现的技术、问题和权衡。

### 章节概览


本书包含以下章节：

#### 导论

贯穿全书的参考材料和背景知识。

1. 导论：RLHF 概述及本书所提供的内容。
2. 关键相关工作：RLHF 技术历史中的关键模型和论文。
3. 训练概述：RLHF 训练目标的设计方式及其理解基础。

#### 核心训练流程

用于优化 language model 以使其与人类偏好对齐的一套技术。

4. Instruction Tuning：将 language model 适配为问答格式。
5. Reward Modeling：从偏好数据中训练 reward model，作为 RL 训练的优化目标（或用于数据过滤）。
6. Reinforcement Learning（即 Policy Gradients）：用于在整个 RLHF 过程中优化 reward model（及其他信号）的核心 RL 技术。
7. 推理与推断时扩展：新型 RL 训练方法在推断时扩展方面相对于 post-training 和 RLHF 的作用。
8. Direct Alignment 算法：直接从成对偏好数据优化 RLHF 目标、而非先学习 reward model 的算法。
9. Rejection Sampling：一种将 reward model 与 instruction tuning 结合使用以对齐模型的基本技术。

#### 数据与偏好

为 RLHF 提供驱动力的数据，以及它所尝试解决的宏观问题的背景介绍。

10. 什么是偏好？：为什么需要人类偏好数据来驱动和理解 RLHF。
11. 偏好数据：如何为 RLHF 收集偏好数据。
12. 合成数据与 AI 反馈：从人类数据向合成数据的转变、AI 反馈的工作原理，以及如何从其他模型中提炼知识。
13. 工具使用与函数调用：训练模型在输出中调用函数或工具的基础知识。

#### 实践注意事项

实现和评估 RLHF 的基本问题与讨论。

14. Over-optimization：关于 RLHF 为何出错以及为何 over-optimization 在以 reward model 为软优化目标时不可避免的定性观察。
15. 正则化：将这些优化工具约束在参数空间有效区域的方法。
16. 评估：language model 中不断演进的评估（及 prompting）角色。
17. 产品、用户体验与特性：随着主要 AI 实验室将 RLHF 用于微妙地将模型与产品匹配，RLHF 在适用性方面的演变。

#### 附录

定义与扩展讨论的参考材料。

- 附录 A - 定义：本书中所用 RL、语言建模及其他机器学习技术的数学定义。
- 附录 B - 风格与信息：RLHF 在提升模型用户体验方面的作用往往被低估，这源于风格在信息传达中所扮演的关键角色。


### 目标读者

本书面向具备 language modeling、reinforcement learning 和通用机器学习入门经验的读者。
本书不会对所有技术进行详尽的说明，而只关注那些对理解 RLHF 至关重要的内容。

### 如何使用本书

本书的创作初衷，主要是因为 RLHF 工作流程中许多重要主题缺乏权威参考资料。
考虑到 LLM 整体的进步速度，加之收集和使用人类数据的复杂性，RLHF 是一个异常偏学术的领域，已发表的结果往往有噪声，且难以在不同环境下复现。
为了培养扎实的直觉，建议读者就每个主题阅读多篇论文，而不是将任何单一结果视为定论。
为此，本书收录了大量学术风格的引用，指向每个论断的权威参考文献。

本书的贡献旨在为你提供尝试玩具实现或深入文献所需的最少知识。
这*不是*一本全面的教科书，而是一本用于快速回顾和入门的简明读本。

本书于 2026 年 4 月定稿，正在进入印刷生产阶段。作为一本以网络为首的书籍，其内容将持续演进，如果你发现了错别字或重要的遗漏，欢迎在 [GitHub](https://github.com/natolambert/rlhf-book) 上贡献修正或建议。

### 关于作者

Nathan Lambert 博士是一位研究人员和作家，专注于构建 language model 的开放科学。他通过机器人学博士学位来到这一领域，并在 ChatGPT 发布后不久组建了一支 RLHF 团队。
他在 Allen Institute for AI（Ai2）和 HuggingFace 任职期间，发布了许多用 RLHF 训练的模型、相应的 dataset 以及训练代码库。
代表性成果包括 [Zephyr-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)、[Tulu 2](https://huggingface.co/allenai/tulu-2-dpo-70b)、[OLMo](https://huggingface.co/allenai/OLMo-7B-Instruct)、[TRL](https://github.com/huggingface/trl)、[Open Instruct](https://github.com/allenai/open-instruct) 等众多项目。
他在 RLHF 领域著述颇丰，包括[众多博客文章](https://www.interconnects.ai/t/rlhf)和[学术论文](https://scholar.google.com/citations?hl=en&user=O4jW7BsAAAAJ&view_op=list_works&sortby=pubdate)。

## RLHF 的未来

随着对 language modeling 的持续投入，传统 RLHF 方法衍生出了许多变体。
RLHF 在口语化使用中已成为多种相互交叠的方法的代名词。
RLHF 是 preference fine-tuning（PreFT）技术的一个子集，后者还包括 Direct Alignment 算法（见第 8 章）——这类方法是 DPO 的下游衍生，通过直接对偏好数据进行梯度更新来解决偏好学习问题，而非先学习一个中间 reward model。
RLHF 是与 language model "post-training" 快速进步最密切相关的工具，而 post-training 涵盖了大规模自回归网络数据预训练之后的所有训练。
本教材是对 RLHF 及其直接相邻方法的广泛概述，包括 instruction tuning 和其他为模型进行 RLHF 训练所需的实现细节。

随着用 RL fine-tuning language model 的更多成功案例不断涌现——例如 OpenAI 的 o1 推理模型——RLHF 将被视为推动 RL 方法进一步投入 fine-tuning 大型 base model 的桥梁。
与此同时，尽管在不久的将来，关注的焦点可能会更集中在 RLHF 中的 RL 部分——作为在高价值任务上最大化性能的手段——但 RLHF 的核心在于，它是一个研究现代 AI 所面临重大问题的视角。
我们如何将人类价值观和目标的复杂性映射到我们日常使用的系统中？
