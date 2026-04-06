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
