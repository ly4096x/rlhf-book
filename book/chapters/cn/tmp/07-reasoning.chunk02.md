> **回复**：`<thinking>` 好的，用户想要一首关于金鱼的短诗。让我先头脑风暴一些与金鱼相关的意象。金鱼通常在鱼缸里，对吧？所以也许从环境入手。"镀金鱼缸"或"水晶球"这样的词语可能有用。
> 
> 接着，想想金鱼的游动方式。它们游动流畅，所以"穿梭"、"滑行"或"旋转"这样的词浮现脑海。也许用"丝质"来形容它们的鱼鳍，以强调其娇柔。
> 
> 然后，考虑金鱼周围的环境。阳光透过水面产生波光粼粼的效果。"阳光起舞"或"涟漪旋转"这样的词语或许能捕捉到这种意境。加入城堡或鹅卵石等元素会增添趣味。
> 
> 主题？也许是金鱼渺小的世界与它对自由的感知之间的对比。鱼缸作为一个宇宙，虽小却完整。以一种神秘感或静默的观察作结，能赋予诗歌深度。需要保持简洁，也许四个诗节，带有押韵方案。让我检查一下节奏和流畅度。避免陈词滥调，但保持易读。好了，让我们把它拼在一起。`</thinking>` 
> 
> 液态玻璃中的镀金梦
> 
> 一只镀金鱼缸，一个水晶球，
> 细小的潮汐在寂静中流淌——
> 一抹橙色，一袭低语的衣裳，
> 你穿梭于阳光下的靛青之中。
> 
> 你的世界虽小，却也无边无际——
> 每块鹅卵石、每座城堡、每株水草舒展。
> 水面将天空折射成蔚蓝，
> 涟漪旋转着你的液态世界。
> 
> 你滑行于我们永不知晓的神话之中，
> 一位沉默的君主，隐匿于薄雾。
> 梦境是否随气泡飘向远方？
> 或是记忆封存于紫水晶？
> 
> 我们注视你描绘你的无尽轨迹——
> 一滴泪珠中的宇宙。

`<thinking>` 标签中的 token 是模型的推理过程。
对于更复杂的问题，推理阶段在产生答案之前可能需要数千个 token。
因此，长上下文语言模型是实现高级 reasoning 行为的先决条件，但这不是本章的重点。

*这种训练如何运作*的核心直觉是：对于给定的模型，我们重复以下循环：

1. 对多个问题采样多个答案，
2. 朝着正确答案的方向进行梯度更新，以及
3. 重复，重新审视相同的数据。

值得注意的是，这种极其简单的方法（在数据分布经过精心设计且训练基础设施稳定的情况下）帮助模型通过反复审视相同问题来学习。
更令人惊叹的是，在这些训练问题上的改进能够泛化到模型从未见过的问题和（某些）领域！

这种简单的方法允许模型在行为空间上进行轻度 search，RL 算法则增加与正确答案相关的行为的可能性。

## 新 reasoning 模型的起源

在此我们详细介绍导致 2025 年 reasoning 模型爆发式增长的高层趋势。

### RL 为何现在奏效？

尽管有许多观点认为"RL 尚不奏效" [@irpan2018deep]，或有论文详细记录 RL 的深层可复现性问题 [@henderson2018deep]，该领域还是克服了这些挑战，找到了高影响力的应用。
其中一些在本书中有所涉及，例如 ChatGPT 的 RLHF 和 DeepSeek R1 的 RLVR，但也存在许多其他应用，包括改进芯片设计 [@mirhoseini2020chip]、掌握视频游戏 [@schrittwieser2020mastering]、自动驾驶 [@cusumano2025robust] 等。
以 RL 为重点的语言模型训练的腾飞表明，该研究领域在许多基本问题上取得了进展，包括：

- **RL 的稳定性问题可以解决**：RL 整个存在期间，限制其采用的因素一直是稳定性。这体现在两个方面。首先，学习本身可能是多变的，并不总是奏效。其次，训练本身比标准语言模型训练更脆弱，更容易出现损失尖峰、崩溃等问题。无数新的模型发布都在使用这种基于预训练 base model 并结合可验证 reward 的 RL 训练风格，学术界的采纳也大幅增加。RL 的技术门槛处于历史最低水平。

- **开源版本已经"存在"**：已有许多工具可用于使用 RLVR 及相关技术训练语言模型。
示例包括 TRL [@vonwerra2022trl]、Open Instruct [@lambert2024t]、veRL [@sheng2024hybridflow] 和 OpenRLHF [@hu2024openrlhf]，其中许多都建立在 RLHF 和 post-training 早期弧线中的优化成果之上。工具的可及性正在推动大量且不断加速的研究工作。

多方资源指出，reasoning 的 RL 训练仅在约 2024 年起的领先模型中才具有可行性，这表明在 reasoning 训练成为可能之前，模型需要具备一定程度的底层能力。

### RL 训练与 inference-time scaling

通过 reinforcement learning 进行训练以引发 reasoning 行为和在可验证领域的性能，与 inference-time scaling 的理念密切相关。
Inference-time scaling，也称为 test-time scaling，是一类通用方法，在 inference 时使用更多计算能力以在下游任务中表现更佳。
在 DeepSeek R1 和 OpenAI o1 发布之前，inference-time scaling 方法已经得到研究，而这两者都极大地推动了对 RL 训练的投入。
示例包括价值引导采样 [@liu2023don] 或带答案提取的重复随机采样 [@brown2024large]。
除此之外，inference-time scaling 还可用于改进 AI 训练的更多方法，而不仅限于 chain-of-thought reasoning 来解决问题，例如让 reward model 深入考虑各种选项 [@ankner2024critique] [@liu2025inference]。

RL 训练是使用 inference-time scaling 规律的捷径，但从长远来看，我们将拥有更多方法来引发所需的 inference-time 权衡以获得最佳性能。
对模型进行大量 RL 训练通常能使其每次回复生成更多 token，这与改进的下游性能高度相关（尽管这种序列长度增加是默认情况，但也有研究明确致力于在*不*依赖这种 inference-time scaling 的情况下提升性能）。
这与早期 RLHF 系统中观察到的长度偏差 [@singhal2023long] 有实质性转变，在早期系统中，人类偏好训练的副作用是增加了平均回复长度，以换取偏好排名上的边际收益。

除核心 RL 训练模型外，还有许多方法正在被探索，以继续突破 reasoning 和 inference-time 算力的极限。
由于这些方法发展迅速，大多超出了本书的讨论范围，但它们包括：通过 instruction tuning 将较大 RL 训练模型的 reasoning 行为蒸馏到较小模型 [@muennighoff2025s1]、组合更多 inference 调用 [@chen2024more] 等。
这里重要的是下游性能与生成 token 数量增加之间的相关性——否则只是浪费能源。


### RLVR 的未来（超越 reasoning）

在许多领域，这些新形式的 RLVR 与开发者的目标更加契合，因为它们专注于性能而非行为。
标准的 fine-tuning API 通常使用参数高效 fine-tuning 方法，例如 LoRA（Low-Rank Adaptation，一种参数高效方法，仅训练小型附加矩阵而非所有模型权重，也称为参数高效 fine-tuning，PEFT），结合指令的 supervised fine-tuning。
开发者输入提示词和补全内容，模型通过更新模型参数来匹配这些补全内容进行调优，从而增加数据中特征在模型生成中的出现频率。

RLVR 专注于匹配答案。
给定查询和正确答案，RLVR 帮助模型学习生成正确答案。
标准 instruction tuning 对数据进行 1 到 2 个 epoch 的损失更新，而 RLVR 则得名于对相同的少量数据点进行数百或数千个 epoch 的训练，从而给模型时间学习新行为。
这可以被视为将 base model 版本中偶尔有效的正向行为，通过 RLVR 强化为稳健的行为。

**针对语言模型的 RL 训练范围持续扩大**：从基础科学层面来看，o1 和 R1 最大的启示在于，我们拥有了更多方式将语言模型训练到潜在有价值的行为。
研究人员和工程师可用的方向越多，我们对 AI 总体发展轨迹就应越乐观。


## 理解 reasoning 训练方法

对 reasoning 的投入引发了语言模型遵循人类指令训练方式的重大演变。
这些方案仍然使用前面章节中讨论的通用组件（如第 3 章对 DeepSeek R1 方案的概述中所讨论的），包括 instruction fine-tuning、reinforcement learning from human feedback 以及 reinforcement learning with verifiable rewards（RLVR）。
核心变化是使用更多的 RLVR，并以不同的顺序应用其他训练技术——传统上，reasoning 模型的核心训练步骤要么是大规模 RL 运行，要么是对另一个经过大量 RLVR 训练的模型的*输出*进行大规模 instruction tuning（称为蒸馏）。

### OpenAI o1 或 DeepSeek R1 之前的 reasoning 研究

在 reasoning 模型起飞之前，学界付出了大量努力来理解如何训练语言模型在可验证领域表现更好。
以下这些工作的主要区别在于：它们的方法论未能扩展到与 DeepSeek R1 及后续模型相同的程度，或者它们产生的模型在整体性能上有所牺牲，以换取更高的数学或编程能力。
这里包含了这些工作的基本思想和动机，以便更全面地描绘 reasoning 模型在整体格局中的涌现过程。

最早尝试在可验证领域训练语言模型的工作包括 self-taught reasoner（STaR）系列工作 [@zelikman2022star] [@Zelikman2024QuietSTaRLM] 和 TRICE [@hoffman2023training]，这两者都在 2022 年至 2023 年间使用真实 reward 信号来鼓励模型中的 chain-of-thought reasoning。
STaR 有效地近似了 policy gradient 算法，但在实践中对样本进行了不同的过滤，并使用交叉熵度量代替对数概率；Quiet-STaR 则以非常相关的思路扩展了这一方法，通过让模型在尝试回答可验证问题之前先生成 token（这有助于提升训练性能），与近期 reasoning 模型的思路高度相关。
TRICE [@hoffman2023training] 也通过生成推理轨迹，然后用自定义的马尔可夫链蒙特卡洛启发式期望最大化算法进行优化来改进 reasoning。
VinePPO [@VinePPO] 在这些工作之后出现，使用的设置更接近现代 reasoning 模型。
VinePPO 使用基于 PPO 的算法，以数学问题正确性的二元 reward 为信号，在 GSM8K 和 MATH 上进行训练。
OpenAI o1 和 DeepSeek R1 之前的其他工作使用代码执行作为训练的反馈信号 [@gehring2024rlefgroundingcodellms]、[@xu2024dpo]，或用于定理证明的验证（此处称为 Reinforcement Learning from Verifier Feedback，RLVF）[@amit2024models]。
Tülu 3 在这些方法的基础上进行了扩展，使用简单的 PPO 训练器对正确答案的补全内容进行 reward——最重要的是同时保持了模型在广泛评估套件上的整体性能。
Tülu 3 和现代 reasoning 训练技术的二元 reward 可与 STaR 的迭代方法或 Quiet-STaR 的对数似然 reward 进行对比。

### 早期 reasoning 模型

继 DeepSeek R1 之后，部分奠基性 reasoning 研究报告的摘要（其中一些附有开放的数据和模型权重）见 @tbl:reasoning_list。

::: {.table-wrap}
| 日期        | 名称                        | 简介                                                                  | 开放权重 | 开放数据 |
