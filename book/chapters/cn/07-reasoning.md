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
| 2025-01-22  | DeepSeek R1 [@guo2025deepseek]             | 基于 RL 对 DeepSeek 的升级，在数学和代码 reasoning 方面大幅提升      |  是      | 否   |
| 2025-01-22  | Kimi 1.5 [@team2025kimi]                  | 在中英文数据上扩展 PPO/GRPO；在 AIME 数学上表现强劲            | 否           | 否        |
| 2025-03-31  | Open-Reasoner-Zero [@hu2025openreasonerzero]   | 对基础模型 RL 的完全开放复现      |  是      |  是   |
| 2025-04-10  | Seed-Thinking 1.5 [@seed2025seed]         | 字节跳动 RL 流水线，带有动态 chain-of-thought 门控                         | 是     | 否   |
| 2025-04-30  | Phi-4 Reasoning [@abdin2025phi4]          | 14B 模型；精心设计的 SFT→RL 流程；在 STEM reasoning 方面表现出色                   | 是      | 否        |
| 2025-05-02  | Llama-Nemotron [@bercovich2025llamanemotron]   | 多规格"reasoning 切换"模型                 |  是      |  是   |
| 2025-05-12  | INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained] | 首个公开记录的全球去中心化 RL 训练运行                    |  是      |  是   |
| 2025-05-12  | Xiaomi MiMo [@xia2025mimo]                | 从预训练到后训练的端到端 reasoning 流水线              | 是          | 否       |
| 2025-05-14  | Qwen 3 [@yang2025qwen3]                   | 将类似 R1 的训练方案应用于新模型                    |  是      | 否   |
| 2025-05-21  | Hunyuan-TurboS [@liu2025hunyuan]          | Mamba-Transformer MoE，自适应长/短 chain-of-thought                        | 否           | 否        |
| 2025-05-28  | Skywork OR-1 [@he2025skyworkor1]          | 避免熵崩溃的 RL 方案；在 AIME 上超越 DeepSeek           |  是      |  是   |
| 2025-06-04  | Xiaomi MiMo VL [@coreteam2025mimovltechnicalreport]                | 将端到端 reasoning 流水线适配至多模态任务              | 是          | 否       |
| 2025-06-04  | OpenThoughts [@guha2025openthoughts]      | 从 QwQ-32B 蒸馏而来的公开 120 万条示例指令数据集                    |  是      |  是   |
| 2025-06-10  | Magistral [@mistral2025magistral]         | 在 Mistral 3 上进行纯 RL 训练；多语言 chain-of-thought；小模型开源      |  是| 否        |
| 2025-06-16 | MiniMax-M1 [@minimax2025minimax_m1] | 开放权重 456B MoE 混合/闪电注意力 reasoning 模型；1M 上下文；使用 CISPO 进行 RL；发布 40K/80K thinking 预算检查点 | 是 | 否 |
| 2025-07-10 | Kimi K2 [@kimiteam2025kimik2]                            | 1T MoE（32B 激活参数），使用 MuonClip（QK 裁剪）保持稳定性；在 15.5T token 上预训练无损失尖峰；多阶段后训练含智能体数据合成 + 联合 RL；发布基础版和后训练版检查点。                               | 是          | 否         |
| 2025-07-28 | GLM-4.5 [@zeng2025glm45] | 开放权重 355B-A32B MoE "ARC" 模型，具有 thinking/非 thinking 模式；23T token 多阶段训练 + 后训练含专家迭代和 RL；发布 GLM-4.5 和 GLM-4.5-Air（MIT 协议）。 | 是 | 否 |
| 2025-08-20 | Nemotron Nano 2 [@nvidia2025nemotronnano2]               | 用于长"thinking traces"的混合 Mamba-Transformer；在 20T token 上进行 FP8 预训练后进行压缩/蒸馏；明确发布多个检查点以及"大部分"预训练/后训练数据集。                                       | 是          | 是（大部分） |
| 2025-09-09 | K2-Think [@llm3602025k2think]                            | 参数高效的数学 reasoning 系统：一个具有 test-time scaling 方案的 32B 开放权重模型；依据发布材料定位为完全开放，包括训练数据/代码。                                                                       | 是          | 是        |
| 2025-09-23 | LongCat-Flash-Thinking [@mlcteam2025longcat]             | 560B MoE reasoning 模型；报告明确介绍了从长 chain-of-thought 冷启动到大规模 RL 的分阶段方案；开源发布。                                                                                                             | 是          | 否         |
| 2025-10-21 | Ring-1T [@ringteam2025everystepevolves]                  | 万亿规模的"thinking 模型"，聚焦 RL 扩展；报告阐述了在 1T 规模下扩展 RL 的瓶颈与解决方案，并发布了开放模型。                                                                                                             | 是          | 否         |
| 2025-11-20 | OLMo 3 Think [@teamolmo2025olmo3]         | 完全开放的"模型流程"发布：报告了整个生命周期（各阶段、检查点和数据点），并将 OLMo 3 Think 32B 定位为旗舰级开放 thinking 模型。                                        | 是          | 是        |
| 2025-12-02 | DeepSeek V3.2 [@deepseekai2025v32]                       | 开放权重 MoE 前沿进展，报告重点介绍注意力效率改进、RL 框架升级以及用于智能体/reasoning 性能的数据合成。                                                                             | 是          | 否         |
| 2025-12-05 | K2-V2 [@liu2025k2] | 700 亿参数稠密"360 开放"从头训练模型；采用 3 档 SFT-only 后训练实现可控 thinking。 | 是 | 是 |
| 2025-12-15 | Nemotron 3 Nano [@nvidia2025nemotron3nano]               | 30B-A3B MoE 混合 Mamba-Transformer；在 25T token 上预训练，包含 SFT + 大规模 RL；明确声明发布权重 + 方案/代码 + 大部分训练数据。                                                                      | 是          | 是（大部分） |
| 2025-12-16 | MiMo-V2-Flash [@mimo2025flash] | 3090 亿 MoE（150 亿激活参数），针对速度优化：混合 SWA/GA 注意力（5:1，128 token 窗口）+ 轻量 MTP；在 27T token 上进行 FP8 预训练；后训练采用 MOPD + 大规模智能体 RL，用于 reasoning/代码。 | 是 | 否 |
表：2025 年值得关注的 reasoning 模型技术报告摘要，这是利用 RLHF 进行大规模 inference-time scaling 的第一年。{#tbl:reasoning_list}
:::


### Training Reasoning Models 的常见实践

本节详细介绍了在训练 reasoning 模型时，用于排列训练阶段和调整数据以最大化性能的常见方法。

需要注意的是，这些论文可能使用了某种技术但未加提及，而其同行却有所说明，因此这些示例仅是已知实现的子集，应作为参考，而非关于最优方案的最终定论。

- **离线难度过滤**：RLVR 的核心直觉是，模型只能从存在梯度的示例中学习。如果 RLVR 的起始模型对某个问题的解决率为 100% 或 0%，则对同一提示的不同补全之间将不存在梯度（即所有策略对 policy gradient 算法而言看起来相同）。许多模型在开始大规模 RL 之前会进行难度过滤，将训练问题限制在起始模型解决率为 20%-80% 的范围内。这些数据通过对训练集中每个提示采样 N 个（例如 16 个）补全并验证正确率来收集。Seed-Thinking 1.5、Open Reasoner Zero、Phi 4、INTELLECT-2、MiMo RL、Skywork OR-1 等均采用了此类方法。
- **批次内在线过滤**（或训练过程中的难度课程）：为了配合离线过滤以找到合适的训练问题，另一个重要问题是：在学习过程中，应以何种顺序向模型呈现问题？为解决这一问题，许多模型采用批次内在线过滤、预构建的课程/数据调度器、将较难问题留到训练后期，或其他方法来提高长期稳定性。Kimi 1.5、Magistral、Llama-Nemotron、INTELLECT-2、MiMo-RL、Hunyuan-TurboS 等均采用了相关思路。
- **移除 KL 惩罚**：随着 reasoning 模型 RL 运行时长（无论以 GPU 总时数、FLOPS 还是 RL 步数衡量）相对于 RLHF 训练显著增加，且 reward 函数变得不那么容易过度优化，许多模型移除了约束 RL 学习 policy 与训练基础模型相似的 KL 惩罚。这使模型在训练期间能够进行更多探索。RAGEN [@wang2025ragenunderstandingselfevolutionllm]、Magistral、OpenReasonerZero、Skywork OR-1 等均采用了此方法。
- **放宽 policy gradient 裁剪**：GRPO 算法的新变体，如 DAPO [@yu2025dapo]，提出了对 GRPO（或 PPO）中双侧裁剪目标的改进，以实现更好的探索。研究还表明，当 reward 不完美时，裁剪可能导致潜在的虚假学习信号 [@shao2025spurious]。RAGEN、Magistral、INTELLECT-2 等均采用了针对不同梯度方向使用不同范围的双侧裁剪方法。
- **离线数据（或完全异步更新）**：随着 RL 解决任务所需补全长度随问题难度的增加而急剧增长（尤其是响应长度的*方差*，其中通常存在极长的异常值），RL 运行中的算力可能处于空闲状态。为解决这一问题，训练正转向异步更新，或调整问题在批次中的组织方式以提高整体吞吐量。Seed-Thinking 1.5、INTELLECT-2 等均采用了部分至完全异步（离线）数据。
- **额外的格式 reward**：为使 reasoning 过程可预测，许多模型添加了少量 reward，以确保模型遵循正确格式，例如在答案前输出 `<think>...</think>`。DeepSeek R1、OpenReasonerZero、Magistral、Skywork OR-1 等均采用了此方法。
- **语言一致性 reward**：与格式 reward 类似，一些多语言 reasoning 模型使用语言一致性 reward，优先选择在 reasoning 过程中不切换语言的模型（以获得更好、更可预测的用户体验）。DeepSeek R1、Magistral 等均采用了此方法。
- **长度惩罚**：许多模型在 RL 训练中使用不同形式的长度惩罚，以稳定长期学习过程或减轻对难题的过度思考。例如，Kimi 1.5 在训练准确率在难度课程中保持较高时，逐步延长目标长度以对抗过度思考；INTELLECT-2 则在整个训练过程中施加小幅长度惩罚。逐步延长训练序列长度可通过迫使模型首先在 thinking 预算有限的领域进行有效 reasoning，再过渡到更长训练以在更复杂问题上高效运用这些行为，从而缓解过度思考。其他模型则使用超长过滤及其他相关实现来提高吞吐量。
- **损失归一化**：关于原始 GRPO 算法的逐组归一化项可能引入长度或难度偏差的讨论（参见 policy gradient 章节或 [@liu2025understanding]）已有一些。因此，Magistral 或 MiMo 等部分模型选择在批次级别而非组级别对损失或优势进行归一化。
- **并行 test-time compute 扩展**：将多个并行独立采样的 rollout 的答案合并，可以相对于使用单个 rollout 的答案带来显著改善。最简单的并行 test-time compute 扩展形式（如 DeepSeek-R1、Phi-4 等所采用的）是使用多数 rollout 返回的答案作为最终答案。更高级的技术是使用经过训练的评分模型从并行 rollout 的答案中选出最佳答案。这一技术尚未被开放 reasoning 模型方案采用（截至 2025 年 6 月），但在 Claude 4 发布公告 [@anthropic2025claude4] 中有所提及，并在 DeepSeek-GRM [@liu2025inference] 中得到应用。

除上述常见技术外，还有许多关于 reasoning 训练如何在不牺牲附加能力的情况下构建有用模型的共同发现：

- **纯文本 reasoning 提升多模态性能**：Magistral、MiMo-VL 等发现，先训练多模态模型，再在此多模态训练之后进行纯文本 reasoning 训练，可以*提升*最终模型的多模态性能。
- **通过系统提示切换 reasoning**（或长度控制）：Llama-Nemotron、Nemotron Nano、Qwen 3、SmolLM 3 等使用特定系统提示（可能结合长度控制 RL 训练 [@aggarwal2025l1]），为用户提供可切换的 thinking 长度开/关功能。其他开放模型，如 OpenAI 的 GPT-OSS 和 LLM360 的 K2-V2 [@liu2025k2]，在系统提示中采用低/中/高 reasoning 努力程度设置，但这类行为的训练方法尚未得到充分记录。

## 展望未来

reasoning 模型领域的演进速度超过近期记忆中 AI 研究的任何其他领域，本章列出的某些常见实践将不可避免地被新技术所取代。

目前已有多项工作致力于系统性地理解 reasoning 训练奏效的原因。OLMo 3 Think [@teamolmo2025olmo3] 代表了对 reasoning 模型完整训练生命周期最全面的公开记录，为研究社区提供了每个阶段的检查点和数据，最终在 220 个 GPU 上完成了近 4 周的训练运行。类似地，关于理解 RL 用于 reasoning 的扩展特性的工作 [@khatri2025art] 正开始将先前仅被实践者直觉感知的算力、数据与性能之间的关系加以正式化。

有一点依然清晰：强化学习已经从"蛋糕上的樱桃"（借用那个比喻）晋升为前沿模型训练的承重组件。本章围绕 RLVR 理念介绍的那些次要技术——难度过滤、格式 reward 等等——并非最终答案，但它们代表了该领域目前对如何从语言模型中激发 reasoning 能力的最佳理解。
