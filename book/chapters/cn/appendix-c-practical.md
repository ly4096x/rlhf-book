<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "风格与信息"
prev-url: "appendix-b-style"
page-title: "附录 C：实践问题"
search-title: "附录 C：实践问题"
next-chapter: "主页"
next-url: "https://rlhfbook.com/"
---

# 实践问题与建议

本附录涵盖了大规模运行 post-training 实验的实际注意事项。
内容以经验教训列表的形式呈现，而非连贯的叙述。

## 1. Post-Training 的计算成本

对 post-training 运行进行成本估算有两种不同的方式。
最大的成本在于开发配方，这很容易达到最终几次训练运行计算量的 10 到 100 倍。
次要成本更易于衡量，即彻底应用一个配方的成本，包括多个随机种子、仔细评估、潜在的工程难题等。

就第一项成本而言，以开发像 Tülu 3 [@lambert2024t] 这样的 post-training 配方为例，团队在拥有最终模型之前，在 7B 规模上运行了数千次实验/评估。

对于最终运行，Olmo 3 报告详细说明了训练最终 32B Think 模型所涉及的内容 [@teamolmo2025olmo3]：

> Post-training follows a different operational pattern in which we run each stage multiple times, sweeping over learning rates and other hyperparameters. The theory for post-training, particularly, RL, is less developed, so we have to run multiple experiments to identify the optimal hyperparameters for a given base model. We hope to address this in future work.
>
> During post-training, checkpoint evaluation consumes a larger proportion of compute resources, in part due to long generations from reasoning models on core benchmarks. For SFT, we swept over four candidate learning rates, on 256 GPUs each, in parallel for 36 hours. Then approximately 12 hours was spent on evaluation, merging, and checkpoint confirmation, totaling approximately two days. DPO training takes less time per run (about 18 hours for a full learning-rate sweep on 64 GPUs per job) but in practice extended over multiple days due to cluster instability. The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days with at least a day of training time lost due to stability issues. After the initial release of Olmo 3, we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.

随着强化学习的规模化成为更标准的实践，这一情况将再次改变 [@khatri2025art]。
继续上述例子，最初的 Olmo 3 32B Think post-training 只花了几周时间，而为了发布改进的 Olmo 3.1 32B Think 模型，团队需要用 RLVR 额外训练 3.5 周。这在*时间*上是一笔可观的成本，而非总计算量上的成本。

## 2. 评估方差

post-training 中一个被低估的挑战是评估方差，尤其是随着推理模型的兴起——这些模型需要使用高于 0 的温度进行采样才能获得最佳评估得分。
对于任何来自模型的采样，输出都会变得更具变异性。
不同的 benchmark 具有截然不同的稳定性特征，这取决于 prompt 的难度方差、评估集中 prompt 的数量、被训练模型的脆弱性等。

在 Olmo 3 期间，团队追踪了用于评估推理模型的不同评测的方差。
下表显示了每项评测的标准差，计算方式为 14 个模型 3 次运行的标准差均值（对每个模型取方差，然后按评测取平均值）：

| 类别 | Benchmark | 标准差 |
|----------|-----------|-----------|
| 高方差 | GPQA | 1.48 |
| | AlpacaEval 3 | 1.24 |
| | IFEval | 0.88 |
| 稳定 | ZebraLogic | 0.56 |
| | Omega | 0.56 |
| | AIME 24 (Avg@32) | 0.54 |
| | HumanEvalPlus | 0.46 |
| | AgiEval | 0.43 |
| | BigBenchHard | 0.39 |
| 非常稳定 | LiveCodeBench (Avg@10) | 0.29 |
| | MBPPPlus | 0.27 |
| | MATH | 0.25 |
| | MMLU | 0.22 |
| | PopQA | 0.16 |

Table: 多次 inference 运行中评估 benchmark 的标准差，按稳定性分类（数据来自 Olmo 3）。 {#tbl:eval_variance}

某些评测（如 LiveCodeBench）既嘈杂又廉价（通过集合中较少的 prompt），因此通过每个模型重新运行评测 10 次，该评测可以从高方差集转移到稳定的设置。这对每项评测都可以做到，但很容易使成本急剧膨胀。

我们还在评估设置中看到方差来源，例如批大小、VLLM 中的张量并行设置（例如，基线的 TP=2），以及在整个基础设施中对长生成进行采样的其他敏感数值。推理模型中方差无处不在。

## 3. 管理训练性能方差

在本书讨论的所有 post-training 配方和工具中，最终模型都会受到有意义的性能方差影响。
理解这种方差的分布、其来源及其影响，对于创建强大的模型至关重要。
训练最终模型的目标是通过改变训练参数和随机种子，采样尽可能多的数据点，以获得尽可能强的模型。
请注意，这需要在模型*实际上*更好与仅仅是从评估噪声中重新抽签所带来的收益之间取得平衡。

上一节关注的是*评估*噪声，更棘手的噪声来源是训练不确定性。
评估噪声可以通过对给定 checkpoint 运行更多测试来管理（均匀降低噪声），而模型只训练一次，并且可以*受益于*正面异常值。

在实践中，训练团队采取许多步骤来从其训练配方中获取最大可能的价值：

1. 对每次最终模型运行的核心优化值（如学习率、批大小等）进行搜索。例如，对于新的 base model，我建议在宽范围内运行 10 个学习率以确保处于最优范围，然后在更紧的最优窗口内重新运行。
2. 对最佳的几个设置运行多个随机种子。随机种子对最终模型可能有有意义的影响，值得在上面花费计算资源。
3. 模型合并（model merging）已被确立为创建强大模型的关键工具。合并可以通过多种方式进行，从在相同数据上合并不同 checkpoint，到合并特定领域的专门模型。通常，合并被视为最终配方中强大而简单的工具，但关于如何在配方中为后续合并准备模型，尚未建立清晰的最佳实践 [@yadav2024matters]。

## 4. 识别糟糕的训练任务

在训练模型时，有一个重要的简单直觉需要建立，即了解不同类型的模型问题。
你希望将大部分时间花在当前数据、算法或配方还不够好的问题上。
另一方面，在设置新配方时，也有很多时候某些方法根本就是坏掉的。

理解这一点的最佳方式是在一个基本静态的评估套件上评估许多模型。这样你就能对哪些测试难以通过 post-training 干预来改动形成直觉（通常是知识密集型评测，如 MMLU）。
当 post-training 设置中出现非常、*非常*严重的问题时，这些基本稳定的评测往往会在一次训练任务中下降 10-20 分。
这是开发工具时最有用的信号之一！
