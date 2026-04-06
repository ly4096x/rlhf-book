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
