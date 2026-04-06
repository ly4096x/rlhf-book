事实上，这些*训练集*包含非常高质量的数据，因此模型可以从在其上进行训练中获益。
如果这些公司*并未*将相应的 evaluation 作为核心指标来追踪，那么在 evaluation 集上进行训练可能是一个实际可行的决策，因为高质量数据是模型开发的主要限制因素。

领先的 AI 实验室通过专注于少数几个关键 evaluation 来进行逐步攀升，并在最后报告核心公开集上的分数。
关键在于，他们用于追踪进展的部分 evaluation（例如 GPT-4 报告 [@achiam2023gpt] 中用于扩展规模的交叉熵损失预测数据集）通常并不公开。

post-training 的 evaluation 与人工评估高度相互依赖。
对于生成式语言模型的人工评估会产生 Elo 排名（在 Anthropic 早期论文如 Constitutional AI 中很流行），而对 reward model 的人工评估则反映一致性程度。
这些也可以通过在 A/B 测试窗口中向用户提供两个不同模型来获得（如[偏好数据章节](https://rlhfbook.com/c/11-preference-data)中所讨论的）。

他们选择关注的有限 evaluation 集合在 evaluation 与训练之间形成了紧密联系。
曾经有一段时间，MMLU 是重点关注的 evaluation 之一。
GPQA 在推理模型兴起期间极为流行，这得益于社区对科学能力的日益关注。
各实验室会修改 evaluation 以使其更适合自身需求，例如 OpenAI 发布了 SWE-Bench-Verified [@openai2024swebench]。
此外，每个前沿实验室还构建或购置了许多公众无法获取的内部 evaluation。

在内部改进 evaluation 对下游训练的关键影响在于**提升比较训练运行时的统计功效**。
通过更换 evaluation，这些实验室降低了其优先信号上的噪声，从而能够做出更明智的训练决策。

这一情况因现代语言模型训练栈中 post-training 的复杂性而进一步加剧。
如今对语言模型进行 evaluation 涉及相当数量的 token 生成（而非仅仅查看答案的对数概率），因此需要一定的计算开销。
普遍认为，前沿实验室会使用一些小技巧来提升许多任务上的性能——最常见的解释是针对特定 evaluation 使用一次性的 prompting。

## Contamination

当前语言模型实践中（即不仅限于 RLHF 和 post-training）的一个重大问题，是有意或无意地将 evaluation 数据集中的数据用于训练。
这被称为*数据集污染（dataset contamination）*，相应地，避免污染的实践称为*去污染（decontamination）*。
为了对数据集进行去污染，需要对训练集和测试集进行搜索，通过词/子词 token 上的 n-gram 重叠，或固定长度的字符子串匹配（例如 50 个字符）来查找匹配项 [@singh2024evaluation]。
数据被污染的方式有很多，但最常见的是在多个阶段从网络抓取训练数据时引入的。
benchmark 通常发布在被爬取的公开网页上，或者用户将问题输入给模型，这些内容随后可能出现在未来模型的候选训练数据中。

例如，在为 Tülu 3 进行 evaluation 套件去污染时，作者发现流行的开放数据集被 RLHF 的常用 evaluation 所污染 [@lambert2024t]。
这些重叠包括：UltraFeedback 对 TruthfulQA 的污染、Evol-CodeAlpaca 对 HumanEval 的污染、NuminaMath 对 MATH 的污染，以及 WildChat 对安全 evaluation 的污染。
这些污染是通过训练 prompt 与 evaluation 集中精确 prompt 之间的 8-gram 重叠发现的。

在其他情况下，模型被发现曾在与 benchmark 非常接近的数据上进行训练，例如保持数学题的文字不变而只修改数字，这可能导致 post-training 阶段出现异常行为，例如当模型在随机奖励上使用 RL 训练时 benchmark 却有所提升——这是一种人为设定，只有当模型存在某些类型的数据污染时，性能才应当提升。
这种 base model 污染（无法确切证明模型行为为何如此）已成为许多早期基于 Qwen 2.5 和 Qwen 3 base model 的 RLVR 工作中一个重要的干扰变量 [@shao2025spurious] [@wu2025reasoning]。

为了理解不披露或不发布训练数据的模型的污染情况，研究者会创建新版本的 benchmark，其中包含对原题稍加改动的问题（例如针对 MATH 的改动 [@huang2025math]），以此观察哪些模型是针对原始格式或题目进行训练的。
这些扰动 benchmark 上的高方差并不能证实污染，这一点难以证明，但可能表明某些模型是针对特定格式训练的，这种训练未必能转化为真实世界的性能。


## Tooling

目前有许多开源 evaluation 工具可供选择。
其中包括：

- 来自英国安全研究所的 Inspect AI [@inspectAI2024]，
- HuggingFace 的 LightEval [@fourrier2023lighteval]，为 Open LLM Leaderboard [@open-llm-leaderboard-v2] 提供支持，
- Eleuther AI 基于其 GPT-Neo-X 模型基础设施构建的 evaluation harness [@gao2023evalharness]（包含良好的 GPT-3 时代 evaluation 设置和配置）[@gpt-neox-20b]，
- AI2 基于 OLMES 构建的库 [@gu2024olmes]，
- Stanford 基础模型研究中心的 HELM [@liang2023helm]，
