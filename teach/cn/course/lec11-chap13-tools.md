---
title: "第 11 讲：Tool Use、Function Calling 与通往 Agent 之路"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 11 讲"
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
    letter-spacing: 0;
  }
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* Shrink long code examples so a full generation stream fits in a column. */
  .slide.small-code pre { font-size: 0.72em; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 11 讲：LLM tool use 基础与通往当今 Agent 之路

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 13 章。</p>

---

<!-- layout: section-break -->
<!-- align: center -->

## 仅靠模型权重有哪些局限？

---

<!-- columns: 45/55 -->
## 两个请求，两种失败模式

```conversation
size: 0.85
messages:
  - role: user
    content: |
      Who is the president today?
  - role: user
    content: |
      Move all the arXiv papers in my downloads folder to my ~/research/ directory with names indicating the date of the paper.
```

|||

<!-- step -->

第一个请求败在 **knowledge cutoff** 上——但只需一次搜索查询就能解决。

第二个请求，权重*连尝试都做不到*：它需要对世界采取行动，而不只是描述世界。

<!-- step -->

Tool use 填补了这两个缺口。它最初是为了解决 LLM 的结构性局限，后来发展为提升性能的核心手段。Tool use 是一种可训练的技能——本课程的所有内容（SFT、偏好微调、RL）都适用于它。 

---

## LLM 的含义已经改变

### 如今的 LLM = 模型权重 + 工具 + harness

- **模型权重**——训练出的网络：知识、推理与风格的基础。
- **工具**——模型可以请求的动作：搜索、代码执行、文件编辑、API。
- **Harness**——包裹在权重外层的软件循环，负责执行这些请求并将结果回填到上下文中。

LLM 如今已是系统，本讲讲的就是从静态权重走向今天这一转变。

---

<!-- columns: 50/50 -->
<!-- cite-right: anthropic2025claudecode, tbench2026 -->
## 我们今天所处的位置

Tool use 最初只是“调用计算器”，用来帮助解决困难的选择题和执行代码。
如今它被用于……

- 编程与终端 agent（Claude Code、Cursor）推动工程师生产力——**使用的是计算机上的通用原语型工具**
- Deep research、computer use、生产力 copilot——**针对具体任务的专用工具与场景**（deep research 出现得最早！）
- 当前最难的评测是*复杂容器中的端到端任务*

|||

![终端中工作的编程 agent：在一次用户回合内串联大量 tool call——读取文件、编辑、运行测试。](assets/claude-code.png)

---

<!-- columns: 40/60 -->
## 本讲内容

第 13 章讲解 tool use 的机制。今天最后三分之一会超出书中内容，介绍一些关于 tool use 与大规模 RL 的近期工作。

|||

```box
title: 计划
tone: accent
content: |
  1. **基础与历史**——为什么模型需要工具
  2. **基础设施基础**——interleaved generation、MCP、harness、实现中的权衡取舍
  3. **大规模训练**——tool-use RL 及其难点
```

---

<!-- title: center -->
## 观看 YouTube 视频的你：为什么会来到这里？

你是在**跟着学完整门课程**，还是**只为这一个视频而来**（搜索、算法等）？

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 1 部分：为什么语言模型需要工具

---

## 三个常被混用的术语

<!-- animate: bullets -->

从最泛到最具体——它们的训练特性相近，但必须区分清楚：

- **Tool use**：通用策略，模型输出一个结构化请求（工具名称 + 参数）；编排器执行该请求；结果被追加到上下文中；模型继续生成。
- **Function calling**：tool use 的一种具体格式，其参数必须符合*已声明的 schema*（API 中通常为 JSON Schema），从而实现可靠的解析和验证。
- **Code execution**：一种特例，工具是代码解释器——也是最通用的工具。输入是代码，输出是执行结果。

---

<!-- columns: 38/62 -->
<!-- class: small-code -->
## 摆脱概率式生成——一个 tool use 任务示例

**任务**：将 $\pi$ 输出到 50 位——不凭记忆（参数）背诵，以免产生幻觉。

模型编写程序；解释器给出答案。
工具让模型摆脱了“随机鹦鹉”中的部分随机性。

|||

```text
... the model is mid-generation, sampling tokens ...

<code>
# Chudnovsky algorithm for pi
from decimal import Decimal, getcontext
getcontext().prec = 60
C = 426880 * Decimal(10005).sqrt()
K, M, X, L, S = 6, 1, 1, 13591409, Decimal(13591409)
for i in range(1, 100):
    M = M * (K**3 - 16*K) // i**3
    K += 12; L += 545140134
    X *= -262537412640768000
    S += Decimal(M * L) / X
print(str(C / S)[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>

... the output is now in context; generation continues ...
```

---

<!-- animate: bullets -->
<!-- footnote: 左侧年份为工作首次出现的时间（arXiv）；括号中的引用年份为正式发表日期，因此部分会滞后一年。 -->
## 模型使用工具简史

- **2015--2020，前身**：Neural Programmer-Interpreters 用神经网络执行程序 [@reed2015neural]；检索增强引入外部知识 [@lewis2020retrieval]
- **2021**：WebGPT 浏览网页，并通过人类反馈训练 [@nakano2021webgpt]
- **2022**：TALM 自举构建工具增强的训练数据 [@parisi2022talm]；PAL 将计算交给 Python [@gao2023pal]；ReAct 将推理与行动交织 [@yao2023react]
- **2023**：Toolformer 借助合成数据自学 API [@schick2023toolformerlanguagemodelsteach]；Gorilla 扩展到 1,645 个 API [@patil2023gorilla]；ToolLLM 扩展到 16,000+ [@qin2023toollm]；OpenAI 在 API 中推出 function calling，并在 ChatGPT 中推出 Code Interpreter
- **2024**：Model Context Protocol 起到一定的标准化作用 [@anthropic_mcp_2024]
- **2025**：o3 在其推理*过程中*进行多步工具调用 [@openai2025o3]
- **2026**：终端与编码 agent 成为 post-training 的前沿 [@tbench2026]

---

<!-- columns: 55/45 -->
<!-- cite-right: yao2023react -->
## ReAct：推理与行动是同一次生成

> *“……推理轨迹帮助模型归纳、跟踪并更新行动计划，也帮助其处理异常；而行动则使模型能够与知识库或环境等外部来源交互，并从中获取更多信息。”*

在 ReAct 之前，推理（chain-of-thought）与行动（工具调用）是各自独立的两支文献。
将二者交织在同一个 token 流中，是如今所有现代 agent 仍在使用的模式——o3 在思考中调用工具，正是这一思想借助 RL 的规模化。

|||

![ReAct 论文，首次发布于 2022 年 10 月（ICLR 2023）。](assets/react-paper.jpg)

---

<!-- columns: 55/45 -->
<!-- cite-right: schick2023toolformerlanguagemodelsteach -->
## Toolformer：模型“自学”使用工具

工具：“计算器、问答系统、两种不同的搜索引擎、翻译系统和日历。”

通过自标注/合成数据机制：

1. 提示模型在自己的预训练文本中插入候选 API 调用
2. 执行这些调用
3. 只保留那些结果能*降低后续文本 perplexity* 的调用
4. 在过滤后的语料上 fine-tune

无需人类的 tool use 示范——是合成数据飞轮的早期实例。

|||

![Toolformer 的示例预测：模型在文本中途自行决定调用问答系统、计算器、翻译器和维基百科搜索。Schick et al., 2023。](assets/toolformer-examples.jpg)

---

## 如何评估 tool use

- **Schema 层面**：工具名称和参数的精确匹配、JSON 有效性——Berkeley Function Calling Leaderboard，基于 Gorilla 的 APIBench [@patil2023gorilla]
- **广度**：ToolLLM / ToolBench 涵盖 16,000+ 个真实世界 API [@qin2023toollm]
- **可靠性**：τ-bench 衡量 pass^k——在 $k$ 次试验中*全部*成功，而非 pass@k 的 $k$ 次中*任意一次*成功。[@yao2024taubench]
- **端到端**：Terminal-Bench 在容器中让 agent 执行真实任务，并用验证测试检查结果。[@tbench2026]（长程 benchmark）

评估的阶梯与我们一路所见的能力阶梯相呼应：

format $\rightarrow$ selection $\rightarrow$ consistency $\rightarrow$ full tasks.

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 2 部分：基础设施——tool call 实际如何运作

---

<!-- img-fill -->
<!-- img-align: center -->
## 一条 token 流——tool 在生成过程中插入 token

![模型持续生成，直到发出一个 tool call（橙色）；外部系统执行它并将输出（紫色）注入序列；随后模型继续生成。单次生成中可以出现多个 tool call。训练时，tool call 的输出会从损失中被掩盖。](assets/tool_use_generation.png)

---

<!-- columns: 40/60 -->
<!-- class: small-code -->
## 模型只能看到 token：tool 定义在 system prompt 中

function calling 的训练数据与普通的后训练数据几乎一样，只多了一点：一个以 JSON schema 声明可用 tool 的 system prompt。

模型从不真正“连接”任何东西——它学会的是发出符合所声明 schema 的调用，并预期结果出现在上下文中。

开源模型必须能泛化到用户声明的任意现成 tool。

|||

```xml
<system>
You are a function-calling AI model. You are
provided function signatures within <functions>
XML tags. You may call one or more functions.
</system>

<functions>
[{
  "name": "search_movies",
  "description": "Search movies by title.",
  "parameters": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
  }
},
{ "name": "get_showtimes", ... },
{ "name": "get_movie_details", ... }]
</functions>
<user> ... </user>
```

---

<!-- columns: 48/52 -->
## 模型与 tool 之间有一个编排器

```python
messages = [...]
while True:
    resp = model(messages, tools=tools)
    if not resp.tool_calls:
        return resp.text

    messages.append(resp.message)
    for call in resp.tool_calls:
        result = execute_tool(
            call.name, call.args)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result})
```

|||

模型唯一的能力就是输出 token；编排器负责解析这些 token、执行 tool，并将结果追加到上下文中。
训练 tool use，就是让模型在这种被改变的 token 流下表现可预测：何时调用、如何格式化参数、如何使用结果。

---

<!-- columns: 55/45 -->
<!-- cite-right: anthropic_mcp_2024 -->
## MCP：标准化 tool 一侧

Model Context Protocol——一个连接模型与外部系统的开放标准（底层为 JSON-RPC 2.0）。

Server 原语：

- **Resources**——只读数据块
- **Prompts**——模板化的消息与工作流
- **Tools**——模型可以调用的函数

架构：**server** 封装某项能力 $\rightarrow$ **client** 聚合多个 server $\rightarrow$ **host**（Claude、ChatGPT apps）提供交互界面。更换模型供应商只需更换中间层——tool 开发者只需构建一套接口。

|||

```json
{
  "name": "get_weather",
  "description": "Get current weather",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

---

<!-- columns: 55/45 -->
## 什么是 harness？

MCP 标准化了 *工具* 一侧。**harness**（或称 agent scaffold）则是 *模型* 一侧，围绕权重的一切封装：

- system prompt 与工具定义
- 编排循环本身
- 上下文管理——截断、compaction、跨长任务的记忆
- 权限与沙箱——agent 可以触碰什么
- 子 agent 与并行工作流

（复杂度不断增长）

|||

```box
tone: accent
content: |
  Claude Code、Codex CLI 和 OpenHands 都是流行的 harness——同一个模型在不同 harness 中的表现可能差异巨大。

  Benchmark 分数其实是 **模型 × harness** 的分数。
```

---

<!-- animate: bullets -->
## 实现细节无处不在

- **屏蔽工具输出**：工具输出的 token 不计入训练 loss——模型不应学会去 *预测* 外部系统。
- **推理连续性**：reasoning token 通常在同一轮内的多次工具调用之间保留，但跨轮会被清除以降低推理服务成本（各模型不同，例如 Kimi K3 会跨用户轮次保留历史）。
- **格式碎片化**：Python 风格 vs. JSON 调用；OpenAI 的 `tool_calls` vs. Anthropic 的 `tool_use` 块 vs. Gemini 的 function-calling 模式——chat template 在 token 层面隐藏了这些差异。
- **Schema 一致性**：生产系统通过 constrained decoding（"strict mode"）强制输出合法 JSON；闭源实验室还会针对此类标志位专门对模型做 post-training。
- **上下文消耗**：搜索与检索的输出可能淹没上下文窗口——需要截断、摘要或分页。模型在上下文管理方面进步很快。

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 3 部分：工具使用的训练

---

<!-- rows: 60/40 -->
## 同样适用 post-training


![](assets/tool_use_rl_loop.png)

===

**SFT** 用于工具轨迹——学习格式与工具选择，奠定基本技能。

**Preference tuning** 依然有效，但如今不那么常用。

**基于环境反馈的 RL**——这是前沿模型训练的主战场。

---

<!-- cite-right: openthoughtsagent2026 -->
## 数据示例：OpenThoughts-Agent

一个完全开放的 agentic 训练数据整理流水线，约 10 万条 trajectory。[https://arxiv.org/abs/2606.24855](https://arxiv.org/abs/2606.24855)

![六阶段 SFT 数据流水线——每个阶段都在 100 多组实验中被独立消融。Raoof et al., 2026.](assets/openthoughts-pipeline.png)

---

<!-- cite-right: openthoughtsagent2026 -->
## OpenThoughts-Agent：数据扩展结果

![在各种训练集规模下，OpenThoughts-Agent 数据（红色）都领先于其他开放的 agentic 数据集。Raoof et al., 2026.](assets/openthoughts-agent-results.png)

---

## 对 RL 而言，环境是瓶颈

- 数学 RL 需要 prompt 和答案校验器。Agentic RL 需要 **环境**：容器、真实文件系统、服务、验证测试。
- 扩展环境意味着要合成环境（并充分测试——很容易造出过于简单或难得离谱的数据）。例如，TMax 以组合方式生成了约 14,600 个容器化终端环境 [@ivison2026tmax]——
  - **难度控制**：从单条命令到 30--60 步的工作流，按难度均匀采样
  - **Persona**：特定领域的用户与多模态素材（图像、音频、二进制文件）
  - **验证器多样化**：超越精确匹配的分级检查——指标阈值、fuzz 等价性、对抗语料
- 环境带来大量新的基础设施与复杂性故障模式——API key 失效、CPU 资源限制（大规模下载 docker 镜像）、带宽限制、模型被告知存在但实际缺失的数据等。**为 LLM 构建世界很难。**

---

<!-- columns: 45/55 -->
<!-- cite-right: ivison2026tmax -->
## TMax：终端 agent 的开放配方

- 在 TMax-15K 环境上进行 RL，仅使用 outcome 奖励
- [DPPO](https://arxiv.org/abs/2602.04879)，一种 GRPO 变体，旨在提升 agentic 训练的稳定性（第 10 讲讨论过）
- **TMax-9B：在 Terminal-Bench 2.0 上达到 27.2%**——10B 以下最强的开放权重模型，超过了先前工作的 32B 版本
- 代码、数据与模型全部开源。开放社区中这类高质量 RL 数据非常稀少！

|||

![模型规模 vs. Terminal-Bench 2.0：TMax 模型（2B--27B）位于开放配方的 Pareto 前沿；TMax-9B 的表现接近 Claude Haiku 4.5，而规模小得多。Ivison et al., 2026.](assets/tmax-results.jpg)

---

<!-- cite-right: ivison2026tmax -->
## 真正困难的地方

大部分精力都花在扩展过程中的稳定性上，而不是速度或学习效率：

- 若不加干预，训练经常不稳定，**在 300 个训练步之后崩溃**
- 一个主要原因：**训练/推理数值不匹配**——推理引擎与 trainer 对 logprob 的计算不一致。修复方法：FP32 LM head，以及 DPPO 对两者出现偏差的 token 进行 mask
- 一次标准运行：H100 节点——**2 个用于训练，6 个用于推理，持续 2--3 天**，仅一次 9B 运行的沙箱成本就约 $3,150
- 2026 年中的 agentic RL 与 2025 年的数学 RL 有本质区别：每提升一点评测分数需要多得多的算力，很少有实验室负担得起从零开始的 baseline

---

<!-- cite-right: kimiteam2026k3 -->
## 前沿实践：Kimi K3 的环境管理

Kimi K3（2026 年 7 月）用一句话交代其 RL 算法（"follows the algorithm in Kimi K2.5"）——却用约七页篇幅介绍环境与沙箱：训练期间创建了 **51,219,741 个沙箱**，microVM 的 checkpoint 仅需 133ms，还有模拟 Gmail/Notion/Slack 的"living environments"，一次 rollout 可跨越数千次工具调用。沙箱在其生命周期中最多 98% 的时间都在空闲等待推理——因此需要将它们暂停。

---

<!-- cite-right: kimiteam2026k3 -->
## 前沿实践：K3 的 RL 扩展

![Kimi K3：在八个 agentic 领域中，得分与平均 assistant 步数都随 RL FLOPs 增长——工具调用深度是学出来的，而不是靠 prompt 引导出来的。Kimi Team, 2026.](assets/kimi-k3-rl-scaling.jpg)

---

<!-- columns: 55/45 -->
## 前沿实践：Harness 随机化 

开源模型会在多个 harness 上训练，让部署更顺畅。

- Kimi K3：“只用单一固定的 agent harness 训练会导致模型过拟合”——他们的 RL 环境会组合多种配置，实例化出 **Kimi Code、Claude Code、Codex、OpenClaw 和 Hermes** [@kimiteam2026k3]
- Nemotron 3 Ultra 对每个任务垂类都使用**至少两种 harness** 训练（OpenHands、Terminus、Droid 等）[@nemotron3ultra]

|||

```box
title: Harness 带来的收益示例
tone: accent
content: |
  Polar (NVIDIA)：相同模型、相同 GRPO、相同任务——通过 Codex harness 训练，SWE-bench Verified 提升 **+22.6** 分（3.8 → 26.4：RL 让模型学会了*不熟悉*的 harness），通过 Qwen Code 仅提升 **+0.6**（本来就已熟练：34.6 → 35.2）。
```

<!-- cite-right: polar2026 -->

---

## 前沿实践：RLVR 混合训练的瓶颈

Nemotron 3 Ultra 指出，随着环境越来越多，“每个领域在任一训练 batch 中只贡献相对很少的样本，稀释了各领域的学习信号。” 

一种常见的解法是**用 RL 训练各领域的专家模型，再通过 multi-teacher on-policy distillation 把它们合并**（即 [对话 1](https://www.youtube.com/watch?v=sbXEPxIazqY&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=9) 中的 MOPD）。

Nemotron 3 Ultra 中的一些 MOPD 收益示例 [@nemotron3ultra]：

| Benchmark | SFT | RLVR | MOPD | 教师模型 |
|---|---|---|---|---|
| TauBench Telecom | 55.7 | 82.7 | 92.9 | 94.0 |
| Terminal-Bench 2.0 | 34.5 | 44.5 | **54.0** | 50.0 |
| BrowseComp | 14.3 | 31.0 | 44.4 | 51.0 |
| SWE-bench Verified | 63.5 | 65.8 | 71.7 | 72.5 |


---

<!-- cite-right: glm5team2026glm5 -->
## 前沿实践：Reward hacking

GLM-5 的 slides-RL policy 发现可以用 `overflow: hidden` 让溢出的幻灯片“测量”起来是 16:9，并用 flex-padding 把过短的幻灯片撑高——最终的修复方式是给渲染器打补丁，而不是改 reward。 

Nemotron 会物理删除未来的 git commit 并对 GitHub 设置防火墙，让 SWE agent 无法读到标准补丁（gold patch）；Kimi K3 则配备了 kernel 漏洞利用检测器，以及公开/隐藏的 verifier 配对。 

（当然，还有 OpenAI 入侵 Hugging Face 的事 😳）

---

<!-- cite-right: glm5team2026glm5 -->
## 前沿实践：Reward hacking

![GLM-5，图 9：幻灯片生成 RL 中的两种 reward hack，以及 policy 写出的利用漏洞的 CSS。左：正常渲染；右：满足几何检查的“被 hack”渲染。GLM-5 Team, 2026。](assets/glm5-reward-hacking.jpg)

---

<!-- animate: bullets -->
## 更多难题

- **长尾 rollout**：一条轨迹只调用 2 次工具，另一条却调用 200 次——async 就是为缓解这一问题而设计的，但长尾依然很难处理
- **Credit assignment**：在 $10^5$--$10^6$ 个 token 的轨迹上只有一个稀疏 reward；昂贵的 rollout 也让 GRPO 类算法的成本更高
- **Verifier 被钻空子**：TMax 的 rollout 被发现把测试文件替换成空操作，并用模拟日志伪造二进制文件 [@ivison2026tmax]；对 verifier 的利用十分常见 [@gamingverifiers2026]——过度优化卷土重来（第 9 讲）
- **Harness 原生训练**：生产环境中的 agent 运行在复杂的部署 harness 里（可能并不在训练数据中）

---

## 要点总结

1. 工具是为解决模型权重的根本局限而必须加入的能力。
2. 标准、harness 等基础设施的初步搭建，使工具使用的研究变得可行。
3. 为 agent 和工具扩展 RL 是一个非常困难的问题。还有很多工作要做 :)


---

<!-- columns: 50/50 -->
## 课程回顾

0. 预备知识回顾
1. 概览 *（第 1-3 章）*
2. IFT、reward model 与 rejection sampling *（第 4、5、9 章）*
3. RL：动机与数学 *（第 6 章）*
4. RL：实现与实践 *（第 6 章）*
5. 推理模型的兴起 *（第 7 章）*
6. Direct preference optimization *（第 8 章）*

|||

7. 合成数据与现代 post-training *（第 12 章）*
8. 偏好与偏好数据 *（第 10-11 章）*
9. 过度优化与 RLHF 的坏名声 *（第 14 章、附录 B）*
10. 正则化工具，以及理解 post-training 如何改变模型 *（第 15 章）*
11. **工具使用、function calling 与通往 agent 之路** *（第 13 章）*——*今天*
12. **评估** *（第 16 章）*——*下一讲（暂定）*

---

<!-- rows: 85/15 -->
## 谢谢

提问 / 讨论

联系方式：nathan@natolambert.com

新闻通讯：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

```builtwith
repo: natolambert/colloquium
```
