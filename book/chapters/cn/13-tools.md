<!-- source-commit: 02c6d49 -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "合成数据与CAI"
prev-url: "12-synthetic-data"
page-title: Tool Use
search-title: "第13章：Tool Use"
next-chapter: "过度优化"
next-url: "14-over-optimization"
---

# Tool Use与Function Calling

语言模型使用工具是扩展其能力的一种自然方式，尤其适用于外部工具包含所需信息的高精度任务，或需要与复杂网络系统交互的agent。
Tool use是语言模型需要通过训练才能掌握的技能，RLHF以及本书中介绍的所有其他方法都可以对其进行精炼。
考虑如下用户问题：

> **用户**：今天的总统是谁？

没有工具的语言模型很难回答这个问题，因为预训练数据存在知识截止日期，但通过一次搜索查询即可轻松获取这一信息。

在深入探讨之前，有必要区分几个常被互换使用的相关术语：

- **Tool use**：模型输出一个结构化请求（工具名称和参数）；编排器执行该工具；结果被追加到上下文中；模型继续生成。
- **Function calling**：tool use的一种形式，其中参数必须符合一组已声明函数的schema（通常为JSON Schema），从而实现可靠的解析和验证。
- **代码执行**：tool use的一种特殊情况，其中"工具"是代码解释器（例如Python）；结果以工具输出的形式返回。

AI模型通过输出特殊token来触发某个端点，从而使用任何外部工具。
这些工具可以是高度专用的工具（如返回特定地点天气的函数），也可以是代码解释器或搜索引擎，它们是构建复杂行为的基础模块。
我们的第一个示例展示了语言模型需要更多最新信息来补充其权重固定于过去数据的局限性，但也存在代码执行等工具，让语言模型能够绕过其概率性、生成式的特点，返回精确的答案。
考虑将pi的近似值打印到小数点后50位的任务（无需从记忆中背诵，以避免幻觉风险）。
具备工具的语言模型可以执行以下操作：

```text
<code>
from decimal import Decimal, getcontext
getcontext().prec = 60

def compute_pi():
    # Chudnovsky algorithm for computing pi
    C = 426880 * Decimal(10005).sqrt()
    K, M, X, L, S = 0, 1, 1, 13591409, Decimal(13591409)
    for i in range(1, 100):
        M = M * (K**3 - 16*K) // ((i)**3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
    return C / S

print(str(compute_pi())[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>
```

本章概述了现代语言模型中tool use的起源、其基本原理与格式，以及当前主流模型在有效利用工具方面的权衡取舍。

"tool use"这一术语的确切起源尚不明确，但这一想法的起源远早于RLHF盛行的ChatGPT时代。
2015年前后的早期示例试图构建早于现代语言模型的系统，例如神经程序解释器（Neural Programmer-Interpreters，NPI）[@reed2015neural]，"一种学习表示和执行程序的循环组合神经网络"。
随着语言模型变得更加普及，许多子领域开始通过与外部能力的集成来提升性能。
为了获取权重之外的信息，许多研究者使用了检索增强生成（retrieval augmented generation）[@lewis2020retrieval]或网页浏览（web browsing）[@nakano2021webgpt]。
此后不久，又有人探索了将语言模型与程序[@gao2023pal]或工具[@parisi2022talm]相结合的方案。

随着该领域的成熟，这些模型除了底层语言建模的大幅改进之外，还获得了更复杂的能力。
例如，ToolFormer能够使用"计算器、问答系统、两种不同的搜索引擎、翻译系统和日历"[@schick2023toolformerlanguagemodelsteach]。
此后不久，Gorilla经过训练可以使用1645个API（来自PyTorch Hub、TensorFlow Hub v2和HuggingFace），其评估框架APIBench成为广受欢迎的Berkeley Function Calling Leaderboard的基础[@patil2023gorilla]。
自这些早期模型以来，调用动作的多样性已大幅增长。

Tool use模型现在与常规语言模型交互深度融合。
Model Context Protocol（MCP）作为一种通用格式出现，用于将语言模型连接到外部数据源（或工具）[@anthropic_mcp_2024]。
随着模型更强大、格式更完善，tool use语言模型被应用于众多场景，包括Microsoft Office或Google Workspace等主流应用中的生产力copilot、科学领域[@bran2023chemcrow]、医疗领域[@li2024mmedagent]、编程agent[@zhang2024codeagent]（如Claude Code或Cursor）、与数据库的集成，以及许多其他自主工作流。

评估tool use模型涉及多个维度：工具名称和参数正确性的精确匹配指标、schema有效性，以及在模拟环境中端到端的任务完成情况。
跨试验的可靠性同样重要——$\tau$-bench引入了pass^k指标（与pass@k不同），用于衡量agent是否能持续成功而非偶尔成功[@yao2024taubench]。
ToolLLM及其ToolBench dataset提供了一个大规模框架，用于在16,000多个真实世界API上训练和评估tool use[@qin2023toollm]，而Berkeley Function Calling Leaderboard（BFCL）仍然是比较模型function calling准确性的热门基准[@patil2023gorilla]。

## 在生成过程中交织Tool Calls

Function calling agent所呈现的数据与其他post-training阶段非常相似。
不同之处在于系统提示中的内容，该内容指示模型可使用哪些工具。
下面展示了一个格式化的数据点示例，其中系统提示和可用工具以JSON格式呈现：
```xml
<system>
You are a function-calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
</system>

<functions>
[
  {
    "name": "search_movies",
    "description": "Search for movies by title and return matching results with IDs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search string for the movie title."
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "get_movie_details",
    "description": "Fetch detailed information about a movie including cast, runtime, and synopsis.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        }
      },
      },
      "required": ["movie_id"]
    }
  },
  {
    "name": "get_showtimes",
    "description": "Get movie showtimes for a given location and date.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        },
        "zip_code": {
          "type": "string",
          "description": "ZIP code for theater location."
        },
        "date": {
          "type": "string",
          "description": "Date for showtimes in YYYY-MM-DD format."
        }
      },
      "required": ["movie_id", "zip_code"]
    }
  }
]
</functions>

<user>
...
</user>
```
在语言模型生成过程中，如果遵循上述示例，它将生成 `search_movies("Star Wars")` 这些 token 来搜索《星球大战》。
这通常被编码在特殊的格式化 token 内，随后插入序列的下一批 token 将包含 tool 的输出。
借助这一机制，模型能够完成比许多简单独立模型更具挑战性的任务。

一种流行的 tool use 形式是代码执行，允许模型对复杂的逻辑或数学问题给出精确答案。
例如，语言模型执行过程中的代码执行可以发生在推理模型的思考 token 阶段。
与 function calling 类似，首先有用于执行代码（由模型生成）的标签，然后是用于输出的独立标签。
```text
<|user|>
What is the 50th Fibonacci number? (Use the standard F_0=0, F_1=1 indexing.)</s>
<|assistant|>
<think>
Okay, I will compute the 50-th Fibonacci number with a simple loop, then return the result.

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib(50)
</code>

<output>
12586269025
</output>
</think>
<answer>
The 50-th Fibonacci number is 12 586 269 025.
</answer>
```

在底层发生的事情是：语言模型将 tool 的输入和输出与标准自回归生成的 token 交织在一起。
使这一切成为可能的编排循环大致如下：

```python
messages = [...]
while True:
    response = model(messages, tools=tools)
    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        result = execute_tool(call.name, call.args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

![Tool use 将模型生成与外部执行交织在一起：模型持续生成 token，直到发出一个 tool call（橙色），外部系统执行该 tool 并将输出（紫色）注入序列，随后模型继续生成。模型可以在单次生成中发出多个 tool call。在训练过程中，tool call 和输出 token 通常会从损失中被掩盖掉。](images/tool_use_generation.png){#fig:tool-use-generation}

针对 tool use 的训练，核心在于让模型在这种不同的 token 流中表现出可预测的行为——知道何时发出 tool call、如何正确格式化参数，以及如何将结果整合到其响应中。
开放模型必须经过训练，才能与用户可能直接接入的各种 tool 协同工作。

## 多步 Tool 推理

OpenAI 的 o3 模型代表了多步 tool use 与语言模型集成方式上的一次重大跨越。
这一行为与社区更早期的研究趋势密切相关。
例如，ReAct [@yao2023react] 展示了动作与推理如何以交错方式整合到一次模型生成中：

> 在本文中，我们探索了利用 LLM 以交错方式同时生成推理轨迹和特定任务动作的方法，从而实现两者之间更强的协同效应：推理轨迹帮助模型推导、跟踪和更新行动计划，并处理异常情况；而动作则使模型能够与知识库或环境等外部信息源进行交互并获取额外信息。

随着 tool use 能力的稳固以及推理模型的兴起，多轮 tool use 已成为一个令人振奋的研究领域 [@wang2025ragenunderstandingselfevolutionllm]。

## Model Context Protocol（MCP）

Model Context Protocol（MCP）是一项用于将语言模型连接到外部数据源和信息系统的开放标准 [@anthropic_mcp_2024]。
在数据层，MCP 使用 JSON-RPC 2.0，并为其原语提供发现和执行方法。
MCP 无需针对每个外部系统进行特定的 tool call 格式化，而是通过标准化协议使模型能够访问丰富的上下文信息。

MCP 是本章 tool use 内容之上的一个简单补充——它规定了应用程序如何以可预测的 JSON schema 将上下文（数据 + 动作）传递给语言模型。
模型与之交互的 MCP server 具有核心原语：资源（只读数据块）、提示词（模板化消息/工作流）和 tool（模型可以调用的函数）。
基于此，MCP 架构可以概括为：

- MCP server 封装特定的数据源或能力。
- MCP client（例如 Claude Desktop、IDE 插件）聚合一个或多个 server。
- Host（例如 Claude 或 ChatGPT 应用）提供用户/LLM 接口；切换模型供应商或后端 tool 仅意味着替换中间的 client。

MCP 使 tool use 模型的开发者能够利用相同的基础设施将其 server 或 client 接入不同的模型，同时模型也拥有一种可预测的格式来集成外部组件。
这两者共同为现实场景中的 tool use 模型构建了一个更加可预测的开发环境。

MCP server 通过标准化的 JSON schema 向 client 暴露 tool：
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

一个实现此 tool 的最简 Python MCP server：
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="get_weather",
        description="Get current weather",
        inputSchema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        weather = fetch_weather(arguments["location"])
        return [TextContent(type="text", text=weather)]
```

## 实现细节

在实现 tool use 模型时，有多项格式化与掩码处理方面的决策需要考量：

- **Python 与 JSON 格式**：本章中，我们提供了将 tool use 格式化为 JSON 数据结构和 Python 代码的两种示例。模型通常会选择其中一种结构，业界不同提供商采用不同的格式。
- **掩码 tool 输出**：训练 tool use 模型时，一个重要细节是将 tool 输出中的 token 从模型的训练损失中屏蔽掉。这确保了模型不会学习预测系统输出——因为这些内容并非模型在实际使用中直接生成的（类似于其他后训练阶段对 prompt 进行掩码的做法）。
- **tool 调用的多轮格式**：在实现 tool calling 模型时，通常的做法是为数据加载格式增加更多结构。后训练 dataset 的标准格式是用户与助手交替出现的消息列表（通常还包含一条系统消息）。tool use 的整体结构相同，但模型的回合被拆分为以每次 tool call 为分隔符的内容子段。以下是一个示例。

```python
messages = [
{
"content": "You are a function calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.",
"function_calls": null,
"functions": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways from the GamerPower API based on the specified type.\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways to retrieve (e.g., game, loot, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]",
"role": "system"
},
{
"content": "Where can I find live giveaways for beta access and games?",
"function_calls": null,
"functions": null,
"role": "user"
},
{
"content": null,
"function_calls": "live_giveaways_by_type(type='beta')\nlive_giveaways_by_type(type='game')",
"functions": null,
"role": "assistant"
}
]
```

- **Tokenization 与消息格式细节**：OpenAI messages 格式中的 tool call 通常通过 chat template（控制发送给模型的消息格式的代码）进行 tokenization，将结构化的 JSON 表示转换为原始 token 流。这一过程因模型架构而异——有些使用特殊 token 来标记 tool call，另一些则在 token 流内部保持结构化格式。[Chat template playground](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B) 提供了一个交互式环境，可用于探索不同模型如何将消息格式转换为 token 流。
- **推理 token 连续性**：随着推理模型的兴起（在给出答案之前会有独立的"推理"token 流），目前存在多种实现方案来处理推理 token 与 tool use 协同工作的问题。部分模型会在单个回合内的多次 tool calling 步骤之间保留推理 token，以维持跨多次 tool 调用的上下文。然而，这些 token 通常在回合之间被清除，以降低服务成本（但并非总是如此——这属于设计决策）。
- **各提供商的 API 格式**（截至 2025 年 7 月）：不同提供商采用概念相似但技术上有所区别的格式。OpenAI 使用带唯一 ID 的 `tool_calls` 数组，Anthropic 使用带 `<thinking>` 标签的详细 `input_schema` 规范，Gemini 则提供 function calling 模式（AUTO/AUTO/NONE）。通过 API 使用这些模型时，可用的 tool 以 JSON 格式定义，模型响应中的 tool 输出则存储在与标准"生成 token"不同的字段中。另一个示例是，开源推理代码库 vLLM 实现了大量解析逻辑，支持多种 tool calling 模式和针对不同模型的解析器，为底层实现细节提供了参考 [@kwon2023efficient]。
- **schema 一致性与受约束解码**：生产系统通常使用受约束解码或"严格模式"选项来强制输出有效的 JSON 和正确的参数类型，从而减少因格式错误输出导致的重试。部分闭源模型提供商会专门针对结构化 JSON 输出的可靠性进行额外的后训练；而对于开源模型，这通常作为 VLLM 等系统中的推理标志来处理。
- **tool 输出的上下文消耗**：tool 输出（尤其是搜索或检索类 tool 返回大量结果时）可能迅速占满模型的上下文窗口。系统必须决定如何对 tool 输出进行截断、摘要或分页，以在保持上下文可管理的同时保留模型继续运行所需的信息。

将这一切与后训练联系起来：tool use 的训练数据从哪里来，使用什么目标函数？
人工编写的 tool 调用轨迹成本高昂，因此大多数现代 tool use 语料库都是合成或自举生成的——例如 Toolformer 风格的自标注 [@schick2023toolformerlanguagemodelsteach]，或 ToolBench 中的大规模生成 [@qin2023toollm]。
在训练目标方面，对 tool 调用轨迹进行监督微调（SFT）可以教会模型基本的格式化和 tool 选择。
这一步骤能够引导模型建立相应能力，通常足以奠定该技能的基础。
对轨迹进行偏好优化（如 DPO）则可以改善模型在何时调用 tool、何时直接作答之间的决策。
