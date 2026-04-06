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
