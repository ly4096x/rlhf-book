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
