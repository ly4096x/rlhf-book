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
