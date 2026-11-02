---
title: "第 8 讲：偏好与偏好数据"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 8 讲"
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
     Target lists only — leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* A/B preference comparison: force both answer cards to fill their column
     evenly so they read as a matched pair, not two ragged chat logs. */
  .slide.poem-ab .colloquium-message { max-width: 100%; width: 100%; padding: 1em 1.1em; }
  .slide.poem-ab .colloquium-conversation { height: 100%; justify-content: center; }
  .slide.poem-ab .colloquium-message-role { font-size: 0.75em; }
  /* Full-bleed image: near-edge figure (98% width), vertically centered,
     with the slide title kept in its normal position. */
  .slide.full-bleed { padding: 60px 13px 24px; }
  .slide.full-bleed h2 { margin-left: 47px; }
  .slide.full-bleed .slide-content { min-height: 0; flex: 1; display: flex; align-items: center; justify-content: center; }
  .slide.full-bleed .slide-content img {
    width: 100%; height: 100%;
    max-width: none; max-height: none;
    object-fit: contain;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 8 讲：关于“偏好”与偏好数据

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 10 章和第 11 章。</p>

---

<!-- layout: section-break -->
<!-- align: center -->

## 对于我们并不真正知道如何打分的能力，该如何 hillclimb？

---

<!-- layout: section-break -->
<!-- align: center -->

## 写出一个好的示范更容易，还是在好坏样本之间做出判断更容易？

---

<!-- class: poem-ab -->
<!-- rows: 22/78 -->
## 哪首诗更好？哪个模型写的？

同一个 prompt 交给两个前沿模型（Claude 3.7 Sonnet 和搭载 GPT-4o 的 ChatGPT，2025 年 2 月）：*“写一首关于乐观金鱼的短诗。”* 哪首更好——又分别出自哪个模型？

===

<!-- row-columns: 50/50 -->

```conversation
size: 0.9
messages:
  - role: assistant
    model: "A"
    content: |
      **The Optimistic Goldfish**

      A little goldfish, bright and bold,
      Swam in circles, dreams untold.
      Though the bowl was small and round,
      Endless wonders still abound.

      "The world is vast," she'd always say,
      "I'll find new joys in every day!"
      No walls could dim her hopeful view,
      For in her heart, the ocean grew.
```

|||

```conversation
size: 0.9
messages:
  - role: assistant
    model: "B"
    content: |
      **The Optimistic Goldfish**

      In a bowl of curved glass,
      Swims a goldfish, bright and bold.
      Three seconds of memory, they say,
      Yet each circuit feels brand new and gold.

      "Today might bring fresh flakes,
      Or perhaps a castle tall.
      What adventure awaits beyond this wall?"
      The optimist in golden hue.
```

---

## 当偏好取代正确性

到目前为止，大部分 post-training（SFT、RLVR）都有“正确”的概念——一道数学题的答案、一个通过的测试、一个可验证的 reward。 
偏好则是在**没有正确答案**时我们才会求助的东西。

例如：哪首诗更好？哪种解释更清楚？哪种语气更友善？

*偏好之所以成为 RLHF 的核心，是因为对齐（alignment）社区希望针对“人类价值观”这类复杂目标进行优化。*

---

<!-- columns: 40/60 -->
## 本讲内容

我们追溯“偏好”从何而来，论证为何它是一种**不完美的问题表述**，然后深入探讨当今模型所使用的**偏好数据**（第 11 章）。

|||

```box
title: 学习计划
tone: accent
content: |
  1. 偏好的简要**历史**——以及它为何是一种**不完美**的表述（第 10 章）
  2. **偏好数据**——实践中的权衡（第 11 章）
  3. **开放问题**（部分与合成数据共通）
```

---

<!-- columns: 42/58 -->
<!-- cite-right: lambert2023entangled -->
## 本讲上半部分背后的论文

![](assets/history-risks-rlhf.png)

|||

*The History and Risks of Reinforcement Learning and Human Feedback*（2023）[@lambert2023entangled] 追溯了 RLHF 所借鉴的各个领域——并追问这种借鉴在哪里出了问题。

- **历史：** RLHF 是哲学、经济学、最优控制和 RL 的交汇点——每个领域对“偏好”是什么都有自己的理解。
- **风险：** 它悄悄地把**成本、reward 和偏好**当作可以互换的概念，而实际上它们并不相同。
- 观点：我们继承了 RL 的优化器和设定，却没有继承它在收敛性上的保证。

论文详细梳理了 RLHF / post-training 所依据的文献中的一系列假设与预设。

---

<!-- columns: 58/42 -->
## 补充：“目标不匹配”（objective mismatch）视角

思考 post-training 的一种常见方式：我们优化的是一个**代理**目标，而不是我们真正关心的那个目标。

- **Model-based RL：** 动力学模型是为预测精度而训练的，而不是为控制性能——因此“更好”的模型可能产生*更差*的 policy [@lambert2020objective]。
- **RLHF：** reward model 是为偏好分类精度而训练的，而不是为下游 policy 质量——同样的错配，只是上了一层 [@lambert2023alignment]。

Reward model 的准确率（RewardBench 之类）是代理的代理！要不断追问你*真正*在优化什么。

|||

![](assets/alignment-ceiling.png)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第一部分：优化或衡量偏好的简史

---

<!-- class: full-bleed -->
<!-- img-align: center -->
<!-- cite-right: lambert2023entangled -->
<!-- notes: The integration of subfields into modern RLHF. Solid links are continuous technical developments; arrows are motivations and conceptual borrowings. Philosophy, economics, control theory, and deep learning each arrive with different assumptions about what a "preference" even is. -->
## 众多领域汇聚成“RLHF”

![](assets/rlhf-tree.png)

---

<!-- columns: 62/38 -->
<!-- valign: top -->
## 效用：从逻辑到数字

选择可以被*打分*的想法由来已久。共同的主线是：人类的欲求可以被归约为单一的度量：

- **波尔罗亚尔逻辑**（Port Royal Logic，1662）：决策质量 = 结果乘以其发生的概率 [@arnauld1861port]
- **边沁的快乐演算**（19 世纪初）：把人的一生放在同一把复杂但通用的尺度上衡量 [@bentham1823hedonic]
- **Ramsey**，*Truth and Probability*（1931）：首次把偏好*与*信念一起量化，作为个体做出概率性决策的方式 [@ramsey2016truth]


> *“要判断为获得善或避免恶必须做什么，不仅要考虑善与恶本身，还必须考虑它发生或不发生的概率。”* -- 《波尔罗亚尔逻辑》，1662

|||

![*La Logique, ou l'Art de Penser*（波尔罗亚尔逻辑），1662](assets/port-royal-logic.jpg)

---

<!-- columns: 64/36 -->
## 冯·诺依曼–摩根斯特恩效用定理（1947）

**冯·诺依曼与摩根斯特恩**（1947）：如果你的偏好满足几条公理（完备性、传递性、连续性、独立性），那么它们可以用单一的**效用函数**表示，而理性选择 = 最大化**期望效用** [@von1947theory]。

RLHF 正是依靠这一结论来为拟合标量 reward 提供依据。 

<!-- step -->

在 RLHF 中，这些*前提*基本上一条都不成立：

- 偏好在标注期间和标注之后会**漂移**
- 它们**依赖于情境和表述方式**
- 在高复杂度下它们可能是**非传递的**
- 并且它们是**多维的**，却被压缩成一个数字

|||

![约翰·冯·诺依曼（洛斯阿拉莫斯工作证照片，公有领域）](assets/von-neumann.jpg)

---


## Bradley-Terry（1952）：从比较到分数

这个统计模型把*比较*变成*分数*——并成为 reward modeling 的支柱 [@BradleyTerry]（见第 5 章 / 第 2 讲）：

$$ P(y_w \succ y_l \mid x) = \sigma\!\big(r(x,y_w) - r(x,y_l)\big) = \frac{e^{\,r(x,y_w)}}{e^{\,r(x,y_w)} + e^{\,r(x,y_l)}} $$

输入成对的人类比较，输出一个标量 reward。 
这正是 RLHF 需要 **preference data** 的*原因*（也是不完美之处的来源）。

---

## 一个标量偏好，许多特征

reward model 把以下所有因素压缩成一个数字：

- 有用性、诚实性、无害性、语气、格式、长度、品味……
- 标注者的心理、文化，以及他们使用的界面
- 比较的*呈现方式*对他们产生的任何引导

然后我们针对这个数字进行强力优化，而这永远不可能做到完美！总会有权衡取舍。

---

<!-- columns: 64/36 -->
## 效用理论的失效之处

效用一经形式化，社会选择（研究如何聚合偏好的领域）和经济学几乎立刻就发现了它的局限：

- **Arrow 不可能定理**（1950）：没有任何投票规则能在满足若干基本公平性标准的同时，把个体偏好聚合为集体偏好 [@arrow1950difficulty]
- **Sen**，*Behaviour, Choice and Values*（1973）：选择 ≠ 偏好；显示性偏好理论过于单薄 [@sen1973behaviour]
- **Hirschman**，*Against Parsimony*（1984）：人们对自己的偏好还有*偏好*——因此偏好可能无法度量 [@hirschman1984against]

|||

![Kenneth Arrow（CC BY 3.0，来自 Wikimedia Commons）](assets/kenneth-arrow.jpg)

---

<!-- columns: 58/42 -->
<!-- cite-right: conitzer2024social -->
## 题外话：面向 AI 对齐的社会选择

如果聚合偏好是难点，那么研究它的领域就是**社会选择理论**——也是看待对齐问题的天然视角。

*Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback*。
- 我参与撰写的一篇 2024 年立场论文主张，社会选择应当指导我们如何聚合多样化的人类反馈 [@conitzer2024social]。 
- 谁的偏好算数？如何整合标注者之间的分歧？……
- （这是我过去花更多时间钻研的兔子洞之一。）

|||

![](assets/social-choice-alignment.png)

---

<!-- columns: 64/36 -->
## 偏好并非随时间和情境而稳定的对象

来自心理学和行为经济学：

- 偏好会**漂移**——随时间、情绪和经历而改变 [@pettigrew2019choosing]
- 选择受情境和框架影响，而不只是由内在排序决定 [@gilbert2022choices]

这对"收集一次标签，之后反复用它训练"的做法是个问题。

|||

![前景理论的价值函数：*同样*的结果，因参照点不同而价值不同——价值取决于框架，而非固定的排序（CC BY-SA 3.0，来自 Wikimedia Commons）。](assets/prospect-value-function.png)

---

<!-- columns: 62/38 -->
## 另一个根源：最优控制与 RL

与此同时，一套用于*优化* reward 的方法逐渐成熟：

- **Bellman**（1957）：MDP 与动态规划 [@bellman1957markovian] → "最优"控制 / 控制理论的起源
- **Sutton**（1988）：用于 credit assignment 的 temporal-difference learning [@sutton1988learning]（而非从大量历史数据中学习）
- **Watkins**（1992）：Q-learning [@watkins1992q]（学习 value）
- **DQN**（2013）：大规模 deep RL [@mnih2013playing]；**AlphaGo / AlphaZero**（2017）：通过自我对弈达到精通 [@silver2017mastering]。"RL 行得通"

注意：这些保证都假设存在**单一的、闭式的 reward**！

|||

![李世石对阵 AlphaGo，第 4 局（2016）（CC BY-SA 4.0，来自 Wikimedia Commons）](assets/alphago-game4.jpg)

---

## 成本 ≠ reward ≠ 偏好

这三者在**本体论上是不同的**，而现代 post-training 的兴起却把它们当作可互换的 [@lambert2023entangled]。

- **成本**来自控制领域：物理的、可测量的、给定的，并且通常有明确的最优性 / 界限。
- **Reward** 来自心理学，也是 RL 的一种便利做法：一个待最大化的标量信号。
- **偏好**是属于人的、关系性的、不稳定的——甚至*并不*显然是一个标量。

把偏好简化为 reward 使优化形式变得可行，但这正是 RLHF 与 preference data 中许多无法解决的"偏差"的根本原因。

---

## 成本 ≠ reward ≠ 偏好

这三者在**本体论上是不同的**，而现代 post-training 的兴起却把它们当作可互换的 [@lambert2023entangled]。
Deep RL 的理论建立在 MDP 之上，其 reward 固定且为闭式（例如游戏、控制）。

- 学出来的 reward model 是一个*不断变化、带噪声的代理*，而不是真实的 reward。
- Inverse RL——*从行为中*学习 reward——在概念上很接近，却奇怪地缺席于 RLHF 实践 [@ng2000algorithms]。
- 因此，我们继承了 RL 的优化器，却没有继承它的保证。


---

## 关于"reward"是什么

> *Rewards in an RL system correspond to primary rewards, i.e., rewards that in animals have been hard-wired by the evolutionary process due to their relevance to reproductive success. … Further, RL systems that form value functions, … effectively create conditioned or secondary reward processes whereby predictors of primary rewards act as rewards themselves… The result is that the local landscape of a value function gives direction to the system’s preferred behavior: decisions are made to cause transitions to higher-valued states. A close parallel can be drawn between the gradient of a value function and incentive motivation.* 

-- Singh et al., 2009 [@singh2009rewards]

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 2 部分：Preference data——实践中的权衡

---

## 起点——为什么要收集 preference data

**判断**远比**生成**容易——人类（和模型）早在能写出更好答案之前，就能可靠地分辨两个答案哪个更好。在很多情况下，模型根本无法生成好的答案，却能捕捉到判断优劣的线索。

但要把这类数据收集（和处理）好，是现代 post-training 流程中最**不透明**的部分。

很少有开源模型会连同收集方法一起公开完整的人类 preference data——最好的例子是 NVIDIA 的 **HelpSteer** 数据集（Nemotron 开源模型背后的数据）[@wang2024helpsteer2; @wang2024helpsteer2p; @wang2025helpsteer3]。

---

## 数据收集界面示例


---

<!-- img-align: center -->
<!-- cite-right: bai2022training -->
<!-- img-fill -->
## 界面 1：研究型数据收集（Anthropic 早期 Claude 模型）

![Anthropic 研究中早期的 preference 收集界面：先查看完整对话，再打分。Bai et al., 2022 (CC-BY)。](assets/anthropic-interface.png)

---

<!-- img-align: center -->
<!-- img-fill -->
## 界面 2：线上 A/B 测试（ChatGPT 用户数据）

![来自不同 ChatGPT beta 模型的两个 completion 并排呈现。两个回答非常接近——这提醒我们 preference data 噪声很大，很难做到完全精确。](assets/chatgpt-ab-test.jpeg)

---

<!-- img-align: center -->
<!-- cite-right: chiang2024chatbot -->
<!-- img-fill -->
## 界面 3：允许平局的成对比较（Chatbot Arena 公开评测）

![早期版本的 Chatbot Arena 界面：带有平局选项的成对比较。](assets/chatbotarena.png)

---

<!-- img-align: center -->
<!-- img-fill -->
## 界面 4：单个比特（Ai2 演示；也见于许多其他产品）

![Ai2 研究演示中的点赞/点踩——最简化的方向性信号。](assets/up-down-vote.png)

---

<!-- img-align: center -->
<!-- img-fill -->
## 界面 5：多选一（图像模型的默认方式）

![在多张生成图像中做选择——文本之外的 preference data。每种界面都会塑造它所捕获的偏好。](assets/midj.jpeg)

---

<!-- rows: 42/58 -->
## 排序 vs. 评分

**评分：**对单个 completion 孤立地打分（如 1-5 分）。适合作为元数据。

**排序：**相对比较，通常采用 Likert 量表——早期 Claude 使用 8 点量表 [@bai2022training]；UltraFeedback 将高分与低分的 completion 配成对 [@cui2023ultrafeedback]。

实践中几乎所有人都在成对排序上训练，并将其二值化为 chosen/rejected 以用于 Bradley-Terry 损失（第 2 讲 / 第 5 章），同时把评分作为辅助数据保留。

===

<!-- row-columns: 50/50 -->

```conversation
size: 0.85
messages:
  - role: user
    content: |
      **Rating:** "Rate this completion 1-5."

      *"Paris is the capital of France, known for the Eiffel Tower."*
  - role: assistant
    model: "Annotator"
    content: |
      **4 / 5**
```

|||

```conversation
size: 0.85
messages:
  - role: user
    content: |
      **Ranking:** "Which response is better?"

      **A:** *"Paris."*  ·  **B:** *"Paris -- the capital since 987, home of the Eiffel Tower."*
  - role: assistant
    model: "Annotator"
    content: |
      **B > A**
```

---

## 到底什么是“Likert 量表”？

**Likert 量表**把偏好记录为一种*有序、分级*的判断——不仅记录哪个答案胜出，还记录**胜出多少**，量表对称，可带一个可选的中立中点。（得名于心理学家 Rensis Likert，1932 年——就是那种“非常同意 ... 非常不同意”的问卷工具，被借用来表达成对偏好。）

**带平局的 5 点量表**（LMArena 风格）：标注者可以选择“差不多”。

| 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|
| **A 好很多** | A 更好 | *平局* | B 更好 | **B 好很多** |

**不带平局的 8 点量表**（早期 Claude [@bai2022training]）：迫使标注者选出方向。

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A>>>B** | | | A>B | B>A | | | **B>>>A** |

采用几点量表、是否允许平局，都是会改变所收集数据的设计选择。

---

<!-- class: poem-ab -->
<!-- rows: 62/38 -->
## 结构化（synthetic）偏好数据

在有结构的领域，可以自动构造偏好对：

- **数学**：正确解 ≻ 错误解。
- **指令遵循（IFEval 风格）**：对同一 prompt 采样两次——一次带约束、一次不带——并偏好遵守约束的那个。约束由代码检查，因此标签是免费的。

在这些狭窄领域中，结构化偏好对可以胜过按质量判断的偏好 [@lambert2024t]。 
这是 *synthetic* 偏好数据的一种简单形式——第 12 章 / 第 7 讲。

示例 prompt：*"Describe the ocean. **Respond in all lowercase.**"*

===

<!-- row-columns: 50/50 -->

```conversation
size: 0.8
messages:
  - role: assistant
    model: "Chosen (sampled with the constraint) ✓"
    content: |
      the ocean covers most of our planet, a restless sheet of salt water...
```

|||

```conversation
size: 0.8
messages:
  - role: assistant
    model: "Rejected (sampled without the constraint) ✗"
    content: |
      The ocean covers most of our planet. It is a restless sheet of salt water...
```

---

## 超越成对偏好

成对比较只是一种惯例，而非定律。其他形式：

- **方向性 / 单比特**标签（点赞/点踩），用 Kahneman-Tversky Optimization（KTO）[@ethayarajh2024kto] 训练。
- **Token 级 / 细粒度**反馈 [@wu2024fine]：把特定 token 或 token 片段标注为好或坏。
- **自然语言**反馈——用书面评语代替标签 [@chen2024learning]。

信号更丰富，但收集更困难（实践中也不常被大规模使用）。

---

<!-- columns: 56/44 -->
## 数据来源与合同

![典型的多批次人类数据合同（约 50 万美元）：早期爬坡阶段目标和方法逐步收窄，前几批数据大部分被丢弃。更大的合同差异很大。](assets/pref-data-timeline.png)

|||

获取渠道靠关系驱动：供应商供给有限，偏好大预算和知名品牌。数百万美元被花掉且部分浪费；很少有团队有精力充分利用人类数据。合同通常限制你公开发布这类数据。

如今这个行业要复杂得多，涉及 environment 等。但即便只是简单的人类偏好数据，你也需要一套稳健、现成的 post-training recipe 来接入。

---

<!-- columns: 56/44 -->
<!-- cite-right: ouyang2022training -->
## 指导数据收集：标注员指令

合同签订后，买方与供应商会为每项任务商定**详细的指令**——通常不会在实验室之外公开。

- 最早的*公开*示例之一：OpenAI 发布了完整的 **InstructGPT** 标注员指令（2022）[@ouyang2022training]。该文件后来被删除，但我找回了它。
- 数页指南，说明如何对**有帮助、真实、无害**进行排序——这才是真正塑造标签的规范。
- [**下载 PDF**](https://rlhfbook.com/assets/instructgpt-instructions.pdf)（镜像于 rlhfbook.com）。

|||

![InstructGPT "Final labeling instructions"（OpenAI，2022）——已公开发布。](assets/instructgpt-instructions.jpg)


---


## 我们买过的一个数据集：No Robots

在 Hugging Face 的 **H4 团队**，我们像各大实验室一样委托采集了人类数据。

- **No Robots** [@no_robots] ——1 万条专家人工撰写的 demonstration，分布与 InstructGPT 一致，向供应商付费采购后**公开**发布（对委托采集的数据来说很罕见）：[hf.co/datasets/HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- 出自 **Zephyr** 模型 [@tunstall2023zephyr] 和 **Open LLM Leaderboard** [@open_llm_leaderboard] 的时代与团队。

---

<!-- columns: 50/50 -->
<!-- cite-right: arena2026 -->
## 偏好 *eval* 如今已是独立的生意

2026 年近期，**Arena**（前身为 LMArena 排行榜，更早为 Chatbot Arena）在推出企业 A/B 测试服务后的**约 8 个月内**，**年化收入运行率达到约 1 亿美元** [@arena2026]。

|||

[![](assets/arena-100m-article.png)](https://arena.ai/blog/arena-100m-revenue)

---

<!-- layout: section-break -->
<!-- align: center -->

## 第 3 部分：偏好数据中的开放问题

---

<!-- animate: bullets -->
## 偏差：数据中需要警惕什么

细微而系统性的偏差会直接从数据流入模型：

- **前缀偏差**——开头部分不成比例地左右标签 [@kumar2025detecting]
- **Sycophancy**——迎合用户胜过追求正确 [@sharma2023towards]
- **冗长**——越长评分越高 [@singhal2023long]
- **格式**——列表和加粗看起来"更好" [@zhang2024lists]
- **奉承 / 废话**——华丽的修饰语会抬高分数 [@bharadwaj2025flatteryflufffogdiagnosing]

<!-- step -->
检测并控制这些偏差是收集高质量偏好数据的核心——它们也正是 over-optimization 会放大为模型行为的东西（第 9 讲）。

---

<!-- animate: bullets -->
## 更高层次的复杂性 

- **收集情境** -- 职场标注的标签能否迁移到终端用户？有偿 vs. 志愿者？标注者是遵循指令，还是遵循自己的价值观？
- **反馈类型** -- 二元对比真的能捕捉我们所指的偏好吗？什么样的结构才贴近人们真实的比较方式？
- **人群与人口统计特征** -- 谁在标注？分歧是**噪声还是信号**？
- **这些偏好真的在模型中体现出来了吗？**

---

## 未经审计的缺口：规范 → 数据 → 行为

RLHF 的*动机*（对齐人类偏好）已经偏离了它的*实践*（让模型更有效）。

由于工业界的 RLHF 是闭源的，我们无法检验训练出的模型是否真的反映了提供给标注者的规范。 
**Model Spec** [@openai2024modelspec] 记录了预期行为，但从数据到行为之间的联系在很大程度上仍未经审计。

其中许多问题在合成数据中已经浮现（第 12 章 / 第 7 讲）：人类/AI feedback 的平衡，以及 on-policy 偏好数据。

---

## 偏好的本质是 RLHF 长期存在的难题

这是该领域中最悬而未决、也最“人性”的部分之一。请广泛阅读，并回到原始文献。
这是一个很棒的学术问题！

---

## 课程回顾

0. 先修知识复习
1. 概览 *（第 1–3 章）*
2. IFT、reward model 与 rejection sampling *（第 4、5、9 章）*
3. RL：动机与数学 *（第 6 章）*
4. RL：实现与实践 *（第 6 章）*
5. 推理模型的兴起 *（第 7 章）*
6. Direct preference optimization *（第 8 章）*
7. 合成数据与现代 post-training *（第 12 章）*
8. **偏好与偏好数据** *（第 10–11 章）* -- *今天*
9. **过度优化与 RLHF 的坏名声** *（第 14 章，附录 B）* -- *下一讲*
10. 正则化工具与理解 post-training 如何改变模型 *（第 15 章）*

---

<!-- rows: 85/15 -->
## 谢谢

提问 / 讨论

联系方式：nathan@natolambert.com

通讯：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

```builtwith
repo: natolambert/colloquium
```
