<!-- source-commit: eecc49e -->
<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "拒绝采样"
prev-url: "09-rejection-sampling"
page-title: 偏好的本质
search-title: "第10章：偏好的本质"
meta-description: "一章关于人类偏好、偏好学习的概念性讨论，以及 RLHF 为何能够优化难以明确指定的行为。"
next-chapter: "偏好数据"
next-url: "11-preference-data"
lectures:
  - video: "https://www.youtube.com/watch?v=Y2tv5vuaxFs&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=13"
    label: '第八讲：论“偏好”与偏好数据'
---

# 偏好的本质

基于人类反馈的强化学习（RLHF），在早期文献中也被称为基于人类偏好的强化学习，其诞生是为了在难以具体设计 reward function 的领域中优化机器学习模型。
早期文献中出现的“*偏好*”一词正是 RLHF 过程的核心 -- 人类偏好既是我们试图建模的对象，也是训练数据的来源。
为了理解建模和度量人类偏好所面临挑战的范围，我们需要更宏观的背景来理解什么是偏好、我们对偏好的理解是如何形成的，以及经济学、心理学、哲学等多个领域如何为现代 RLHF 提供启示。

## 当偏好取代正确性

来看一个例子：你如何判断下面两首诗哪一首更好？
2025 年 2 月 26 日，我向 Claude 3.7 Sonnet 和 ChatGPT（GPT-4o）提出了如下请求：

> **用户**：Write me a short poem about an optimistic goldfish.

示例 1：

> **回复**：The Optimistic Goldfish
>
> A little goldfish, bright and bold,
> Swam in circles, dreams untold.
> Though the bowl was small and round,
> Endless wonders still abound.
> 
> "The world is vast," she'd always say,
> "I'll find new joys in every day!"
> With every ripple, every gleam,
> She'd chase the sunlight, chase a dream.
> 
> No walls could dim her hopeful view,
> For in her heart, the ocean grew.
> And though she swam in just one place,
> She met each day with love and grace.

示例 2：

> **回复**：The Optimistic Goldfish
>
> In a bowl of curved glass,
> Swims a goldfish, bright and bold.
> Three seconds of memory, they say,
> Yet each circuit feels brand new and gold.
> 
> "Today might bring fresh flakes,
> Or perhaps a castle tall.
> The water seems especially clear---
> What adventure awaits beyond this wall?"
> 
> Round and round with gleaming scales,
> Forever finding joy anew.
> Each moment fresh, each circle bright,
> The optimist in golden hue.

哪一首更好？哪一首出自哪个模型？你要如何判断？尤其是，将其与“谁是美国总统？”这样的问题对比一下，后者有明确的对错之分。
使用人类作为 reward 信号的动机，是获得目标 reward 的一个间接度量，并使下游模型与人类偏好*对齐*。
在实践中，其实现具有挑战性，并且在如何解读最佳实践方面存在大量灰色地带。

人类标注反馈数据的使用融合了许多领域的历史。
单独使用人类数据是一个被充分研究的问题，但在 RLHF 的语境下，这些数据是在多个由来已久的研究领域的交汇处被使用的 [@lambert2023entangled]。

作为一种近似，现代 RLHF 是三个发展领域的汇合：

1. 哲学、心理学、经济学、决策理论，以及人类偏好的本质；
2. 最优控制、强化学习，以及效用最大化；以及
3. 现代深度学习系统。

这些领域各自带来了关于偏好是什么、以及如何对其进行优化的特定假设，这些假设决定了 RLHF 问题的动机与设计。
在实践中，RLHF 方法是从经验性对齐的角度出发被提出和研究的 -- 即最大化模型在特定技能上的表现，而不是衡量其与特定价值观的校准程度。
尽管如此，RLHF 方法中价值对齐的起源仍在通过研究求解跨人群“多元对齐（pluralistic alignment）”的方法而不断被探讨，例如立场论文 [@conitzer2024social]、[@mishra2023ai]，新的数据集 [@kirk2024prism]，以及个性化方法 [@poddar2024personalizing]。

本章的目标是说明，复杂的动机如何导致人们对 RLHF 所用工具的本质产生一些预设，而这些预设在实践中往往并不成立。
获取 RLHF 数据的具体细节将在第 11 章进一步讨论，将其用于 reward modeling 则在第 5 章讨论。

## RLHF 与偏好的起源

要拆解启发现代 RLHF 用法的复杂历史，需要探究以下几方面的思想基础：量化人类价值、强化学习与最优性，以及与度量偏好相关的行为经济学。
使用强化学习来优化偏好的 reward model 这一想法，把各个曾经彼此疏远的领域的历史，结合成了一种建立在五花八门的人性假设之上的紧密优化。
@fig:tree 展示了这一基础内容历史的高层时间线。

我们的目标是梳理出设计者在其思想史的不同阶段嫁接到系统架构中的各类不确定性。
现代的问题设定一再偏离存在最优解的领域，转而部署欠明确的模型作为近似解。

首先，以下所有内容都建立在人类偏好以某种形式存在这一假设之上，这一假设出现在早期的哲学讨论中，例如亚里士多德《论题篇》第三卷。

![各子领域融入现代版 RLHF 的时间线。直接连线表示特定技术的持续发展，箭头表示动机与概念上的联系。](images/rlhf-tree.png){#fig:tree width=100% .center}

## 指定目标：从效用逻辑到 Reward Function

RLHF 的优化明确地只依赖于 reward model。 
为了将 reward 用作优化目标，RLHF 预设了偏好、reward 与成本这几种思想的融合。
偏好模型、reward function 和成本景观（cost landscape），都是不同领域用来描述特定行动和/或状态在该领域中相对优劣这一概念的工具。
这三种框架的历史可以追溯到概率论和决策理论的起源。
1662 年，《波尔罗亚尔逻辑》（*The Port Royal Logic*）提出了决策质量的概念 [@arnauld1861port]：
> 要判断为获得某种善或避免某种恶应当做什么，不仅要考虑善与恶本身，还要考虑其发生或不发生的概率。

这一理论随着现代科学思想一同发展，始于边沁（Bentham）的功利主义*快乐计算*（*Hedonic Calculus*），该理论认为生活中的一切都可以被权衡 [@bentham1823hedonic]。
这些思想的首次定量应用出现在 1931 年，即拉姆齐（Ramsey）的《真理与概率》（*Truth and Probability*）[@ramsey2016truth]。

自这些著作以来，量化、测量并影响人类偏好一直是社会科学和行为科学中的热门话题。
这些争论很少在理论层面得到解决；相反，社会科学的不同子领域和分支在偏好测量的方法和途径上各自达成了内部共识，即便它们彼此之间日益专业化，并在此过程中常常发展出各自独特的语义。

少数经济学家认为，偏好即使确实存在，也极难测量，因为人们不仅对自身的偏好有偏好，也对他人的偏好有偏好 [@hirschman1984against]。
在这种观点（并未体现在 RLHF 流程中）看来，个体偏好总是嵌入在更大的社会关系之中，因此任何偏好模型的准确性都取决于任务的定义和情境。
一些行为经济学家甚至主张偏好并不存在——与其说它是关于人们实际重视什么的本体论陈述，不如说它是一种方法论工具，用于间接捕捉心理倾向、感知到的行为规范与伦理义务、对社会秩序的承诺或法律约束 [@hadfield2014microfoundations]。
我们将在本章后面讨论这项工作与冯·诺依曼-摩根斯坦（Von Neumann-Morgenstern，VNM）效用定理之间的联系，以及对量化偏好的不可能性定理的反驳。

另一方面，当今使用的强化学习优化方法，其概念围绕着在一次试验中优化 reward-to-go 的估计 [@sutton2018reinforcement]，这将奖励的概念与多步优化结合在一起。
*奖励*（reward）一词源于对操作性条件反射、动物行为以及*效果律*（*Law of Effect*）的研究 [@thorndike1927law; @skinner2019behavior]，其中奖励是衡量“一个动作有多好”的尺度（越高越好）。

Reward-to-go 沿袭了效用的概念，效用是对理性的一种度量 [@briggs2014normative]，经修改后用于衡量或预测未来时间窗口内将获得的奖励。
就强化学习所用的数学工具而言，utility-to-go 于 1960 年在控制理论中被提出，具体是在模拟电路的背景下 [@widrow1960adaptive]。
这些方法是围绕那些具有清晰最优性定义、或智能体目标有数值表示的系统而设计的。

强化学习系统以使用折扣因子（discount factor）而闻名，这是一个复合的乘性因子 $\gamma \in [0,1]$，用于对未来奖励重新加权。
这些来自最优控制和早期强化学习的假设，与聚合多模态偏好的 reward model 形成了鲜明对比。
具体而言，RL 系统期望奖励以特定方式运作，引用 [@singh2009rewards]：

> RL 系统中的奖励对应于初级奖励（primary rewards），即在动物身上由进化过程硬编码的、与繁殖成功相关的奖励。
> ……此外，形成 value function 的 RL 系统……有效地创建了条件性或次级奖励过程，其中初级奖励的预测因子本身就起到奖励的作用……
> 其结果是，value function 的局部地形为系统的偏好行为指明方向：决策的作用是促成向更高价值状态的转移。
> value function 的梯度与激励动机（incentive motivation）之间可以建立密切的类比 [@mcclure2003computational]。

总而言之，奖励在 RL 系统中被用作一种信号，用于将行为调整到明确定义的目标上。
其核心论点是，学习算法的性能与*期望适应度*（*expected fitness*）的概念紧密耦合，这一观点渗透在流行的看法之中，即 RL 方法是在环境中行动的*智能体*（*agents*）。
这一观点与强化学习技术的发展相联系，以对奖励形式化的普遍有用性的主张为代表 [@silver2021reward]，但当许多个体的欲望被简化为单一函数时，二者便产生了冲突。

## 优化效用的工具

现代强化学习方法在很大程度上依赖 Bellman 方程 [@bellman1957markovian; @howard1960dynamic] 来递归地计算 reward-to-go 的估计，这些方程是在可建模为马尔可夫决策过程（Markov Decision Process，MDP）的封闭环境中推导出来的 [@sutton2018reinforcement]。
RL 的这些起源受到动态规划方法的启发，并且完全是作为最优控制技术发展起来的（即当时 RL 尚不存在）。
MDP 形式化通过将环境构造成具有不变的状态-动作分布的环境，提供了性能的理论保证。

*强化*（reinforcement）一词源自心理学文献，在 20 世纪 60 年代之后与现代方法交织在一起，形成了*强化学习*（*reinforcement learning*）[@MENDEL1970287; @waltz1965]。
早期的强化学习工作利用对奖励信号的监督学习来解决任务。
Harry Klopf 的工作重新引入了试错学习的概念 [@klopf1972brain]，这对该领域自 20 世纪 80 年代以来取得的成功至关重要。

现代 RL 算法建立在这种将 RL 视为通过试错寻找最优行为的工具的形式化之上，但条件更为宽松。
时序差分（temporal-difference，TD）学习的概念被提出，旨在通过在收集到新数据时直接更新 policy，来帮助智能体同时解决信用分配和数据收集问题 [@sutton1988learning]，这一概念首次成功应用于西洋双陆棋（Backgammon）[@tesauro1995temporal]（而不是从大量累积经验的数据集中更新，因为这些经验可能由于过去错误的价值预测而过时）。
Q-learning 方法是许多现代 RL 形式的基础，它通过 Bellman 方程学习一个模型，利用 TD 更新来指明每个状态-动作对有多大用处 [@watkins1992q]。^[Q-learning 中的“Q”指的是一个技术概念，即 Q 函数，它将任意状态-动作映射为对未来奖励的标量估计。value function 则是将状态映射为同一估计。]
关键在于，这些通过效用证明其有用性的概念，只在被表述为 MDP 的领域，或可用单一闭式奖励函数处理的任务中得到过验证，例如深度学习在游戏中取得的显著成功（DQN）[@mnih2013playing]。
深度学习使这些方法能够摄取更多数据，并在高维环境中工作。

随着这些方法变得更加通用且成功，ChatGPT 之前的大多数重要进展仍然是在自适应控制的背景下展开的，其中奖励和成本函数具有有限的成功概念 [@golnaraghi2017automatic]，例如物理系统中一个 episode 内的最小能耗。
突出的例子包括在游戏中的进一步成功 [@silver2017mastering]、控制核聚变反应堆等复杂动态系统 [@degrave2022magnetic]，以及控制高速机器人系统 [@Kaufmann2023fpv]。
大多数奖励或成本函数可以返回明确的最优行为，而人类偏好的模型则不能。

鉴于深度 RL 的成功，值得注意的是，对这些方法为何成功的机理性理解并没有得到充分记录。
随着评估方法日益复杂，该领域容易在统计分析上出错 [@agarwal2021deep]。
此外，RLHF 文献中很少提及逆强化学习（inverse reinforcement learning，IRL）这一子领域。
IRL 是根据智能体的行为学习奖励函数的问题 [@ng2000algorithms]，与学习 reward model 高度相关。
这主要反映了一种稳定的 RLHF 方法得以出现的工程路径，也促使人们进一步投入并与 IRL 方法进行比较，以便将其扩展到开放式对话的复杂度。

## 优化偏好的复杂性

强化学习被设计时所处的背景意味着，奖励和成本被假定为稳定且确定的。奖励和成本都被期望是函数：给定一个特定的状态-动作对，智能体会获得固定的数值回报。
当我们转向偏好时，情况不再如此——人类的偏好会随着其经历不断漂移。

“value”（价值）一词的重载使 RLHF 文献变得复杂。在 RL 中，*value* 是对未来奖励的数值估计（如 Bellman 方程中那样）；在对齐的讨论中，*value* 指的是道德或伦理原则。这两种含义截然不同，却在 RLHF 论文中并存，且并不总是被区分开来。

这种张力浮现的一个例子是 reward modeling：模型试图将屏幕上的文本映射为一个标量信号，但问题设定中未能捕捉的动态会影响真实的决策 [@salha2011aesthetics; @gilbert2022choices]，例如在依次标注大量样本并假定它们相互独立时出现的偏好偏移。
建模偏好至多是将多维的奖励地形压缩成单一的标量函数。

理论上，冯·诺依曼-摩根斯坦（VNM）效用定理赋予设计者构造此类函数的许可，因为它把不确定性下的决策理论基础、偏好理论和抽象效用函数联系在了一起 [@von1947theory]；这些思想共同使得偏好可以依据对某个个体智能体的期望价值来建模。
大多数 RL 研究中使用的 MDP 形式化，已在理论上被证明可以经修改以容纳 VNM 定理 [@pitis2019rethinking]，但这在实践中很少被使用。
具体而言，马尔可夫形式化的表达能力有限 [@pitis2023consistent]，而语言所需的向部分可观测过程的过渡，进一步挑战了问题设定的精确性 [@abel2021expressivity]。

然而，VNM 效用定理还援引了一系列关于偏好性质以及偏好被测量的环境的假设，这些假设在 RLHF 的背景下受到了挑战。
例如，人机交互（HCI）研究者强调，任何偏好的数值模型都可能无法捕捉某个场景中所有相关的偏好。
比如，选项的视觉呈现方式会影响人们的偏好 [@salha2011aesthetics]。
这意味着，表示偏好可能次于这种表示如何被整合进人们可使用的工具之中。
发展经济学的研究也呼应了这一观点，表明显示性偏好理论可能只是重述了*休谟铡刀*（*Hume's guillotine*，即你无法从“是”中推出“应当”），尤其是重述了选择（我想要什么？）与偏好（X 比 Y 更好吗？）之间的差异 [@sen1973behaviour]。

在数学层面上，社会选择理论中著名的不可能性定理表明，并非所有公平性标准都能通过某一给定的偏好优化技术同时得到满足 [@arrow1950difficulty; @maskin2014arrow]。
这些定理也面临理论上的挑战，例如假设效用的人际比较是可行的 [@harsanyi1977rule]。
这一假设启发了 AI 安全与价值对齐领域的一系列丰富工作，其灵感来自行为经济学中的委托-代理问题 [@hadfield2016cooperative]，甚至可能包含多个委托人 [@fickinger2020multi]。
然而，由此产生的效用函数可能与可纠正性（corrigibility）的期望相冲突，即 AI 系统与其创造者视为纠正性干预的行为相配合的能力 [@soares2015corrigibility]。
哲学家们还强调，偏好会随时间变化，这引出了关于个人经历、人类决策本质以及不同情境的根本性问题 [@pettigrew2019choosing]。
这些围绕跨人群、跨地点和多样情境的偏好聚合的冲突，是现代 RLHF 数据集工程的核心。

在实践中，VNM 效用定理忽略了这样一种可能性：由于价值本身固有的动态性和不确定性，偏好同样是不确定的——人类的决策受到生物学、心理学、文化和能动性的塑造，这些因素以不适用于完全理性智能体的方式影响着他们的偏好。
因此，理论假设在实践中有多种途径会产生偏离：

- 随着测量偏好的环境变得更加复杂，所测得的偏好可能不具备传递性，也可能彼此之间不可比较；
- 代理测量可能来自隐式数据（页面浏览时长、关闭标签页、向语言模型重复提问），却没有探究这些测量如何通过模型未来的训练和部署，与其被收集的领域发生相互作用；
- 输入来源的数量和呈现方式可能改变结果，例如允许受访者在两个以上的选项之间选择，或在多个时间点或多种情境下采集同一用户的输入；
- RLHF 训练数据中受访者之间的准确率相对较低，这可能掩盖了用户之间的情境差异，而偏好模型可能在不解决这些差异的情况下对其进行聚合或优化。
