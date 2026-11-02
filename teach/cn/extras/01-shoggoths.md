---
title: "番外 1：Shoggoth 是怎么回事？"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "番外 1"
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
  .slide--title-sidebar h1 .title-subtitle {
    display: block;
    margin-top: 0.6em;
    font-size: 0.45em;
    font-weight: 400;
    letter-spacing: 0;
    opacity: 0.75;
  }
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* colloquium 0.2.3: .slide-content is flex:1, which absorbs the section's
     valign-center. Let the wrapper shrink to content so the heading and body
     center together via the theme's justify-content rule. */
  .slide.valign-center .slide-content:not(.colloquium-grid):not(.colloquium-rows) {
    flex: 0 0 auto;
  }
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

# Shoggoth 是怎么回事？ <span class="title-subtitle">课程番外视频 1：post-training 必备背景知识。</span>

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。 </p>

---

<!-- rows: 50/50 -->

## 这到底是什么鬼？

<!-- row-columns: 50/50 -->
![](assets/course-thumb-lec1.jpg)

|||

![](assets/course-thumb-lec2.jpg)

===

<!-- row-columns: 50/50 -->
![](assets/course-thumb-lec9.jpg)

|||

![](assets/course-thumb-lec13.jpg)

---

<!-- columns: 60/40 -->
## 1931 年：来自南极的怪物

Shoggoth 出自 H.P. Lovecraft 的《疯狂山脉》（*At the Mountains of Madness*，1931 年写成，1936 年发表于 *Astounding Stories*）[@lovecraft1936mountains]：

> “那是一个可怕得无法形容的东西，比任何地铁列车都要庞大——一团无定形的原生质气泡聚合体，微微自发荧光，无数临时的眼睛如同一颗颗泛着绿光的脓疱，在那堵充满整条隧道的前壁上不断生成又消散。它朝我们碾压而来，踩碎惊慌失措的企鹅，滑行过光亮的地面——它和它的同类曾经那样邪恶地把这片地面清扫得一尘不染。”

Shoggoth 最初是作为生物仆役被创造和培育出来的，但渐渐发展出了自己的心智，并反叛了它们的创造者。

|||

![*Shoggoth*，Nottsuo 绘（Wikimedia Commons）](assets/shoggoth-nottsuo.jpg)

---

## 2022 年 9 月：Simulators，梗图出现之前

Janus 的 [*Simulators*](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators)（LessWrong）认为，早期对齐文献中的 agent / oracle / genie 这几种框架，都不是理解 base LLM 的最佳方式。他们提出：

- base model 是其训练分布的**模拟器**（simulator）。
- 输出中出现的角色是**拟像**（simulacra）——由模型生成，但并不等同于模型本身。
- 输出中没有任何单一角色是“模型的身份”。

（摘要由 Claude 撰写，ChatGPT 表示没问题）

---

<!-- img-align: center -->
## 2022 年 12 月：梗图诞生

ChatGPT 发布一个月后，[@TetraspaceWest 画了两只 shoggoth](https://twitter.com/TetraspaceWest/status/1608966939929636864)：base **GPT-3** 是未经驯化的怪物，而 **GPT-3 + RLHF** 是同一只怪物，用一条触手举着一张小小的笑脸面具（[梗图历史](https://knowyourmeme.com/memes/shoggoth-with-smiley-face-artificial-intelligence)）。


![](assets/tetraspace-shoggoth-2022.png)

---

<!-- columns: 55/45 -->
## 2023 年 2 月：梗图变成训练示意图

[@anthrupad 的版本](https://x.com/anthrupad/status/1622349563922362368)（2023 年 2 月 5 日，右图）为每个训练阶段加上了标签：怪物是**无监督学习（pretraining）**，小人脸是 **supervised fine-tuning**，棍子上的笑脸是 **RLHF，“顶上的樱桃”**。

后来它不断被引用！

- [Helen Toner 的帖子](https://twitter.com/hlntnr/status/1632030583462285312)（2023 年 3 月）用它向政策圈的读者解释训练流程。
- [Chip Huyen 的 RLHF 讲解](https://huyenchip.com/2023/05/02/rlhf.html)（2023 年 5 月）把它放进了一篇广为阅读的工程参考文章。

|||

![传播最广的版本，作者 [\@anthrupad](https://x.com/anthrupad)](assets/shoggoth-meme-mask.jpeg)

---

<!-- columns: 60/40 -->
## 2023 年春：梗图走向大众

- **2023 年 2 月**：Elon Musk [发布](https://web.archive.org/web/20230223004549/https://twitter.com/elonmusk/status/1628491148124884992)了一个版本 -- 在他删除之前获得约 72k 点赞。
- **2023 年 2 月**：[Bing/Sydney](https://www.nytimes.com/2023/02/16/technology/bing-chatbot-microsoft-chatgpt.html) 让模型的“性格”成为全民话题。
- **2023 年 5 月**：Kevin Roose 在《纽约时报》上介绍这个梗图 -- ["Why an Octopus-like Creature Has Come to Symbolize the State of A.I."](https://www.nytimes.com/2023/05/30/technology/shoggoth-meme-ai.html) [@roose2023shoggoth]。

|||

[![](assets/nyt-shoggoth-article.png)](https://www.nytimes.com/2023/05/30/technology/shoggoth-meme-ai.html)

---

<!-- columns: 58/42 -->
## “我能和 shoggoth 说句话吗？”

传说的第二条支线：当助手的 persona 出现破绽时，人们觉得那是*一瞥其下的真身*。

例如 [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)（Rumbelow & Watkins，LessWrong，2023 年 2 月）：glitch token 会引发怪异、脱离 persona 的行为。

|||

![](assets/yudkowsky-shoggoth-tweet.png)

---

<!-- columns: 40/60 -->
## Shoggoth 依然活着

从很多方面看，这个梗比以往更火。它已不太关乎 RLHF，而是关乎我们对这些模型了解得如此之少。尤其是最新的 RL scaling 正在酝酿出很强的东西。

|||


![Interconnects NeurIPS 派对上的 Shoggoth 贴纸，2025 年 12 月。](assets/shoggoth-stickers-neurips.jpeg)

---

<!-- rows: 85/15 -->
## 谢谢

延伸阅读：[Reckoning with the Shoggoth of AI](https://www.interconnects.ai/p/the-shoggoth-rlhf)（2023 年 11 月）

联系方式：nathan@natolambert.com

通讯：[interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

```builtwith
repo: natolambert/colloquium
```

---

<!-- after: references -->

## 2023 年 1 月：Scott Alexander 把它变成一套本体论

[*Janus' Simulators*](https://www.astralcodexten.com/p/janus-simulators)（Astral Codex Ten，2023 年 1 月 26 日）的小节标题是“左边的无面具 Shoggoth”和“右边戴面具的 Shoggoth”。关于 RLHF 对 simulator 做了什么：

> “如果你惩罚它模拟坏角色，它就会开始模拟好角色。现在它只会模拟一个角色，即 HHH Assistant。”

<!-- step -->

由此形成的链条：base model → 众多角色的 simulator → post-training 选出**一个默认角色**，即有帮助、诚实、无害（HHH）的 Assistant。

这个版本的梗最接近本课程对 post-training 的描述。

---

<!-- after: references -->

## 2023 年 3 月：一直都是面具？

Robert_AIZI 在 LessWrong 上写道：[*Why do we assume there is a "real" shoggoth behind the LLM? Why not masks all the way down?*](https://www.lesswrong.com/posts/bYzkipnDqzMgBaLr8/why-do-we-assume-there-is-a-real-shoggoth-behind-the-llm-why)

> “我们真的有证据表明 LLM 里存在一个‘真实身份’吗？还是它只是一堆面具？”

评论区也成了传说的一部分：

- **RogerDearnaley**：“我一直以为 shoggoth *就是*一层层面具：所以它才有那么多眼睛。”
- **Gwern**：面具本身不会计算 -- 面具之下必须有东西在表演，而且 jailbreak 往往是把一个 persona 换成另一个。

梗图背后的技术问题：post-training 是掩盖了一个稳定的底层 agent，是在潜在 persona 中做选择，还是在一个没有单一自我的系统里构造出默认 persona？
