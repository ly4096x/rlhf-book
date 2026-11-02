---
title: "第 6 讲：Direct Preference Optimization"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "RLHF Book 中文版"
  center: "第 6 讲"
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
---

<!-- Source note: build with `make teach`, which copies assets/ into the output. A single-file `colloquium build -o ...` does NOT copy assets/, so the meme + displacement images 404 in that standalone build. -->
<!-- Reveal convention: each derivation step is a run of duplicate slides with identical title; one aligned line (with its reason) is appended per slide so equations build up in place and the layout never recenters. Every equation from chapter 8 is shown. -->

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# 第 6 讲：Direct Preference Optimization

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training 课程。第 8 章：Direct Alignment 算法。</p>

---

<!-- animate: bullets -->
## 为什么 DPO 过去、现在都如此重要

Direct Preference Optimization 论文 [@rafailov2024direct] 是 RLHF 和 post-training 研究在可及性上的一次突破。
一些背景：
- DPO 论文发布于 2023 年 5 月，当时许多团队都在艰难尝试开源复现 RLHF 流程。大多数“对齐”模型其实只是 SFT。
- 直到 2023 年秋天前后，人们才摸清让 DPO 奏效的数据和设置
- Zephyr-Beta、Tülu 2 等模型在那年秋天打开了 post-training 研究的闸门。2024 年迭代 DPO 方法的兴奋感，不亚于 2025 年的 RL 方法或 2026 年的 tool-use 方法。
- **术语说明**：我把这类在静态偏好数据上像 DPO 一样运作的算法统称为 Direct Alignment Algorithms（DAAs）。这个名字没怎么流行起来，但我们会看到它有大量变体！

---

<!-- img-align: center -->
## DPO 之争（DPO 对 RL）

![DPO 发布时，引发了关于如何最好地做 RLHF 和偏好学习的激烈争论。梗图来源：Tom Goldstein。](assets/dpo_meme.jpeg)

---

<!-- columns: 50/50 -->
## 本讲内容

我们将**从零推导 Direct Preference Optimization（DPO）**，然后看它在实践中如何使用。

DPO 的承诺：

- 不需要单独的 reward model
- 不需要 reinforcement learning 循环
- 只需在偏好对上使用一个可直接求导的损失函数

|||

分四步走：

1. 求解 RLHF 目标的**最优 policy**
2. 反过来，把 **reward 写成 policy 的函数**
3. 代入 **Bradley-Terry** 偏好模型
4. 得到 **DPO loss** 及其**梯度**

*Rafailov et al., 2023 — "Your Language Model is Secretly a Reward Model"* [@rafailov2024direct]。

---

<!-- rows: 60/40 -->
## 回顾：DPO 在流程中的位置

<!-- row-columns: 48/52 -->
经典 RLHF 由三个部分组成：

1. 收集人类偏好对
2. 训练 reward model $r_\phi(x,y)$
3. 在 KL 惩罚下，用 RL（如 PPO）针对 $r_\phi$ 优化 policy

|||

DPO 把第 2 步和第 3 步合并为**一个监督式损失**。

我们将要证明的关键结论：

> 最优的 RLHF policy 与 reward model 是同一个对象的两种视角。知道其中一个，就能以闭式解得到另一个。

因此我们可以*直接*在偏好上训练 policy。

===

![现代 post-training 会运行多轮反馈与优化；DPO 是该循环中可互换的优化器之一。](assets/rlhf-complex.png)

---

<!-- columns: 55/45 -->
## DPO 概览：我们要去哪里

整个推导最终落在两个对象上。

**最优 policy** —— RLHF 解的闭式形式：

$$ \pi^{*}(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac{1}{\beta} r(x,y)\big) $$

**DPO loss** —— 实际训练用的损失：

$$ \mathcal{L}_{\text{DPO}} = -\mathbb{E}\Big[\log\sigma\big(\beta\log\tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta\log\tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big)\Big] $$

|||

实际上，这个 loss 只需要几行代码：

```python
pi_logratios  = policy_chosen_logps  - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

从这里到代码实现之间的所有内容，都是在解释*为什么*这就是正确的 loss。

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO 推导 P1/4：推导最优 policy

---

<!-- valign: top -->
<!-- align: center -->
## 从 RLHF 优化问题出发

$$
\begin{aligned}
 & \max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}\big[\,r(x,y)\,\big] - \beta\,\mathcal{D}_{\text{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big) && \text{the RLHF objective}
\end{aligned}
$$

我们要找到使这个式子最优的 $\pi$！但不使用 RL。

---

<!-- valign: top -->
<!-- title: center -->
## 把 KL 折进期望中

$$
\begin{aligned}
 & \max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}\big[\,r(x,y)\,\big] - \beta\,\mathcal{D}_{\text{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big) && \text{the RLHF objective}\\[6pt]
={}& \max_{\pi}\ \mathbb{E}\Big[\, r(x,y) - \beta\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \,\Big] && \text{write KL}=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi}\big[\log\tfrac{\pi}{\pi_{\text{ref}}}\big]
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 把 KL 折进期望中

$$
\begin{aligned}
 & \max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}\big[\,r(x,y)\,\big] - \beta\,\mathcal{D}_{\text{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big) && \text{the RLHF objective}\\[6pt]
={}& \max_{\pi}\ \mathbb{E}\Big[\, r(x,y) - \beta\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \,\Big] && \text{write KL}=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi}\big[\log\tfrac{\pi}{\pi_{\text{ref}}}\big]\\[6pt]
={}& \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{split into two terms}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 转为最小化并化简

$$
\begin{aligned}
 & \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{where we left off}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 转为最小化并化简

$$
\begin{aligned}
 & \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{where we left off}\\[6pt]
={}& \min_{\pi}\Big( -\mathbb{E}[r(x,y)] + \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{multiply by } -1:\ \max\to\min
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 转为最小化并化简

$$
\begin{aligned}
 & \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{where we left off}\\[6pt]
={}& \min_{\pi}\Big( -\mathbb{E}[r(x,y)] + \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{multiply by } -1:\ \max\to\min\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\Big[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\Big] && \text{divide by }\beta\text{, recombine}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- align: center -->
<!-- animate: bullets -->
## 将优化目标折叠为一个 log-ratio

接着上次的推导，利用 $\tfrac{1}{\beta} r = \log e^{\,r/\beta}$，把括号折叠成单个 log-ratio：

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\Big[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\Big] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]
\end{aligned}
$$

- 现在 $\pi$ 是与 $\pi_{\text{ref}}\,e^{r/\beta}$ 比较：即**按 reward 重新加权的参考分布**——每个 response 的概率都乘以 $e^{r/\beta}$，reward 越高，这个因子越大（$\beta$ 决定重新加权的力度）。
- 如果这个目标是一个概率分布 $q$，那么这个 $\min_{\pi}$ 就是一个 **KL divergence**，$\mathbb{E}_{y\sim\pi}\!\big[\log\tfrac{\pi}{q}\big] = \mathcal{D}_{\text{KL}}(\pi\,\|\,q)$——可以*精确地*以闭式解求得最小值（[Gibbs 不等式](https://en.wikipedia.org/wiki/Gibbs%27_inequality)）。

---

<!-- valign: top -->
<!-- align: center -->
## 引入配分函数 $Z(x)$

我们要最小化 $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$。

为什么我们希望 $\pi$ 与这个分母相匹配？

---

<!-- valign: top -->
<!-- title: center -->
## 引入配分函数 $Z(x)$

我们要最小化 $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$。

为什么我们希望 $\pi$ 与这个分母相匹配？因为 $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big]$ 是一个 **KL divergence** $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$——它 $\ge 0$，且**仅当 $\pi = q$ 时**等于 $0$。因此，最小化的解就是分母中的那个分布 $q$。

---

<!-- valign: top -->
<!-- title: center -->
## 引入配分函数 $Z(x)$

我们要最小化 $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$。

为什么我们希望 $\pi$ 与这个分母相匹配？因为 $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big]$ 是一个 **KL divergence** $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$——它 $\ge 0$，且**仅当 $\pi = q$ 时**等于 $0$。因此，最小化的解就是分母中的那个分布 $q$。

问题在于：我们的分母 $\pi_{\text{ref}}\,e^{r/\beta}$ **不是一个分布**——对 $y$ 求和后并不等于 $1$。 

---

<!-- valign: top -->
<!-- title: center -->
## 引入配分函数 $Z(x)$

我们要最小化 $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$。

为什么我们希望 $\pi$ 与这个分母相匹配？因为 $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big]$ 是一个 **KL divergence** $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$——它 $\ge 0$，且**仅当 $\pi = q$ 时**等于 $0$。因此，最小化的解就是分母中的那个分布 $q$。

问题在于：我们的分母 $\pi_{\text{ref}}\,e^{r/\beta}$ **不是一个分布**——对 $y$ 求和后并不等于 $1$。 
我们用配分函数对它做归一化：

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r(x,y)\big)
\qquad\text{(depends on $x$ and $r$, not on $\pi$)}
$$

<!-- animate: bullets -->
- 这样 $q(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,e^{\,r/\beta}$ **就是**一个合法的分布了。
- 目标函数变成了真正的 $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$——恰好在 $\pi = q$ 处取得最小值（稍后用 [Gibbs 不等式](https://en.wikipedia.org/wiki/Gibbs%27_inequality)严格证明）。
- 接下来我们变形目标函数的内部，让 $Z(x)$ 出现。

---

<!-- valign: top -->
<!-- title: center -->
## 用配分函数 $Z(x)$ 重新整理目标函数

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 用配分函数 $Z(x)$ 重新整理目标函数

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \,\right] && \text{add } 0 = \log Z - \log Z
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 用配分函数 $Z(x)$ 重新整理目标函数

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \,\right] && \text{add } 0 = \log Z - \log Z\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \big( \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \log Z(x) \big) - \log Z(x) - \tfrac{1}{\beta} r(x,y) \,\right] && \text{group terms}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 用配分函数 $Z(x)$ 重新整理目标函数

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \,\right] && \text{add } 0 = \log Z - \log Z\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \big( \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \log Z(x) \big) - \log Z(x) - \tfrac{1}{\beta} r(x,y) \,\right] && \text{group terms}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\frac{\pi(y\mid x)}{\tfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)} - \log Z(x) - \tfrac{1}{\beta} r(x,y) \,\right] && \log a + \log b = \log ab
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 把目标函数化为 KL divergence

将 $\tfrac{1}{\beta} r = \log e^{\,r/\beta}$ 并入目标函数第一项的分母（同样是 log 运算法则），并记 $q(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}$ —— 它是 $y$ 上的一个合法分布。目标函数为：

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi}\Big[\, \log\tfrac{\pi(y\mid x)}{q(y\mid x)} - \log Z(x) \,\Big]
$$

<!-- step -->

$\log Z(x)$ 不依赖于 $y$，因此将它提到内层期望之外：

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\Big[\, \mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi(y\mid x)}{q(y\mid x)}\big] - \log Z(x) \,\Big]
$$

<!-- step -->

内层期望恰好就是一个 **KL divergence**，$\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big] = \mathcal{D}_{\text{KL}}(\pi\,\|\,q)$：

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\big[\, \mathcal{D}_{\text{KL}}\big(\pi(y\mid x)\,\|\,q(y\mid x)\big) - \log Z(x) \,\big]
$$

---

<!-- valign: top -->
<!-- title: center -->
## 用 Gibbs 不等式读出 $\pi^{*}$

目前的结果 -- 目标函数现在是一个 KL 加上一个与 $\pi$ 无关的常数：

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\big[\, \mathcal{D}_{\text{KL}}\big(\pi(y\mid x)\,\|\,q(y\mid x)\big) - \log Z(x) \,\big]
$$

- $\log Z(x)$ 不依赖于 $\pi$ —— 只有 KL 项可以被优化。
- KL 满足 $\ge 0$，且**仅当 $\pi = q$ 时**等于 $0$（[Gibbs 不等式](https://en.wikipedia.org/wiki/Gibbs%27_inequality)），因此最小值点为 $\pi^{*} = q$：

$$
\boxed{\ \ \pi^{*}(y\mid x) = q(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r(x,y)\big)\ \ }
$$

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO 推导 P2/4：从 policy 反推 reward（实现 DPO 所需）

---

<!-- valign: top -->
<!-- title: center -->
## 反解 $\pi^{*}$ 得到隐式 reward

$$
\begin{aligned}
 & \pi^{*}(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big)  && 
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 反解 $\pi^{*}$ 得到隐式 reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 反解 $\pi^{*}$ 得到隐式 reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}\\[6pt]
 & \log \pi^{*}(y\mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r^{*}(x,y) && \log abc = \log a+\log b+\log c
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 反解 $\pi^{*}$ 得到隐式 reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}\\[6pt]
 & \log \pi^{*}(y\mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r^{*}(x,y) && \log abc = \log a+\log b+\log c\\[6pt]
 & \tfrac{1}{\beta} r^{*}(x,y) = \log \pi^{*}(y\mid x) - \log \pi_{\text{ref}}(y\mid x) + \log Z(x) && \text{rearrange for } r^{*}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## 反解 $\pi^{*}$ 得到隐式 reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}\\[6pt]
 & \log \pi^{*}(y\mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r^{*}(x,y) && \log abc = \log a+\log b+\log c\\[6pt]
 & \tfrac{1}{\beta} r^{*}(x,y) = \log \pi^{*}(y\mid x) - \log \pi_{\text{ref}}(y\mid x) + \log Z(x) && \text{rearrange for } r^{*}\\[6pt]
 & r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x) && \text{multiply by }\beta
\end{aligned}
$$

$$
\boxed{\ \ r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)\ \ }\qquad\text{\small your LM is secretly a reward model}
$$

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO 推导 P3/4：连接 Bradley-Terry 偏好模型（我们的对齐数据长什么样） 

---

<!-- valign: top -->
<!-- title: center -->
## 回顾 Bradley-Terry 模型

两个回复之间的偏好是其 reward 的 softmax：

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{\exp\!\big(r^{*}(x,y_1)\big)}{\exp\!\big(r^{*}(x,y_1)\big) + \exp\!\big(r^{*}(x,y_2)\big)}
$$

现在代入隐式 reward $r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$。

*第一次接触？Bradley-Terry 模型请见[第 2 讲](https://rlhfbook.com/teach/course/lec2-chap4-5-9/)和[第 5 章：Reward Modeling](https://rlhfbook.com/c/05-reward-models.html)。*

---

<!-- valign: top -->
<!-- align: center -->
## 代入 reward，然后消去 $Z(x)$

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta \log Z(x)\big)}{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta \log Z(x)\big) + \exp\!\big(\beta \log \tfrac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} + \beta \log Z(x)\big)}
$$

- 每一项都带有相同的因子 $\exp(\beta \log Z(x)) = Z(x)^{\beta}$。

<!-- step -->

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\big)}{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\big) + \exp\!\big(\beta \log \tfrac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}\big)}
$$

- 分解 $e^{a+b}=e^a e^b$ 并消去公共的 $Z(x)^{\beta}$——**难以计算的项消失了**。

---

<!-- valign: top -->
<!-- title: center -->
## 上下同除，得到 sigmoid

记 $\Delta_i = \beta \log \tfrac{\pi^{*}(y_i\mid x)}{\pi_{\text{ref}}(y_i\mid x)}$，则消去后的形式为 $p^{*} = \dfrac{e^{\Delta_1}}{e^{\Delta_1} + e^{\Delta_2}}$。分子分母同乘 $e^{-\Delta_1}$：

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{e^{\Delta_1}\,e^{-\Delta_1}}{e^{\Delta_1}\,e^{-\Delta_1} + e^{\Delta_2}\,e^{-\Delta_1}} = \frac{1}{1 + e^{\,\Delta_2 - \Delta_1}}
$$

- 分子变为 $1$；分母是**差值** $\Delta_2 - \Delta_1$ 的指数。

<!-- step -->

令 $\sigma(z) = \tfrac{1}{1+e^{-z}}$，这就是一个 sigmoid（$\Delta_i = \beta \log \tfrac{\pi^{*}(y_i\mid x)}{\pi_{\text{ref}}(y_i\mid x)}$）：

$$
\boxed{\ \ p^{*}(y_1 \succ y_2 \mid x) = \sigma\!\Big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} - \beta \log \tfrac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}\Big)\ \ }
$$

- **两个 log-ratio 之差的 sigmoid**——即 Bradley-Terry，只是用 policy 取代了 reward model。

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO 推导 P4/4：损失函数与梯度

---

<!-- valign: top -->
<!-- title: center -->
## DPO loss

最小化所观测到的偏好（$y_c \succ y_r$）的负对数似然——即让这一概率变得更大：

$$
\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\,\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\big[ \log p(y_c \succ y_r \mid x) \big]
$$

<!-- step -->

代入 $p$ 的 sigmoid 形式，以可训练的 policy $\pi_{\theta}$ 作为隐式 reward：

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\Big[ \log \sigma\big(\beta \log \tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big) \Big]
$$

它可以直接求导——**无需 reward model，无需采样，无需 RL 循环。**

---

<!-- valign: top -->
<!-- title: center -->
## DPO 梯度

先对上面的 loss function 求梯度。

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\nabla_{\theta}\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \log \sigma\big(\beta \log \tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big) \Big]
$$

令 $u = \big(\beta \log \tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big)$——即 $\sigma$ 内部的项。

<!-- step -->

链式法则，先处理外层运算 $\tfrac{d}{dz}\log\sigma(z) = \tfrac{\sigma'(z)}{\sigma(z)}$：

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \tfrac{\sigma'(u)}{\sigma(u)}\,\nabla_{\theta} u \Big]
$$

<!-- step -->

代入 $\sigma'(u) = \sigma(u)\big(1-\sigma(u)\big)$ 并消去 $\sigma(u)$：

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \tfrac{\sigma(u)\big(1-\sigma(u)\big)}{\sigma(u)}\,\nabla_{\theta} u \Big] = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \big(1-\sigma(u)\big)\,\nabla_{\theta} u \Big]
$$

---

<!-- valign: top -->
<!-- title: center -->
## DPO 梯度

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \big(1-\sigma(u)\big)\,\nabla_{\theta} u \Big]
$$

反射恒等式 $1-\sigma(u) = \sigma(-u)$ 会翻转自变量的符号，得到紧凑形式：

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \sigma(-u)\,\nabla_{\theta} u \Big]
$$

<!-- step -->

现在对 $u$ 求导，其中 $-u = \beta\log\tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)} - \beta\log\tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)}$。只有 $\pi_{\theta}$ 依赖于 $\theta$——$\pi_{\text{ref}}$ 是冻结的，因此它的 $\log$ 项消去：

$$
\nabla_{\theta} u = \beta\big[\nabla_{\theta}\log\pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log\pi_{\theta}(y_r\mid x)\big]
$$

<!-- step -->

代入 $\nabla_{\theta} u$，并将 $\sigma(-u)$ 完整展开：

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\beta\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \sigma\big(\beta\log\tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)} - \beta\log\tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)}\big)\,\big[\nabla_{\theta}\log\pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log\pi_{\theta}(y_r\mid x)\big] \Big]
$$

---

<!-- valign: top -->
<!-- title: center -->
<!-- animate: bullets -->
## DPO 梯度

$$
\begin{aligned}
\nabla_{\theta}\mathcal{L}_{\text{DPO}} &= -\,\beta\, \mathbb{E}_{(x,y_c,y_r)}\Big[\, \underbrace{w}_{\text{how wrong}} \cdot \big( \nabla_{\theta}\log\pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log\pi_{\theta}(y_r\mid x) \big)\,\Big] \\[6pt]
&\hspace{6em}\text{where}\quad w = \sigma\big(r_\theta(x,y_r) - r_\theta(x,y_c)\big)
\end{aligned}
$$

- **权重 $w \in (0,1)$** 在模型*错得越厉害*时越大——当模型把 rejected response 排在 chosen 之上时，loss 更高。
- **梯度差分项**会提高 $y_c$ 的似然，并降低 $y_r$ 的似然。
- **$\beta$** 控制步长，在正确排序与偏离 $\pi_{\text{ref}}$ 的程度之间权衡（源自 RLHF 目标中的 KL 惩罚）。

---

<!-- rows: 55/45 -->
<!-- title: center -->
## 回顾：DPO 的推导

$$
\begin{aligned}
\textbf{Objective} \;&:\; \max_{\pi}\, \mathbb{E}\big[r(x,y)\big] - \beta\, \mathcal{D}_{\text{KL}}(\pi \,\|\, \pi_{\text{ref}}) \\[4pt]
\Rightarrow\ \textbf{Optimal policy} \;&:\; \pi^{*}(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac{1}{\beta} r(x,y)\big) \\[4pt]
\Rightarrow\ \textbf{Implicit reward} \;&:\; r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x) \\[4pt]
\Rightarrow\ \textbf{Bradley-Terry} \;&:\; p^{*}(y_1 \succ y_2) = \sigma\!\big(r^{*}(x,y_1) - r^{*}(x,y_2)\big)\ \ (Z \text{ cancels}) \\[4pt]
\Rightarrow\ \textbf{DPO loss} \;&:\; -\log \sigma\!\big(\beta \log \tfrac{\pi_{\theta}(y_c)}{\pi_{\text{ref}}(y_c)} - \beta \log \tfrac{\pi_{\theta}(y_r)}{\pi_{\text{ref}}(y_r)}\big)
\end{aligned}
$$

===

reward model 从来不需要单独构建——它一直就藏在 policy 内部。

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO 的弱点、变体与实现细节

---

<!-- columns: 55/45 -->
## 一个隐患：chosen 的概率可能下降

DPO loss 只关心 chosen 与 rejected 对数比值之间的**差值（margin）**，而不关心它们的绝对值。
因此，模型可以通过让 *rejected* 的概率下降得比 chosen **更快**来降低 loss，即使 **chosen 的概率也在下降**。

这是通过推导中的配分函数 $Z(x)$ 传导的。

- 称为 **likelihood displacement** [@razin2024unintentional] [@ren2024learning]；据推测会把概率推向未被处理的、分布外的行为。
- 这也是一些实践者在 chosen response 上加入 SFT 项，或使用 Cal-DPO [@xiao2024cal] / AlphaPO [@gupta2025alphapo] 等修正方法的原因。


|||

![DPO 中 likelihood displacement 的示意图。](assets/dpo_displacement.png)

<!-- step -->

**真实训练中：**[Olmo 1B DPO](https://wandb.ai/rlhf-book/core/runs/fzy8k8go)——chosen/rejected 的 reward margin 在扩大，但 *chosen* 的 log-prob 本身仍可能下降。


---

## $\beta$ 参数与静态 KL

$\beta$ 决定了 KL 约束相对于奖励最大化的强度：

- **较大的 $\beta$** → policy 保持在 $\pi_{\text{ref}}$ 附近；几乎不动。
- **较小的 $\beta$** → policy 可以自由偏离；可能**过度优化**。

关键在于，DPO 最终的 KL 距离是**静态**的：它直接一步到达由数据集和所选 $\beta$ 所隐含的*最优*解。 
在线 RL 则基于新采样的 batch 和逐样本的 KL 惩罚逐步更新。有些 RL 训练甚至包含动态调整的 KL 控制器。

---

<!-- valign: top -->
## 直接对齐算法大观

每个变体都是通过调整损失函数来修补某个局限，往往只需改一行。我开始把所有这些变体统称为直接对齐算法（Direct Alignment Algorithms，DAAs）。先看两个：

<!-- step -->

- **Identity Preference Optimization (IPO)** [@azar2024general] — 把偏好概率从 Bradley-Terry 形式上软化，以抑制过拟合。

$$ \mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x,y_w,y_l)}\!\left[\left(\log\tfrac{\pi_\theta(y_w\mid x)\,\pi_{\text{ref}}(y_l\mid x)}{\pi_\theta(y_l\mid x)\,\pi_{\text{ref}}(y_w\mid x)} - \tfrac{1}{2\beta}\right)^2\right] $$

<!-- step -->

- **Conservative DPO (cDPO)** [@rafailov2024direct]（出自原始 DPO 论文）— 假设偏好标签带有噪声（以概率 $\varepsilon$ 被翻转），并据此软化损失目标。

<!-- step -->

- **Offset DPO (ODPO)** [@amini2024direct] — 要求 chosen 比 rejected 高出一个 margin 偏移量，偏移量随偏好强度缩放。

---

<!-- valign: top -->
## 直接对齐算法大观

另外两个算法则**完全去掉了 reference model**：

<!-- step -->

- **Odds Ratio Preference Optimization (ORPO)** [@hong2024reference] — 加入一项指向 chosen 回复的 odds ratio 拉力，直接并入 SFT 损失。odds ratio 只用到 $\pi_\theta$，因此可以去掉 reference model。

$$ \mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} - \lambda\,\mathbb{E}\!\left[\log \sigma\!\left(\log \tfrac{\text{odds}_\theta(y_w\mid x)}{\text{odds}_\theta(y_l\mid x)}\right)\right], \quad \text{odds}_\theta(y) = \tfrac{\pi_\theta(y)}{1-\pi_\theta(y)} $$

<!-- step -->

- **Simple Preference Optimization (SimPO)** [@meng2025simpo] — 把 reward 按长度归一化为平均 log-prob，并设定目标 margin $\gamma$。$\tfrac{1}{|y|}$ 用于按长度归一化；$\gamma$ 是目标 margin；不含 $\pi_{\text{ref}}$。

$$ \mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\tfrac{\beta}{|y_w|}\log \pi_\theta(y_w\mid x) - \tfrac{\beta}{|y_l|}\log \pi_\theta(y_l\mid x) - \gamma\right)\right] $$

<!-- step -->

算法本身的影响**远小于**基础模型和数据。尽管如此，许多论文仍在不断做小幅算法改动。实际存在的变体远比这几页幻灯片上的多。

---

## 实现其实非常简单

训练中无需生成，也不需要单独的 reward model，损失的核心只有几行代码：

```python
# log-prob gaps for policy and frozen reference
pi_logratios  = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# positive when policy shifts mass toward the chosen completion
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

**小技巧：** $\pi_{\text{ref}}$ 是冻结的，因此可以预先计算并缓存它的 log-prob，把峰值显存降低约 50%。参考代码：`code/direct_alignment/`。

---

<!-- animate: bullets -->
## DAAs 适用于合成偏好数据

这些算法需要的是*反馈*数据，不一定是*人类*反馈数据，AI 反馈同样有效。

- 现代 DPO 大多使用由强模型标注的偏好。**UltraFeedback** [@cui2023ultrafeedback] 是第一个有影响力的*开源*合成偏好数据集，帮助开启了 DPO 时代；Tülu 3 [@lambert2024t]、SmolLM 2 [@allal2025smollm2] 等随后推出了规模更大的合成反馈方案。
- **On-policy 数据**（部分 completion 来自你正在调优的模型）有助于对比损失在复杂的 post-training 方案中优化到正确的 token 空间（我们在 Tülu 3 中专门研究过这一点）。
- **Delta Learning** [@geng2025the]：后续工作认为，chosen 与 rejected 之间的*差距*比它们由哪些模型生成更重要（例如 Qwen3-32B 作 chosen，Qwen3-0.6B 作 rejected）。我们在 Olmo 3 中用到了这一点！

---

<!-- columns: 50/50 -->
## DPO 与 RL：离线与在线

**DPO 与其他 DAAs**

- 在事先收集好的固定数据集上训练。
- 更简单、更稳定，迭代数据很快。
- 受限于该数据集的**覆盖范围**，性能上限略低。

|||

**PPO / policy gradient 是在线的**

- 训练过程中生成新的 completion，并用 reward model 打分。
- 可以探索新的区域，因此峰值性能通常更高 [@ivison2024unpacking] [@xu2024dpo] [@tajwar2024preference]。
- 计算量更大、活动部件更多（内存中有四个模型），工程复杂度也更高。

---

## 如今 DPO 处在什么位置？

既然这么多模型都用 DPO 取得了不错的效果，而且它如此简单，为什么它似乎很少被提起，尤其是在前沿模型中？

---

<!-- columns: 52/48 -->
## 对 DPO 当下角色的几点假设

- **Olmo 3**（2025 年 11 月）/ **SmolLM 3**（2025 年 7 月）-- 在 SFT 之后用 DPO 处理推理轨迹，以简单的流程提升性能。
- **NVIDIA Nemotron 3**（2025 年 12 月）-- *Mixed Preference Optimization*：在一个离线阶段中结合 DPO 与 Binary Classifier Optimization，数据由 generative reward model 打分。
- **Liquid AI LFM2**（2025 年 11 月）-- 在半在线数据上使用长度归一化的 DPO，最后再接一轮 RLVR。

|||

---

<!-- columns: 52/48 -->
## 对 DPO 当下角色的几点假设

- **Olmo 3**（2025 年 11 月）/ **SmolLM 3**（2025 年 7 月）-- 在 SFT 之后用 DPO 处理推理轨迹，以简单的流程提升性能。
- **NVIDIA Nemotron 3**（2025 年 12 月）-- *Mixed Preference Optimization*：在一个离线阶段中结合 DPO 与 Binary Classifier Optimization，数据由 generative reward model 打分。
- **Liquid AI LFM2**（2025 年 11 月）-- 在半在线数据上使用长度归一化的 DPO，最后再接一轮 RLVR。

|||

**我的看法：**

- DPO 在 Olmo 这类“古怪”的、重度依赖蒸馏的方案中效果很好，例如分布更尖锐的模型，训练数据来自众多 teacher 模型。
- DPO 在其他场景下依然有效，但大多数实验室有足够的工程资源去做峰值性能更高的其他方案。
- 一句话总结：DPO 是通往一个不错、扎实的模型的路径，但通不到最好的模型。而且在*更干净*的训练方案中，DPO 的作用可能更小。
