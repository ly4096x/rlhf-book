
接下来，我们需要引入一个配分函数 $Z(x)$：

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

配分函数作为非归一化密度 $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 的归一化因子，从而使其成为对于每个固定的 $x$，关于 $y$ 的合法概率函数。这一需求的具体原因将在推导过程中逐渐明朗。

将其代入后，我们得到中间变换形式：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

为了理解如何得到这一结果，请考虑 @eq:dpo_deriv_4 方括号内优化的内部部分：

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

然后，在两边加上 $\log Z(x) - \log Z(x)$：

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

再对各项进行分组：

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

利用 $\log(x) + \log(y) = \log(x\cdot y)$（并将 $Z$ 移至分母），得到：

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

接下来，将 $\frac{1}{\beta}r(x,y)$ 展开为 $\log \exp \frac{1}{\beta}r(x,y)$，并做相同操作，即可得到 @eq:dpo_deriv_5，在此略作改写：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

有了这一优化形式，我们需要实际求解最优 policy $\pi^*$。由于我们引入了配分函数 $Z(x)$，使得 $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 成为关于 $y$ 的合法概率分布，因此可以认识到内层期望实际上是一个真正的 KL divergence！

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

由于 $\log Z(x)$ 项与最终结果无关，可以忽略。这样就只剩下我们正在学习的 policy 与一个涉及配分函数、$\beta$、reward 以及 reference policy 的形式之间的 KL divergence。Gibbs 不等式告诉我们，该 KL divergence 在距离为 0 时取得最小值，而这仅在两个量相等时成立！因此，我们得到最优 policy：

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}


#### 2. 推导 Bradley Terry 模型的 DPO 目标函数

首先，回顾第 5 章关于 Reward Modeling 以及第 11 章关于 Preference Data 的内容，Bradley-Terry 人类偏好模型表述如下：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

通过对 @eq:dpo_opt_policy 进行变换，可以求解最优 reward。首先对两边取对数：

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

利用 $\log(abc) = \log a + \log b + \log c$ 展开右侧：

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

整理以求解 $r^*(x,y)$：

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

两边乘以 $\beta$：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

随后，将 reward 代入 @eq:bradley_terry_dpo 所示的 Bradley-Terry 方程，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

通过将指数表达式从 $e^{a+b}$ 分解为 $e^a e^b$，再消去 $e^{\log(Z(x))}$ 项，化简得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

然后，将分子和分母同乘以 $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

最后，根据 sigmoid 函数的定义 $\sigma(x) = \frac{1}{1+e^{-x}}$，我们得到：

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

这是在最优 policy $\pi^*$ 下，Bradley-Terry 模型给出的 preference data 的似然概率。回顾第 5 章关于 Reward Modeling 的内容，我们已推导出 Bradley-Terry 目标函数为最大化似然，即等价地最小化负对数似然，由此得到损失函数：
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

这就是 DPO 的损失函数，其形式如 @eq:dpo_core 所示。DPO 论文还额外推导了 Plackett-Luce 模型下的目标函数，但该形式在实践中较少使用 [@rafailov2024direct]。

#### 3. 推导 Bradley Terry DPO 梯度

我们在 @eq:dpo_gradient 中使用了 DPO 梯度来解释模型学习的直觉。要推导该梯度，需要对 @eq:dpo_loss_deriv4 关于模型参数求梯度。

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

首先，这个式子可以改写。我们知道 sigmoid 函数的导数 $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$，对数的导数 $\frac{d}{dx} \log x = \frac{1}{x}$，以及 sigmoid 的性质 $\sigma(-x)=1-\sigma(x)$，因此可以对上式进行变形。

令 $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$（即 sigmoid 内部的表达式），则有：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

展开并利用上述 sigmoid 和对数的表达式，得到前面引入的梯度：

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi(y_c|x)-\nabla_{\theta}\log\pi(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 数值问题、局限性与替代方法

DPO 算法的许多变体已被提出，以解决 DPO 的不足。例如，由于缺乏 rollout 过程（reward model 无法对生成结果进行评分），DPO 对每对 preference data 赋予相同的权重。而实际上，正如第 11 章关于 Preference Data 所示，有许多方式可以用比二元标签更丰富的标签来捕捉偏好信息。已有多种算法被提出，以重新平衡优化过程，使其不再对每对数据一视同仁。
