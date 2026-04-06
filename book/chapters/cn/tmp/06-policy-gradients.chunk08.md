```python
masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.0909, 0.0909, 0.0909, 0.0909, 0.0000, 0.0000, 0.0000],
# [0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]])
```

输出结果表明，采用策略 1（`masked_mean`）时，短序列的每个 token 梯度（0.25）大于长序列（0.14）。
策略 2 和策略 3 则使各序列间的每个 token 梯度趋于均等。
值得注意的是，若使用梯度累积——即在进行一次反向传播步骤前将多个 mini-batch 的梯度求和——上述结果可能会有显著变化：在这种情况下，短序列与长序列之间的权重平衡可能发生反转。

在实际操作中，最优策略取决于具体的训练配置。
在 RLHF 中，通常优先选择数值稳定性最佳或 loss 方差最小的方法。

#### 相关内容：MDP 框架与 Bandit 框架

loss 聚合方式的选择与我们如何定义 RL 问题的框架密切相关。
**MDP（token 级别）**视角将每个 token $a_t$ 视为一个动作，状态 $s_t$ 为当前的前缀序列。
在实际中，当我们通过学习得到的 value function $V(s_t)$（例如 GAE [@schulman2015high]）来计算 token 级别的 advantage function，并逐 token 施加 KL divergence 惩罚时，采用的正是这种框架。
带有学习 value network 的 PPO 是这一框架的典型代表 [@schulman2017proximal]。

相比之下，**bandit（序列级别）**视角将整个输出视为单一动作，对应一个标量 reward $R$。
在代码层面，这意味着计算序列级别的 advantage $A_{\text{seq}}$，并将其广播至所有 token。
RLOO 和 GRPO 风格的 advantage 通常用于这种 bandit 式设置 [@kool2019buy] [@ahmadian2024back] [@shao2024deepseekmath]。
DPO 和 A-LoL 等直接对齐方法也定义了序列级别的目标，尽管它们并非 policy gradient 估计器 [@baheti2023leftover]。

需要注意的是，许多 GRPO 实现采用 bandit 式 advantage，同时在 loss 中额外加入逐 token 的 KL 项；而许多 PPO/RLOO 实现则在计算 advantage 之前将 KL 折叠进 reward 中——两种做法在实践中均有使用。

下面的示例展示了两种方法的对比：

```python
# === Bandit-style（序列级别）===
# 每个序列对应一个标量 reward；advantage 广播至所有 token
reward = torch.tensor([3.0, 1.0])       # (B,) 例如 reward model 的评分
baseline = reward.mean()                 # 简单基线（RLOO 使用 leave-one-out）
advantage_seq = reward - baseline        # (B,)
advantages = advantage_seq[:, None].expand(-1, seq_len)  # (B, L)
# tensor([[ 1.,  1.,  1.,  1.],    <- 所有 token 的 advantage 相同
#         [-1., -1., -1., -1.]])

# === MDP-style（token 级别）===
# 逐 token reward + 学习得到的 V(s_t)；每个 token 有其自身的 advantage
# （也可使用逐 token 的 KL shaping、格式 reward 或其他 token 级别信号）
advantages = gae(per_token_rewards, values, done_mask, gamma=1.0, lam=0.95)
# tensor([[ 0.2,  0.5,  0.8,  1.5],    <- 随位置变化
#         [-0.3, -0.5, -0.8, -1.4]])
```

这一框架上的区别也解释了为何在几乎所有 RLHF 实现中，折扣因子 $\gamma$ 均被设置为 1.0。
在标准 RL 中，折扣（$\gamma < 1$）至关重要：它在多步骤 episode 中平衡了对短期与长期 reward 的优化，这对智能体学习有效的长期行为不可或缺。
但在 RLHF 设置中，即便采用 token 级别的 MDP 视角，优化的归纳偏置仍是整体输出的质量——reward 信号对整个响应进行评分，而非对单个 token 评分。
对较早出现的 token 进行折扣会任意降低其贡献权重，缺乏合理的理论依据。
随着 agentic RL 场景的日趋成熟——模型在其中执行真正的多步骤动作，如工具调用、代码执行和网页浏览——折扣因子可能再度变得相关，因为这类场景涉及真正具有不同长期后果的序贯决策。

### 异步 RL 系统

policy gradient 算法的默认实现方式是所谓的 **on-policy** 执行：智能体（语言模型）在采取动作（生成输出）后，先对动作进行评分，再更新模型。
policy gradient 的理论推导依赖于所有动作严格遵循 on-policy 原则，即模型始终与最新 trial/rollout 的结果保持同步。
然而在实践中，维持严格的 on-policy 执行会显著降低训练速度 [@noukhovitch2024asynchronous]——而且无论如何，完美同步在技术上都是不可能实现的。
因此，近期关于语言模型的大量实证结果在理论上都略微偏离了相应的数学证明。
实际发生的情况是：研究者根据什么真正有效来设计算法与系统。

![依据 Noukhovitch et al. 2024，同步与异步 RL 训练的生成-更新阶段对比。](images/async_v_synch_rl.png){#fig:async}

常用的解决方案是在独立的 GPU 节点上持续并行运行推理和训练，并使用专门为高效同步运行两者而设计的软件，如 @fig:async 底部所示。
在面向语言模型的主流开源 RL 工具中，通常的做法是使用 Ray 等分布式进程管理库，借助 vLLM 等高效推理引擎，在 policy gradient 学习循环与推理循环之间传递信息。
在这类系统中，负责执行 RL 更新步骤的 GPU 被称为"learner"，负责从语言模型采样的 GPU 被称为"actor"。
提升训练异步程度时面临的主要挑战在于保持训练稳定性和维持学习信号。

![一个分布式 RL 系统示例：系统通过两个队列向 learner GPU 和 actor GPU 传递数据，两者均可通过 Ray 等分布式计算库进行同步。Olmo Team 2025，CC-BY 许可。](images/distributed-rl.png){#fig:async_system}

这类系统的设计前提是：接近 on-policy 的数据足以支撑稳定的学习。
在这里，生成阶段与更新阶段可以轻松同步，避免训练系统任一部分出现计算资源空闲——即在 @fig:async_system 中将模型权重从 learner 传递至 actor 的过程。
对于推理模型而言，每个答案需要生成 10K 至 100K+ 个 token，其极长的推理特性使得 rollout 的生成成为远比其他环节更突出的瓶颈。
在同步性较强的 RL 基础设施上训练推理模型时，一个常见问题是：批次中某个 prompt 的回答可能需要消耗远多于其他 prompt 的生成时间（无论是更多的 token 还是更多的工具调用），导致大部分分配的计算资源在等待该回答完成期间处于闲置状态。
针对这一长度不匹配问题的第二种解决方案称为**序列级别打包（sequence-level packing）**：通过巧妙的 masking 将较短的样本叠加在同一批次中，使模型能够持续推进 rollout，并在批次内各样本间更好地分配长度归一化。
分布式 RL 基础设施的完整复杂性超出了本书的讨论范围，因为它还可能引发许多其他降低训练速度或导致不稳定的微妙问题。

随着推理模型的兴起，研究者进一步致力于将训练与推理循环转变为完全 off-policy 的模式：policy gradient 更新的训练批次由多个实例中最近完成的 rollout 填充 [@wu2025llamarl] [@fu2025areal]。
完全异步的训练也将使跨多个数据中心扩展 RL 训练规模变得更加容易，因为这样可以灵活增加 learner 节点（执行 policy gradient 步骤）与 actor（尝试求解问题）之间权重同步的时间间隔 [@primeintellectteam2025intellect2reasoningmodeltrained]。

相关方法正在探索完全 off-policy 的 policy gradient 算法 [@leroux2025topr]。

### 截断重要性采样（TIS）

截断重要性采样（Truncated Importance Sampling，TIS）是现代异步语言模型 RL 框架中用于稳定训练的重要工具。
重要性采样是一种修正方法，将从某一分布中采集的样本重新加权，以估计另一分布下的期望（如 @eq:IS_identity 中所介绍）。
截断重要性采样 [@ionides2008truncated] 使用 $\min(\rho, C)$（其中 $C$ 为某常数）对这些权重进行截断，以小幅偏差换取 policy gradient 中有界的方差。

这是一种应用于 policy gradient 的重要性采样修正，但与 PPO 和 CISPO 中的双边 clipping（将比率约束在 1 附近）不同，TIS 采用单侧上限：比率可以自由低于 1，但被截断至 $C$ 以防止极端的上加权。
在 PPO、GRPO、CISPO 及相关算法中，比率 $\rho_t^{\text{policy}} = \pi_\theta(a_t \mid s) / \pi_{\theta_{\text{old}}}(a_t \mid s)$ 用于修正同一 RL 批次中多次梯度步骤带来的策略漂移。
当我们转向以异步性为核心的真实 RL 框架时，还可能存在更大来源的数值差异（同样需要重要性采样进行数值修正）。
即便采样器与 learner 共享完全相同的参数 $\theta$，它们的有效 token 分布也可能存在差异，因为推理引擎（如 vLLM）与训练框架（如 FSDP）使用了不同的 kernel、精度和并行策略 [@yao2025offpolicy]。
因此，将同一策略在两个系统上的评估加以区分是有意义的，分别记为 $\pi_\theta^{\text{sampler}}$ 和 $\pi_\theta^{\text{learner}}$，并定义对应的比率及其截断形式：

$$
\rho_t^{\text{learner}} = \frac{\pi_\theta^{\text{learner}}(a_t \mid s, a_{<t})}{\pi_\theta^{\text{sampler}}(a_t \mid s, a_{<t})}, \qquad \tilde{\rho}_t^{\text{learner}} = \min(\rho_t^{\text{learner}},\; C).
$$ {#eq:tis_backend}

这两种修正相辅相成，但引入 policy gradient 实现中的原因各不相同——前者补偿 RL 批次训练过程中的策略漂移，后者则补偿实现层面引入的分布偏差——二者可以同时应用。
它们的组合方式取决于具体算法：

**带 TIS 的 REINFORCE**（单次梯度步骤）：不存在策略漂移（$\pi_\theta = \pi_{\theta_\text{old}}$），因此唯一的不匹配来自 learner 与 sampler 之间。
此时 $\pi_{\theta_\text{old}} = \pi_\text{gen}$，TIS 直接修正 learner–sampler 之间的差距：

$$
\nabla_\theta J \approx \mathbb{E}_{a \sim \pi_\theta^{\text{sampler}}} \left[ \tilde{\rho}_t^{\text{learner}} \cdot A_t \cdot \nabla_\theta \log \pi_\theta^{\text{learner}}(a_t \mid s, a_{<t}) \right].
$$ {#eq:reinforce_tis}
