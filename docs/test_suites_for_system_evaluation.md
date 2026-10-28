# 🧪《云山问学》系统全场景多轮研讨标准化测试剧本集
> **测试环境说明**：当前知识库为空（已清空本地文献），重点考察系统的 **全网联邦文献发现（arXiv / CrossRef / GitHub）**、**纯第一性原理数理推演（LaTeX 严密证明）** 与 **LangGraph 原生记忆通道状态流转（INIT $\to$ INHERIT $\to$ APPEND $\to$ REPLACE）**。

---

## 📑 目录索引
1. [【剧本一】双体量子纠缠、von Neumann 熵与局域退相干全流程推演（4轮）](#剧本一双体量子纠缠von-neumann-熵与局域退相干全流程推演4轮)
2. [【剧本二】NISQ 量子算法与贫瘠高原（Barren Plateaus）理论证明（3轮）](#剧本二nisq-量子算法与贫瘠高原barren-plateaus理论证明3轮)
3. [【剧本三】全网前沿文献追踪 $\to$ 架构横向大比拼 $\to$ 线性推导 $\to$ 话题重置（4轮）](#剧本三全网前沿文献追踪--架构横向大比拼--线性推导--话题重置4轮)
4. [【剧本四】生成模型第一性原理推演（DDPM $\to$ Flow Matching 最优传输）（3轮）](#剧本四生成模型第一性原理推演ddpm--flow-matching-最优传输3轮)
5. [【剧本五】跨学科理论设计与负向排除（PINN 求解 Navier-Stokes 方程）（3轮）](#剧本五跨学科理论设计与负向排除pinn-求解-navier-stokes-方程3轮)

---

## ⚛️ 剧本一：双体量子纠缠、von Neumann 熵与局域退相干全流程推演（4轮）
* **测试目标**：高阶张量与矩阵运算、代词严格继承（`INHERIT`）、环境开放系统动力学演化。

### 第 1 轮：第一性原理基础公式推导
- 📋 **用户输入（复制使用）**：
  ```text
  请根据第一性原理严格推导双体量子纠缠系统中的 von Neumann 纠缠熵计算公式，写出偏迹（Partial Trace）与约化密度矩阵的数学定义
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`（公式推导专家）
  - 记忆状态通道：`INIT -> 【双体量子纠缠与 von Neumann 熵】`
- 🎯 **检验要点**：
  - 严格定义双体 Hilbert 空间 $\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B$ 与复合态密度矩阵 $\rho_{AB} = |\psi\rangle\langle\psi|$；
  - 给出偏迹定义 $\rho_A = \text{Tr}_B(\rho_{AB}) = \sum_k (I_A \otimes \langle k|_B) \rho_{AB} (I_A \otimes |k\rangle_B)$；
  - 写出 von Neumann 熵 $S(\rho_A) = -\text{Tr}(\rho_A \ln \rho_A) = -\sum_i \lambda_i \ln \lambda_i$。

---

### 第 2 轮：代词继承与 Bell 态具体对角化特解
- 📋 **用户输入（复制使用）**：
  ```text
  那么它在标准最大纠缠态 Bell 态（|\Phi^+\rangle = \frac{|00\rangle+|11\rangle}{\sqrt{2}}）下的约化密度矩阵与纠缠熵具体数值是多少？请写出详细对角化计算过程
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent` / `TutorAgent`
  - 记忆状态通道：`INHERIT -> 【双体量子纠缠与 von Neumann 熵】`（0ms 零开销继承）
- 🎯 **检验要点**：
  - 准确将 $|\Phi^+\rangle$ 展开为密度矩阵并对 $B$ 求偏迹，得到最大混态 $\rho_A = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}I_2$；
  - 本征值 $\lambda_1 = \lambda_2 = \frac{1}{2}$；
  - 计算得出最大纠缠熵 $S(\rho_A) = -(\frac{1}{2}\ln\frac{1}{2} + \frac{1}{2}\ln\frac{1}{2}) = \ln 2 \approx 0.693$（或 $1\text{ bit}$）。

---

### 第 3 轮：环境退相干与 Lindblad 耗散主方程演化
- 📋 **用户输入（复制使用）**：
  ```text
  如果让子系统 A 受到局域退相干（Phase Damping）噪声环境影响，系统的纠缠熵会如何随时间衰减演化？请给出 Lindblad 主方程与定量衰减规律
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INHERIT -> 【双体量子纠缠与 von Neumann 熵】`
- 🎯 **检验要点**：
  - 引入 Lindblad 主方程 $\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \mathcal{D}[\rho]$；
  - Phase Damping 算符 $L = \sqrt{\gamma} \sigma_z$；
  - 证明密度矩阵非对角相干项以指数速度衰减 $\rho_{01}(t) = \rho_{01}(0) e^{-\gamma t}$，导致纠缠熵向混态演化。

---

### 第 4 轮：高阶拓展与 Rényi 熵横向对比
- 📋 **用户输入（复制使用）**：
  ```text
  对比一下 von Neumann 熵与 Rényi 熵在表征量子多体系统相变（如拓扑纠缠熵）时的计算差异与适用场景
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`ComparisonAgent`
  - 记忆状态通道：`INHERIT` 或 `APPEND`
- 🎯 **检验要点**：
  - 给出 Rényi 熵定义 $S_\alpha(\rho) = \frac{1}{1-\alpha}\ln \text{Tr}(\rho^\alpha)$，并证明 $\lim_{\alpha \to 1} S_\alpha(\rho) = S(\rho)$；
  - 对比两者在 DMRG（密度矩阵重整化群）和量子蒙特卡洛计算中的算力开销差异。

---

## ⚛️ 剧本二：NISQ 量子算法与贫瘠高原（Barren Plateaus）理论证明（3轮）
* **测试目标**：量子-经典混合回路推导、方差消失机理证明、主体追加（`APPEND`）。

### 第 1 轮：变分量子本征求解器（VQE）理论推导
- 📋 **用户输入（复制使用）**：
  ```text
  请详细推导变分量子本征求解器（VQE）的数学原理，说明它如何利用 Ritz 变分原理在量子-经典混合回路中求解分子哈密顿量基态能量
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INIT -> 【VQE 变分量子本征求解器】`
- 🎯 **检验要点**：
  - Ritz 变分原理证明：$\langle \psi(\theta)|H|\psi(\theta)\rangle \ge E_0$；
  - 分子哈密顿量在 Pauli 基底上的线性展开 $H = \sum_i c_i P_i$；
  - 量子处理器负责测量期望值 $\langle P_i \rangle$，经典优化器负责梯度更新 $\theta \leftarrow \theta - \eta \nabla_\theta \langle H \rangle$。

---

### 第 2 轮：贫瘠高原（Barren Plateaus）方差指数消失证明
- 📋 **用户输入（复制使用）**：
  ```text
  它在深层参数化量子电路中遇到的“贫瘠高原（Barren Plateaus）”现象在数学上是如何定义的？请证明其代价函数梯度的方差为什么随比特数呈指数衰减
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INHERIT -> 【VQE 变分量子本征求解器】`
- 🎯 **检验要点**：
  - 基于 Haar 测度与 2-design 电路假设，证明 $\mathbb{E}_\theta[\partial_\theta C] = 0$；
  - 运用 Weingarten 函数积分，证明梯度方差 $\text{Var}_\theta[\partial_\theta C] \in \mathcal{O}\left(\frac{1}{2^n}\right)$，即随量子比特数 $n$ 发生灾难性指数衰减。

---

### 第 3 轮：主体追加与局域代价函数破解方案
- 📋 **用户输入（复制使用）**：
  ```text
  再追加考虑局域代价函数（Local Cost Function）与对称性保持拟设（Symmetry-preserving Ansatz），请分析它们是如何在理论上打破贫瘠高原的？
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`ComparisonAgent` / `DerivationAgent`
  - 记忆状态通道：`APPEND -> 【VQE & 局域代价函数/对称拟设】`
- 🎯 **检验要点**：
  - 将全局可观测量 $H = I - |0\dots 0\rangle\langle 0\dots 0|$ 替换为局域和 $H_{local} = \frac{1}{n}\sum_{j=1}^n (I - |0\rangle\langle 0|_j)$；
  - 证明局域代价函数的方差在浅层电路下仅按多项式衰减 $\mathcal{O}(1/\text{poly}(n))$，梯度可训练。

---

## 🤖 剧本三：全网前沿文献追踪 $\to$ 架构横向大比拼 $\to$ 线性推导 $\to$ 话题重置（4轮）
* **测试目标**：全网真实文献多源检索（arXiv / CrossRef / GitHub）、选型决策、数学推导与主体彻底切换（`REPLACE`）。

### 第 1 轮：全网多源前沿学术文献发现
- 📋 **用户输入（复制使用）**：
  ```text
  请检索并介绍关于 Transformer 最新变体（如 2025/2026 年高效注意力与 Mamba/SSM 混合架构）的前沿研究与代表论文
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`OnlinePaperSearchAgent`（全域文献发现专员）
  - 检索执行：并发调用 arXiv + CrossRef + GitHub
  - 记忆状态通道：`INIT -> 【Transformer 最新变体与前沿架构】`
- 🎯 **检验要点**：
  - 输出准确的论文标题、作者、发表年份（arXiv / 顶会）与 GitHub 开源实现链接；
  - 涵盖 FlashAttention-3、Mamba-2、Jamba 或线性注意力等主流代表作。

---

### 第 2 轮：代词继承与低显存极限选型对比
- 📋 **用户输入（复制使用）**：
  ```text
  那么在 24GB 单卡显存限制下，哪个变体在长上下文场景下的吞吐与效果最佳？请做详细选型对比
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`ComparisonAgent`
  - 记忆状态通道：`INHERIT -> 【Transformer 最新变体与前沿架构】`
- 🎯 **检验要点**：
  - 开门见山指名第 1 轮涉及的变体；
  - 给出显存消耗对比（KV Cache 显存占用 $O(N)$ vs SSM 恒定状态 $O(1)$）；
  - 给出明确的工程选型建议（如“推理侧首选 Mamba/SSM 混合，训练侧首选 FlashAttention + GQA”）。

---

### 第 3 轮：线性注意力 $O(N^2) \to O(N)$ 核技巧数学推导
- 📋 **用户输入（复制使用）**：
  ```text
  请写出数学公式，详细推导并证明线性注意力（Linear Attention）是如何通过核技巧将时间复杂度从 O(N^2) 降至 O(N) 的？
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INHERIT`
- 🎯 **检验要点**：
  - 标准 Softmax 注意力公式：$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$；
  - 引入核特征映射 $\phi(x)$，将相似度写为 $\phi(Q)\phi(K)^T$；
  - 运用矩阵乘法结合律：$(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$，优先计算 $\phi(K)^T V \in \mathbb{R}^{d \times d}$，将计算复杂度从 $\mathcal{O}(N^2 d)$ 彻底降为 $\mathcal{O}(N d^2)$。

---

### 第 4 轮：显式重置话题（清除旧偏见）
- 📋 **用户输入（复制使用）**：
  ```text
  好了，大模型先聊到这。我们换个全新话题，请详细推导狄拉克方程（Dirac Equation）的相对论协变形式与负能量态的物理诠释
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`REPLACE -> 【狄拉克方程与相对论协变性】`（旧记忆完全清零）
- 🎯 **检验要点**：
  - 彻底摆脱前 3 轮的 Transformer/显存思维定势，完全切入理论物理通道；
  - 推导 $(i\hbar \gamma^\mu \partial_\mu - mc)\psi = 0$ 与 Clifford 代数 $\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}$。

---

## 🎨 剧本四：生成模型第一性原理推演（DDPM $\to$ Flow Matching）（3轮）
* **测试目标**：扩散随机微分方程、重参数化技巧、最优传输（Optimal Transport）路径直线化。

### 第 1 轮：DDPM 前向加噪与 ELBO 损失函数推导
- 📋 **用户输入（复制使用）**：
  ```text
  请从第一性原理推导扩散模型 DDPM 的前向加噪高斯马尔可夫链与变分下界（ELBO）损失函数
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INIT -> 【DDPM 扩散模型】`
- 🎯 **检验要点**：
  - 推导前向边缘分布 $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$；
  - 将负对数似然拆解为 KL 散度之和 $\mathcal{L}_{VLB} = \sum_{t=1}^T D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_\theta(x_{t-1}|x_t))$。

---

### 第 2 轮：代词继承与噪声预测等价性证明
- 📋 **用户输入（复制使用）**：
  ```text
  它在反向去噪采样过程中，为什么预测噪声 \epsilon_\theta 与预测均值 \mu_\theta 在数学上是等价的？
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INHERIT -> 【DDPM 扩散模型】`
- 🎯 **检验要点**：
  - 代入后验真实均值 $\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$；
  - 代入 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$，消元化简证明 $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$。

---

### 第 3 轮：主体追加与 Flow Matching 最优传输对比
- 📋 **用户输入（复制使用）**：
  ```text
  再追加考虑 Flow Matching 架构，请对比它与扩散模型在最优传输（Optimal Transport）路径直线化上的核心差异
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`ComparisonAgent`
  - 记忆状态通道：`APPEND -> 【DDPM & Flow Matching】`
- 🎯 **检验要点**：
  - 对比 SDE 布朗运动弯曲扩散轨迹与 Flow Matching 条件向量场 $v_t(x) = \frac{d\psi_t(x)}{dt}$；
  - 证明 OT-CFM（Optimal Transport Conditional Flow Matching）生成直线 ODE 轨迹，大幅减少采样步数（从数十步降至几步）。

---

## 🌊 剧本五：跨学科理论设计与负向排除（PINN 求解 Navier-Stokes）（3轮）
* **测试目标**：PDE 残差网络设计、硬约束条件嵌入、负向排除约束（`negative_constraints`）。

### 第 1 轮：物理信息神经网络（PINN）架构设计
- 📋 **用户输入（复制使用）**：
  ```text
  我想设计一套基于物理信息神经网络（PINN）求解不可压缩 Navier-Stokes 方程的方案，请给出核心网络架构与残差损失函数
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent` / `TutorAgent`
  - 记忆状态通道：`INIT -> 【PINN 求解 Navier-Stokes 方程】`
- 🎯 **检验要点**：
  - 定义输入 $(x, y, z, t)$ 与输出 $(u, v, w, p)$；
  - 构建自动微分（AD）计算残差 $\mathcal{L}_{pde} = \|\frac{\partial u}{\partial t} + (u\cdot\nabla)u + \frac{1}{\rho}\nabla p - \nu \nabla^2 u\|^2 + \|\nabla \cdot u\|^2$。

---

### 第 2 轮：流函数硬约束嵌入
- 📋 **用户输入（复制使用）**：
  ```text
  在这个方案中，如何通过流函数（Stream Function）等硬约束方法，强制保证速度散度为 0（\nabla \cdot u = 0）？
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`DerivationAgent`
  - 记忆状态通道：`INHERIT -> 【PINN 求解 Navier-Stokes 方程】`
- 🎯 **检验要点**：
  - 设神经网络输出标量流函数 $\psi(x, y, t)$；
  - 令 $u = \frac{\partial \psi}{\partial y}, v = -\frac{\partial \psi}{\partial x}$；
  - 证明 $\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = \frac{\partial^2 \psi}{\partial x \partial y} - \frac{\partial^2 \psi}{\partial y \partial x} \equiv 0$（由偏导数可交换性严格恒等于 0，无需将其放入 Loss 软惩罚）。

---

### 第 3 轮：负向排除与隐式神经表示前沿
- 📋 **用户输入（复制使用）**：
  ```text
  请评估该方案在处理高雷诺数湍流时的理论瓶颈。注意：绝对不要提传统的有限元（FEM）或自适应网格细化（AMR）方案，只谈隐式神经表示本身的优化方向
  ```
- 🔍 **后台状态机预期**：
  - 调度智能体：`TutorAgent`
  - 负向抑制识别：`negative_constraints: ['有限元', 'FEM', '网格细化', 'AMR']`
  - 记忆状态通道：`INHERIT`
- 🎯 **检验要点**：
  - 系统严格排除网格法，聚焦于高频谱偏差（Spectral Bias）、Fourier 特征映射（Fourier Feature Mapping）、时空因果加权与物理自适应配点采样。
