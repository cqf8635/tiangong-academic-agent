# 🌌 六大学科多轮科研研讨全集测试题库 (Multi-Disciplinary Benchmark Suites)

本套题库涵盖 **大模型、深度学习、自然语言处理、哲学、相对论、天文学** 六大核心领域。
每一组题库均严格设计为包含 **5 轮深度多轮递进**：
- **第 1 轮**：领域立项/文献检索/第一性原理发起 (`INIT`)
- **第 2 轮**：数理推导与机理公式深入 (`INHERIT`)
- **第 3 轮**：经典主流理论横向对比 (`compare_contrast` | `INHERIT`)
- **第 4 轮**：追加新方法/新学派多向权衡 (`APPEND`)
- **第 5 轮**：外科手术式基于前序对比，设计严密科研方案 (`research_design` | `INHERIT`)

---

## 目录
1. [领域一：大语言模型（LLM）· 测试时计算与长上下文](#领域一大语言模型llm)
2. [领域二：深度学习（Deep Learning）· 状态空间模型与现代架构](#领域二深度学习deep-learning)
3. [领域三：自然语言处理（NLP）· 模型编辑与幻觉机理](#领域三自然语言处理nlp)
4. [领域四：心灵哲学（Philosophy）· 现象意识与解释鸿沟](#领域四心灵哲学philosophy)
5. [领域五：相对论（Relativity）· 广义相对论与黑洞奇点](#领域五相对论relativity)
6. [领域六：天文学（Astronomy）· 快射电暴与多信使天体物理](#领域六天文学astronomy)

---

## 领域一：大语言模型（LLM）

### 剧本定位：测试时计算扩展（Test-Time Compute）与推理架构演进
- **第 1 轮（文献检索与前沿追踪）**：
  > 请检索近两年关于大语言模型测试时计算扩展（Test-Time Compute Scaling）与自适应长链条推理思考（Reasoning Models）的代表性最新论文，说明主要方法和贡献。
  *预期*：`intent: arxiv_paper_search`, `subject_action: INIT`, 触发 2024-2026 时间窗口在线检索。

- **第 2 轮（数理机制深入）**：
  > 详细解释测试时计算中的蒙特卡洛树搜索（MCTS）与过程奖励模型（PRM）是如何在数学上评估逐步生成置信度与价值估计的？
  *预期*：`intent: concept_simplify` 或 `stepwise_derivation`, `subject_action: INHERIT`。

- **第 3 轮（多方案对比）**：
  > 比较隐式思考链采样扩展（如多次生成多数投票）与显式搜索解码（Search-based Decoding）在计算预算、探索多样性与语义漂移方面的差异。
  *预期*：`intent: compare_contrast`, `subject_action: INHERIT`。

- **第 4 轮（追加新机制对比）**：
  > 再加入投机解码（Speculative Decoding）和注意力 KV 缓存动态压缩策略，重新评估它们在长序列高吞吐推理下的优缺点与适用场景。
  *预期*：`intent: compare_contrast`, `subject_action: APPEND`。

- **第 5 轮（基于前序对比设计研究方案）**：
  > 基于前面的比较，设计一个面向超长数学证明场景的低延迟自适应混合推理方案，包含假设、动态计算调度流程、基线对照、可观测量与验证标准。
  *预期*：`intent: research_design`, 0 毫秒外科手术式锁定第 4 轮对比定论，输出 2000~3000 字严密科研方案。

---

## 领域二：深度学习（Deep Learning）

### 剧本定位：从连续动力系统推导 Mamba/SSM 架构并设计混合层
- **第 1 轮（第一性原理推导）**：
  > 请从连续时间线性时不变动力学系统出发，推导状态空间模型（SSM）到零阶保持（ZOH）离散化的严密数学映射公式。
  *预期*：`intent: stepwise_derivation`, `subject_action: INIT`, LaTeX 推导矩阵指数 $e^{\Delta A}$。

- **第 2 轮（算法演进与硬件加速）**：
  > 写出带有输入依赖的选择性状态空间（S6 / Mamba）时变状态方程，证明其线性时间复杂度与硬件感知并行关联扫描（Parallel Scan）的加速原理。
  *预期*：`intent: stepwise_derivation`, `subject_action: INHERIT`。

- **第 3 轮（架构对比）**：
  > 对比 Mamba 状态空间架构与经典自注意力 Transformer 在长程联想记忆（Associative Recall）与模态外推上的理论极限与优缺点。
  *预期*：`intent: compare_contrast`, `subject_action: INHERIT`。

- **第 4 轮（追加新型注意力变体）**：
  > 再追加门控线性注意力（Gated Linear Attention, GLA）和循环记忆机制（Recurrent Attention），重新比较它们与 Mamba、标准 Transformer 的综合权衡。
  *预期*：`intent: compare_contrast`, `subject_action: APPEND`。

- **第 5 轮（方案设计）**：
  > 基于前面的比较，设计一个融合状态空间层与稀疏自注意力的多模态超长序列混合网络架构（Hybrid Backbone）实验方案，包含层间配比假设、流水线流程、基线模型与长文本评测验证标准。
  *预期*：`intent: research_design`, 锁定第 4 轮架构对比，生成实验设计。

---

## 领域三：自然语言处理（NLP）

### 剧本定位：大模型参数知识编辑（Model Editing）与幻觉溯源
- **第 1 轮（前沿顶会检索）**：
  > 检索近两年关于大语言模型参数化知识编辑（如 ROME、MEMIT）与事实定位纠错的代表性顶会论文，总结核心技术流派。
  *预期*：`intent: arxiv_paper_search`, `subject_action: INIT`。

- **第 2 轮（数理推导与闭式解）**：
  > 详细推导因果中介分析（Causal Tracing）与线性关联记忆编辑（ROME）在 MLP 权重上通过秩-1 约束最小二乘法求解的最优闭式解公式。
  *预期*：`intent: stepwise_derivation`, `subject_action: INHERIT`。

- **第 3 轮（技术路线对比）**：
  > 比较直接权重编辑方法（ROME/MEMIT）与检索增强生成（RAG）、记忆上下文编辑在局部性（Locality）、泛化性（Generalization）与灾难性遗忘方面的差异。
  *预期*：`intent: compare_contrast`, `subject_action: INHERIT`。

- **第 4 轮（追加表示工程）**：
  > 如果再引入表示工程（Representation Engineering）与隐藏层激活引导（Activation Steering），重新评估它们在抑制语义幻觉方面的鲁棒性与计算开销。
  *预期*：`intent: compare_contrast`, `subject_action: APPEND`。

- **第 5 轮（研究方案设计）**：
  > 基于前面的比较，设计一个在不损伤模型通用推理能力的前提下，实时修正医疗垂直领域过时实体知识的动态模型更新方案，包含核心假设、更新流程、对照基线与抗遗忘验证标准。
  *预期*：`intent: research_design`, 锁定对比定论，设计闭环验证。

---

## 领域四：心灵哲学（Philosophy）

### 剧本定位：现象意识、解释鸿沟与模态逻辑僵尸论证
- **第 1 轮（核心概念与问题界定）**：
  > 阐述心灵哲学中查尔默斯（David Chalmers）关于意识的“难问题”（The Hard Problem of Consciousness）的核心命题，以及物理主义所面临的解释鸿沟（Explanatory Gap）。
  *预期*：`intent: concept_simplify`, `subject_action: INIT`, 领域进入哲学与认知科学。

- **第 2 轮（思想实验横向对比）**：
  > 对比内格尔的“蝙蝠论证”（What is it like to be a bat?）与杰克逊的“知识论证”（黑白玛丽 Mary’s Room）在反驳还原论物理主义时的论证结构与认识论差异。
  *预期*：`intent: compare_contrast`, `subject_action: INHERIT`。

- **第 3 轮（模态逻辑形式化推演）**：
  > 利用二维语义学（Two-Dimensional Semantics）的模态逻辑框架，逐步推导哲学僵尸论证（Philosophical Zombie Argument）从认识可设想性（Epistemic Conceivability）到形而上学可能性（Metaphysical Possibility）的形式化论证链条。
  *预期*：`intent: stepwise_derivation`, `subject_action: INHERIT`, 严密模态符号推导。

- **第 4 轮（追加非主流哲学流派）**：
  > 追加引入丹尼特的幻相主义（Illusionism）与罗素式泛心论（Russellian Monism），重新比较它们消除解释鸿沟的哲学代价与理论自洽性。
  *预期*：`intent: compare_contrast`, `subject_action: APPEND`。

- **第 5 轮（哲学与认知科学交叉方案设计）**：
  > 基于前面的哲学比较，针对当前具备元认知自我反思行为的大语言模型，设计一个形式化检验其是否具备功能性“现象状态报告自洽性”的哲学-认知科学交叉研究方案，包含假设、操作性定义、可观测行为基线与反驳验证标准。
  *预期*：`intent: research_design`, 锁定哲学流派优缺点，输出交叉实证方案。

---

## 领域五：相对论（Relativity）

### 剧本定位：广义相对论引力场方程、黑洞奇点与无毛定理检验
- **第 1 轮（作用量变分推导）**：
  > 从爱因斯坦-希尔伯特作用量出发，利用度规逆变分原理逐步推导真空爱因斯坦引力场方程，并解释边界 Gibbons-Hawking-York（GHY）表面项的处理机理。
  *预期*：`intent: stepwise_derivation`, `subject_action: INIT`, 场论张量推导。

- **第 2 轮（精确解与坐标奇点推演）**：
  > 推导静态球对称史瓦西度规（Schwarzschild Metric），并计算克雷奇曼标量不变量（Kretschmann Scalar），证明 r = 2M 处仅为坐标奇点而 r = 0 处为物理真实奇点。
  *预期*：`intent: stepwise_derivation`, `subject_action: INHERIT`。

- **第 3 轮（黑洞流派对比）**：
  > 比较静态史瓦西黑洞与旋转克尔黑洞（Kerr Metric）在视界结构、能层（Ergosphere）以及彭罗斯过程（Penrose Process）能量提取效率上的差异。
  *预期*：`intent: compare_contrast`, `subject_action: INHERIT`。

- **第 4 轮（追加极值与热力学机制）**：
  > 再加入带电克尔-纽曼黑洞与极值黑洞（Extremal Black Hole），重新比较不同黑洞度规在霍金辐射温度、视界面积与黑洞热力学定律上的异同。
  *预期*：`intent: compare_contrast`, `subject_action: APPEND`。

- **第 5 轮（引力波检验方案设计）**：
  > 基于前面的度规与视界分析，设计一个利用空间引力波天文台准简正模（Quasi-Normal Modes, QNM）频谱检验黑洞无毛定理（No-hair Theorem）的数值相对论计算方案，包含度规微扰假设、波形振荡提取流程、基线波形与偏差验证标准。
  *预期*：`intent: research_design`, 锁定黑洞度规定论，输出 2500 字方案。

---

## 领域六：天文学与天体物理（Astronomy & Astrophysics）

### 剧本定位：快射电暴（FRB）起源、磁星耀斑与多信使联动
- **第 1 轮（前沿观测文献检索）**：
  > 检索近两年关于快射电暴（Fast Radio Bursts, FRBs）起源机制与银河系内磁星（Magnetar）X射线耀斑关联的代表性最新天文观测论文，说明主要观测突破与证据。
  *预期*：`intent: arxiv_paper_search`, `subject_action: INIT`。

- **第 2 轮（天体辐射公式推导）**：
  > 推导快射电暴的色散量（Dispersion Measure, DM）与星系际介质（IGM）自由电子积分列密度的数学关系（Macquart 关系），说明如何用它约束宇宙可见重子物质分布。
  *预期*：`intent: stepwise_derivation`, `subject_action: INHERIT`。

- **第 3 轮（辐射机制模型对比）**：
  > 对比磁星磁层内部受激发射模型（Magnetospheric Maser）与远端相对论激波模型（Relativistic Blast Waves）在解释超精细微结构偏振角突变时的优缺点。
  *预期*：`intent: compare_contrast`, `subject_action: INHERIT`。

- **第 4 轮（追加多信使物理源）**：
  > 追加考虑双中子星并合（BNS Mergers）产生的引力波多信使千新星（Kilonova）事件，重新对比不同高能天体爆发源在射电暂现源光变曲线衰减上的特征差异。
  *预期*：`intent: compare_contrast`, `subject_action: APPEND`。

- **第 5 轮（天文观测联动方案设计）**：
  > 基于前面的理论模型与观测特征比较，设计一套结合大型单口径射电望远镜（如 FAST）与空间高能巡天望远镜的重复暴实时触发联动观测方案，包含多波段触发判据、采样分析流程、基线统计模型与物理源鉴别标准。
  *预期*：`intent: research_design`, 锁定辐射模型定论，输出高水准科研设计。
