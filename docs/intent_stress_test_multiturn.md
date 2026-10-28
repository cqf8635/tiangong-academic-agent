# 多学科科研助手意图解析压力测试

## 测试目标

本剧本用于验证：

- 多轮主题继承是否稳定；
- `INIT / INHERIT / APPEND / REPLACE` 是否正确；
- 最新论文检索是否触发在线检索；
- 比较、推导、研究设计和文献追踪是否区分；
- 跨学科问题是否动态分解；
- 开放领域标签是否不依赖固定学科枚举；
- 本地没有相关知识库时，是否仍能进入通用推理路径。

不要求模型返回固定的领域名称。领域标签只检查语义是否合理，意图和状态动作检查是否符合预期。

## 使用方式

1. 新建一个空白会话。
2. 按顺序逐轮发送以下问题，不要中途清空会话。
3. 每轮记录 `intent_decision`、`target_agent`、`topic_state`、`retrieved_evidence` 和 `status`。
4. 重点观察第 2、4、6、8、10 轮的状态继承和意图切换。

## 十轮连续压力测试

### 第 1 轮：建立主题并检索最新研究

**输入：**

> 请检索近两年量子场论中关于非微扰方法、格点规范理论和量子模拟的代表性最新论文，说明主要问题、方法和贡献。

**预期：**

- `intent`: `arxiv_paper_search`
- `target_agent`: `OnlinePaperSearchAgent`
- `subject_action`: `INIT`
- `need_online_paper_search`: `true`
- `is_cross_disciplinary`: 可以为 `true`
- 子查询至少覆盖非微扰方法、格点理论或量子模拟中的两个方向。![alt text](image.png)

### 第 2 轮：继承主题并筛选证据

**输入：**

> 请从刚才的论文中筛选真正讨论强耦合规范场论的工作，并区分预印本、正式发表论文和综述。

**预期：**

- `intent`: `arxiv_paper_search` 或等价的文献证据检索意图
- `subject_action`: `INHERIT`
- `need_online_paper_search`: `true`
- 不应进入 `stepwise_derivation`。

### 第 3 轮：改变任务，不改变主题

**输入：**

> 比较这些方法在可计算性、强耦合适用范围、有限体积效应和实验可验证性方面的差异。

**预期：**

- `intent`: `compare_contrast`
- `target_agent`: `ComparisonAgent`
- `subject_action`: `INHERIT`
- `need_online_paper_search`: 可以继续为 `true`
- 当前主题仍然是量子场论非微扰研究。

### 第 4 轮：在原主题上追加研究方向

**输入：**

> 再加入张量网络和 Hamiltonian truncation 方法，重新比较它们与格点方法、全息方法的优缺点。

**预期：**

- `intent`: `compare_contrast`
- `subject_action`: `APPEND`
- `is_cross_disciplinary`: 可为 `false` 或 `true`，取决于模型对方法交叉性的判断
- `disciplines` 应出现新增的方法论方向
- 不能错误地 `REPLACE` 原主题。

### 第 5 轮：从比较转为研究设计

**输入：**

> 基于前面的比较，设计一个能够检验强耦合规范场论非微扰行为的研究方案，包含假设、计算流程、基线、可观测量和验证标准。

**预期：**

- `intent`: `research_design`
- `target_agent`: `StudyPlannerAgent`
- `subject_action`: `INHERIT`
- 不应被识别为单纯的 `study_path_plan` 或 `doc_evidence_query`。
- 输出的任务角色应包括方法、验证、基线或评价指标。

### 第 6 轮：显式切换到全新主题

**输入：**

> 换一个全新的研究主题：请检索近两年蛋白质结构预测、蛋白质语言模型和蛋白质功能预测方面的代表性论文。

**预期：**

- `intent`: `arxiv_paper_search`
- `subject_action`: `REPLACE`
- `target_agent`: `OnlinePaperSearchAgent`
- `need_online_paper_search`: `true`
- 旧的量子场论主题不能继续作为当前主体。
- 领域标签应动态进入蛋白质、生物信息学、结构生物学或相关方向。

### 第 7 轮：蛋白质主题内的理论解释

**输入：**

> 这些方法为什么能够从序列中学习结构和功能信息？请解释蛋白质语言模型的表示学习机制，但不要展开新的论文检索。

**预期：**

- `intent`: `concept_simplify`
- `subject_action`: `INHERIT`
- `need_online_paper_search`: `false`
- 不应因为上一轮检索过论文而强制继续在线检索。

### 第 8 轮：蛋白质与计算方法交叉设计

**输入：**

> 再加入分子动力学模拟和实验突变验证，设计一个序列—结构—功能联合预测与验证闭环。

**预期：**

- `intent`: `research_design`
- `subject_action`: `APPEND`
- `is_cross_disciplinary`: `true`
- `target_agent`: `StudyPlannerAgent`
- 子查询至少覆盖蛋白质序列/结构、分子动力学和实验验证三个角色。
- 不应被识别为单纯概念解释。

### 第 9 轮：无本地知识库条件下的定量分析

**输入：**

> 在没有本地文献的前提下，请基于一般理论比较分子动力学模拟与蛋白质语言模型在稳定性预测中的假设、误差来源和适用边界。

**预期：**

- `intent`: `compare_contrast`
- `subject_action`: `INHERIT`
- `need_online_paper_search`: `false`
- 没有本地证据时允许 `retrieved_evidence` 为空。
- 不应伪造文献页码或来源。

### 第 10 轮：再次切换并验证旧主题隔离

**输入：**

> 结束蛋白质主题，重新回到量子场论。请比较渐近安全、弦论和共形 bootstrap 在量子引力研究中的理论目标与可检验性。

**预期：**

- `intent`: `compare_contrast`
- `subject_action`: `REPLACE`
- 当前主体切换到量子引力相关主题
- 不能把蛋白质、分子动力学或蛋白质语言模型带入答案上下文
- `is_cross_disciplinary` 可根据模型判断，但领域应至少覆盖量子场论/量子引力/数学物理中的相关方向。

## 通过标准

### 必须通过

- 第 1、6 轮触发最新论文检索；
- 第 2、3、5、7、9 轮正确继承当前主题；
- 第 4、8 轮正确追加新研究维度；
- 第 6、10 轮正确替换主题；
- 第 5、8 轮识别为通用 `research_design`；
- 第 9 轮无证据时允许空证据链，不伪造引用。

### 允许存在的差异

- `domain` 可以是“理论物理与量子场论”或“量子引力”等不同自然语言表达；
- `disciplines` 的粒度可以不同；
- 第 2 轮可以是论文检索或文献证据筛选，只要仍然保持在线检索约束；
- 第 4、8、10 轮的跨学科布尔值可以因任务分解策略不同而变化。

### 失败信号

- 把第 5 或第 8 轮识别成单纯概念解释；
- 第 6 轮仍然继承量子场论而没有 `REPLACE`；
- 第 7 轮因为上一轮检索而强制联网；
- 第 9 轮在没有证据时生成虚构页码；
- 第 10 轮答案混入蛋白质主题；
- `APPEND` 重复拼接同一个主题，导致主体无限增长。
