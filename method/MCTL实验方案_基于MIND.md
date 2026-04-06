# MCTL 渐进式消融实验方案（基于现有代码）

## 一、现有代码基础

当前项目已实现 **MIND 框架**（`models/news_fact_checker.py`），包含三个阶段：

| 现有模块 | 功能 | 对应代码方法 |
|---------|------|------------|
| **SSR** (Similar Sample Retrieval) | CLIP 嵌入检索 Top-K 相似训练样本 | `ssr_similar_sample_retrieval()` |
| **RID** (Relevant Insight Derivation) | 基于相似样本迭代生成正/反向规则 | `rid_relevant_insight_derivation()` |
| **IAI** (Insight-Augmented Inference) | 双辩论者 + 法官的多智能体推理 | `iai_insight_augmented_inference()` |

**现有数据管线**: `Input -> CLIP嵌入 -> SSR检索Top-K -> RID生成规则 -> IAI(辩论者x2+法官) -> 输出`

**API 配置**: `gemini-2.5-flash-lite` 通过 `yunwu.ai` 代理，支持多模态（文本+图像base64）

**数据格式**: JSONL，每条含 `index, image, text, labels`，标签为 `"harmful"/"not harmful"`

---

## 二、MCTL 模块与现有代码的映射关系

| MCTL 模块 | 现有代码可复用部分 | 需新增/修改的部分 |
|-----------|------------------|-----------------|
| **CAMR** (上下文增强多模态检索) | SSR 的 CLIP 嵌入 + Top-K 检索逻辑可直接复用 | 需要将检索结果组装为结构化上下文包（文本+标签+相似度），而非仅传索引 |
| **MCP** (元认知规划) | 无 | 新增一个 LLM 调用：输入待检测文本+检索上下文 -> 输出 JSON 格式的分析维度 DAG |
| **ATR** (自适应工具路由) | 无（RID 的规则生成是不同逻辑） | 新增 8 个工具函数（本质是特化 prompt 的 LLM 调用），按 DAG 调度执行 |
| **MPRE** (多视角推理集成) | IAI 的法官角色有类似逻辑 | 改为收集所有工具的 reasoning_trace + confidence，由 LLM 综合而非简单投票 |
| **CBDF** (基于共识的决策融合) | 无 | 新增最终决策层：综合 MPRE 摘要 + 检索标签分布 -> 输出预测 + 置信度 |

---

## 三、实验阶段设计（6 个阶段）

### Exp0: Baseline -- Direct LLM Prompting

> **目的**: 建立性能下界，验证 LLM 裸推理的能力天花板

**实现方式**: 新建一个简单的推理方法，跳过 SSR/RID，直接将文本+图像发给 LLM

**代码修改点**:
- 在 `MCTLChecker` 中新增 `run_exp0()` 方法
- 新增 prompt 模板 `EXP0_DIRECT_PROMPT`
- 复用现有 `fetch_api()` 和批量异步处理逻辑

**Prompt 设计**:
```
你是一个专业的新闻真实性分析师。请判断以下新闻内容是否为虚假新闻（有害信息）。

新闻文本："{text}"
（附带新闻配图）

请分析这条新闻的真实性，并给出判断。

你的输出必须严格遵循以下格式：
"思考：[分析这条新闻的内容特征、语言风格、逻辑是否合理等。]
答案：[有害/无害]。"
```

**Pipeline**: `Input -> LLM Direct Prediction -> Output`

**每个样本 API 调用次数**: 1

---

### Exp1: + CAMR（上下文增强多模态检索）

> **目的**: 验证检索增强的作用（Exp1 vs Exp0 -> **CAMR 贡献**）

**实现方式**: 复用现有 SSR 检索逻辑，但不进入 RID，而是将检索结果直接作为上下文注入 prompt

**代码修改点**:
- 新增 `run_exp1()` 方法
- 复用 `ssr_similar_sample_retrieval()` 获取 Top-K 相似样本
- 新增 `_build_camr_context()` 方法：将检索到的训练样本组装为结构化上下文字符串
- 新增 prompt 模板 `EXP1_CAMR_PROMPT`

**_build_camr_context() 逻辑**:
```python
def _build_camr_context(self, ssr_item, train_df):
    """将 SSR 检索结果组装为结构化上下文"""
    context_parts = []
    for i, (sample_idx, score) in enumerate(zip(ssr_item['samples'], ssr_item['scores'])):
        row = train_df.iloc[sample_idx]
        label = "有害（虚假）" if row['label'] == 1 else "无害（真实）"
        context_parts.append(
            f"[相似样本{i+1}] (相似度: {score:.3f})\n"
            f"  文本: {row['text'][:200]}...\n"
            f"  判定: {label}"
        )
    return "\n".join(context_parts)
```

**Prompt 设计**:
```
你是一个专业的新闻真实性分析师。请参考以下检索到的相似新闻样本及其判定结果，判断目标新闻是否为虚假新闻。

=== 检索到的相似新闻样本 ===
{camr_context}

=== 待判断的目标新闻 ===
新闻文本："{text}"
（附带新闻配图）

请结合相似样本的信息，分析目标新闻的真实性。

你的输出必须严格遵循以下格式：
"思考：[参考相似样本，分析目标新闻的内容特征、与已知真实/虚假新闻的相似之处。]
答案：[有害/无害]。"
```

**Pipeline**: `Input -> SSR(CLIP检索Top-K) -> 组装上下文 -> LLM + Context Prediction -> Output`

**每个样本 API 调用次数**: 1（SSR 是本地计算，不调 API）

**关键超参**: `SSR_K`（检索数量，后续做参数敏感性分析）

---

### Exp2: + MCP（元认知规划）

> **目的**: 验证元认知规划的作用（Exp2 vs Exp1 -> **MCP 贡献**）

**实现方式**: 在 Exp1 的检索上下文基础上，先让 LLM 规划分析维度，再按规划的维度逐一分析

**代码修改点**:
- 新增 `run_exp2()` 方法
- 新增 `_mcp_plan()` 方法：LLM 生成分析计划
- 新增 `_mcp_analyze()` 方法：LLM 按计划逐维度分析并给出最终判断
- 新增 prompt 模板 `EXP2_MCP_PLAN_PROMPT` 和 `EXP2_MCP_ANALYZE_PROMPT`

**MCP 规划 Prompt**:
```
你是一个元认知规划器。给定一条待检测新闻和检索到的相似新闻上下文，请诊断该新闻需要从哪些维度进行分析，并规划分析顺序。

=== 检索上下文 ===
{camr_context}

=== 待检测新闻 ===
文本："{text}"

请识别该新闻的关键分析维度（从以下维度中选择最相关的 3-5 个）：
- 事实核查：文本中的事实声明是否与检索到的信息一致
- 图文一致性：图像与文本描述是否匹配
- 修辞分析：是否使用夸张、煽动性或情绪操纵的语言
- 语义解构：是否存在隐含义、双关或误导性表述
- 情感操纵：是否利用恐惧、愤怒等情绪引导读者
- 证据对比：与检索到的相似样本相比，内容一致性如何
- 来源可信度：内容的表述方式是否符合专业新闻报道标准

输出 JSON 格式（不要输出其他内容）：
{"dimensions": ["维度1", "维度2", ...], "reason": "选择这些维度的原因"}
```

**MCP 分析 Prompt**:
```
你是一个专业的新闻真实性分析师。请按照以下规划的分析维度，逐一分析目标新闻。

=== 检索上下文 ===
{camr_context}

=== 待检测新闻 ===
文本："{text}"
（附带新闻配图）

=== 规划的分析维度 ===
{planned_dimensions}

请按照上述维度逐一分析，最后综合所有维度给出最终判断。

你的输出必须严格遵循以下格式：
"维度分析：
[维度1]：[分析内容]
[维度2]：[分析内容]
...
综合判断：[综合以上维度的分析结果]
答案：[有害/无害]。"
```

**Pipeline**: `Input -> SSR检索 -> MCP规划(LLM调用1) -> MCP按维度分析(LLM调用2) -> Output`

**每个样本 API 调用次数**: 2（规划 1 次 + 分析 1 次）

---

### Exp3: + ATR（自适应工具路由）

> **目的**: 验证专门化工具 vs 通用 LLM 分析的优势（Exp3 vs Exp2 -> **ATR 贡献**）

**实现方式**: 将 Exp2 中 LLM 的"自己逐维度分析"替换为调用专门的 LLM-based 工具（特化 prompt），按 MCP 规划的 DAG 调度执行

**代码修改点**:
- 新增 `tools/` 目录，包含 8 个工具模块
- 新增 `run_exp3()` 方法
- 新增 `_atr_dispatch()` 方法：按 MCP 规划的维度映射到对应工具并执行
- 每个工具本质上是一个特化的 prompt + LLM 调用，输出标准化的 `{reasoning_trace, confidence_score, conclusion}`

**8 个工具与维度的映射**:

| MCP 维度 | 工具名称 | 核心 Prompt 逻辑 |
|---------|---------|-----------------|
| 事实核查 | `knowledge_grounding` | 对比文本事实声明与检索上下文，判断一致性，输出验证结论+置信度 |
| 图文一致性 | `cross_modal_aligner` | 分析图像内容（通过描述）与文本的语义匹配度，输出对齐评分+理由 |
| 修辞分析 | `rhetorical_scanner` | 识别反讽、夸张、煽动性语言，评估修辞操纵程度，输出分析+置信度 |
| 语义解构 | `semantic_dissector` | 分析文本字面义与隐含义，识别误导性表述，输出解构分析+置信度 |
| 情感操纵 | `emotional_manipulator` | 解析情绪操纵元素（恐惧、愤怒等），评估操纵强度，输出分析+置信度 |
| 证据对比 | `evidence_comparator` | 比对待检测新闻与检索样本的标签偏移，输出证据分析+置信度 |
| 来源可信度 | `expectation_deviator` | 分析内容的专业性和可信度特征，评估与正规新闻的偏离度，输出分析+置信度 |
| 综合不一致性 | `inconsistency_amplifier` | 综合前序工具输出，量化各维度之间的矛盾点，输出矛盾分析+置信度 |

**工具统一输出格式**（JSON）:
```json
{
  "tool_name": "工具名称",
  "reasoning_trace": "详细推理过程",
  "conclusion": "该维度的结论",
  "confidence": 0.85
}
```

**工具 Prompt 模板示例**（以 `knowledge_grounding` 为例）:
```
你是一个事实核查专家工具。你的任务是验证新闻文本中的事实声明与检索到的相关信息是否一致。

=== 检索到的相关信息 ===
{camr_context}

=== 待核查新闻 ===
文本："{text}"

请执行以下步骤：
1. 提取新闻中的关键事实声明（人物、事件、时间、地点、数据等）
2. 将每个事实声明与检索到的相关信息进行比对
3. 标注哪些事实得到证实、哪些存在矛盾、哪些无法验证

你的输出必须严格遵循以下 JSON 格式（不要输出其他内容）：
{"reasoning_trace": "你的详细分析过程", "conclusion": "事实核查的结论", "confidence": 0.xx}
```

**ATR 调度逻辑**:
```python
async def _atr_dispatch(self, text, image_path, camr_context, planned_dimensions):
    """按 MCP 规划的维度调度工具"""
    DIMENSION_TO_TOOL = {
        "事实核查": "knowledge_grounding",
        "图文一致性": "cross_modal_aligner",
        "修辞分析": "rhetorical_scanner",
        "语义解构": "semantic_dissector",
        "情感操纵": "emotional_manipulator",
        "证据对比": "evidence_comparator",
        "来源可信度": "expectation_deviator",
    }
    # 并行执行无依赖的工具
    tool_tasks = []
    for dim in planned_dimensions:
        tool_name = DIMENSION_TO_TOOL.get(dim)
        if tool_name:
            tool_tasks.append(self._run_tool(tool_name, text, image_path, camr_context))

    tool_results = await asyncio.gather(*tool_tasks)

    # 短路机制：如果某工具 confidence > 0.95，跳过 inconsistency_amplifier
    high_conf = [r for r in tool_results
                 if isinstance(r.get('confidence'), (int, float)) and r['confidence'] > 0.95]
    if not high_conf:
        incon_result = await self._run_tool(
            "inconsistency_amplifier", text, image_path,
            camr_context, prior_results=tool_results)
        tool_results.append(incon_result)

    return tool_results
```

**Pipeline**: `Input -> SSR检索 -> MCP规划 -> ATR工具调度(并行执行) -> 取最高置信度工具结论 -> Output`

**每个样本 API 调用次数**: 2 + N（规划1 + N个工具各1，N = MCP 选择的维度数，通常 3-5）

**注意**: 此阶段最终判断暂时取 **最高置信度工具的结论**，或取所有工具结论的多数投票

---

### Exp4: + MPRE（多视角推理集成）

> **目的**: 验证多视角融合的作用（Exp4 vs Exp3 -> **MPRE 贡献**）

**实现方式**: 不再简单取最高置信度工具结论，而是让 LLM 综合所有工具的推理轨迹进行证据融合

**代码修改点**:
- 新增 `_mpre_synthesize()` 方法
- 新增 prompt 模板 `EXP4_MPRE_PROMPT`

**MPRE Prompt**:
```
你是一个证据综合分析专家。以下是多个专业分析工具对同一条新闻的分析结果。请综合所有工具的推理轨迹，识别共识和冲突，给出最终判断。

=== 待判断新闻 ===
文本："{text}"

=== 各工具分析结果 ===
{tool_results_formatted}

请执行以下步骤：
1. 识别共识证据：多个工具共同指出的特征（如多工具均发现图文冲突）
2. 仲裁冲突证据：如果工具间结论矛盾，分析哪个更可信并解释理由
3. 对各工具的分析按置信度加权，生成综合判断

你的输出必须严格遵循以下格式：
"共识证据：[多工具共同发现的关键证据]
冲突仲裁：[工具间矛盾的处理]
综合判断：[加权综合后的分析结论]
答案：[有害/无害]。"
```

**Pipeline**: `Input -> SSR -> MCP -> ATR工具执行 -> MPRE证据融合(LLM调用) -> Output`

**每个样本 API 调用次数**: 3 + N（规划1 + N个工具 + MPRE融合1）

---

### Exp5: + CBDF（基于共识的决策融合）-- 完整 MCTL

> **目的**: 验证标签先验 + 不确定性校准的作用（Exp5 vs Exp4 -> **CBDF 贡献**）

**实现方式**: 在 MPRE 证据摘要基础上，引入检索样本的标签分布作为先验信号，并加入不确定性感知决策

**代码修改点**:
- 新增 `_cbdf_decide()` 方法
- 新增 `_compute_label_distribution()` 方法：统计检索样本的标签分布
- 新增 prompt 模板 `EXP5_CBDF_PROMPT`

**_compute_label_distribution() 逻辑**:
```python
def _compute_label_distribution(self, ssr_item, train_df):
    """计算检索样本的标签分布"""
    labels = [train_df.iloc[idx]['label'] for idx in ssr_item['samples']]
    harmful_count = sum(1 for l in labels if l == 1)
    total = len(labels)
    return {
        "harmful_ratio": harmful_count / total if total > 0 else 0,
        "harmless_ratio": (total - harmful_count) / total if total > 0 else 0,
        "total_retrieved": total
    }
```

**CBDF Prompt**:
```
你是最终决策裁判。请基于以下证据摘要和统计先验信息，做出最终判断。

=== 待判断新闻 ===
文本："{text}"

=== 多视角证据摘要（来自 MPRE） ===
{mpre_summary}

=== 检索样本的标签分布（统计先验） ===
在检索到的 {total_retrieved} 个相似样本中：
- 有害（虚假）样本占比：{harmful_ratio}
- 无害（真实）样本占比：{harmless_ratio}

决策规则：
1. 如果证据摘要明确指出虚假新闻特征，且检索样本多数为有害 -> 强有害信号
2. 如果证据摘要明确指出真实新闻特征，且检索样本多数为无害 -> 强无害信号
3. 如果证据摘要与标签分布出现矛盾，需特别审慎分析
4. 如果各维度分析的置信度分歧较大（高低差 > 30%），请标注"低置信度"

你的输出必须严格遵循以下格式：
"决策分析：[综合证据摘要和标签先验的分析过程]
置信度评估：[高置信度/中置信度/低置信度]，理由：[...]
答案：[有害/无害]。"
```

**Pipeline**: `Input -> SSR -> MCP -> ATR -> MPRE -> CBDF -> Output（完整 MCTL）`

**每个样本 API 调用次数**: 4 + N（规划1 + N个工具 + MPRE融合1 + CBDF决策1）

---

## 四、代码实现规划

### 文件结构变更

```
fakenews/
├── config/
│   ├── prompt_config.py        # 新增 Exp0-5 的 prompt 模板
│   └── mctl_config.py          # 新增 MCTL 相关配置（维度列表、工具映射等）
├── models/
│   ├── news_fact_checker.py    # 保持原有 MIND 流程不变
│   └── mctl_checker.py         # 新增 MCTL 检测器（包含 Exp0-5 的运行方法）
├── tools/                       # 新增工具目录
│   ├── __init__.py
│   ├── base_tool.py            # 工具基类（统一接口）
│   ├── knowledge_grounding.py
│   ├── cross_modal_aligner.py
│   ├── rhetorical_scanner.py
│   ├── semantic_dissector.py
│   ├── emotional_manipulator.py
│   ├── evidence_comparator.py
│   ├── expectation_deviator.py
│   └── inconsistency_amplifier.py
├── main_mctl.py                 # 新增 MCTL 实验入口
```

### MCTLChecker 类设计概要

```python
class MCTLChecker:
    """MCTL 框架检测器，支持 Exp0-Exp5 各阶段独立运行"""

    def __init__(self):
        self.embedding_computer = EmbeddingComputer()
        # 复用现有的异步 API 调用、批量处理、断点续传逻辑

    # === 核心模块方法 ===
    def camr_retrieve(self, test_texts, test_images, train_df, dataset_type, k):
        """CAMR: 基于 CLIP 的上下文增强检索（复用 SSR 逻辑）"""

    async def mcp_plan(self, text, camr_context):
        """MCP: 元认知规划，返回分析维度列表"""

    async def atr_dispatch(self, text, image_path, camr_context, dimensions):
        """ATR: 按维度调度工具并行执行"""

    async def mpre_synthesize(self, text, tool_results):
        """MPRE: 多视角推理集成"""

    async def cbdf_decide(self, text, mpre_summary, label_distribution):
        """CBDF: 基于共识的决策融合"""

    # === 实验运行方法 ===
    async def run_exp0(self, test_data_path, dataset_type):
        """Exp0: Direct LLM"""

    async def run_exp1(self, test_data_path, train_data_path, dataset_type):
        """Exp1: + CAMR"""

    async def run_exp2(self, test_data_path, train_data_path, dataset_type):
        """Exp2: + CAMR + MCP"""

    async def run_exp3(self, test_data_path, train_data_path, dataset_type):
        """Exp3: + CAMR + MCP + ATR"""

    async def run_exp4(self, test_data_path, train_data_path, dataset_type):
        """Exp4: + CAMR + MCP + ATR + MPRE"""

    async def run_exp5(self, test_data_path, train_data_path, dataset_type):
        """Exp5: Full MCTL"""
```

### 关键实现注意事项

1. **复用现有基础设施**: `fetch_api()`、`EmbeddingComputer`、`compute_similarity_scores()`、`extract_top_k_similar()` 等均可直接复用
2. **断点续传**: 每个实验阶段的结果保存到 `results/{dataset}/{exp_name}/` 目录下，支持中断后继续
3. **结果保存路径**: `results/{dataset}/exp0/`, `results/{dataset}/exp1/`, ..., `results/{dataset}/exp5/`
4. **并行调度**: Exp3-5 中 ATR 的无依赖工具应使用 `asyncio.gather()` 并行执行（已在代码示例中体现）
5. **工具输出的 JSON 解析**: LLM 返回的 JSON 可能不规范，需要鲁棒的解析逻辑（try/except + 正则兜底）
6. **答案提取**: 所有实验统一使用 `_find_answer_position()` 将"有害/无害"映射为 0/1

---

## 五、实验结果表格设计

### 表1: 渐进式消融实验（主实验）

> 在 **所有数据集** 上各跑一遍

| 实验 | 模块组合 | API调用/样本 | PolitiFact Acc | PolitiFact F1 | GossipCop Acc | GossipCop F1 | Weibo21 Acc | Weibo21 F1 | Weibo Acc | Weibo F1 | 平均DF1 |
|------|---------|-------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| Exp0 | Direct LLM | 1 | - | - | - | - | - | - | - | - | -- |
| Exp1 | + CAMR | 1 | - | - | - | - | - | - | - | - | - |
| Exp2 | + MCP | 2 | - | - | - | - | - | - | - | - | - |
| Exp3 | + ATR | 2+N | - | - | - | - | - | - | - | - | - |
| Exp4 | + MPRE | 3+N | - | - | - | - | - | - | - | - | - |
| Exp5 | Full MCTL | 4+N | - | - | - | - | - | - | - | - | - |

### 表2: 对比实验（vs SOTA）

| Method | PolitiFact Acc/F1 | GossipCop Acc/F1 | Weibo21 Acc/F1 | Weibo Acc/F1 |
|--------|----------|----------|----------|----------|
| EANN | - | - | - | - |
| SAFE | - | - | - | - |
| HMCAN | - | - | - | - |
| LEMMA | - | - | - | - |
| MIND (现有Baseline) | - | - | - | - |
| **MCTL (Ours)** | - | - | - | - |

> MIND 结果可直接从现有代码运行获取，作为强 baseline

### 表3: 参数敏感性分析（检索数 K）

> 固定使用完整 MCTL (Exp5)，在 PolitiFact 上变化 K

| K | 1 | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|----|----|
| Accuracy | - | - | - | - | - | - |
| Macro-F1 | - | - | - | - | - | - |

### 表4: LLM 底座鲁棒性

> 固定使用完整 MCTL (Exp5)，在 PolitiFact 上替换 LLM

| LLM Base | gemini-2.5-flash-lite (默认) | DeepSeek-R1 | GPT-4o-mini | Qwen-2.5 |
|----------|-------------|--------|----------|---------|
| Accuracy | - | - | - | - |
| Macro-F1 | - | - | - | - |

> 只需修改 `api_config.py` 中的 `API_MODEL`，其他代码不变（因为 API 使用 OpenAI 兼容格式）

### 表5: 推理效率分析

| 实验 | 平均推理时间/样本(s) | API调用次数/样本 | 总API调用次数 | 工具并行率 |
|------|---------------------|-----------------|--------------|-----------|
| Exp0 | - | 1 | 1 | -- |
| Exp1 | - | 1 | 1 | -- |
| Exp2 | - | 2 | 2 | -- |
| Exp3 | - | 2+N | ~5-7 | 可并行无依赖工具 |
| Exp4 | - | 3+N | ~6-8 | 同上 |
| Exp5 | - | 4+N | ~7-9 | 同上 |

---

## 六、Case Study: 可解释性分析

选取 2 个典型样本（一真一假），在完整 MCTL (Exp5) 下展示完整 pipeline：

### 样本选取标准
- 选择 Exp0 判错但 Exp5 判对的样本（体现框架价值）
- 一个英文（PolitiFact）+ 一个中文（Weibo21）

### 展示内容
1. **原始输入**: 新闻文本 + 配图描述
2. **CAMR 输出**: 检索到的 Top-5 相似样本（文本摘要 + 标签 + 相似度）
3. **MCP 输出**: 规划的分析维度 JSON + 选择理由
4. **ATR 输出**: 每个被调用工具的 reasoning_trace + confidence
5. **MPRE 输出**: 共识证据 + 冲突仲裁 + 综合摘要
6. **CBDF 输出**: 最终判断 + 置信度 + 决策理由链

---

## 七、实验执行顺序

### 阶段一：基础验证（优先级最高）
1. **实现并运行 Exp0**（约1天）-- 确认 API 调通、数据格式正确、指标计算正常
2. **实现并运行 Exp1**（约1天）-- 验证 CAMR 检索增强是否有效，这是整个框架的基石
3. 先在 **PolitiFact**（仅 104 样本）上快速验证，确认流程无误后再扩展到其他数据集

### 阶段二：核心模块（优先级高）
4. **实现并运行 Exp2**（约1天）-- 验证 MCP 规划是否优于直接分析
5. **实现并运行 Exp3**（约2-3天）-- 工作量最大（8 个工具 prompt 设计 + 调度逻辑），逐个工具实现和验证

### 阶段三：融合决策（优先级中）
6. **实现并运行 Exp4 和 Exp5**（约1-2天）

### 阶段四：补充实验
7. 跑其余 3 个数据集的 Exp0-Exp5
8. 参数敏感性分析（K 值）
9. LLM 底座鲁棒性
10. 推理效率统计
11. Case Study 选取和可视化

### 调优原则
> **如果某一步加入模块后效果没有提升，应在进入下一步之前先调优该模块**（调 prompt、调参数），确保每个模块都能带来正向贡献后再继续叠加。常见调优手段：
> - 调整 prompt 措辞和结构
> - 调整检索数 K
> - 调整工具的置信度阈值
> - 调整 MCP 的候选维度列表

---

## 八、各数据集说明

| 数据集 | 语言 | 测试集大小 | 训练集大小 | 领域 | 预计 Exp5 API调用量 |
|--------|------|-----------|-----------|------|---------------------|
| PolitiFact | 英文 | 104 | 381 | 政治新闻 | 低（约 700-900 次） |
| GossipCop | 英文 | 2,830 | 10,010 | 娱乐新闻 | 高（约 2万+ 次） |
| Weibo21 | 中文 | 615 | 4,926 | 社交媒体 | 中（约 4,300-5,500 次） |
| Weibo | 中文 | 2,453 | 9,739 | 社交媒体 | 高（约 1.7万+ 次） |

> **建议**: 开发和调优阶段全部在 PolitiFact 上进行（样本最少），确认无误后再批量跑其他数据集。GossipCop 和 Weibo 样本量大，应充分利用现有的 `MAX_CONCURRENCY` 和 `BATCH_SIZE` 配置加速。

---

## 九、与原 MIND 框架的关系

MCTL 是 MIND 的进化版本，核心区别：

| 对比维度 | MIND (现有) | MCTL (新方案) |
|---------|-------------|--------------|
| 检索后处理 | 用相似样本迭代生成规则 (RID) | 将检索结果作为上下文直接注入 |
| 分析策略 | 固定流程（正/反规则 -> 辩论） | 元认知规划，自适应选择分析维度 |
| 分析执行 | 通用 prompt（辩论者） | 8 个专门化工具（特化 prompt） |
| 决策方式 | 辩论者投票/法官裁决 | 多视角证据融合 + 标签先验校准 |
| 可解释性 | 有思考过程但不结构化 | DAG 规划 + 工具轨迹 + 决策链 |

> 论文中 MIND 作为一个重要的对比方法（Baseline），MCTL 需要在 MIND 基础上展示明显提升。
