# GossipCop 数据集效果分析与改进方案

## 一、当前结果诊断

### 1.1 核心指标（Exp5 Full MCTL, K=5）

| 指标 | GossipCop | PolitiFact（对比） |
|------|-----------|-------------------|
| Accuracy | **0.7746** | 0.9126 |
| Fake-Precision | **0.4503** | 0.8846 |
| Fake-Recall | 0.6260 | 0.7931 |
| Fake-F1 | **0.5238** | 0.8364 |
| Real-F1 | 0.8545 | — |

**结论：GossipCop 比 PolitiFact 差 ~14% Accuracy，Fake-F1 差 ~31%。核心问题是 FP 过多（188个），即大量真实新闻被误判为虚假。**

### 1.2 五大根因分析

#### 根因1：数据集严重不平衡 + 领域特殊性
- GossipCop: **80.5% 真实 / 19.5% 虚假**（1014 real vs 246 fake）
- PolitiFact: 72.1% 真实 / 27.9% 虚假（相对平衡）
- **GossipCop 是娱乐/名人八卦新闻**，即使是真实新闻也天然使用煽动性、夸张性语言（"Next Level!", "BREAKING", "OMG"），这与 prompt 中定义的"虚假新闻特征"高度重叠

#### 根因2：工具级偏差严重
各工具在 GossipCop 上的准确率和 fake 预测偏向：

| 工具 | 准确率 | fake 预测率 | 使用次数 | 问题 |
|------|--------|-----------|---------|------|
| knowledge_grounding | 66.09% | 37.72% | 1103 | 最常用但准确率低 |
| evidence_comparator | 78.89% | 24.61% | 971 | 相对最好 |
| expectation_deviator | 75.32% | 23.38% | 770 | 尚可 |
| **rhetorical_scanner** | **60.33%** | **58.68%** | 242 | 近6成预测为fake，严重偏差 |
| **emotional_manipulator** | **58.06%** | **79.84%** | 124 | 80%预测为fake，完全失效 |
| inconsistency_amplifier | 41.18% | 58.82% | 17 | 准确率最低 |

**关键发现：rhetorical_scanner 和 emotional_manipulator 在娱乐新闻上几乎全部预测 fake，因为八卦新闻天然就是"煽动性+情绪化"的写作风格。**

#### 根因3：Consensus Bypass 机制错误传播
- 71.35% (899/1260) 样本走 bypass 通道
- Bypass 准确率: 84.87%，但 **136 个 bypass 错误占全部 284 错误的 47.9%**
- 当2+工具以>=0.5置信度一致判断时直接跳过 MPRE/CBDF，但工具本身就有系统性偏差

#### 根因4：检索标签噪声传播
- 真实新闻中有 126 个样本（12.4%）的 Top-5 检索结果多数为 fake
- 在这些样本中，**FP 率高达 51.59%**
- knowledge_grounding 和 evidence_comparator 严重依赖检索结果，错误的检索标签直接误导判断

#### 根因5：底层模型能力受限
- 使用 gemini-2.5-flash-lite，属于轻量模型
- 对于细粒度的真假新闻区分（尤其是娱乐领域），模型辨别能力不足

---

## 二、改进方案

### 方案A：Prompt 层面改进（最可行，改动最小）

#### A1. 针对 GossipCop 领域定制 Prompt（优先级最高）

**问题**：当前 prompt 中的"虚假新闻特征"对于娱乐新闻来说过于通用，导致真实八卦新闻因"煽动性语言"被误判。

**改进**：为 GossipCop 数据集添加领域感知的 prompt 模板。

```python
# 新增：GossipCop 专用 Baseline Prompt
BASELINE_PROMPT_GOSSIP = """You are a professional fake news detector specializing in celebrity and entertainment news.

IMPORTANT DOMAIN CONTEXT: You are analyzing celebrity/entertainment gossip news. In this domain:
- Sensational language ("OMG!", "Next Level!", "BREAKING") is NORMAL and does NOT indicate fake news
- Emotionally charged writing is the standard style for entertainment journalism
- Clickbait-style headlines are common even in legitimate entertainment outlets (E! News, People, TMZ, Us Weekly)
- The key distinction is whether the FACTUAL CLAIMS are true, not whether the writing style is dramatic

Focus your analysis on:
1. Are the specific factual claims verifiable? (names, dates, events, quotes)
2. Does the story cite or reference real, identifiable sources?
3. Are there internal logical contradictions in the story?
4. Is the core event/claim plausible given known context?

Do NOT use these as fake indicators for entertainment news:
- Dramatic or sensational language/headlines
- Emotional tone or exclamation marks
- Clickbait-style framing
- Celebrity gossip speculation (which is normal in this genre)

News text:
\"{text}\"

Your output must strictly follow this format:
\"Thinking: [Focus on verifying factual claims rather than judging writing style. Consider whether the core claims are plausible and consistent.]
Answer: [real/fake].\"
"""
```

#### A2. 修改 ATR 工具 Prompt，加入领域适配指令

**问题**：rhetorical_scanner (60.33% acc, 58.68% fake rate) 和 emotional_manipulator (58.06% acc, 79.84% fake rate) 在娱乐新闻上严重偏向 fake。

**改进**：在工具 prompt 中注入领域校准指令。

```python
# 在 TOOL_PROMPT_EN 模板中加入领域上下文占位符 {domain_context}
TOOL_PROMPT_EN_V2 = """You are acting as the "{tool_name}" -- a specialized analysis tool for fake news detection.

Your specific task: {tool_desc}

{domain_context}

Retrieved similar articles from verified database:
{retrieved_context}

{dependency_context}

Target news to analyze:
\"{text}\"

Perform your specialized analysis focused ONLY on your designated dimension. Be thorough but focused.

Your output must strictly follow this format:
\"Analysis: [Your detailed analysis focused on your specific dimension]
Confidence: [A number between 0.0 and 1.0 indicating how confident you are in your assessment.]
Judgment: [real/fake]\"
"""

# 领域上下文（在构建 prompt 时按数据集注入）
GOSSIP_DOMAIN_CONTEXT = """CRITICAL DOMAIN NOTE: This is celebrity/entertainment news. In this genre:
- Sensational language, clickbait headlines, and emotional writing are STANDARD editorial practices, even in credible outlets
- Do NOT treat dramatic writing style as evidence of fakeness
- Focus ONLY on factual accuracy, logical consistency, and source verifiability
- A story can be written in a highly dramatic, clickbait style and still be completely TRUE"""

# 非娱乐数据集可以用空字符串或通用提示
DEFAULT_DOMAIN_CONTEXT = ""
```

#### A3. MPRE Prompt 增加领域偏差校正指令

```python
# 在 MPRE prompt 末尾拼接（仅 GossipCop）
MPRE_GOSSIP_CALIBRATION = """
IMPORTANT CALIBRATION NOTE: For entertainment/celebrity news, some tools (especially
rhetorical_scanner and emotional_manipulator) may have systematic bias toward predicting
\"fake\" because dramatic/sensational language is normal in this domain. When synthesizing:
- Give LOWER weight to rhetorical_scanner and emotional_manipulator findings if they
  flag \"fake\" primarily based on writing style/tone
- Give HIGHER weight to knowledge_grounding and evidence_comparator findings that check
  actual factual accuracy
- A story being \"sensational\" does NOT make it fake in entertainment journalism
"""
```

#### A4. CBDF Prompt 增加先验偏差提醒

```python
# 在 CBDF prompt 末尾拼接（仅 GossipCop）
CBDF_GOSSIP_CALIBRATION = """
DATASET PRIOR: This dataset is approximately 80% real news and 20% fake news.
When evidence is ambiguous or tools disagree, lean toward \"real\" as the base rate
strongly favors real news in this domain. Only predict \"fake\" when you have strong,
specific factual evidence (not just stylistic concerns).
"""
```

### 方案B：机制/算法层面改进

#### B1. 工具可靠性加权（替代等权投票）

**问题**：当前 weighted_vote 仅按置信度加权，未考虑工具在特定数据集上的历史准确率。

**改进**：引入工具可靠性先验权重。

```python
# 根据各工具在验证集上的准确率设定权重
TOOL_RELIABILITY_WEIGHTS = {
    "gossip": {
        "evidence_comparator": 1.2,      # 78.89% acc, 最可靠
        "expectation_deviator": 1.1,      # 75.32% acc
        "cross_modal_aligner": 1.0,       # 72.41% acc
        "knowledge_grounding": 0.8,       # 66.09% acc, 降权
        "rhetorical_scanner": 0.4,        # 60.33% acc, 大幅降权
        "emotional_manipulator": 0.3,     # 58.06% acc, 几乎不采信
        "inconsistency_amplifier": 0.3,   # 41.18% acc
    },
}

def compute_weighted_vote_v2(tool_results, dataset_type="gossip"):
    """改进版加权投票：同时考虑置信度和工具可靠性"""
    weights = TOOL_RELIABILITY_WEIGHTS.get(dataset_type, {})
    fake_score, real_score = 0.0, 0.0
    for name, result in tool_results.items():
        conf = result["confidence_score"]
        reliability = weights.get(name, 1.0)
        weight = conf * reliability
        pred = normalize_prediction(result["prediction"])
        if pred == 1:
            fake_score += weight
        elif pred == 0:
            real_score += weight
    total = fake_score + real_score
    margin = abs(fake_score - real_score) / total if total > 0 else 0
    majority = "fake" if fake_score > real_score else "real"
    return {"fake_score": round(fake_score, 2), "real_score": round(real_score, 2),
            "majority": majority, "margin": round(margin, 3)}
```

#### B2. Consensus Bypass 条件收紧

**问题**：当前 bypass 条件为 "2+ 工具>=0.5置信度一致 + margin>=0.5"，在 GossipCop 上 71.35% 样本被 bypass，但 136 个 bypass 是错的。

**改进**：

```python
# 收紧 bypass 条件
CONFIDENCE_FILTER_THRESHOLD_GOSSIP = 0.7  # 从 0.5 提高到 0.7
BYPASS_MIN_TOOLS = 3  # 从 2 提高到 3
BYPASS_MARGIN_THRESHOLD = 0.7  # 从 0.5 提高到 0.7

# 额外条件：如果检索标签分布与工具共识方向不一致，禁止 bypass
def should_bypass_v2(high_conf_preds, weighted_vote, label_dist, dataset_type):
    if len(high_conf_preds) < BYPASS_MIN_TOOLS:
        return False
    if len(set(high_conf_preds)) != 1:
        return False
    if weighted_vote["margin"] < BYPASS_MARGIN_THRESHOLD:
        return False
    # 新增：检索标签一致性检查
    consensus_dir = "fake" if high_conf_preds[0] == 1 else "real"
    retr_majority = "fake" if label_dist["fake"] > label_dist["real"] else "real"
    if consensus_dir != retr_majority and label_dist["fake"] != label_dist["real"]:
        return False  # 检索先验不一致时，不允许 bypass
    return True
```

#### B3. 动态工具选择：禁用/降权不可靠工具

**问题**：MCP 规划阶段可能选择 rhetorical_scanner 和 emotional_manipulator，但这些工具在 GossipCop 上几乎失效。

**改进**：

```python
# 在 MCP 规划后，按数据集过滤不可靠工具
DISABLED_TOOLS_PER_DATASET = {
    "gossip": ["emotional_manipulator", "inconsistency_amplifier"],
    # rhetorical_scanner 不完全禁用但降权
}

DOWNWEIGHTED_TOOLS_PER_DATASET = {
    "gossip": ["rhetorical_scanner"],
}

def filter_tool_sequence(tool_sequence, dataset_type):
    """过滤掉在特定数据集上不可靠的工具"""
    disabled = DISABLED_TOOLS_PER_DATASET.get(dataset_type, [])
    return [t for t in tool_sequence if t not in disabled]
```

#### B4. 检索质量门控

**问题**：当 Top-5 检索标签多数为 fake 时，真实新闻的 FP 率高达 51.59%。

**改进**：当检索质量存疑时，降低检索上下文的影响权重。

```python
def assess_retrieval_quality(retrieved, dataset_type):
    """评估检索结果质量，返回可信度"""
    if not retrieved:
        return 0.0
    avg_sim = sum(r["similarity"] for r in retrieved) / len(retrieved)
    label_counts = {"real": 0, "fake": 0}
    for r in retrieved:
        label_counts[r["label_str"]] += 1

    # 如果检索样本标签分布与数据集先验严重偏离，降低可信度
    dataset_prior = {"gossip": 0.195, "polifact": 0.279}  # fake 比例
    expected_fake = dataset_prior.get(dataset_type, 0.3)
    actual_fake = label_counts["fake"] / len(retrieved)
    deviation = abs(actual_fake - expected_fake)

    quality = avg_sim * (1.0 - deviation * 0.5)
    return quality
```

### 方案C：数据/模型层面改进（效果最好但成本最高）

#### C1. 升级底层模型
- 将 gemini-2.5-flash-lite 升级为 gemini-2.5-flash 或 gemini-2.5-pro
- 更强的模型在细粒度判断上表现更好，尤其是需要理解领域惯例（娱乐新闻的写作风格）

#### C2. 增加检索 Top-K
- 当前 K=5，可以尝试 K=7 或 K=10
- 更多检索样本能稀释个别错误标签的影响
- 但需注意 context 长度限制

#### C3. Few-shot 示例注入
- 在 prompt 中加入 2-3 个 GossipCop 特有的正确判断示例

```python
GOSSIP_FEW_SHOT_EXAMPLES = """
Here are examples of correctly classified celebrity news:

Example 1 (REAL): "Kim Kardashian Breaks the Internet With Bold Magazine Cover! PHOTOS"
-> Despite the sensational headline, this reports a real event (Kim's Paper magazine cover).
   Dramatic language is normal for entertainment news. -> Answer: real

Example 2 (FAKE): "Tom Hanks Secretly Divorced Rita Wilson After 30 Years - Insider Reveals Shocking Truth"
-> The factual claim is false (Tom Hanks and Rita Wilson remain married).
   The "insider reveals" framing is a red flag for fabricated celebrity gossip. -> Answer: fake

Example 3 (REAL): "OMG! Taylor Swift and Travis Kelce Spotted Together AGAIN - Relationship Confirmed!"
-> Highly sensational language but reports a verified public event.
   Multiple credible sources confirm. -> Answer: real
"""
```

---

## 三、推荐实施路线

### 第一阶段（最小改动，预计提升 5-10% Accuracy）
1. **A1**: 为 GossipCop 定制领域 prompt（改 prompt_config.py）
2. **A2**: 在工具 prompt 中注入领域校准（改工具 prompt 模板）
3. **B3**: 禁用 emotional_manipulator，降权 rhetorical_scanner

### 第二阶段（中等改动，预计再提升 3-5%）
4. **B1**: 实现工具可靠性加权投票
5. **B2**: 收紧 bypass 条件
6. **A3+A4**: MPRE/CBDF prompt 加领域校准

### 第三阶段（较大改动）
7. **C3**: 加入 few-shot 示例
8. **C1**: 升级模型
9. **B4**: 检索质量门控

---

## 四、关于"是否可以通过调整 Prompt 输入方式来改进"

**答案：是的，prompt 改进是最直接且最有效的改进方向。** 具体理由：

1. **根因匹配**：GossipCop 效果差的核心原因是"领域风格与虚假新闻特征的混淆"，这正是 prompt 指令可以解决的问题
2. **工具偏差的根源在 prompt**：rhetorical_scanner 和 emotional_manipulator 之所以偏向 fake，是因为它们的 prompt 描述让模型将"煽动性语言"等同于"虚假"，而在娱乐新闻中这是正常现象
3. **低成本高回报**：只需修改 prompt_config.py 和工具 prompt 模板，无需改算法逻辑
4. **可叠加**：prompt 改进与算法改进（B1-B4）互不冲突，可以同时使用

**最关键的 prompt 改进方向：**
- 告诉模型"煽动性语言在娱乐新闻中是正常的"
- 将判断焦点从"写作风格"转移到"事实准确性"
- 为不同领域的数据集使用不同的 prompt 模板（领域自适应）
