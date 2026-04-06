# MCTL Exp5 通用改进方案

## 改进动机

GossipCop 数据集上 Exp5 的 Accuracy 仅 77.46%（Fake-F1=0.5238），远低于 PolitiFact 的 91.26%。分析发现问题根源不在于特定数据集，而在于系统存在三个**通用设计缺陷**：

1. 工具 prompt 将"写作风格"与"虚假"硬绑定，无法适应不同新闻领域的写作惯例
2. 所有工具等权投票，不管工具自身在当前语境下是否可靠
3. Bypass 条件过松，2 个工具一致就跳过仲裁，容错空间不足

## 五项通用改进

### 改进1：领域感知 MCP 规划

**位置**：`MCP_PLANNING_PROMPT_V2_EN/ZH`

在 MCP 规划阶段增加 **Step 0: 领域识别**，让 LLM 先自动判断新闻领域（政治/娱乐/科学/体育等），再据此选择合适的分析维度。

关键设计：
- 明确指出不同领域的写作惯例差异（如"娱乐新闻的煽动性语言是标准做法"）
- 引导 MCP 在选择 emotional_tone 和 rhetorical_analysis 维度时考虑领域适配
- 输出新增 `"domain"` 字段，记录识别结果

**为什么通用**：基于新闻内容自动判断，不硬编码任何数据集规则。

### 改进2：检索校准工具分析

**位置**：`TOOL_PROMPT_V2_EN/ZH`

将原始工具 prompt 中的"绝对判断标准"替换为"相对于检索基线的判断"。要求工具在分析前先观察检索到的**真实**样本的写作风格，以此作为该领域的"正常"基线。

核心指令：
> "Only flag features that DEVIATE from this domain baseline as suspicious. Do NOT flag features that are standard for this type of journalism."

**为什么通用**：基线从检索数据中动态获取。对于政治新闻，基线是正式严肃的；对于娱乐新闻，基线是煽动夸张的。

### 改进3：动态工具可靠性估计

**位置**：`compute_dynamic_tool_weights()` + `compute_weighted_vote_dynamic()`

运行时基于三个信号自动计算每个工具的可靠性权重：

| 信号 | 权重 | 逻辑 |
|------|------|------|
| 检索先验对齐 | ±0.3 × retr_confidence | 预测与检索标签分布一致 → 加权；矛盾 → 减权 |
| 工具间一致性 | ±0.1 | 与多数工具一致 → 小幅加权 |
| 过度自信离群惩罚 | -0.2 | 置信度≥0.75 但与检索先验矛盾 → 惩罚 |

检索先验信号设为主信号（±0.3），工具间一致性为辅助信号（±0.1），避免多数工具系统性偏差时相互抱团。

**为什么通用**：不依赖任何预设权重表，权重完全由当前样本的检索结果和工具间关系决定。

### 改进4：保守 Bypass 策略

**位置**：Bypass 条件判断逻辑

| 参数 | 原始值 | 改进值 |
|------|--------|--------|
| 最少工具数 | 2 | **3** |
| margin 阈值 | 0.5 | **0.6** |
| 检索先验对齐 | 不检查 | **必须对齐** |

新增条件：工具共识方向必须与检索标签分布方向一致（或检索平局）才允许 bypass。

**为什么通用**：这是容错性改进。Bypass 跳过了 MPRE 和 CBDF 两个仲裁阶段，条件理应严格。

### 改进5：基率感知 MPRE/CBDF

**位置**：`MPRE_CALIBRATION_EN/ZH` + `CBDF_BASE_RATE_EN/ZH`

在 MPRE prompt 末尾追加领域校准指令，提醒 MPRE 关注事实准确性而非写作风格。
在 CBDF prompt 末尾追加基率感知指令，引导贝叶斯推理：

> "If the majority of retrieved similar samples are labeled 'real', you need STRONGER evidence to conclude 'fake'. Writing style alone is NOT sufficient evidence to override a strong 'real' base rate."

**为什么通用**：检索基率由数据驱动，不同数据集、不同样本的基率不同，自动适应。

## 修改范围

| 文件 | 改动 | 影响 |
|------|------|------|
| `models/baseline_CAMR_MCP_ATR_MPRE_CBDF.py` | 所有 5 项改进 | 仅 Exp5 |
| `config/` | **无改动** | — |
| `models/baseline_CAMR_MCP_ATR.py` (Exp3) | **无改动** | — |
| `models/baseline_CAMR_MCP_ATR_MPRE.py` (Exp4) | **无改动** | — |
| 其他 Exp0-4 文件 | **无改动** | — |

**所有改动集中在一个文件中，Exp0-4 结果完全不受影响。**
**无任何 `if dataset == "xxx"` 判断，所有机制通用于所有数据集。**

## 运行方式

```bash
# 清除旧结果后重新运行
mv results/gossip/exp5_full_MCTL_k5.jsonl results/gossip/exp5_full_MCTL_k5_v1.jsonl
python main.py --dataset gossip --exp exp5

# 其他数据集同样可以受益
mv results/polifact/exp5_full_MCTL_k5.jsonl results/polifact/exp5_full_MCTL_k5_v1.jsonl
python main.py --dataset polifact --exp exp5
```

## 原始文件备份

保存在 `backup/` 目录下。
