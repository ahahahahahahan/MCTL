"""
MCTL 实验 Prompt 模板配置
"""

# ============================================================
# Exp0: Baseline — Direct LLM Prompting
# 纯 LLM 零样本判断，不使用任何模块
# ============================================================

# 英文数据集 prompt（PolitiFact, GossipCop）
BASELINE_PROMPT_EN = '''You are a professional fake news detector. Your task is to determine whether the following news is real or fake based on its text content and the accompanying image.

Fake news characteristics:
1. Contains false information or misleading content
2. Uses sensationalist or emotionally manipulative language
3. Lacks credible sources or evidence
4. Contains logical inconsistencies or contradictions
5. Headline does not match the actual content

Real news characteristics:
1. Based on verifiable facts and credible sources
2. Uses objective and neutral language
3. Provides context and multiple perspectives
4. Consistent and logically coherent content

News text:
"{text}"

Based on the text and the image (if provided), determine whether this news is real or fake.

Your output must strictly follow this format:
"Thinking: [Analyze the news content, consider the text, image, and whether it meets the above criteria.]
Answer: [real/fake]."'''

# 中文数据集 prompt（Weibo21, Weibo）
BASELINE_PROMPT_ZH = '''你是一名专业的虚假新闻检测专家。你的任务是根据以下新闻的文本内容和配图，判断该新闻是真实的还是虚假的。

虚假新闻的特征：
1. 包含虚假信息或误导性内容
2. 使用煽动性或情绪操纵性语言
3. 缺乏可信来源或证据
4. 存在逻辑不一致或矛盾之处
5. 标题与实际内容不符

真实新闻的特征：
1. 基于可验证的事实和可信来源
2. 使用客观、中性的语言
3. 提供上下文和多角度视角
4. 内容一致且逻辑连贯

新闻文本：
"{text}"

请根据文本和图片（如有），判断该新闻是真实的还是虚假的。

你的输出必须严格遵循以下格式：
"思考：[分析新闻内容，考虑文字、图片含义，以及是否符合上述标准。]
答案：[真实/虚假]。"'''

# ============================================================
# Exp1: + CAMR（上下文增强多模态检索）
# 在 Baseline 基础上加入检索模块，提供 Top-K 相似样本上下文
# Pipeline: Input -> CAMR检索 -> LLM + Context Prediction -> Output
# ============================================================

# 英文数据集 CAMR prompt（PolitiFact, GossipCop）
CAMR_PROMPT_EN = '''You are a professional fake news detector. Your task is to determine whether the following news is real or fake.

To help you make a more informed judgment, we have retrieved several similar news articles from a verified database. Please use them as reference:

{retrieved_context}

Now, analyze the following target news:

News text:
"{text}"

Instructions:
1. Compare the target news with the retrieved similar articles above.
2. Pay attention to whether the retrieved articles with similar topics/images were labeled as real or fake.
3. Consider the consistency between the target news and the retrieved context.
4. Look for fake news characteristics: false claims, sensationalism, emotional manipulation, lack of credible sources, logical inconsistencies.
5. Look for real news characteristics: verifiable facts, neutral language, credible sources, logical coherence.

Your output must strictly follow this format:
"Thinking: [Analyze the target news by comparing it with the retrieved context. Consider text content, image (if provided), and the patterns from similar articles.]
Answer: [real/fake]."'''

# 中文数据集 CAMR prompt（Weibo21, Weibo）
CAMR_PROMPT_ZH = '''你是一名专业的虚假新闻检测专家。你的任务是判断以下新闻是真实的还是虚假的。

为了帮助你做出更准确的判断，我们从已验证的数据库中检索了若干相似新闻样本，请参考：

{retrieved_context}

现在，请分析以下待检测新闻：

新闻文本：
"{text}"

分析要求：
1. 将待检测新闻与上面的检索到的相似样本进行比较。
2. 注意相似主题/图片的已标注样本是真实还是虚假的。
3. 考虑待检测新闻与检索上下文之间的一致性。
4. 关注虚假新闻特征：虚假信息、煽动性语言、情绪操纵、缺乏可信来源、逻辑矛盾。
5. 关注真实新闻特征：可验证事实、客观语言、可信来源、逻辑连贯。

你的输出必须严格遵循以下格式：
"思考：[通过与检索上下文对比分析待检测新闻内容，考虑文字、图片含义以及相似样本的规律。]
答案：[真实/虚假]。"'''

# ============================================================
# Exp2: + MCP（元认知规划）
# 在 CAMR 基础上加入元认知规划模块
# 第1阶段: 规划 — 诊断新闻需要哪些分析维度，生成 DAG
# 第2阶段: 分析 — 按 DAG 逐维度分析后做最终判断
# Pipeline: Input -> CAMR检索 -> MCP规划DAG -> LLM逐维度分析 -> Output
# ============================================================

# ---- 第1阶段: 规划 Prompt ----

MCP_PLANNING_PROMPT_EN = '''You are a meta-cognitive planner for fake news detection. Your task is to analyze the given news and its retrieved context, then determine which analysis dimensions are needed to verify this news.

Retrieved similar articles from verified database:
{retrieved_context}

Target news to analyze:
"{text}"

Based on the news content and retrieved context, diagnose the core cognitive needs for detecting whether this news is real or fake.

Step 1: Identify which of the following semantic analysis dimensions are relevant for this specific news (select only the necessary ones, typically 3-5):
- factual_verification: Check if claims can be verified against known facts and retrieved context
- emotional_tone: Detect sensationalist, fear-inducing, or emotionally manipulative language
- source_credibility: Assess whether credible sources are cited or if the source appears unreliable
- rhetorical_analysis: Identify irony, exaggeration, clickbait, or propaganda techniques
- cross_modal_consistency: Check if the image matches the text content (when image is available)
- logical_coherence: Look for internal contradictions or logical fallacies
- temporal_context: Verify time-sensitive claims against the event timeline from retrieved context
- comparative_evidence: Compare patterns with retrieved similar articles and their labels

Step 2: Define the execution order as a DAG (directed acyclic graph), where some dimensions may depend on others.

Output ONLY a JSON object in this exact format (no other text):
```json
{{
  "semantic_layers": ["dim1", "dim2", "dim3"],
  "tool_dag": [
    {{"tool": "dim1", "deps": []}},
    {{"tool": "dim2", "deps": []}},
    {{"tool": "dim3", "deps": ["dim1"]}}
  ]
}}
```'''

MCP_PLANNING_PROMPT_ZH = '''你是一个虚假新闻检测的元认知规划器。你的任务是分析给定的新闻及其检索到的上下文，然后确定需要哪些分析维度来验证这条新闻。

从已验证数据库中检索到的相似样本：
{retrieved_context}

待分析的目标新闻：
"{text}"

基于新闻内容和检索上下文，诊断检测该新闻真伪所需的核心认知需求。

第1步：识别以下哪些语义分析维度与该新闻相关（仅选择必要的维度，通常3-5个）：
- factual_verification: 核查声称的事实是否能与已知事实和检索上下文相互验证
- emotional_tone: 检测煽动性、恐惧诱导性或情绪操纵性语言
- source_credibility: 评估是否引用了可信来源，或来源是否可疑
- rhetorical_analysis: 识别反讽、夸张、标题党或宣传手法
- cross_modal_consistency: 检查图片是否与文本内容一致（当有图片时）
- logical_coherence: 查找内部矛盾或逻辑谬误
- temporal_context: 根据检索上下文中的事件时间线验证时效性声明
- comparative_evidence: 与检索到的相似样本及其标签进行模式比较

第2步：将分析维度定义为 DAG（有向无环图）执行顺序，某些维度可能依赖于其他维度的结果。

仅输出如下格式的 JSON 对象（不要输出其他文本）：
```json
{{
  "semantic_layers": ["dim1", "dim2", "dim3"],
  "tool_dag": [
    {{"tool": "dim1", "deps": []}},
    {{"tool": "dim2", "deps": []}},
    {{"tool": "dim3", "deps": ["dim1"]}}
  ]
}}
```'''

# ---- 第2阶段: 逐维度分析 + 判断 Prompt ----

MCP_ANALYSIS_PROMPT_EN = '''You are a professional fake news detector performing a structured multi-dimensional analysis.

Retrieved similar articles from verified database:
{retrieved_context}

Target news to analyze:
"{text}"

A meta-cognitive planner has determined that the following analysis dimensions are needed for this news, in this execution order:
{dimensions}

DAG structure (showing dependencies between dimensions):
{dag_structure}

Please perform the analysis as follows:
1. Analyze the target news along EACH of the above dimensions, following the DAG execution order.
2. For each dimension, provide a brief finding and a confidence assessment (high/medium/low).
3. After completing all dimension analyses, synthesize your findings to make a final judgment.

Your output must strictly follow this format:
"Dimension Analysis:
[For each dimension, write: Dimension_name: Finding (Confidence: high/medium/low)]

Synthesis: [Integrate all dimension findings, noting agreements and conflicts between dimensions.]
Thinking: [Based on the synthesis, explain your reasoning for the final judgment.]
Answer: [real/fake]."'''

MCP_ANALYSIS_PROMPT_ZH = '''你是一名专业的虚假新闻检测专家，正在执行结构化多维度分析。

从已验证数据库中检索到的相似样本：
{retrieved_context}

待分析的目标新闻：
"{text}"

元认知规划器已确定该新闻需要以下分析维度，按此执行顺序：
{dimensions}

DAG 结构（显示维度间的依赖关系）：
{dag_structure}

请按以下步骤进行分析：
1. 按照 DAG 执行顺序，逐一对目标新闻进行各维度分析。
2. 对每个维度，给出简要发现和置信度评估（高/中/低）。
3. 完成所有维度分析后，综合所有发现做出最终判断。

你的输出必须严格遵循以下格式：
"维度分析：
[对每个维度，写：维度名称：发现内容（置信度：高/中/低）]

综合：[整合所有维度的发现，指出各维度间的一致性和冲突。]
思考：[基于综合分析，解释你做出最终判断的推理过程。]
答案：[真实/虚假]。"'''

# ============================================================
# Exp4: + MPRE（多视角推理集成）
# 综合所有工具的推理轨迹，识别共识、仲裁冲突、加权融合
# Pipeline: Input -> CAMR -> MCP -> ATR -> MPRE证据融合 -> Output
# ============================================================

MPRE_PROMPT_EN = '''You are a multi-perspective reasoning integrator for fake news detection. Your task is to synthesize the analysis results from {num_tools} specialized detection tools and make a final, well-reasoned judgment.

Retrieved similar articles from verified database:
{retrieved_context}

Target news:
"{text}"

Analysis results from all specialized tools:
{tool_evidence}

Your task is to integrate these multi-perspective analyses:

Step 1 - Consensus Identification: Identify which tools AGREE on their findings. What evidence is consistently supported across multiple tools? (e.g., multiple tools flagging image-text inconsistency)

Step 2 - Conflict Arbitration: Identify where tools DISAGREE. For each conflict, determine which tool's analysis is more reliable based on:
  - The tool's confidence score
  - The specificity and quality of its reasoning
  - Whether the tool's focus area is most relevant to this particular news

Step 3 - Weighted Synthesis: Generate a final assessment that:
  - Gives more weight to high-confidence findings
  - Prioritizes consensus evidence over isolated findings
  - Resolves conflicts with clear justification

Your output must strictly follow this format:
"Consensus: [List the key findings that multiple tools agree on]
Conflicts: [List any disagreements between tools and your arbitration]
Synthesis: [Your weighted integration of all evidence]
Thinking: [Your final reasoning based on the synthesis]
Answer: [real/fake]."'''

MPRE_PROMPT_ZH = '''你是一个虚假新闻检测的多视角推理集成器。你的任务是综合 {num_tools} 个专门检测工具的分析结果，做出最终的、有充分理据的判断。

从已验证数据库中检索到的相似样本：
{retrieved_context}

待检测新闻：
"{text}"

所有专门工具的分析结果：
{tool_evidence}

你的任务是整合这些多视角分析：

第1步 - 共识识别：找出哪些工具的发现是一致的。哪些证据被多个工具共同支持？（例如，多个工具同时标记了图文不一致）

第2步 - 冲突仲裁：找出工具之间的分歧。对每个冲突，根据以下因素判断哪个工具的分析更可靠：
  - 工具的置信度分数
  - 推理的具体性和质量
  - 该工具的分析维度是否与该新闻最相关

第3步 - 加权综合：生成最终评估：
  - 给予高置信度发现更大权重
  - 优先考虑共识证据而非孤立发现
  - 以清晰的理由解决冲突

你的输出必须严格遵循以下格式：
"共识：[列出多个工具一致同意的关键发现]
冲突：[列出工具之间的分歧以及你的仲裁结果]
综合：[你对所有证据的加权整合]
思考：[基于综合分析的最终推理]
答案：[真实/虚假]。"'''

# ============================================================
# Exp5: + CBDF（基于共识的决策融合）
# 引入检索标签分布先验 + 不确定性感知决策
# Pipeline: Input -> CAMR -> MCP -> ATR -> MPRE -> CBDF -> Output
# ============================================================

CBDF_PROMPT_EN = '''You are the final decision-maker in a multi-stage fake news detection system. Your task is to make the final verdict by integrating the multi-perspective analysis results with the statistical prior from retrieved similar articles.

Target news:
"{text}"

=== Evidence Source 1: Multi-Perspective Reasoning Result ===
{mpre_result}

=== Evidence Source 2: Retrieved Label Distribution (Statistical Prior) ===
{label_distribution}

=== Evidence Source 3: Uncertainty Assessment ===
{uncertainty_info}
Tool consensus direction: {weighted_vote_majority} (margin: {weighted_vote_margin})

=== Decision Rules ===
1. If the MPRE result clearly identifies fake news characteristics AND the retrieved samples are mostly fake → Strong fake signal, predict fake with high confidence.
2. If the MPRE result clearly identifies real news characteristics AND the retrieved samples are mostly real → Strong real signal, predict real with high confidence.
3. If MPRE and retrieved label distribution AGREE → Follow the consensus direction.
4. If MPRE and retrieved label distribution DISAGREE → Prioritize MPRE analysis (evidence-based) over statistical prior, but note the conflict.
5. Consider the uncertainty level: if tool confidence divergence is high and predictions disagree, lower your overall confidence.

Make your final decision. Your output must strictly follow this format:
"Reasoning: [Integrate MPRE evidence with retrieved label distribution. Note whether they agree or conflict. Consider uncertainty.]
Confidence: [0.0-1.0, your confidence in the final prediction]
Answer: [real/fake]."'''

CBDF_PROMPT_ZH = '''你是多阶段虚假新闻检测系统的最终决策者。你的任务是整合多视角分析结果与检索样本的统计先验，做出最终裁决。

待检测新闻：
"{text}"

=== 证据来源1：多视角推理综合结果 ===
{mpre_result}

=== 证据来源2：检索标签分布（统计先验） ===
{label_distribution}

=== 证据来源3：不确定性评估 ===
{uncertainty_info}
工具共识方向：{weighted_vote_majority}（优势幅度：{weighted_vote_margin}）

=== 决策规则 ===
1. 若 MPRE 结果明确识别出虚假新闻特征，且检索样本多数为虚假 → 强虚假信号，高置信度判定虚假。
2. 若 MPRE 结果明确识别出真实新闻特征，且检索样本多数为真实 → 强真实信号，高置信度判定真实。
3. 若 MPRE 与检索标签分布一致 → 遵循共识方向。
4. 若 MPRE 与检索标签分布不一致 → 优先采信 MPRE 分析（基于证据），但需注明冲突。
5. 考虑不确定性水平：若工具置信度分歧大且预测不一致，降低总体置信度。

请做出最终决策。你的输出必须严格遵循以下格式：
"推理：[整合 MPRE 证据与检索标签分布，指出二者是否一致或冲突，考虑不确定性。]
置信度：[0.0-1.0，你对最终预测的置信度]
答案：[真实/虚假]。"'''


# ============================================================
# 数据集专用 Prompt 补丁（轻量版 v4）
# 只保留核心平台提醒，不做重度指导，避免跷跷板效应
# ============================================================

DATASET_PROMPT_PATCH = {
    "weibo": {
        "mcp_planning": """

补充：当前数据来自微博（社交媒体）。微博内容天然信息碎片化，不要选择 rhetorical_analysis 和 emotional_tone 维度。优先选择 factual_verification、comparative_evidence、cross_modal_consistency。""",

        "tool_prompt": """

平台提醒：当前内容来自微博（社交媒体而非正规新闻）。微博帖子天然不会提供完整来源引用。"信息不完整"和"缺乏来源"不构成虚假的充分证据。请聚焦于是否存在明确的事实性矛盾或图文不符。""",

        "mpre": "",
        "cbdf": "",
    },
    "weibo21": {
        "mcp_planning": """

补充：当前数据来自微博（社交媒体）。优先选择 factual_verification、comparative_evidence、cross_modal_consistency。""",

        "tool_prompt": """

平台提醒：当前内容来自微博。"信息不完整"不构成虚假的充分证据。聚焦事实性矛盾。""",

        "mpre": "",
        "cbdf": "",
    },
    "gossip": {
        "mcp_planning": "",
        "tool_prompt": "",
        "mpre": "",
        "cbdf": "",
    },
    "polifact": {
        "mcp_planning": "",
        "tool_prompt": "",
        "mpre": "",
        "cbdf": "",
    },
}
