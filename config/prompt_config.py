"""
MCTL Experiment Prompt Template Configuration
"""

# ============================================================
# Exp0: Baseline — Direct LLM Prompting
# Pure LLM zero-shot judgment, no additional modules
# ============================================================

BASELINE_PROMPT_EN = '''Determine whether the following news is real or fake.

News text:
"{text}"

Your output must strictly follow this format:
"Thinking: [Your analysis.]
Answer: [real/fake]."'''

# ============================================================
# Exp1: + CAMR (Context-Augmented Multimodal Retrieval)
# Add retrieval module on top of Baseline, providing Top-K similar sample context
# Pipeline: Input -> CAMR Retrieval -> LLM + Context Prediction -> Output
# ============================================================

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

# ============================================================
# Exp2: + MCP (Meta-Cognitive Planning)
# Add meta-cognitive planning module on top of CAMR
# Phase 1: Planning — Diagnose which analysis dimensions the news needs, generate DAG
# Phase 2: Analysis — Analyze each dimension per DAG, then make final judgment
# Pipeline: Input -> CAMR Retrieval -> MCP Plan DAG -> LLM Per-dimension Analysis -> Output
# ============================================================

# ---- Phase 1: Planning Prompt ----

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

# ---- Phase 2: Per-dimension Analysis + Judgment Prompt ----

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

# ============================================================
# Exp4: + MPRE (Multi-Perspective Reasoning Ensemble)
# Synthesize reasoning traces from all tools, identify consensus, arbitrate conflicts, weighted fusion
# Pipeline: Input -> CAMR -> MCP -> ATR -> MPRE Evidence Fusion -> Output
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

# ============================================================
# Exp5: + CBDF (Consensus-Based Decision Fusion)
# Introduce retrieved label distribution prior + uncertainty-aware decision
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


# ============================================================
# Dataset-specific prompt patches (lightweight v4)
# Only keep core platform reminders, avoid heavy guidance to prevent seesaw effect
# ============================================================

DATASET_PROMPT_PATCH = {
    "weibo": {
        "mcp_planning": """
Note: Weibo (Chinese social media). Do NOT select rhetorical_analysis or emotional_tone. Prioritize factual_verification, comparative_evidence, cross_modal_consistency.""",

        "tool_prompt": """
Platform: Weibo. Informal writing, emotional expression, and incomplete sourcing are platform norms — not evidence of falsity. Focus on factual accuracy and image-text consistency.""",

        "mpre": "",
        "cbdf": "",
    },
    "weibo21": {
        "mcp_planning": """
Note: Weibo (Chinese social media). Prioritize factual_verification, comparative_evidence, cross_modal_consistency.""",

        "tool_prompt": """
Platform: Weibo. Fragmented information and lack of formal structure are platform norms — not evidence of falsity. Focus on factual accuracy.""",

        "mpre": "",
        "cbdf": "",
    },
    "gossip": {
        "mcp_planning": """
Note: GossipCop (celebrity/entertainment news). Do NOT select emotional_tone or rhetorical_analysis — sensational language is the domain norm. Prioritize factual_verification, source_credibility, comparative_evidence.""",

        "tool_prompt": """
Domain: Entertainment news. Sensational headlines, dramatic language, and emotional framing are standard in this domain — not indicators of fake news. Judge only by factual accuracy of claims.""",

        "mpre": """
Domain note: ~80% of entertainment news is real. Sensational tone is normal for this domain. If a tool flags emotional/rhetorical style as suspicious, discount that finding.""",

        "cbdf": "",
    },
    "polifact": {
        "mcp_planning": """
Note: PolitiFact (political news). Prioritize factual_verification, logical_coherence, source_credibility. Political rhetoric is expected — focus on verifiable claims.""",

        "tool_prompt": """
Domain: Political news. Persuasive framing and partisan language are standard in political discourse — not evidence of falsity. Focus on whether factual claims are accurate.""",

        "mpre": "",
        "cbdf": "",
    },
    "xfacta": {
        "mcp_planning": """
Note: Cross-domain factual claims. Prioritize factual_verification, comparative_evidence, logical_coherence.""",

        "tool_prompt": """
Domain: Factual claims across diverse topics. Focus strictly on verifiability of core claims against retrieved evidence.""",

        "mpre": "",
        "cbdf": "",
    },
}
