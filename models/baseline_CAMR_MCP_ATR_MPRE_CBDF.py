"""
Exp5: + CBDF（Consensus-Based Decision Fusion）— 完整 MCTL
在 Exp4 (MPRE) 基础上引入检索标签分布先验 + 不确定性感知决策
Pipeline: Input -> CAMR -> MCP -> ATR -> MPRE -> CBDF -> Output

v2 改进（通用机制，无数据集特判，自动适应所有领域）：
  1. 领域感知 MCP 规划：规划阶段自动识别新闻领域并调整分析策略
  2. 检索校准工具分析：工具以检索样本为"领域基线"做相对判断
  3. 动态工具可靠性：基于工具间一致性和检索先验自动估计权重
  4. 保守 Bypass：收紧直通条件 + 检索先验对齐检查
  5. 基率感知 CBDF：将检索标签分布作为贝叶斯先验显式注入
"""
import asyncio
import aiohttp
import time
import json
import os
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS, API_TIMEOUT,
    MCP_PLANNING_PROMPT_EN, MCP_PLANNING_PROMPT_ZH,
    MPRE_PROMPT_EN, MPRE_PROMPT_ZH,
    CBDF_PROMPT_EN, CBDF_PROMPT_ZH,
)
from models.baseline_CAMR import CAMRRetriever
from models.baseline_CAMR_MCP import parse_planning_output, get_dag_execution_order
from models.baseline_CAMR_MCP_ATR import (
    TOOL_DEFINITIONS, DIMENSION_TO_TOOL,
    TOOL_PROMPT_EN, TOOL_PROMPT_ZH,
    parse_tool_output,
)
from models.baseline_CAMR_MCP_ATR_MPRE import (
    format_tool_evidence_filtered,
    compute_weighted_vote,
    CONFIDENCE_FILTER_THRESHOLD,
    SHORT_CIRCUIT_THRESHOLD,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics

ZH_DATASETS = {"weibo", "weibo21"}

# 不确定性阈值
UNCERTAINTY_STD_THRESHOLD = 0.15

# ============================================================
# [改进4] 保守 Bypass 参数（通用收紧）
# 原始：min_tools=2, margin=0.5
# 改进：min_tools=3, margin=0.6，并要求检索先验对齐
# 原因：bypass 跳过 MPRE/CBDF 仲裁，条件过松会传播工具偏差
# ============================================================
BYPASS_MIN_TOOLS = 3       # 至少 3 个高置信度工具一致才 bypass
BYPASS_MARGIN_THRESHOLD = 0.6  # 加权投票优势幅度至少 0.6


# ============================================================
# [改进1] 领域感知 MCP 规划 Prompt（v2）
# 与原版区别：增加 Step 0 "领域识别"，让 LLM 先判断新闻领域
# 再据此调整分析维度权重。例如：
#   - 娱乐新闻 → 降低 rhetorical_analysis/emotional_tone 优先级
#   - 政治新闻 → 保持 rhetorical_analysis 高优先级
#   - 科学新闻 → 提高 factual_verification 优先级
# 通用机制：基于内容自动适配，无需硬编码数据集规则
# ============================================================
MCP_PLANNING_PROMPT_V2_EN = '''You are a meta-cognitive planner for fake news detection. Your task is to analyze the given news and its retrieved context, then determine which analysis dimensions are needed to verify this news.

Retrieved similar articles from verified database:
{retrieved_context}

Target news to analyze:
"{text}"

Step 0: First, identify the news domain (e.g., politics, entertainment/celebrity, health/science, sports, business, general). This is critical because different domains have different "normal" writing conventions:
- Entertainment/celebrity news: Sensational language, clickbait headlines, and emotional writing are STANDARD practice even in legitimate outlets. Rhetorical and emotional analysis are LESS useful for detecting fakes in this domain.
- Political news: Propaganda techniques and rhetorical manipulation are strong fake signals. Rhetorical analysis is highly relevant.
- Health/science news: Factual verification against known science is critical. Source credibility matters most.
- General/breaking news: Temporal verification and source cross-referencing are most important.

Step 1: Based on the identified domain, select which of the following semantic analysis dimensions are relevant for this specific news (select only the necessary ones, typically 3-5). Prioritize dimensions that are most discriminative for this domain:
- factual_verification: Check if claims can be verified against known facts and retrieved context
- emotional_tone: Detect sensationalist or emotionally manipulative language — but ONLY flag this if it deviates from the norm for this news domain
- source_credibility: Assess whether credible sources are cited or if the source appears unreliable
- rhetorical_analysis: Identify propaganda techniques — but ONLY flag this if the rhetoric goes beyond what is standard for this news domain
- cross_modal_consistency: Check if the image matches the text content (when image is available)
- logical_coherence: Look for internal contradictions or logical fallacies
- temporal_context: Verify time-sensitive claims against the event timeline from retrieved context
- comparative_evidence: Compare patterns with retrieved similar articles and their labels

Step 2: Define the execution order as a DAG (directed acyclic graph), where some dimensions may depend on others.

Output ONLY a JSON object in this exact format (no other text):
```json
{{
  "domain": "identified_domain",
  "semantic_layers": ["dim1", "dim2", "dim3"],
  "tool_dag": [
    {{"tool": "dim1", "deps": []}},
    {{"tool": "dim2", "deps": []}},
    {{"tool": "dim3", "deps": ["dim1"]}}
  ]
}}
```'''

MCP_PLANNING_PROMPT_V2_ZH = '''你是一个虚假新闻检测的元认知规划器。你的任务是分析给定的新闻及其检索到的上下文，然后确定需要哪些分析维度来验证这条新闻。

从已验证数据库中检索到的相似样本：
{retrieved_context}

待分析的目标新闻：
"{text}"

第0步：首先识别新闻领域（如：政治、娱乐/名人、健康/科学、体育、商业、综合）。这至关重要，因为不同领域有不同的"正常"写作惯例：
- 娱乐/名人新闻：煽动性语言、标题党、情绪化写作是合法媒体的标准做法。修辞和情感分析在该领域对检测虚假新闻的作用较小。
- 政治新闻：宣传手法和修辞操纵是强虚假信号，修辞分析高度相关。
- 健康/科学新闻：基于已知科学的事实核查至关重要，来源可信度最为重要。
- 综合/突发新闻：时效性验证和来源交叉核实最为重要。

第1步：基于识别的领域，选择以下哪些语义分析维度与该新闻相关（仅选择必要的维度，通常3-5个）。优先选���在该领域最具鉴别力的维度：
- factual_verification: 核查声称的事实是否能与已知事实和检索上下文相互验证
- emotional_tone: 检测煽动性或情绪操纵性语言——但仅在偏离该新闻领域正常风格时才标记
- source_credibility: 评估是否引用了可信来源，或来源是否可疑
- rhetorical_analysis: 识别宣传手法——但仅在修辞手法超出该新闻领域的标准做法时才标记
- cross_modal_consistency: 检查图片是否与文本内容一致（当有图片时）
- logical_coherence: 查找内部矛盾或逻辑谬误
- temporal_context: 根据检索上下文中的事件时间线验证时效性声明
- comparative_evidence: 与检索到的相似样本及其标签进行模式比较

第2步：将分析维度定义为 DAG（有向无环图）执行顺序。

仅输出如下格式的 JSON 对象（不要输出其他文本）：
```json
{{
  "domain": "identified_domain",
  "semantic_layers": ["dim1", "dim2", "dim3"],
  "tool_dag": [
    {{"tool": "dim1", "deps": []}},
    {{"tool": "dim2", "deps": []}},
    {{"tool": "dim3", "deps": ["dim1"]}}
  ]
}}
```'''


# ============================================================
# [改进2] 检索校准工具分析 Prompt（v2）
# 与原版区别：增加"领域基线校准"指令
# 要求工具先观察检索到的真实样本的写作风格，
# 以此作为"正常"基线，只标记偏离基线的特征为可疑
# 通用机制：基线从检索数据中自动获取，非硬编码
# ============================================================
TOOL_PROMPT_V2_EN = '''You are acting as the "{tool_name}" — a specialized analysis tool for fake news detection.

Your specific task: {tool_desc}

Retrieved similar articles from verified database:
{retrieved_context}

IMPORTANT — Domain Calibration: Before analyzing the target news, examine the writing style and characteristics of the retrieved articles labeled as "real" above. These establish the BASELINE for what is "normal" in this particular news domain. When performing your analysis:
- Only flag features that DEVIATE from this domain baseline as suspicious
- Do NOT flag features that are standard for this type of journalism (e.g., sensational language may be normal in entertainment news but suspicious in scientific reporting)
- Focus on whether the factual claims are accurate, not whether the writing style matches your expectations

{dependency_context}

Target news to analyze:
"{text}"

Perform your specialized analysis focused ONLY on your designated dimension. Be thorough but focused.

Your output must strictly follow this format:
"Analysis: [Your detailed analysis focused on your specific dimension. Explicitly note what the domain baseline is and whether the target deviates from it.]
Confidence: [A number between 0.0 and 1.0 indicating how confident you are in your assessment. Higher means more certain about whether this is real or fake.]
Judgment: [real/fake]"'''

TOOL_PROMPT_V2_ZH = '''你正在扮演"{tool_name}"——一个用于虚假新闻检测的专门分析工具。

你的具体任务：{tool_desc}

从已验证数据库中检索到的相似样本：
{retrieved_context}

重要——领域基线校准：在分析目标新闻之前，先观察上方被标注为"真实"的检索文章的写作风格和特征。这些建立了该新闻领域"正常"写作的基线。执行分析时：
- 仅将偏离该领域基线的特征标记为可疑
- 不要将该类新闻的标准写作特征标记为可疑（例如，煽动性语言在娱乐新闻中是正常的，但在科学报道中则可疑）
- 聚焦于事实性声明是否准确，而非写作风格是否符合你的预期

{dependency_context}

待分析的目标新闻：
"{text}"

请仅针对你负责的分析维度进行专门分析，做到深入但聚焦。

你的输出必须严格遵循以下格式：
"分析：[你针对该维度的详细分析。明确指出领域基线是什么，以及目标新闻是否偏离了基线。]
置信度：[0.0到1.0之间的数字，表示你对判断的确信程度。越高表示越确定新闻是真实还是虚假。]
判断：[真实/虚假]"'''


# ============================================================
# [改进3] 动态工具可靠性估计
# 基于三个信号自动估算每个工具的可靠性权重：
#   a) 检索先验对齐：工具预测是否与检索标签分布一致
#   b) 工具间一致性：工具预测是否与多数工具方向一致
#   c) 置信度合理性：过度自信但与多数相悖 → 降权
# 通用机制：自动适应任何数据集和领域
# ============================================================
def compute_dynamic_tool_weights(tool_results: Dict[str, Dict],
                                  label_dist: Dict[str, int]) -> Dict[str, float]:
    """
    动态估计各工具可靠性权重。
    输入:
      - tool_results: {tool_name: {confidence_score, prediction, ...}}
      - label_dist: {"real": n, "fake": m} 检索标签分布
    输出:
      - {tool_name: weight} 权重在 [0.3, 1.5] 区间
    """
    if not tool_results:
        return {}

    # 1. 计算多数投票方向（简单多数）
    pred_counts = {"fake": 0, "real": 0}
    for result in tool_results.values():
        pred = normalize_prediction(result["prediction"])
        if pred == 1:
            pred_counts["fake"] += 1
        elif pred == 0:
            pred_counts["real"] += 1
    majority_dir = "fake" if pred_counts["fake"] > pred_counts["real"] else "real"

    # 2. 检索先验方向
    retr_majority = "fake" if label_dist.get("fake", 0) > label_dist.get("real", 0) else "real"
    total_retr = sum(label_dist.values())
    retr_confidence = abs(label_dist.get("fake", 0) - label_dist.get("real", 0)) / max(total_retr, 1)

    weights = {}
    for name, result in tool_results.items():
        weight = 1.0
        conf = result["confidence_score"]
        pred = normalize_prediction(result["prediction"])
        pred_dir = "fake" if pred == 1 else "real" if pred == 0 else "unknown"

        # 信号 a: 检索先验对齐（主信号，权重 ±0.3）
        # 检索标签基于人工标注，是最可靠的参考
        if pred_dir == retr_majority:
            weight += 0.3 * retr_confidence  # 检索分布越偏，奖惩越大
        elif pred_dir != "unknown":
            weight -= 0.3 * retr_confidence

        # 信号 b: 工具间一致性（辅助信号，权重 ±0.1）
        # 比检索先验弱，避免多数工具系统性偏差时相互抱团
        if pred_dir == majority_dir:
            weight += 0.1
        elif pred_dir != "unknown":
            weight -= 0.1

        # 信号 c: 过度自信离群惩罚
        # 高置信度但与检索先验相悖 → 重罚（可能是工具领域偏差）
        if (conf >= 0.75 and pred_dir != retr_majority and pred_dir != "unknown"):
            weight -= 0.2  # 与检索先验矛盾的高置信预测

        # 限制权重范围
        weight = max(0.3, min(1.5, weight))
        weights[name] = round(weight, 3)

    return weights


def compute_weighted_vote_dynamic(tool_results: Dict[str, Dict],
                                   dynamic_weights: Dict[str, float],
                                   min_confidence: float = CONFIDENCE_FILTER_THRESHOLD) -> dict:
    """
    动态加权投票：置信度 × 动态可靠性权重。
    """
    fake_score = 0.0
    real_score = 0.0
    for name, result in tool_results.items():
        conf = result["confidence_score"]
        if conf < min_confidence:
            continue
        reliability = dynamic_weights.get(name, 1.0)
        weight = conf * reliability
        pred = normalize_prediction(result["prediction"])
        if pred == 1:
            fake_score += weight
        elif pred == 0:
            real_score += weight

    total = fake_score + real_score
    if total == 0:
        return {"fake_score": 0, "real_score": 0, "majority": "uncertain", "margin": 0}

    majority = "fake" if fake_score > real_score else "real"
    margin = abs(fake_score - real_score) / total
    return {
        "fake_score": round(fake_score, 3),
        "real_score": round(real_score, 3),
        "majority": majority,
        "margin": round(margin, 3),
    }


# ============================================================
# [改进5] 基率感知 MPRE 补充指令
# 在 MPRE prompt 末尾追加：提醒 MPRE 关注领域基线
# ============================================================
MPRE_CALIBRATION_EN = '''
IMPORTANT — Domain-Calibrated Synthesis:
When integrating tool results, be aware that some tools (especially rhetorical and emotional analysis) may flag features that are NORMAL for certain news domains as "fake" signals. Before concluding:
- Check whether the flagged features (e.g., sensational language) also appear in the retrieved REAL samples — if so, they are domain-standard, not fake indicators
- Give higher weight to tools that assess factual accuracy (knowledge grounding, evidence comparison) over tools that assess writing style
- Only conclude "fake" when there is clear FACTUAL evidence, not just stylistic deviations'''

MPRE_CALIBRATION_ZH = '''
重要——领域校准综合：
整合工具结果时，注意某些工具（尤其是修辞和情感分析）可能将某些新闻领域的正常特征标记为"虚假"信号。在得出结论之前：
- 检查被标记的特征（如煽动性语言）是否也出现在检索到的真实样本中——如果是，则为领域标准，非虚假指标
- 给予评估事实准确性的工具（知识锚定、证据对比）更高权重，而非评估写作风格的工具
- 仅在有明确的事实性证据时才判定"虚假"，而非仅基于风格偏差'''

# ============================================================
# [改进5] 基率感知 CBDF 补充指令
# 在 CBDF prompt 末尾追加：检索基率先验指导
# ============================================================
CBDF_BASE_RATE_EN = '''
IMPORTANT — Base Rate Awareness:
The retrieved label distribution above provides a statistical prior for this type of news. Use it as a Bayesian prior:
- If the majority of retrieved similar samples are labeled "real", you need STRONGER evidence to conclude "fake" (and vice versa)
- When tool evidence is ambiguous or conflicting, lean toward the base rate direction
- Only override the base rate when you have clear, specific factual evidence
- Writing style alone (sensationalism, clickbait, emotional language) is NOT sufficient evidence to override a strong "real" base rate — these may be normal for this news domain'''

CBDF_BASE_RATE_ZH = '''
重要——基率感知：
上述检索标签分布为该类新闻提供了统计先验。请将其作为贝叶斯先验使用：
- 如果检索到的相似样本大多标注为"真实"，你需要更强的证据才能判定"虚假"（反之亦然）
- 当工具证据模糊或矛盾时，倾向于基率方向
- 仅在有明确、具体的事实性证据时才推翻基率
- 仅凭写作风格（煽动性、标题党、情绪化语言）不足以推翻强"真实"基率——这些可能是该新闻领域的正常风格'''


def compute_uncertainty(tool_results_list: List[Dict]) -> Dict[str, Any]:
    """
    计算工具间不确定性指标。
    返回: {std, max_divergence, is_uncertain, confidence_scores}
    """
    import numpy as np

    scores = [t["confidence"] for t in tool_results_list if t.get("confidence") is not None]
    preds = [t["prediction"] for t in tool_results_list if t.get("prediction")]

    if len(scores) < 2:
        return {"std": 0, "max_divergence": 0, "is_uncertain": False, "pred_agreement": True, "scores": scores}

    std = float(np.std(scores))
    max_div = max(scores) - min(scores)

    pred_normalized = [normalize_prediction(p) for p in preds]
    pred_normalized = [p for p in pred_normalized if p is not None]
    pred_agreement = len(set(pred_normalized)) <= 1 if pred_normalized else True

    is_uncertain = (std > UNCERTAINTY_STD_THRESHOLD) or (max_div > 0.3 and not pred_agreement)

    return {
        "std": round(std, 4),
        "max_divergence": round(max_div, 4),
        "is_uncertain": is_uncertain,
        "pred_agreement": pred_agreement,
        "scores": scores,
    }


class MCTLDetector:
    """Exp5: 完整 MCTL 框架 (CAMR + MCP + ATR + MPRE + CBDF)"""

    def __init__(self, top_k: int = 5, clip_model: str = "ViT-B/32"):
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.request_delay = REQUEST_DELAY
        self.top_k = top_k
        self.clip_model = clip_model

    def _get_result_path(self, dataset_type: str) -> str:
        results_dir = DATASET_CONFIGS[dataset_type]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"exp5_full_MCTL_k{self.top_k}.jsonl")

    def _load_existing_results(self, result_path: str) -> List[Dict]:
        if not os.path.exists(result_path):
            return []
        results = []
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if "index" in item:
                            results.append(item)
                    except json.JSONDecodeError:
                        continue
        return results

    async def _call_llm(self, session: aiohttp.ClientSession, prompt: str,
                        image_path: Optional[str] = None) -> str:
        return await fetch_api(
            session=session,
            prompt=prompt,
            image_path=image_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            request_delay=self.request_delay,
        )

    def _map_dimensions_to_tools(self, execution_order: List[str]) -> List[str]:
        tools = []
        seen = set()
        for dim in execution_order:
            tool_name = DIMENSION_TO_TOOL.get(dim, "semantic_dissector")
            if tool_name not in seen:
                tools.append(tool_name)
                seen.add(tool_name)
        return tools

    # ============================================================
    # [改进2] 使用 v2 工具 prompt 模板（含领域基线校准指令）
    # ============================================================
    def _build_tool_prompt(self, tool_name: str, text: str,
                           retrieved_context: str,
                           prev_results: Dict[str, Dict],
                           tool_dag: List[Dict],
                           is_zh: bool) -> str:
        tool_def = TOOL_DEFINITIONS.get(tool_name, TOOL_DEFINITIONS["semantic_dissector"])

        # [改进2] 使用 v2 模板替代原始模板
        if is_zh:
            t_name, t_desc, template = tool_def["name_zh"], tool_def["desc_zh"], TOOL_PROMPT_V2_ZH
        else:
            t_name, t_desc, template = tool_def["name_en"], tool_def["desc_en"], TOOL_PROMPT_V2_EN

        dep_context = ""
        deps = []
        for node in tool_dag:
            if node.get("tool") == tool_name:
                deps = node.get("deps", [])
                break
            mapped = DIMENSION_TO_TOOL.get(node.get("tool", ""), "")
            if mapped == tool_name:
                deps = [DIMENSION_TO_TOOL.get(d, d) for d in node.get("deps", [])]
                break

        if deps and prev_results:
            dep_lines = []
            for dep in deps:
                if dep in prev_results:
                    r = prev_results[dep]
                    if is_zh:
                        dep_lines.append(
                            f"[前序工具 - {TOOL_DEFINITIONS.get(dep, {}).get('name_zh', dep)}] "
                            f"的分析结果：\n{r['reasoning_trace'][:300]}\n"
                            f"（置信度: {r['confidence_score']:.2f}, 判断: {r['prediction']}）"
                        )
                    else:
                        dep_lines.append(
                            f"[Prerequisite tool - {TOOL_DEFINITIONS.get(dep, {}).get('name_en', dep)}] "
                            f"analysis result:\n{r['reasoning_trace'][:300]}\n"
                            f"(Confidence: {r['confidence_score']:.2f}, Judgment: {r['prediction']})"
                        )
            if dep_lines:
                dep_context = ("前序工具的分析结果（供你参考）：\n" if is_zh else
                               "Analysis results from prerequisite tools (for your reference):\n") + "\n\n".join(dep_lines)

        if not dep_context:
            dep_context = "（无前序工具依赖）" if is_zh else "(No prerequisite tool dependencies)"

        return template.format(
            tool_name=t_name, tool_desc=t_desc,
            retrieved_context=retrieved_context,
            dependency_context=dep_context, text=text,
        )

    # ============================================================
    # [改进5] MPRE prompt 追加领域校准指令
    # ============================================================
    def _build_mpre_prompt(self, text, context_str, evidence_str, num_tools,
                           weighted_vote, is_zh):
        template = MPRE_PROMPT_ZH if is_zh else MPRE_PROMPT_EN
        if is_zh:
            vote_signal = (
                f"\n工具置信度加权投票结果：\n"
                f"  虚假加权得分: {weighted_vote['fake_score']:.2f}, "
                f"真实加权得分: {weighted_vote['real_score']:.2f}\n"
                f"  多数票方向: {weighted_vote['majority']}, "
                f"优势幅度: {weighted_vote['margin']:.1%}\n"
                f"  注意：此投票结果仅供参考，你应基于具体证据做出独立判断。"
                f"但如果你的判断与多数票方向不同，请确保有充分的理由。"
            )
            calibration = MPRE_CALIBRATION_ZH
        else:
            vote_signal = (
                f"\nConfidence-weighted tool voting result:\n"
                f"  Fake weighted score: {weighted_vote['fake_score']:.2f}, "
                f"Real weighted score: {weighted_vote['real_score']:.2f}\n"
                f"  Majority direction: {weighted_vote['majority']}, "
                f"Margin: {weighted_vote['margin']:.1%}\n"
                f"  Note: This vote is for reference only. Make your independent judgment based on evidence. "
                f"However, if your judgment differs from the majority, ensure you have strong justification."
            )
            calibration = MPRE_CALIBRATION_EN
        base = template.format(text=text, retrieved_context=context_str,
                               tool_evidence=evidence_str, num_tools=num_tools)
        return base + "\n" + vote_signal + "\n" + calibration

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)
        is_zh = dataset_type in ZH_DATASETS

        print(f"\n{'='*60}")
        print(f"Exp5: Full MCTL v2 (K={self.top_k}) — 数据集: {dataset_type}")
        print(f"  改进: 领域感知MCP + 检索校准工具 + 动态权重 + 保守Bypass + 基率感知CBDF")
        print(f"{'='*60}")

        start_time = time.time()
        retriever.build_index(config["train_path"], config["images_dir"])

        print("--- 加载测试数据 ---")
        test_df = preprocess_data(config["test_path"])
        test_texts = test_df["text"].tolist()
        true_labels = test_df["label"].tolist()
        test_images = test_df["image"].tolist() if "image" in test_df.columns else [None] * len(test_texts)
        images_dir = config["images_dir"]

        print(f"测试样本数: {len(test_texts)}")
        print(f"标签分布: 真实(0)={true_labels.count(0)}, 虚假(1)={true_labels.count(1)}")

        print(f"--- 预计算 Top-{self.top_k} 检索结果 ---")
        all_retrieved = []
        for i in tqdm(range(len(test_texts)), desc="CAMR检索"):
            img_path = None
            if test_images[i]:
                p = os.path.join(images_dir, test_images[i])
                if os.path.exists(p):
                    img_path = p
            retrieved = retriever.retrieve(test_texts[i], img_path, self.top_k)
            all_retrieved.append(retrieved)

        result_path = self._get_result_path(dataset_type)
        existing_results = self._load_existing_results(result_path)
        processed_indices = {r["index"] for r in existing_results}
        if processed_indices:
            print(f"已有 {len(processed_indices)} 个已处理样本，继续处理...")

        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_results = list(existing_results)
        all_true, all_pred = [], []
        for r in existing_results:
            if r.get("predict") is not None and r.get("label") is not None:
                all_true.append(r["label"])
                all_pred.append(r["predict"])

        f_write = open(result_path, "a" if existing_results else "w", encoding="utf-8")

        # [改进1] 使用 v2 MCP 规划 prompt（含领域识别）
        planning_template = MCP_PLANNING_PROMPT_V2_ZH if is_zh else MCP_PLANNING_PROMPT_V2_EN
        cbdf_template = CBDF_PROMPT_ZH if is_zh else CBDF_PROMPT_EN

        async def process_item(session, idx, text, image_filename, label, retrieved):
            if idx in processed_indices:
                return None

            context_str = retriever.format_context(retrieved, is_zh=is_zh)
            image_path = None
            if image_filename:
                img_path = os.path.join(images_dir, image_filename)
                if os.path.exists(img_path):
                    image_path = img_path

            api_calls = 0

            # 检索标签分布
            label_dist = {"real": 0, "fake": 0}
            for r in retrieved:
                label_dist[r["label_str"]] += 1
            label_prior = label_dist["fake"] / max(sum(label_dist.values()), 1)

            # ---- MCP 规划（改进1: 领域感知） ----
            planning_prompt = planning_template.format(text=text, retrieved_context=context_str)
            async with semaphore:
                planning_response = await self._call_llm(session, planning_prompt, image_path)
            api_calls += 1

            planning_result = parse_planning_output(planning_response)
            semantic_layers = planning_result.get("semantic_layers", [])
            tool_dag = planning_result.get("tool_dag", [])
            detected_domain = planning_result.get("domain", "unknown")  # [改进1] 提取识别的领域
            execution_order = get_dag_execution_order(tool_dag)
            if not execution_order and semantic_layers:
                execution_order = semantic_layers

            # ---- ATR 工具执行（改进2: 检索校准 prompt） ----
            tool_sequence = self._map_dimensions_to_tools(execution_order)
            if not tool_sequence:
                tool_sequence = ["knowledge_grounding", "rhetorical_scanner", "evidence_comparator"]

            tool_results = {}
            executed_tools = []
            short_circuited = False

            for tool_name in tool_sequence:
                tool_prompt = self._build_tool_prompt(
                    tool_name, text, context_str, tool_results, tool_dag, is_zh
                )
                async with semaphore:
                    tool_response = await self._call_llm(session, tool_prompt, image_path)
                api_calls += 1

                parsed = parse_tool_output(tool_response)
                tool_results[tool_name] = parsed
                executed_tools.append({
                    "tool": tool_name,
                    "confidence": parsed["confidence_score"],
                    "prediction": parsed["prediction"],
                    "reasoning_snippet": parsed["reasoning_trace"][:200],
                })

                if (len(executed_tools) >= 2 and
                        parsed["confidence_score"] >= SHORT_CIRCUIT_THRESHOLD):
                    short_circuited = True
                    break

            # ---- 改进3: 动态工具可靠性估计 ----
            dynamic_weights = compute_dynamic_tool_weights(tool_results, label_dist)

            # ---- MPRE（改进3+4: 动态加权投票 + 保守 bypass） ----
            weighted_vote = compute_weighted_vote_dynamic(tool_results, dynamic_weights)

            consensus_bypass = False
            high_conf_preds = []
            for t in executed_tools:
                if t["confidence"] >= CONFIDENCE_FILTER_THRESHOLD:
                    p = normalize_prediction(t["prediction"])
                    if p is not None:
                        high_conf_preds.append(p)

            # [改进4] 保守 Bypass: 至少 3 个工具 + margin >= 0.6 + 检索先验对齐
            if (len(high_conf_preds) >= BYPASS_MIN_TOOLS and
                    len(set(high_conf_preds)) == 1 and
                    weighted_vote["margin"] >= BYPASS_MARGIN_THRESHOLD):
                # 额外检查：检索先验对齐
                consensus_dir = "fake" if high_conf_preds[0] == 1 else "real"
                retr_majority = "fake" if label_dist["fake"] > label_dist["real"] else "real"
                # 仅当检索先验一致（或检索平局）时允许 bypass
                if consensus_dir == retr_majority or label_dist["fake"] == label_dist["real"]:
                    consensus_bypass = True
                    mpre_prediction = high_conf_preds[0]
                    mpre_summary = f"All {len(high_conf_preds)} tools agree: {'fake' if mpre_prediction == 1 else 'real'}"

            if not consensus_bypass:
                evidence_str, included, _ = format_tool_evidence_filtered(
                    tool_results, is_zh=is_zh, min_confidence=CONFIDENCE_FILTER_THRESHOLD
                )
                if included == 0:
                    evidence_str, included, _ = format_tool_evidence_filtered(
                        tool_results, is_zh=is_zh, min_confidence=0.0
                    )
                mpre_prompt = self._build_mpre_prompt(
                    text, context_str, evidence_str, included, weighted_vote, is_zh
                )
                async with semaphore:
                    mpre_response = await self._call_llm(session, mpre_prompt, image_path)
                api_calls += 1
                mpre_summary = mpre_response[:600]
                mpre_answer = extract_answer(mpre_response)
                mpre_prediction = normalize_prediction(mpre_answer)

            # ---- CBDF: 不确定性感知决策融合 ----
            uncertainty = compute_uncertainty(executed_tools)
            cbdf_bypass = False
            needs_review = False

            retrieval_majority = "fake" if label_dist["fake"] > label_dist["real"] else "real"
            mpre_direction = "fake" if mpre_prediction == 1 else "real" if mpre_prediction == 0 else "unknown"

            if (consensus_bypass and
                    weighted_vote["margin"] >= BYPASS_MARGIN_THRESHOLD and
                    (retrieval_majority == mpre_direction or label_dist["fake"] == label_dist["real"])):
                cbdf_bypass = True
                predict = mpre_prediction
                answer = "fake" if predict == 1 else "real"
                cbdf_response = f"[CBDF bypass] Consensus + retrieval prior aligned: {answer}"
            else:
                if is_zh:
                    label_dist_str = (f"检索到的 {self.top_k} 个相似样本中，"
                                      f"{label_dist['fake']} 个为虚假，{label_dist['real']} 个为真实 "
                                      f"(虚假先验概率: {label_prior:.0%})")
                    uncertainty_str = (f"工具置信度标准差: {uncertainty['std']:.3f}, "
                                       f"最大分歧: {uncertainty['max_divergence']:.3f}, "
                                       f"预测一致性: {'一致' if uncertainty['pred_agreement'] else '不一致'}")
                    mpre_result_str = (f"MPRE 综合结果: {'虚假' if mpre_prediction == 1 else '真实' if mpre_prediction == 0 else '未知'}\n"
                                       f"摘要: {mpre_summary[:400]}")
                    base_rate_note = CBDF_BASE_RATE_ZH
                else:
                    label_dist_str = (f"Among {self.top_k} retrieved similar samples, "
                                      f"{label_dist['fake']} are fake, {label_dist['real']} are real "
                                      f"(fake prior: {label_prior:.0%})")
                    uncertainty_str = (f"Tool confidence std: {uncertainty['std']:.3f}, "
                                       f"Max divergence: {uncertainty['max_divergence']:.3f}, "
                                       f"Prediction agreement: {'yes' if uncertainty['pred_agreement'] else 'no'}")
                    mpre_result_str = (f"MPRE synthesis result: {'fake' if mpre_prediction == 1 else 'real' if mpre_prediction == 0 else 'unknown'}\n"
                                       f"Summary: {mpre_summary[:400]}")
                    base_rate_note = CBDF_BASE_RATE_EN

                cbdf_prompt = cbdf_template.format(
                    text=text,
                    label_distribution=label_dist_str,
                    mpre_result=mpre_result_str,
                    uncertainty_info=uncertainty_str,
                    weighted_vote_majority=weighted_vote["majority"],
                    weighted_vote_margin=f"{weighted_vote['margin']:.1%}",
                )
                # [改进5] 追加基率感知指令
                cbdf_prompt = cbdf_prompt + "\n" + base_rate_note

                async with semaphore:
                    cbdf_response = await self._call_llm(session, cbdf_prompt, image_path)
                api_calls += 1

                answer = extract_answer(cbdf_response)
                predict = normalize_prediction(answer)
                needs_review = uncertainty["is_uncertain"] and weighted_vote["margin"] < 0.3

            fake_votes = sum(1 for t in executed_tools if normalize_prediction(t["prediction"]) == 1)
            real_votes = sum(1 for t in executed_tools if normalize_prediction(t["prediction"]) == 0)

            return {
                "index": idx,
                "text": text[:200],
                "label": label,
                "predict": predict,
                "answer_raw": answer,
                "cbdf_response": cbdf_response[:500],
                "mpre_prediction": mpre_prediction,
                "consensus_bypass": consensus_bypass,
                "detected_domain": detected_domain,   # [改进1] 记录识别的领域
                "semantic_layers": semantic_layers,
                "tool_sequence": [t["tool"] for t in executed_tools],
                "tool_results": executed_tools,
                "tool_votes": {"fake": fake_votes, "real": real_votes},
                "dynamic_weights": dynamic_weights,    # [改进3] 记录动态权重
                "weighted_vote": weighted_vote,
                "retrieved_labels": label_dist,
                "label_prior": round(label_prior, 3),
                "uncertainty": {
                    "std": uncertainty["std"],
                    "max_divergence": uncertainty["max_divergence"],
                    "is_uncertain": uncertainty["is_uncertain"],
                },
                "needs_human_review": needs_review,
                "short_circuited": short_circuited,
                "api_calls": api_calls,
                "top1_similarity": retrieved[0]["similarity"] if retrieved else 0,
            }

        try:
            async with aiohttp.ClientSession() as session:
                items = [
                    (i, test_texts[i],
                     test_images[i] if i < len(test_images) else None,
                     true_labels[i], all_retrieved[i])
                    for i in range(len(test_texts))
                    if i not in processed_indices
                ]

                with tqdm(total=len(items), desc=f"Exp5[{dataset_type}]", unit="样本") as pbar:
                    for batch_start in range(0, len(items), self.batch_size):
                        batch = items[batch_start:batch_start + self.batch_size]
                        tasks = [
                            process_item(session, idx, text, img, label, retr)
                            for idx, text, img, label, retr in batch
                        ]
                        batch_results = await asyncio.gather(*tasks)

                        for result in batch_results:
                            if result is None:
                                continue
                            all_results.append(result)
                            json.dump(result, f_write, ensure_ascii=False)
                            f_write.write("\n")
                            f_write.flush()

                            if result["predict"] is not None and result["label"] is not None:
                                all_true.append(result["label"])
                                all_pred.append(result["predict"])

                        pbar.update(len(batch))
                        if all_true:
                            acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
                            pbar.set_postfix({"acc": f"{acc:.2%}", "有效": len(all_true)})
                        await asyncio.sleep(BATCH_DELAY)
        finally:
            if all_true:
                metrics = calculate_metrics(all_true, all_pred)
                valid_results = [r for r in all_results if "tool_sequence" in r]
                total_api = sum(r.get("api_calls", 0) for r in valid_results)
                avg_tools = sum(len(r["tool_sequence"]) for r in valid_results) / max(len(valid_results), 1)
                sc_count = sum(1 for r in valid_results if r.get("short_circuited", False))
                bypass_count = sum(1 for r in valid_results if r.get("consensus_bypass", False))
                review_count = sum(1 for r in valid_results if r.get("needs_human_review", False))
                summary = {
                    "summary": True,
                    "dataset": dataset_type,
                    "experiment": "exp5_full_MCTL",
                    "top_k": self.top_k,
                    "total_samples": len(test_texts),
                    "valid_predictions": len(all_true),
                    "total_api_calls": total_api,
                    "avg_tools_per_sample": round(avg_tools, 2),
                    "short_circuit_count": sc_count,
                    "consensus_bypass_count": bypass_count,
                    "needs_human_review_count": review_count,
                    "metrics": metrics,
                }
                json.dump(summary, f_write, ensure_ascii=False)
                f_write.write("\n")
            f_write.close()

        total_time = time.time() - start_time
        metrics = calculate_metrics(all_true, all_pred) if all_true else {}

        valid_results = [r for r in all_results if "tool_sequence" in r]
        total_api = sum(r.get("api_calls", 0) for r in valid_results)
        avg_tools = sum(len(r["tool_sequence"]) for r in valid_results) / max(len(valid_results), 1)
        sc_count = sum(1 for r in valid_results if r.get("short_circuited", False))
        bypass_count = sum(1 for r in valid_results if r.get("consensus_bypass", False))
        review_count = sum(1 for r in valid_results if r.get("needs_human_review", False))

        print(f"\n--- Exp5: Full MCTL v2 结果 [{dataset_type}] (K={self.top_k}) ---")
        print(f"总样本: {len(test_texts)}, 有效预测: {len(all_true)}")
        print(f"总 API 调用: {total_api}, 平均工具数/样本: {avg_tools:.2f}")
        print(f"短路: {sc_count}, 共识直通: {bypass_count}, 需人工审核: {review_count}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1:        {metrics.get('f1', 0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0):.4f}")
        print(f"耗时: {total_time:.1f}s")
        print(f"结果保存至: {result_path}")

        return {
            "dataset": dataset_type, "experiment": "exp5_full_MCTL",
            "top_k": self.top_k, "total_samples": len(test_texts),
            "valid_predictions": len(all_true), "total_api_calls": total_api,
            "avg_tools_per_sample": round(avg_tools, 2),
            "short_circuit_count": sc_count, "consensus_bypass_count": bypass_count,
            "needs_human_review_count": review_count,
            "metrics": metrics, "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
        if dataset_type:
            datasets = [dataset_type]
        else:
            datasets = list(DATASET_CONFIGS.keys())

        retriever = CAMRRetriever(clip_model=self.clip_model, top_k=self.top_k)
        results = {}
        for ds in datasets:
            print(f"\n>>> 开始处理数据集: {ds}")
            result = asyncio.run(self.run_dataset(ds, retriever))
            results[ds] = result

        if len(results) > 1:
            print(f"\n{'='*60}")
            print(f"Exp5: Full MCTL v2 (K={self.top_k}) 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
