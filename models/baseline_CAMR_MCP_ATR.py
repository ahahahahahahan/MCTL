"""
Exp3: + ATR（Adaptive Tool Router）
在 Exp2 (CAMR+MCP) 基础上加入 8 个专门 LLM-based 工具
MCP 规划 DAG → ATR 按拓扑序调度工具 → 支持短路执行 → 取最后工具输出作为判断
Pipeline: Input -> CAMR检索 -> MCP规划DAG -> ATR工具调度执行 -> Output
"""
import asyncio
import aiohttp
import time
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS, API_TIMEOUT,
    MCP_PLANNING_PROMPT_EN, MCP_PLANNING_PROMPT_ZH,
)
from models.baseline_CAMR import CAMRRetriever
from models.baseline_CAMR_MCP import parse_planning_output, get_dag_execution_order
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics

ZH_DATASETS = {"weibo", "weibo21"}

# ============================================================
# 8 个专门化 LLM 工具定义
# ============================================================
TOOL_DEFINITIONS = {
    "semantic_dissector": {
        "name_en": "Semantic Dissector",
        "name_zh": "语义解构器",
        "desc_en": "Analyze the literal and implied meaning of the text. Identify double meanings, metaphors, hidden implications, and whether the surface meaning differs from the actual intent.",
        "desc_zh": "分析文本的字面含义和隐含含义。识别双关语、隐喻、隐藏暗示，以及表面含义是否与实际意图不同。",
    },
    "rhetorical_scanner": {
        "name_en": "Rhetorical Scanner",
        "name_zh": "修辞探测器",
        "desc_en": "Identify rhetorical manipulation techniques: irony, exaggeration, clickbait headlines, propaganda tactics, sensationalist language (e.g., '!!!' or 'BREAKING'), and persuasion devices.",
        "desc_zh": "识别修辞操纵手法：反讽、夸张、标题党、宣传策略、煽动性语言（如'!!!'或'震惊'），以及说服手段。",
    },
    "knowledge_grounding": {
        "name_en": "Knowledge Grounding",
        "name_zh": "知识锚定器",
        "desc_en": "Verify factual claims against the retrieved context and known facts. Check if specific claims (names, dates, events, statistics) are consistent with the retrieved articles and general knowledge.",
        "desc_zh": "根据检索上下文和已知事实核查事实性声明。检查具体声明（人名、日期、事件、统计数据）是否与检索到的文章和常识一致。",
    },
    "expectation_deviator": {
        "name_en": "Expectation Deviator",
        "name_zh": "期望偏移器",
        "desc_en": "Assess the deviation between what the image/headline leads readers to expect and what the actual content delivers. Identify bait-and-switch tactics and misleading framing.",
        "desc_zh": "评估图片/标题引导读者产生的期望与实际内容之间的偏差。识别诱导点击策略和误导性框架。",
    },
    "cross_modal_aligner": {
        "name_en": "Cross-Modal Aligner",
        "name_zh": "跨模态对齐器",
        "desc_en": "Evaluate the semantic consistency between the text and the image. Determine if the image genuinely illustrates the reported event or is unrelated/manipulated/out-of-context.",
        "desc_zh": "评估文本和图片之间的语义一致性。判断图片是否真正展示了所报道的事件，还是无关/被篡改/脱离原始上下文的。",
    },
    "inconsistency_amplifier": {
        "name_en": "Inconsistency Amplifier",
        "name_zh": "不一致放大器",
        "desc_en": "Systematically quantify contradictions across text, image, and retrieved context. Identify internal contradictions within the article and external contradictions with verified information.",
        "desc_zh": "系统性地量化文本、图片和检索上下文之间的矛盾。识别文章内部矛盾以及与已验证信息的外部矛盾。",
    },
    "emotional_manipulator": {
        "name_en": "Emotional Manipulator Detector",
        "name_zh": "情感操纵检测器",
        "desc_en": "Detect emotional manipulation elements: fear-inducing language, outrage triggers, appeal to tribal identity, urgency fabrication, and emotional framing designed to bypass rational evaluation.",
        "desc_zh": "检测情感操纵元素：恐惧诱导语言、愤怒触发器、群体认同诉求、紧迫感制造，以及旨在绕过理性评估的情感框架。",
    },
    "evidence_comparator": {
        "name_en": "Evidence Comparator",
        "name_zh": "证据对比器",
        "desc_en": "Compare the target news with retrieved similar samples. Analyze the label distribution of similar articles, identify pattern matches with known fake/real news, and assess source reliability patterns.",
        "desc_zh": "将目标新闻与检索到的相似样本进行比较。分析相似文章的标签分布，识别与已知虚假/真实新闻的模式匹配，评估来源可靠性模式。",
    },
}

# MCP 维度 → ATR 工具映射
DIMENSION_TO_TOOL = {
    "factual_verification": "knowledge_grounding",
    "emotional_tone": "emotional_manipulator",
    "source_credibility": "evidence_comparator",
    "rhetorical_analysis": "rhetorical_scanner",
    "cross_modal_consistency": "cross_modal_aligner",
    "logical_coherence": "inconsistency_amplifier",
    "temporal_context": "expectation_deviator",
    "comparative_evidence": "evidence_comparator",
}

# 短路置信度阈值
SHORT_CIRCUIT_THRESHOLD = 0.9

# ============================================================
# 工具执行 Prompt 模板
# ============================================================
TOOL_PROMPT_EN = '''You are acting as the "{tool_name}" — a specialized analysis tool for fake news detection.

Your specific task: {tool_desc}

Retrieved similar articles from verified database:
{retrieved_context}

{dependency_context}

Target news to analyze:
"{text}"

Perform your specialized analysis focused ONLY on your designated dimension. Be thorough but focused.

Your output must strictly follow this format:
"Analysis: [Your detailed analysis focused on your specific dimension]
Confidence: [A number between 0.0 and 1.0 indicating how confident you are in your assessment. Higher means more certain about whether this is real or fake.]
Judgment: [real/fake]"'''

TOOL_PROMPT_ZH = '''你正在扮演"{tool_name}"——一个用于虚假新闻检测的专门分析工具。

你的具体任务：{tool_desc}

从已验证数据库中检索到的相似样本：
{retrieved_context}

{dependency_context}

待分析的目标新闻：
"{text}"

请仅针对你负责的分析维度进行专门分析，做到深入但聚焦。

你的输出必须严格遵循以下格式：
"分析：[你针对该维度的详细分析]
置信度：[0.0到1.0之间的数字，表示你对判断的确信程度。越高表示越确定新闻是真实还是虚假。]
判断：[真实/虚假]"'''


def parse_tool_output(response: str) -> Dict[str, Any]:
    """解析工具输出，提取 reasoning_trace、confidence_score 和 prediction"""
    result = {
        "reasoning_trace": response[:500],
        "confidence_score": 0.5,
        "prediction": None,
    }

    # 提取置信度
    conf_patterns = [
        r'[Cc]onfidence:\s*([\d.]+)',
        r'置信度[：:]\s*([\d.]+)',
    ]
    for pattern in conf_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    result["confidence_score"] = score
                    break
            except ValueError:
                pass

    # 提取判断
    judge_patterns = [
        r'[Jj]udgment:\s*\[?\s*(real|fake)\s*\]?',
        r'判断[：:]\s*\[?\s*(真实|虚假)\s*\]?',
    ]
    for pattern in judge_patterns:
        match = re.search(pattern, response)
        if match:
            result["prediction"] = match.group(1).strip()
            break

    # 如果没匹配到，用通用提取
    if result["prediction"] is None:
        answer = extract_answer(response)
        result["prediction"] = answer

    return result


class BaselineCAMRMCPATRDetector:
    """Exp3: Baseline + CAMR + MCP + ATR 检测器"""

    def __init__(self, top_k: int = 5, clip_model: str = "ViT-B/32",
                 short_circuit: bool = True):
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.request_delay = REQUEST_DELAY
        self.top_k = top_k
        self.clip_model = clip_model
        self.short_circuit = short_circuit

    def _get_result_path(self, dataset_type: str) -> str:
        results_dir = DATASET_CONFIGS[dataset_type]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"exp3_baseline_CAMR_MCP_ATR_k{self.top_k}.jsonl")

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
        """LLM 调用（不带信号量，由外部控制并发）"""
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
        """将 MCP 规划的维度映射为 ATR 工具名"""
        tools = []
        seen = set()
        for dim in execution_order:
            tool_name = DIMENSION_TO_TOOL.get(dim, "semantic_dissector")
            if tool_name not in seen:
                tools.append(tool_name)
                seen.add(tool_name)
        return tools

    def _build_tool_prompt(self, tool_name: str, text: str,
                           retrieved_context: str,
                           prev_results: Dict[str, Dict],
                           tool_dag: List[Dict],
                           is_zh: bool) -> str:
        """为指定工具构建专门化 prompt"""
        tool_def = TOOL_DEFINITIONS.get(tool_name, TOOL_DEFINITIONS["semantic_dissector"])

        if is_zh:
            t_name = tool_def["name_zh"]
            t_desc = tool_def["desc_zh"]
            template = TOOL_PROMPT_ZH
        else:
            t_name = tool_def["name_en"]
            t_desc = tool_def["desc_en"]
            template = TOOL_PROMPT_EN

        # 构建依赖上下文（前序工具的分析结果）
        dep_context = ""
        # 找到当前工具在 DAG 中的依赖
        deps = []
        for node in tool_dag:
            if node.get("tool") == tool_name:
                deps = node.get("deps", [])
                break
            # 也检查映射后的名称
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
                if is_zh:
                    dep_context = "前序工具的分析结果（供你参考）：\n" + "\n\n".join(dep_lines)
                else:
                    dep_context = "Analysis results from prerequisite tools (for your reference):\n" + "\n\n".join(dep_lines)

        if not dep_context:
            if is_zh:
                dep_context = "（无前序工具依赖）"
            else:
                dep_context = "(No prerequisite tool dependencies)"

        return template.format(
            tool_name=t_name,
            tool_desc=t_desc,
            retrieved_context=retrieved_context,
            dependency_context=dep_context,
            text=text,
        )

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        """对单个数据集运行 Exp3"""
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)
        is_zh = dataset_type in ZH_DATASETS

        print(f"\n{'='*60}")
        print(f"Exp3: Baseline + CAMR + MCP + ATR (K={self.top_k}) — 数据集: {dataset_type}")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. 构建检索索引
        retriever.build_index(config["train_path"], config["images_dir"])

        # 2. 加载测试数据
        print("--- 加载测试数据 ---")
        test_df = preprocess_data(config["test_path"])
        test_texts = test_df["text"].tolist()
        true_labels = test_df["label"].tolist()
        test_images = test_df["image"].tolist() if "image" in test_df.columns else [None] * len(test_texts)
        images_dir = config["images_dir"]

        print(f"测试样本数: {len(test_texts)}")
        print(f"标签分布: 真实(0)={true_labels.count(0)}, 虚假(1)={true_labels.count(1)}")

        # 3. 预计算检索结果
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

        # 4. 断点续传
        result_path = self._get_result_path(dataset_type)
        existing_results = self._load_existing_results(result_path)
        processed_indices = {r["index"] for r in existing_results}
        if processed_indices:
            print(f"已有 {len(processed_indices)} 个已处理样本，继续处理...")

        # 5. 逐样本处理（因工具间有依赖，每个样本内部串行执行工具）
        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_results = list(existing_results)
        all_true = []
        all_pred = []
        for r in existing_results:
            if r.get("predict") is not None and r.get("label") is not None:
                all_true.append(r["label"])
                all_pred.append(r["predict"])

        f_write = open(result_path, "a" if existing_results else "w", encoding="utf-8")

        planning_template = MCP_PLANNING_PROMPT_ZH if is_zh else MCP_PLANNING_PROMPT_EN

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

            # ---- 第1阶段: MCP 规划 ----
            planning_prompt = planning_template.format(
                text=text, retrieved_context=context_str
            )
            async with semaphore:
                planning_response = await self._call_llm(session, planning_prompt, image_path)
            api_calls += 1

            planning_result = parse_planning_output(planning_response)
            semantic_layers = planning_result.get("semantic_layers", [])
            tool_dag = planning_result.get("tool_dag", [])
            execution_order = get_dag_execution_order(tool_dag)
            if not execution_order and semantic_layers:
                execution_order = semantic_layers

            # ---- 第2阶段: ATR 工具调度执行 ----
            tool_sequence = self._map_dimensions_to_tools(execution_order)

            # 确保至少有一个工具
            if not tool_sequence:
                tool_sequence = ["knowledge_grounding", "rhetorical_scanner", "evidence_comparator"]

            tool_results = {}
            short_circuited = False
            executed_tools = []

            for tool_name in tool_sequence:
                tool_prompt = self._build_tool_prompt(
                    tool_name, text, context_str,
                    tool_results, tool_dag, is_zh
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

                # 短路判断
                if self.short_circuit and parsed["confidence_score"] >= SHORT_CIRCUIT_THRESHOLD:
                    short_circuited = True
                    break

            # ---- 取最后一个执行工具的输出作为最终判断 ----
            last_tool = executed_tools[-1] if executed_tools else None
            if last_tool:
                final_answer = last_tool["prediction"]
                predict = normalize_prediction(final_answer)
            else:
                final_answer = "未知"
                predict = None

            # 检索标签分布
            label_dist = {"real": 0, "fake": 0}
            for r in retrieved:
                label_dist[r["label_str"]] += 1

            return {
                "index": idx,
                "text": text[:200],
                "label": label,
                "predict": predict,
                "answer_raw": final_answer,
                "semantic_layers": semantic_layers,
                "tool_sequence": [t["tool"] for t in executed_tools],
                "tool_results": executed_tools,
                "short_circuited": short_circuited,
                "api_calls": api_calls,
                "retrieved_labels": label_dist,
                "top1_similarity": retrieved[0]["similarity"] if retrieved else 0,
            }

        try:
            async with aiohttp.ClientSession() as session:
                items = [
                    (i, test_texts[i],
                     test_images[i] if i < len(test_images) else None,
                     true_labels[i],
                     all_retrieved[i])
                    for i in range(len(test_texts))
                    if i not in processed_indices
                ]

                with tqdm(total=len(items), desc=f"Exp3[{dataset_type}]", unit="样本") as pbar:
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
                # 统计
                total_api = sum(r.get("api_calls", 0) for r in all_results if "api_calls" in r)
                sc_count = sum(1 for r in all_results if r.get("short_circuited", False))
                avg_tools = (
                    sum(len(r.get("tool_sequence", [])) for r in all_results if "tool_sequence" in r)
                    / max(len([r for r in all_results if "tool_sequence" in r]), 1)
                )
                summary = {
                    "summary": True,
                    "dataset": dataset_type,
                    "experiment": "exp3_baseline_CAMR_MCP_ATR",
                    "top_k": self.top_k,
                    "total_samples": len(test_texts),
                    "valid_predictions": len(all_true),
                    "total_api_calls": total_api,
                    "avg_tools_per_sample": round(avg_tools, 2),
                    "short_circuit_count": sc_count,
                    "metrics": metrics,
                }
                json.dump(summary, f_write, ensure_ascii=False)
                f_write.write("\n")
            f_write.close()

        # 6. 最终指标
        total_time = time.time() - start_time
        metrics = calculate_metrics(all_true, all_pred) if all_true else {}

        total_api = sum(r.get("api_calls", 0) for r in all_results if "api_calls" in r)
        sc_count = sum(1 for r in all_results if r.get("short_circuited", False))
        valid_results = [r for r in all_results if "tool_sequence" in r]
        avg_tools = (
            sum(len(r["tool_sequence"]) for r in valid_results) / max(len(valid_results), 1)
        )

        print(f"\n--- Exp3: +ATR 结果 [{dataset_type}] (K={self.top_k}) ---")
        print(f"总样本: {len(test_texts)}, 有效预测: {len(all_true)}")
        print(f"总 API 调用: {total_api}, 平均工具数/样本: {avg_tools:.2f}")
        print(f"短路触发: {sc_count}/{len(valid_results)}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1:        {metrics.get('f1', 0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0):.4f}")
        print(f"耗时: {total_time:.1f}s")
        print(f"结果保存至: {result_path}")

        return {
            "dataset": dataset_type,
            "experiment": "exp3_baseline_CAMR_MCP_ATR",
            "top_k": self.top_k,
            "total_samples": len(test_texts),
            "valid_predictions": len(all_true),
            "total_api_calls": total_api,
            "avg_tools_per_sample": round(avg_tools, 2),
            "short_circuit_count": sc_count,
            "metrics": metrics,
            "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
        """
        运行 Exp3: Baseline + CAMR + MCP + ATR

        Args:
            dataset_type: 指定数据集，为 None 则运行所有数据集
        """
        if dataset_type:
            datasets = [dataset_type]
        else:
            datasets = list(DATASET_CONFIGS.keys())

        retriever = CAMRRetriever(
            clip_model=self.clip_model,
            top_k=self.top_k
        )

        results = {}
        for ds in datasets:
            print(f"\n>>> 开始处理数据集: {ds}")
            result = asyncio.run(self.run_dataset(ds, retriever))
            results[ds] = result

        if len(results) > 1:
            print(f"\n{'='*60}")
            print(f"Exp3: +ATR (K={self.top_k}) 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
