"""
Exp4: + MPRE（Multi-Perspective Reasoning Ensemble）
在 Exp3 (ATR) 基础上加入多视角推理集成
不再取最后工具输出，而是 LLM 综合所有工具推理轨迹做融合判断
Pipeline: Input -> CAMR -> MCP -> ATR工具执行 -> MPRE证据融合 -> Output

调优策略：
  1. 过滤低置信度工具（<0.5）的证据，避免噪声干扰 MPRE
  2. 在 MPRE prompt 中注入置信度加权多数票信号，锚定融合方向
  3. 恢复适度短路（阈值 0.95），极高置信度时提前结束避免后续工具引入噪声
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
)
from models.baseline_CAMR import CAMRRetriever
from models.baseline_CAMR_MCP import parse_planning_output, get_dag_execution_order
from models.baseline_CAMR_MCP_ATR import (
    TOOL_DEFINITIONS, DIMENSION_TO_TOOL,
    TOOL_PROMPT_EN, TOOL_PROMPT_ZH,
    parse_tool_output,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics

ZH_DATASETS = {"weibo", "weibo21"}

# 调优参数
CONFIDENCE_FILTER_THRESHOLD = 0.5   # 过滤低置信度工具证据
SHORT_CIRCUIT_THRESHOLD = 0.95      # 适度短路阈值


def format_tool_evidence_filtered(tool_results: Dict[str, Dict], is_zh: bool,
                                  min_confidence: float = CONFIDENCE_FILTER_THRESHOLD) -> tuple:
    """
    将工具结果格式化为 MPRE 证据字符串，过滤低置信度工具。
    返回: (evidence_str, included_count, filtered_count)
    """
    lines = []
    filtered = 0
    for i, (tool_name, result) in enumerate(tool_results.items(), 1):
        if result["confidence_score"] < min_confidence:
            filtered += 1
            continue
        tool_def = TOOL_DEFINITIONS.get(tool_name, {})
        if is_zh:
            display_name = tool_def.get("name_zh", tool_name)
            lines.append(
                f"[工具 {i}: {display_name}]\n"
                f"分析：{result['reasoning_trace'][:400]}\n"
                f"置信度：{result['confidence_score']:.2f}\n"
                f"判断：{result['prediction']}"
            )
        else:
            display_name = tool_def.get("name_en", tool_name)
            lines.append(
                f"[Tool {i}: {display_name}]\n"
                f"Analysis: {result['reasoning_trace'][:400]}\n"
                f"Confidence: {result['confidence_score']:.2f}\n"
                f"Judgment: {result['prediction']}"
            )
    return "\n\n".join(lines), len(lines), filtered


def compute_weighted_vote(tool_results: Dict[str, Dict],
                          min_confidence: float = CONFIDENCE_FILTER_THRESHOLD) -> dict:
    """
    计算置信度加权投票。
    返回: {fake_score, real_score, majority, margin}
    """
    fake_score = 0.0
    real_score = 0.0
    for result in tool_results.values():
        conf = result["confidence_score"]
        if conf < min_confidence:
            continue
        pred = normalize_prediction(result["prediction"])
        if pred == 1:
            fake_score += conf
        elif pred == 0:
            real_score += conf

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


class BaselineCAMRMCPATRMPREDetector:
    """Exp4: Baseline + CAMR + MCP + ATR + MPRE 检测器"""

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
        return os.path.join(results_dir, f"exp4_CAMR_MCP_ATR_MPRE_k{self.top_k}.jsonl")

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

    def _build_tool_prompt(self, tool_name: str, text: str,
                           retrieved_context: str,
                           prev_results: Dict[str, Dict],
                           tool_dag: List[Dict],
                           is_zh: bool) -> str:
        tool_def = TOOL_DEFINITIONS.get(tool_name, TOOL_DEFINITIONS["semantic_dissector"])

        if is_zh:
            t_name = tool_def["name_zh"]
            t_desc = tool_def["desc_zh"]
            template = TOOL_PROMPT_ZH
        else:
            t_name = tool_def["name_en"]
            t_desc = tool_def["desc_en"]
            template = TOOL_PROMPT_EN

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

    def _build_mpre_prompt(self, text: str, context_str: str,
                           evidence_str: str, num_tools: int,
                           weighted_vote: dict, is_zh: bool) -> str:
        """构建带加权投票信号的 MPRE prompt"""
        template = MPRE_PROMPT_ZH if is_zh else MPRE_PROMPT_EN

        # 注入加权投票信号
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

        base_prompt = template.format(
            text=text,
            retrieved_context=context_str,
            tool_evidence=evidence_str,
            num_tools=num_tools,
        )

        # 在 prompt 末尾的输出格式指令之前插入投票信号
        return base_prompt + "\n" + vote_signal

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)
        is_zh = dataset_type in ZH_DATASETS

        print(f"\n{'='*60}")
        print(f"Exp4: +MPRE (K={self.top_k}) — 数据集: {dataset_type}")
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

        # 5. 推理
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

            # ---- 第2阶段: ATR 工具执行（适度短路 0.95） ----
            tool_sequence = self._map_dimensions_to_tools(execution_order)
            if not tool_sequence:
                tool_sequence = ["knowledge_grounding", "rhetorical_scanner", "evidence_comparator"]

            tool_results = {}
            executed_tools = []
            short_circuited = False

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

                # 适度短路：仅在极高置信度时触发（需至少2个工具）
                if (len(executed_tools) >= 2 and
                        parsed["confidence_score"] >= SHORT_CIRCUIT_THRESHOLD):
                    short_circuited = True
                    break

            # ---- 第3阶段: MPRE 多视角推理集成 ----
            # 计算加权投票
            weighted_vote = compute_weighted_vote(tool_results)

            # 共识直通：高置信度工具全部一致时，跳过 MPRE 直接输出
            consensus_bypass = False
            high_conf_preds = []
            for t in executed_tools:
                if t["confidence"] >= CONFIDENCE_FILTER_THRESHOLD:
                    p = normalize_prediction(t["prediction"])
                    if p is not None:
                        high_conf_preds.append(p)

            if (len(high_conf_preds) >= 2 and
                    len(set(high_conf_preds)) == 1 and
                    weighted_vote["margin"] >= 0.5):
                # 所有高置信度工具一致且优势明显 → 直接输出
                consensus_bypass = True
                predict = high_conf_preds[0]
                answer = "fake" if predict == 1 else "real"
                mpre_response = f"[Consensus bypass] All {len(high_conf_preds)} high-confidence tools agree: {answer}"
            else:
                # 存在冲突或低置信度 → 调用 MPRE 仲裁
                evidence_str, included, filtered = format_tool_evidence_filtered(
                    tool_results, is_zh=is_zh, min_confidence=CONFIDENCE_FILTER_THRESHOLD
                )
                if included == 0:
                    evidence_str, included, filtered = format_tool_evidence_filtered(
                        tool_results, is_zh=is_zh, min_confidence=0.0
                    )

                mpre_prompt = self._build_mpre_prompt(
                    text, context_str, evidence_str, included,
                    weighted_vote, is_zh
                )
                async with semaphore:
                    mpre_response = await self._call_llm(session, mpre_prompt, image_path)
                api_calls += 1

                answer = extract_answer(mpre_response)
                predict = normalize_prediction(answer)

            # 统计
            tool_predictions = [t["prediction"] for t in executed_tools if t["prediction"]]
            fake_votes = sum(1 for p in tool_predictions if normalize_prediction(p) == 1)
            real_votes = sum(1 for p in tool_predictions if normalize_prediction(p) == 0)

            label_dist = {"real": 0, "fake": 0}
            for r in retrieved:
                label_dist[r["label_str"]] += 1

            return {
                "index": idx,
                "text": text[:200],
                "label": label,
                "predict": predict,
                "answer_raw": answer,
                "mpre_response": mpre_response[:500],
                "semantic_layers": semantic_layers,
                "tool_sequence": [t["tool"] for t in executed_tools],
                "tool_results": executed_tools,
                "tool_votes": {"fake": fake_votes, "real": real_votes},
                "weighted_vote": weighted_vote,
                "consensus_bypass": consensus_bypass,
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

                with tqdm(total=len(items), desc=f"Exp4[{dataset_type}]", unit="样本") as pbar:
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
                avg_tools = (
                    sum(len(r["tool_sequence"]) for r in valid_results)
                    / max(len(valid_results), 1)
                )
                sc_count = sum(1 for r in valid_results if r.get("short_circuited", False))
                summary = {
                    "summary": True,
                    "dataset": dataset_type,
                    "experiment": "exp4_CAMR_MCP_ATR_MPRE",
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

        valid_results = [r for r in all_results if "tool_sequence" in r]
        total_api = sum(r.get("api_calls", 0) for r in valid_results)
        avg_tools = (
            sum(len(r["tool_sequence"]) for r in valid_results)
            / max(len(valid_results), 1)
        )
        sc_count = sum(1 for r in valid_results if r.get("short_circuited", False))
        bypass_count = sum(1 for r in valid_results if r.get("consensus_bypass", False))

        print(f"\n--- Exp4: +MPRE 结果 [{dataset_type}] (K={self.top_k}) ---")
        print(f"总样本: {len(test_texts)}, 有效预测: {len(all_true)}")
        print(f"总 API 调用: {total_api}, 平均工具数/样本: {avg_tools:.2f}")
        print(f"短路触发: {sc_count}/{len(valid_results)}, 共识直通: {bypass_count}/{len(valid_results)}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1:        {metrics.get('f1', 0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0):.4f}")
        print(f"耗时: {total_time:.1f}s")
        print(f"结果保存至: {result_path}")

        return {
            "dataset": dataset_type,
            "experiment": "exp4_CAMR_MCP_ATR_MPRE",
            "top_k": self.top_k,
            "total_samples": len(test_texts),
            "valid_predictions": len(all_true),
            "total_api_calls": total_api,
            "avg_tools_per_sample": round(avg_tools, 2),
            "short_circuit_count": sc_count,
            "evidence_filtered": filt_count,
            "metrics": metrics,
            "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
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
            print(f"Exp4: +MPRE (K={self.top_k}) 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
