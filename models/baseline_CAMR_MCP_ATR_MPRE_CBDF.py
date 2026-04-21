"""
Exp5: + CBDF（Consensus-Based Decision Fusion）— 完整 MCTL
在 Exp4 (MPRE) 基础上引入检索标签分布先验 + 不确定性感知决策
Pipeline: Input -> CAMR -> MCP -> ATR -> MPRE -> CBDF -> Output
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
    MCP_PLANNING_PROMPT_EN,
    MPRE_PROMPT_EN,
    CBDF_PROMPT_EN,
)
from models.baseline_CAMR import CAMRRetriever
from models.baseline_CAMR_MCP import parse_planning_output, get_dag_execution_order
from models.baseline_CAMR_MCP_ATR import (
    TOOL_DEFINITIONS, DIMENSION_TO_TOOL,
    TOOL_PROMPT_EN,
    parse_tool_output,
)
from models.baseline_CAMR_MCP_ATR_MPRE import (
    format_tool_evidence_filtered,
    compute_weighted_vote,
    CONFIDENCE_FILTER_THRESHOLD,
    SHORT_CIRCUIT_THRESHOLD,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics

# 不确定性阈值：工具置信度标准差超过此值则标记为 uncertain
UNCERTAINTY_STD_THRESHOLD = 0.15  # 对应 ~30% 分歧范围


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

    # 判断分歧：置信度标准差高 或 预测不一致
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

    def _build_tool_prompt(self, tool_name: str, text: str,
                           retrieved_context: str,
                           prev_results: Dict[str, Dict],
                           tool_dag: List[Dict]) -> str:
        tool_def = TOOL_DEFINITIONS.get(tool_name, TOOL_DEFINITIONS["semantic_dissector"])
        t_name, t_desc, template = tool_def["name_en"], tool_def["desc_en"], TOOL_PROMPT_EN

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
                    dep_lines.append(
                        f"[Prerequisite tool - {TOOL_DEFINITIONS.get(dep, {}).get('name_en', dep)}] "
                        f"analysis result:\n{r['reasoning_trace'][:300]}\n"
                        f"(Confidence: {r['confidence_score']:.2f}, Judgment: {r['prediction']})"
                    )
            if dep_lines:
                dep_context = ("Analysis results from prerequisite tools (for your reference):\n") + "\n\n".join(dep_lines)

        if not dep_context:
            dep_context = "(No prerequisite tool dependencies)"

        return template.format(
            tool_name=t_name, tool_desc=t_desc,
            retrieved_context=retrieved_context,
            dependency_context=dep_context, text=text,
        )

    def _build_mpre_prompt(self, text, context_str, evidence_str, num_tools,
                           weighted_vote):
        template = MPRE_PROMPT_EN
        vote_signal = (
            f"\nConfidence-weighted tool voting result:\n"
            f"  Fake weighted score: {weighted_vote['fake_score']:.2f}, "
            f"Real weighted score: {weighted_vote['real_score']:.2f}\n"
            f"  Majority direction: {weighted_vote['majority']}, "
            f"Margin: {weighted_vote['margin']:.1%}\n"
            f"  Note: This vote is for reference only. Make your independent judgment based on evidence. "
            f"However, if your judgment differs from the majority, ensure you have strong justification."
        )
        base = template.format(text=text, retrieved_context=context_str,
                               tool_evidence=evidence_str, num_tools=num_tools)
        return base + "\n" + vote_signal

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)

        print(f"\n{'='*60}")
        print(f"Exp5: Full MCTL (K={self.top_k}) — 数据集: {dataset_type}")
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
        planning_template = MCP_PLANNING_PROMPT_EN
        cbdf_template = CBDF_PROMPT_EN

        async def process_item(session, idx, text, image_filename, label, retrieved):
            if idx in processed_indices:
                return None

            context_str = retriever.format_context(retrieved)
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

            # ---- MCP 规划 ----
            planning_prompt = planning_template.format(text=text, retrieved_context=context_str)
            async with semaphore:
                planning_response = await self._call_llm(session, planning_prompt, image_path)
            api_calls += 1

            planning_result = parse_planning_output(planning_response)
            semantic_layers = planning_result.get("semantic_layers", [])
            tool_dag = planning_result.get("tool_dag", [])
            execution_order = get_dag_execution_order(tool_dag)
            if not execution_order and semantic_layers:
                execution_order = semantic_layers

            # ---- ATR 工具执行（按 DAG 层级并行） ----
            tool_sequence = self._map_dimensions_to_tools(execution_order)
            if not tool_sequence:
                tool_sequence = ["knowledge_grounding", "rhetorical_scanner", "evidence_comparator"]

            # 构建工具依赖图，按层级分组
            tool_dep_map = {}
            for node in tool_dag:
                mapped = DIMENSION_TO_TOOL.get(node.get("tool", ""), node.get("tool", ""))
                deps = [DIMENSION_TO_TOOL.get(d, d) for d in node.get("deps", [])]
                tool_dep_map[mapped] = [d for d in deps if d in tool_sequence]

            tool_levels = []
            remaining = list(tool_sequence)
            resolved = set()
            while remaining:
                level = [t for t in remaining
                         if all(d in resolved for d in tool_dep_map.get(t, []))]
                if not level:
                    level = remaining[:]
                tool_levels.append(level)
                resolved.update(level)
                remaining = [t for t in remaining if t not in resolved]

            tool_results = {}
            executed_tools = []
            short_circuited = False

            for level in tool_levels:
                if short_circuited:
                    break

                async def _run_tool(tn):
                    tool_prompt = self._build_tool_prompt(
                        tn, text, context_str, tool_results, tool_dag
                    )
                    async with semaphore:
                        return tn, await self._call_llm(session, tool_prompt, image_path)

                level_results = await asyncio.gather(*[_run_tool(t) for t in level])
                api_calls += len(level_results)

                for tool_name, tool_response in level_results:
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

            # ---- MPRE（含共识直通 — 放宽条件） ----
            weighted_vote = compute_weighted_vote(tool_results)
            consensus_bypass = False
            high_conf_preds = []
            for t in executed_tools:
                if t["confidence"] >= CONFIDENCE_FILTER_THRESHOLD:
                    p = normalize_prediction(t["prediction"])
                    if p is not None:
                        high_conf_preds.append(p)

            if (len(high_conf_preds) >= 2 and
                    len(set(high_conf_preds)) == 1 and
                    weighted_vote["margin"] >= 0.3):
                consensus_bypass = True
                mpre_prediction = high_conf_preds[0]
                mpre_summary = f"All {len(high_conf_preds)} tools agree: {'fake' if mpre_prediction == 1 else 'real'}"
            else:
                evidence_str, included, _ = format_tool_evidence_filtered(
                    tool_results, min_confidence=CONFIDENCE_FILTER_THRESHOLD
                )
                if included == 0:
                    evidence_str, included, _ = format_tool_evidence_filtered(
                    tool_results, min_confidence=0.0
                    )
                mpre_prompt = self._build_mpre_prompt(
                    text, context_str, evidence_str, included, weighted_vote
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

            # 判断是否需要 CBDF 介入
            # 条件：共识直通 + 检索标签先验与共识一致 → 跳过 CBDF
            retrieval_majority = "fake" if label_dist["fake"] > label_dist["real"] else "real"
            mpre_direction = "fake" if mpre_prediction == 1 else "real" if mpre_prediction == 0 else "unknown"

            if (consensus_bypass and
                    weighted_vote["margin"] >= 0.3 and
                    (retrieval_majority == mpre_direction or label_dist["fake"] == label_dist["real"])):
                # 工具共识 + 检索先验不强烈冲突 → 无需 CBDF
                cbdf_bypass = True
                predict = mpre_prediction
                answer = "fake" if predict == 1 else "real"
                cbdf_response = f"[CBDF bypass] Consensus + retrieval prior aligned: {answer}"
            else:
                # 存在冲突或不确定性 → 调用 CBDF
                label_dist_str = (f"Among {self.top_k} retrieved similar samples, "
                                  f"{label_dist['fake']} are fake, {label_dist['real']} are real "
                                  f"(fake prior: {label_prior:.0%})")
                uncertainty_str = (f"Tool confidence std: {uncertainty['std']:.3f}, "
                                   f"Max divergence: {uncertainty['max_divergence']:.3f}, "
                                   f"Prediction agreement: {'yes' if uncertainty['pred_agreement'] else 'no'}")
                mpre_result_str = (f"MPRE synthesis result: {'fake' if mpre_prediction == 1 else 'real' if mpre_prediction == 0 else 'unknown'}\n"
                                   f"Summary: {mpre_summary[:400]}")

                cbdf_prompt = cbdf_template.format(
                    text=text,
                    label_distribution=label_dist_str,
                    mpre_result=mpre_result_str,
                    uncertainty_info=uncertainty_str,
                    weighted_vote_majority=weighted_vote["majority"],
                    weighted_vote_margin=f"{weighted_vote['margin']:.1%}",
                )
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
                "semantic_layers": semantic_layers,
                "tool_sequence": [t["tool"] for t in executed_tools],
                "tool_results": executed_tools,
                "tool_votes": {"fake": fake_votes, "real": real_votes},
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

        print(f"\n--- Exp5: Full MCTL 结果 [{dataset_type}] (K={self.top_k}) ---")
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
            print(f"Exp5: Full MCTL (K={self.top_k}) 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results