"""
Exp2: + MCP（Meta-Cognitive Planning）
在 Exp1 (CAMR) 基础上加入元认知规划模块
LLM 先诊断新闻需要哪些分析维度，生成任务 DAG，然后按 DAG 逐维度分析后做判断
Pipeline: Input -> CAMR检索 -> MCP规划DAG -> LLM逐维度分析 -> Output
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
    MCP_ANALYSIS_PROMPT_EN,
)
from models.baseline_CAMR import CAMRRetriever
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics


def parse_planning_output(response: str) -> Dict[str, Any]:
    """
    解析 MCP 规划输出，提取分析维度和 DAG 结构。
    尝试从 LLM 响应中提取 JSON，失败则用默认规划。
    """
    # 尝试提取 JSON 块
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{[\s\S]*"semantic_layers"[\s\S]*\})',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if "semantic_layers" in result:
                    return result
            except json.JSONDecodeError:
                continue

    # 尝试直接解析整个响应
    try:
        result = json.loads(response)
        if "semantic_layers" in result:
            return result
    except json.JSONDecodeError:
        pass

    # 解析失败，使用默认规划
    return {
        "semantic_layers": [
            "factual_verification",
            "emotional_tone",
            "source_credibility",
            "cross_modal_consistency"
        ],
        "tool_dag": [
            {"tool": "factual_verification", "deps": []},
            {"tool": "emotional_tone", "deps": []},
            {"tool": "source_credibility", "deps": ["factual_verification"]},
            {"tool": "cross_modal_consistency", "deps": ["emotional_tone"]},
        ],
        "parse_failed": True,
    }


def get_dag_execution_order(tool_dag: List[Dict]) -> List[str]:
    """
    拓扑排序获取 DAG 执行顺序。
    """
    if not tool_dag:
        return []

    graph = {}
    in_degree = {}
    for node in tool_dag:
        name = node.get("tool", "")
        deps = node.get("deps", [])
        if name not in graph:
            graph[name] = []
            in_degree[name] = 0
        for dep in deps:
            if dep not in graph:
                graph[dep] = []
                in_degree[dep] = 0
            graph[dep].append(name)
            in_degree[name] = in_degree.get(name, 0) + 1

    queue = [n for n in in_degree if in_degree[n] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 添加可能遗漏的节点
    for node in tool_dag:
        name = node.get("tool", "")
        if name not in order:
            order.append(name)

    return order


class BaselineCAMRMCPDetector:
    """Exp2: Baseline + CAMR + MCP 检测器"""

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
        return os.path.join(results_dir, f"exp2_baseline_CAMR_MCP_k{self.top_k}.jsonl")

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
                        image_path: Optional[str] = None,
                        semaphore: asyncio.Semaphore = None) -> str:
        """统一的 LLM 调用封装"""
        if semaphore:
            async with semaphore:
                return await fetch_api(
                    session=session,
                    prompt=prompt,
                    image_path=image_path,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    request_delay=self.request_delay,
                )
        else:
            return await fetch_api(
                session=session,
                prompt=prompt,
                image_path=image_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                request_delay=self.request_delay,
            )

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        """对单个数据集运行 Exp2: Baseline + CAMR + MCP"""
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)

        print(f"\n{'='*60}")
        print(f"Exp2: Baseline + CAMR + MCP (K={self.top_k}) — 数据集: {dataset_type}")
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

        # 5. 异步推理（两阶段：规划 + 分析）
        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_results = list(existing_results)
        all_true = []
        all_pred = []
        for r in existing_results:
            if r.get("predict") is not None and r.get("label") is not None:
                all_true.append(r["label"])
                all_pred.append(r["predict"])

        f_write = open(result_path, "a" if existing_results else "w", encoding="utf-8")

        # 选择 prompt 模板
        planning_template = MCP_PLANNING_PROMPT_EN
        analysis_template = MCP_ANALYSIS_PROMPT_EN

        async def process_item(session, idx, text, image_filename, label, retrieved):
            if idx in processed_indices:
                return None

            context_str = retriever.format_context(retrieved)

            image_path = None
            if image_filename:
                img_path = os.path.join(images_dir, image_filename)
                if os.path.exists(img_path):
                    image_path = img_path

            # ---- 第1阶段: MCP 规划 ----
            planning_prompt = planning_template.format(
                text=text,
                retrieved_context=context_str
            )
            planning_response = await self._call_llm(
                session, planning_prompt, image_path, semaphore
            )
            planning_result = parse_planning_output(planning_response)

            semantic_layers = planning_result.get("semantic_layers", [])
            tool_dag = planning_result.get("tool_dag", [])
            execution_order = get_dag_execution_order(tool_dag)

            # 如果执行顺序为空，使用 semantic_layers
            if not execution_order and semantic_layers:
                execution_order = semantic_layers

            # ---- 第2阶段: 逐维度分析 + 判断 ----
            # 将规划信息组织为分析 prompt
            dimensions_str = "\n".join(
                f"  {i+1}. {dim}" for i, dim in enumerate(execution_order)
            )
            dag_str = json.dumps(tool_dag, ensure_ascii=False, indent=2) if tool_dag else "N/A"

            analysis_prompt = analysis_template.format(
                text=text,
                retrieved_context=context_str,
                dimensions=dimensions_str,
                dag_structure=dag_str,
            )
            analysis_response = await self._call_llm(
                session, analysis_prompt, image_path, semaphore
            )

            answer = extract_answer(analysis_response)
            predict = normalize_prediction(answer)

            # 检索标签分布
            label_dist = {"real": 0, "fake": 0}
            for r in retrieved:
                label_dist[r["label_str"]] += 1

            return {
                "index": idx,
                "text": text[:200],
                "label": label,
                "predict": predict,
                "answer_raw": answer,
                "response": analysis_response[:500],
                "planning_response": planning_response[:500],
                "semantic_layers": semantic_layers,
                "execution_order": execution_order,
                "parse_failed": planning_result.get("parse_failed", False),
                "retrieved_labels": label_dist,
                "top1_similarity": retrieved[0]["similarity"] if retrieved else 0,
                "api_calls": 2,
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

                with tqdm(total=len(items), desc=f"Exp2[{dataset_type}]", unit="样本") as pbar:
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
                # 统计规划解析成功率
                parse_ok = sum(1 for r in all_results if "parse_failed" in r and not r["parse_failed"])
                parse_total = sum(1 for r in all_results if "parse_failed" in r)
                summary = {
                    "summary": True,
                    "dataset": dataset_type,
                    "experiment": "exp2_baseline_CAMR_MCP",
                    "top_k": self.top_k,
                    "total_samples": len(test_texts),
                    "valid_predictions": len(all_true),
                    "planning_parse_rate": f"{parse_ok}/{parse_total}" if parse_total > 0 else "N/A",
                    "metrics": metrics,
                }
                json.dump(summary, f_write, ensure_ascii=False)
                f_write.write("\n")
            f_write.close()

        # 6. 最终指标
        total_time = time.time() - start_time
        metrics = calculate_metrics(all_true, all_pred) if all_true else {}

        # 统计规划解析
        parse_ok = sum(1 for r in all_results if "parse_failed" in r and not r["parse_failed"])
        parse_total = sum(1 for r in all_results if "parse_failed" in r)

        print(f"\n--- Exp2: Baseline + CAMR + MCP 结果 [{dataset_type}] (K={self.top_k}) ---")
        print(f"总样本: {len(test_texts)}, 有效预测: {len(all_true)}")
        print(f"规划解析成功率: {parse_ok}/{parse_total}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1:        {metrics.get('f1', 0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0):.4f}")
        print(f"耗时: {total_time:.1f}s")
        print(f"结果保存至: {result_path}")

        return {
            "dataset": dataset_type,
            "experiment": "exp2_baseline_CAMR_MCP",
            "top_k": self.top_k,
            "total_samples": len(test_texts),
            "valid_predictions": len(all_true),
            "planning_parse_rate": f"{parse_ok}/{parse_total}",
            "metrics": metrics,
            "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
        """
        运行 Exp2: Baseline + CAMR + MCP

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
            print(f"Exp2: Baseline + CAMR + MCP (K={self.top_k}) 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
