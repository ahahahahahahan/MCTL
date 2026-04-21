"""
Exp0-COT: Baseline + Chain-of-Thought Prompting
在纯 LLM 零样本基础上，加入 COT 思维链引导逐步推理
Pipeline: Input -> LLM Chain-of-Thought Reasoning -> Output
"""
import asyncio
import aiohttp
import time
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS, API_TIMEOUT,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics


# ============================================================
# COT Prompt: 引导 LLM 分步骤推理后再给出判断
# 与 Baseline 的区别：Baseline 只要求 "Thinking + Answer"，
# COT 显式拆解为 5 个推理步骤，迫使模型逐步分析后再下结论
# ============================================================

BASELINE_COT_PROMPT_EN = '''You are a professional fake news detector. Determine whether the following news is real or fake by reasoning step-by-step.

News text:
"{text}"

Please analyze this news using the following Chain-of-Thought reasoning steps:

Step 1 — Claim Identification: What are the core factual claims made in this news? List the key assertions.

Step 2 — Language & Tone Analysis: Is the language neutral and objective, or does it use sensationalist, emotionally manipulative, or exaggerated expressions?

Step 3 — Logical Consistency: Are there any internal contradictions, logical fallacies, or unsupported leaps in reasoning?

Step 4 — Source & Evidence: Does the news cite credible sources? Are the claims supported by verifiable evidence? Are there missing attributions?

Step 5 — Overall Assessment: Synthesize your findings from Steps 1-4. What is the overall pattern — does the evidence point toward real or fake?

Your output must strictly follow this format:
"Step 1 — Claim Identification: [Your analysis]
Step 2 — Language & Tone: [Your analysis]
Step 3 — Logical Consistency: [Your analysis]
Step 4 — Source & Evidence: [Your analysis]
Step 5 — Overall Assessment: [Your synthesis]
Thinking: [Your final reasoning based on the above steps]
Answer: [real/fake]."'''


class BaselineCOTDetector:
    """Exp0-COT: 基线 + Chain-of-Thought 思维链推理"""

    def __init__(self):
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.request_delay = REQUEST_DELAY

    def _build_prompt(self, text: str, dataset_type: str) -> str:
        return BASELINE_COT_PROMPT_EN.format(text=text)

    def _get_result_path(self, dataset_type: str) -> str:
        results_dir = DATASET_CONFIGS[dataset_type]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, "exp0_baseline_cot.jsonl")

    def _load_existing_results(self, result_path: str) -> List[Dict]:
        if not os.path.exists(result_path):
            return []
        results = []
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'index' in item:
                            results.append(item)
                    except json.JSONDecodeError:
                        continue
        return results

    async def run_dataset(self, dataset_type: str) -> Dict[str, Any]:
        """对单个数据集运行 Exp0-COT"""
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)

        print(f"\n{'='*60}")
        print(f"Exp0-COT Baseline+COT — 数据集: {dataset_type}")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. 加载数据
        print("--- 加载测试数据 ---")
        test_df = preprocess_data(config["test_path"])
        test_texts = test_df["text"].tolist()
        true_labels = test_df["label"].tolist()
        test_images = test_df["image"].tolist() if "image" in test_df.columns else [None] * len(test_texts)
        images_dir = config["images_dir"]

        print(f"测试样本数: {len(test_texts)}")
        print(f"标签分布: 真实(0)={true_labels.count(0)}, 虚假(1)={true_labels.count(1)}")

        # 2. 断点续传
        result_path = self._get_result_path(dataset_type)
        existing_results = self._load_existing_results(result_path)
        processed_indices = {r['index'] for r in existing_results}
        if processed_indices:
            print(f"已有 {len(processed_indices)} 个已处理样本，继续处理...")

        # 3. 异步推理
        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_results = list(existing_results)

        all_true = []
        all_pred = []
        for r in existing_results:
            if r.get('predict') is not None and r.get('label') is not None:
                all_true.append(r['label'])
                all_pred.append(r['predict'])

        f_write = open(result_path, 'a' if existing_results else 'w', encoding='utf-8')

        async def process_item(session, idx, text, image_filename, label):
            if idx in processed_indices:
                return None

            prompt = self._build_prompt(text, dataset_type)
            image_path = None
            if image_filename:
                img_path = os.path.join(images_dir, image_filename)
                if os.path.exists(img_path):
                    image_path = img_path

            async with semaphore:
                response = await fetch_api(
                    session=session,
                    prompt=prompt,
                    image_path=image_path,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    request_delay=self.request_delay
                )

            answer = extract_answer(response)
            predict = normalize_prediction(answer)

            return {
                'index': idx,
                'text': text[:200],
                'label': label,
                'predict': predict,
                'answer_raw': answer,
                'response': response[:1000],
            }

        try:
            async with aiohttp.ClientSession() as session:
                items = [
                    (i, test_texts[i], test_images[i] if i < len(test_images) else None, true_labels[i])
                    for i in range(len(test_texts))
                    if i not in processed_indices
                ]

                with tqdm(total=len(items), desc=f"Exp0-COT[{dataset_type}]", unit="样本") as pbar:
                    for batch_start in range(0, len(items), self.batch_size):
                        batch = items[batch_start:batch_start + self.batch_size]
                        tasks = [
                            process_item(session, idx, text, img, label)
                            for idx, text, img, label in batch
                        ]
                        batch_results = await asyncio.gather(*tasks)

                        for result in batch_results:
                            if result is None:
                                continue
                            all_results.append(result)
                            json.dump(result, f_write, ensure_ascii=False)
                            f_write.write('\n')
                            f_write.flush()

                            if result['predict'] is not None and result['label'] is not None:
                                all_true.append(result['label'])
                                all_pred.append(result['predict'])

                        pbar.update(len(batch))

                        if all_true:
                            acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
                            pbar.set_postfix({"acc": f"{acc:.2%}", "有效": len(all_true)})

                        await asyncio.sleep(BATCH_DELAY)
        finally:
            if all_true:
                metrics = calculate_metrics(all_true, all_pred)
                summary = {
                    'summary': True,
                    'dataset': dataset_type,
                    'total_samples': len(test_texts),
                    'valid_predictions': len(all_true),
                    'metrics': metrics,
                }
                json.dump(summary, f_write, ensure_ascii=False)
                f_write.write('\n')
            f_write.close()

        # 4. 计算最终指标
        total_time = time.time() - start_time
        metrics = calculate_metrics(all_true, all_pred) if all_true else {}

        print(f"\n--- Exp0-COT Baseline+COT 结果 [{dataset_type}] ---")
        print(f"总样本: {len(test_texts)}, 有效预测: {len(all_true)}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1:        {metrics.get('f1', 0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0):.4f}")
        print(f"耗时: {total_time:.1f}s")
        print(f"结果保存至: {result_path}")

        return {
            "dataset": dataset_type,
            "total_samples": len(test_texts),
            "valid_predictions": len(all_true),
            "metrics": metrics,
            "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
        """
        运行 Exp0-COT

        Args:
            dataset_type: 指定数据集，为 None 则运行所有数据集
        """
        if dataset_type:
            datasets = [dataset_type]
        else:
            datasets = list(DATASET_CONFIGS.keys())

        results = {}
        for ds in datasets:
            print(f"\n>>> 开始处理数据集: {ds}")
            result = asyncio.run(self.run_dataset(ds))
            results[ds] = result

        if len(results) > 1:
            print(f"\n{'='*60}")
            print("Exp0-COT Baseline+COT 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get('metrics', {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
