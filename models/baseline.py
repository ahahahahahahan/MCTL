"""
Exp0: Baseline — Direct LLM Prompting
纯 LLM 零样本判断，不使用任何模块
Pipeline: Input -> LLM Direct Prediction -> Output
"""
import asyncio
import aiohttp
import time
import json
import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS, API_TIMEOUT,
    BASELINE_PROMPT_EN, BASELINE_PROMPT_ZH,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics


# 中文数据集列表
ZH_DATASETS = {"weibo", "weibo21"}


class BaselineDetector:
    """Exp0: 直接 LLM 推理 Baseline"""

    def __init__(self):
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.request_delay = REQUEST_DELAY

    def _get_prompt_template(self, dataset_type: str) -> str:
        if dataset_type in ZH_DATASETS:
            return BASELINE_PROMPT_ZH
        return BASELINE_PROMPT_EN

    def _build_prompt(self, text: str, dataset_type: str) -> str:
        template = self._get_prompt_template(dataset_type)
        return template.format(text=text)

    def _get_result_path(self, dataset_type: str) -> str:
        results_dir = DATASET_CONFIGS[dataset_type]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, "exp0_baseline.jsonl")

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
                        # 跳过最终统计行
                        if 'index' in item:
                            results.append(item)
                    except json.JSONDecodeError:
                        continue
        return results

    async def run_dataset(self, dataset_type: str) -> Dict[str, Any]:
        """对单个数据集运行 Exp0 Baseline"""
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)

        print(f"\n{'='*60}")
        print(f"Exp0 Baseline — 数据集: {dataset_type}")
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

        # 统计已有结果
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
                'response': response[:500],
            }

        try:
            async with aiohttp.ClientSession() as session:
                # 准备待处理项
                items = [
                    (i, test_texts[i], test_images[i] if i < len(test_images) else None, true_labels[i])
                    for i in range(len(test_texts))
                    if i not in processed_indices
                ]

                with tqdm(total=len(items), desc=f"Exp0[{dataset_type}]", unit="样本") as pbar:
                    # 分批处理
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

                        # 实时显示准确率
                        if all_true:
                            acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
                            pbar.set_postfix({"acc": f"{acc:.2%}", "有效": len(all_true)})

                        # 批次间延迟
                        await asyncio.sleep(BATCH_DELAY)
        finally:
            # 写入最终统计
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

        print(f"\n--- Exp0 Baseline 结果 [{dataset_type}] ---")
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
        运行 Exp0 Baseline

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

        # 汇总输出
        if len(results) > 1:
            print(f"\n{'='*60}")
            print("Exp0 Baseline 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get('metrics', {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
