"""
Exp1: + CAMR（Context-Augmented Multimodal Retrieval）
在 Baseline 基础上加入检索模块，用 CLIP 嵌入检索相似训练样本作为上下文
Pipeline: Input -> CAMR检索 -> LLM + Context Prediction -> Output
"""
import asyncio
import aiohttp
import time
import json
import os
import numpy as np
import torch
import clip
import faiss
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS, API_TIMEOUT,
    CAMR_PROMPT_EN, CAMR_PROMPT_ZH,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics

ZH_DATASETS = {"weibo", "weibo21"}


class CAMRRetriever:
    """CAMR: 基于 CLIP 嵌入的上下文增强多模态检索模块"""

    def __init__(self, clip_model: str = "ViT-B/32", device: str = None, top_k: int = 5):
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CAMR] 加载 CLIP 模型 {clip_model} on {self.device} ...")
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()

        # 索引相关
        self.index = None
        self.train_data: List[Dict] = []

    @torch.no_grad()
    def _encode_text(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """编码文本为 CLIP 嵌入"""
        all_features = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype("float32")

    @torch.no_grad()
    def _encode_image(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """编码图像为 CLIP 嵌入，图片不存在则返回零向量"""
        dim = self.model.visual.output_dim
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            for j, path in enumerate(batch_paths):
                if path and os.path.exists(path):
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_tensors.append(self.preprocess(img))
                        valid_indices.append(j)
                    except Exception:
                        pass

            batch_features = np.zeros((len(batch_paths), dim), dtype="float32")
            if batch_tensors:
                images = torch.stack(batch_tensors).to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                feat_np = features.cpu().numpy().astype("float32")
                for idx_in_feat, idx_in_batch in enumerate(valid_indices):
                    batch_features[idx_in_batch] = feat_np[idx_in_feat]
            all_features.append(batch_features)
        return np.vstack(all_features)

    def _fuse_embeddings(self, text_emb: np.ndarray, image_emb: np.ndarray,
                         text_weight: float = 0.6) -> np.ndarray:
        """融合文本和图像嵌入"""
        img_norm = np.linalg.norm(image_emb, axis=1, keepdims=True)
        has_image = (img_norm > 1e-6).flatten()

        fused = np.copy(text_emb)
        if has_image.any():
            fused[has_image] = (text_weight * text_emb[has_image] +
                                (1 - text_weight) * image_emb[has_image])
            norms = np.linalg.norm(fused[has_image], axis=1, keepdims=True)
            fused[has_image] = fused[has_image] / (norms + 1e-8)

        return fused

    def build_index(self, train_path: str, images_dir: str):
        """从训练集构建 FAISS 索引"""
        from utils import preprocess_data as prep

        print("[CAMR] 加载训练数据 ...")
        train_df = prep(train_path)
        texts = train_df["text"].tolist()
        labels = train_df["label"].tolist()
        image_filenames = train_df["image"].tolist() if "image" in train_df.columns else [None] * len(texts)

        image_paths = []
        for fn in image_filenames:
            if fn:
                image_paths.append(os.path.join(images_dir, fn))
            else:
                image_paths.append(None)

        self.train_data = []
        for i in range(len(texts)):
            self.train_data.append({
                "text": texts[i],
                "label": labels[i],
                "label_str": "fake" if labels[i] == 1 else "real",
                "image": image_filenames[i],
            })

        print(f"[CAMR] 编码 {len(texts)} 条训练文本 ...")
        text_emb = self._encode_text(texts)
        print(f"[CAMR] 编码 {len(image_paths)} 张训练图片 ...")
        image_emb = self._encode_image(image_paths)

        fused_emb = self._fuse_embeddings(text_emb, image_emb)

        dim = fused_emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(fused_emb)
        print(f"[CAMR] FAISS 索引构建完成, 维度={dim}, 样本数={self.index.ntotal}")

    def retrieve(self, text: str, image_path: Optional[str] = None,
                 top_k: int = None) -> List[Dict]:
        """检索 Top-K 相似训练样本"""
        k = top_k or self.top_k
        if self.index is None:
            raise RuntimeError("FAISS 索引未构建，请先调用 build_index()")

        text_emb = self._encode_text([text])
        image_emb = self._encode_image([image_path])
        query_emb = self._fuse_embeddings(text_emb, image_emb)

        scores, indices = self.index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.train_data[idx])
            item["similarity"] = float(score)
            results.append(item)
        return results

    def format_context(self, retrieved: List[Dict], is_zh: bool = False) -> str:
        """将检索结果格式化为 prompt 上下文字符串"""
        if not retrieved:
            if is_zh:
                return "（未检索到相似样本）"
            return "(No similar articles retrieved)"

        lines = []
        for i, item in enumerate(retrieved, 1):
            label = item["label_str"]
            sim = item["similarity"]
            text_snippet = item["text"][:300]
            if is_zh:
                label_display = "虚假" if label == "fake" else "真实"
                lines.append(
                    f"[相似样本 {i}] (相似度: {sim:.3f}, 标签: {label_display})\n"
                    f"内容摘要: {text_snippet}..."
                )
            else:
                lines.append(
                    f"[Similar Article {i}] (Similarity: {sim:.3f}, Label: {label})\n"
                    f"Content summary: {text_snippet}..."
                )
        return "\n\n".join(lines)


class BaselineCAMRDetector:
    """Exp1: Baseline + CAMR 检索增强检测器"""

    def __init__(self, top_k: int = 5, clip_model: str = "ViT-B/32"):
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.request_delay = REQUEST_DELAY
        self.top_k = top_k
        self.clip_model = clip_model

    def _get_prompt_template(self, dataset_type: str) -> str:
        if dataset_type in ZH_DATASETS:
            return CAMR_PROMPT_ZH
        return CAMR_PROMPT_EN

    def _build_prompt(self, text: str, retrieved_context: str, dataset_type: str) -> str:
        template = self._get_prompt_template(dataset_type)
        return template.format(text=text, retrieved_context=retrieved_context)

    def _get_result_path(self, dataset_type: str) -> str:
        results_dir = DATASET_CONFIGS[dataset_type]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"exp1_baseline_CAMR_k{self.top_k}.jsonl")

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

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        """对单个数据集运行 Exp1: Baseline + CAMR"""
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)
        is_zh = dataset_type in ZH_DATASETS

        print(f"\n{'='*60}")
        print(f"Exp1: Baseline + CAMR (K={self.top_k}) — 数据集: {dataset_type}")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. 构建检索索引（基于训练集）
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

        # 3. 预计算所有测试样本的检索结果
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

        # 5. 异步推理
        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_results = list(existing_results)
        all_true = []
        all_pred = []
        for r in existing_results:
            if r.get("predict") is not None and r.get("label") is not None:
                all_true.append(r["label"])
                all_pred.append(r["predict"])

        f_write = open(result_path, "a" if existing_results else "w", encoding="utf-8")

        async def process_item(session, idx, text, image_filename, label, retrieved):
            if idx in processed_indices:
                return None

            context_str = retriever.format_context(retrieved, is_zh=is_zh)
            prompt = self._build_prompt(text, context_str, dataset_type)

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

            label_dist = {"real": 0, "fake": 0}
            for r in retrieved:
                label_dist[r["label_str"]] += 1

            return {
                "index": idx,
                "text": text[:200],
                "label": label,
                "predict": predict,
                "answer_raw": answer,
                "response": response[:500],
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

                with tqdm(total=len(items), desc=f"Exp1[{dataset_type}]", unit="样本") as pbar:
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
                summary = {
                    "summary": True,
                    "dataset": dataset_type,
                    "experiment": "exp1_baseline_CAMR",
                    "top_k": self.top_k,
                    "total_samples": len(test_texts),
                    "valid_predictions": len(all_true),
                    "metrics": metrics,
                }
                json.dump(summary, f_write, ensure_ascii=False)
                f_write.write("\n")
            f_write.close()

        # 6. 最终指标
        total_time = time.time() - start_time
        metrics = calculate_metrics(all_true, all_pred) if all_true else {}

        print(f"\n--- Exp1: Baseline + CAMR 结果 [{dataset_type}] (K={self.top_k}) ---")
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
            "experiment": "exp1_baseline_CAMR",
            "top_k": self.top_k,
            "total_samples": len(test_texts),
            "valid_predictions": len(all_true),
            "metrics": metrics,
            "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
        """
        运行 Exp1: Baseline + CAMR

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
            print(f"Exp1: Baseline + CAMR (K={self.top_k}) 汇总结果")
            print(f"{'='*60}")
            print(f"{'数据集':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(f"{ds:<12} {m.get('accuracy',0):>10.4f} {m.get('macro_f1',0):>10.4f} "
                      f"{m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f}")

        return results
