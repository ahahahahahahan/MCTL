"""
评估指标计算工具函数
"""
from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def calculate_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    """
    计算评估指标

    Args:
        true_labels: 真实标签列表 (0=真实, 1=虚假)
        pred_labels: 预测标签列表 (0=真实, 1=虚假)

    Returns:
        包含各种评估指标的字典
    """
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "macro_f1": 0.0}

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='binary', zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
    }
