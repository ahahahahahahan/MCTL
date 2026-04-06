"""
数据处理工具函数
"""
import json
import os
import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效的 JSON 行: {e}")
    return pd.DataFrame(data)


def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    数据预处理
    标签映射: harmful -> 1 (虚假/fake), not harmful -> 0 (真实/real)
    """
    df = load_data(data_path)

    def extract_label(labels):
        if isinstance(labels, list) and len(labels) > 0:
            label_str = str(labels[0]).lower()
            if 'not harmful' in label_str:
                return 0  # 真实
            elif 'harmful' in label_str:
                return 1  # 虚假
        return 1  # 默认虚假

    df['label'] = df['labels'].apply(extract_label)
    df['text'] = df['text'].astype(str).str.strip()
    return df
