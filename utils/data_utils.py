"""
Data processing utility functions
"""
import json
import os
import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping invalid JSON line: {e}")
    return pd.DataFrame(data)


def preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Data preprocessing.
    Label mapping: harmful -> 1 (fake), not harmful -> 0 (real)
    """
    df = load_data(data_path)

    def extract_label(labels):
        if isinstance(labels, list) and len(labels) > 0:
            label_str = str(labels[0]).lower()
            if 'not harmful' in label_str:
                return 0  # real
            elif 'harmful' in label_str:
                return 1  # fake
        return None

    df['label'] = df['labels'].apply(extract_label)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str).str.strip()
    return df
