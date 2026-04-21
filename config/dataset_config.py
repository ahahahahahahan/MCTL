"""
Dataset configuration - reusing data from fakenews project
"""
DATA_ROOT = "/home/liangjun/Desktop/HY/code/fakenews/data"

DATASET_CONFIGS = {
    "polifact": {
        "test_path": f"{DATA_ROOT}/polifact/test.jsonl",
        "train_path": f"{DATA_ROOT}/polifact/train.jsonl",
        "images_dir": f"{DATA_ROOT}/polifact/images",
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/polifact",
    },
    "gossip": {
        "test_path": f"{DATA_ROOT}/gossip/test.jsonl",
        "train_path": f"{DATA_ROOT}/gossip/train.jsonl",
        "images_dir": f"{DATA_ROOT}/gossip/images",
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/gossip",
    },
    "weibo21": {
        "test_path": f"{DATA_ROOT}/weibo21/test.jsonl",
        "train_path": f"{DATA_ROOT}/weibo21/train.jsonl",
        "images_dir": f"{DATA_ROOT}/weibo21/images",
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/weibo21",
    },
    "weibo": {
        "test_path": f"{DATA_ROOT}/weibo/test.jsonl",
        "train_path": f"{DATA_ROOT}/weibo/train.jsonl",
        "images_dir": f"{DATA_ROOT}/weibo/images",
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/weibo",
    },
    "xfacta": {
        "test_path": f"{DATA_ROOT}/xfacta/test.jsonl",
        "train_path": f"{DATA_ROOT}/xfacta/train.jsonl",
        "images_dir": f"{DATA_ROOT}/xfacta/images",
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/xfacta",
    },
}
