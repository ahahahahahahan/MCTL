"""
MCTL 实验主程序
用法:
    python main.py --exp exp0                   # Exp0: Baseline
    python main.py --exp exp1                   # Exp1: +CAMR
    python main.py --exp exp2                   # Exp2: +CAMR+MCP
    python main.py --exp exp3                   # Exp3: +CAMR+MCP+ATR
    python main.py --exp exp4                   # Exp4: +CAMR+MCP+ATR+MPRE
    python main.py --exp exp5                   # Exp5: Full MCTL (+CBDF)
    python main.py --dataset polifact --exp exp5
"""
import argparse
from models import (
    BaselineDetector,
    BaselineCAMRDetector,
    BaselineCAMRMCPDetector,
    BaselineCAMRMCPATRDetector,
    BaselineCAMRMCPATRMPREDetector,
    MCTLDetector,
)


def main():
    parser = argparse.ArgumentParser(description="MCTL 实验")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["polifact", "gossip", "weibo21", "weibo"])
    parser.add_argument("--exp", type=str, default="exp0",
                        choices=["exp0", "exp1", "exp2", "exp3", "exp4", "exp5"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    detectors = {
        "exp0": lambda: BaselineDetector(),
        "exp1": lambda: BaselineCAMRDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp2": lambda: BaselineCAMRMCPDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp3": lambda: BaselineCAMRMCPATRDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp4": lambda: BaselineCAMRMCPATRMPREDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp5": lambda: MCTLDetector(top_k=args.top_k, clip_model=args.clip_model),
    }

    detector = detectors[args.exp]()
    results = detector.run(dataset_type=args.dataset)


if __name__ == "__main__":
    main()
