#!/usr/bin/env python
"""CLI entry point.

Examples
--------
python run_pipeline.py --source sample                 # synthetic data, works offline
python run_pipeline.py --source uci                    # real UCI Online Retail (downloads once)
python run_pipeline.py --source file --path my.csv     # any CSV/XLSX with the Online-Retail schema
python run_pipeline.py --source uci --k 5              # force k instead of auto-select
"""
import argparse

from src.config import MAX_K, MIN_K
from src.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Supermarket customer segmentation (PySpark + MLlib)")
    ap.add_argument("--source", choices=["sample", "uci", "file"], default="sample")
    ap.add_argument("--path", help="path to CSV/XLSX when --source file")
    ap.add_argument("--k", type=int, help="fixed number of clusters (default: best silhouette)")
    ap.add_argument("--min-k", type=int, default=MIN_K)
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--n-customers", type=int, default=1500, help="size of synthetic sample")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(source=a.source, path=a.path, k=a.k, min_k=a.min_k, max_k=a.max_k,
        n_customers=a.n_customers, seed=a.seed)
