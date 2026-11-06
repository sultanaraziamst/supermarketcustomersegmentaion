"""Data acquisition: UCI download (real data), synthetic sample, or any local file."""
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from .config import SAMPLE_FILE, UCI_RAW_FILE
from .sample_data import write_sample

UCI_ONLINE_RETAIL_ID = 352   # https://archive.ics.uci.edu/dataset/352/online+retail


def download_uci(force: bool = False) -> Path:
    """Download UCI 'Online Retail' via the official `ucimlrepo` package and
    store it as an ISO-dated CSV. (UCI id=352, ~541k rows, 2010-12 .. 2011-12.)"""
    if UCI_RAW_FILE.exists() and not force:
        print(f"[data] using cached {UCI_RAW_FILE}")
        return UCI_RAW_FILE
    from ucimlrepo import fetch_ucirepo

    print("[data] downloading UCI Online Retail (id=352) ...")
    ds = fetch_ucirepo(id=UCI_ONLINE_RETAIL_ID)
    df = ds.data.original.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(UCI_RAW_FILE, index=False)
    print(f"[data] saved {len(df):,} rows -> {UCI_RAW_FILE}")
    return UCI_RAW_FILE


def resolve_source(source: str, path: str | None = None, n_customers: int = 1500) -> Path:
    """source: 'uci' | 'sample' | 'file' (needs path)."""
    if source == "uci":
        return download_uci()
    if source == "sample":
        if not SAMPLE_FILE.exists():
            write_sample(n_customers=n_customers)
        return SAMPLE_FILE
    if source == "file":
        if not path:
            raise ValueError("--path is required when --source file")
        return Path(path)
    raise ValueError(f"Unknown source {source!r}")


def read_raw(spark: SparkSession, path: Path) -> DataFrame:
    """Read CSV (or xlsx via pandas) as all-string columns; types are cast in preprocessing."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        pdf = pd.read_excel(path, dtype=str)
        return spark.createDataFrame(pdf.astype(str).replace({"nan": None, "NaT": None}))
    return spark.read.option("header", True).option("multiLine", False).csv(str(path))
