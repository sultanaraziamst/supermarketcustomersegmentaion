"""Central configuration: paths, column names and modelling parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

UCI_RAW_FILE = DATA_RAW / "online_retail_uci.csv"      # real data (UCI id=352)
SAMPLE_FILE = DATA_RAW / "supermarket_sample.csv"      # synthetic data, same schema

RANDOM_SEED = 42

# Stock codes that are fees/adjustments, not real products
NON_PRODUCT_CODES = {"POST", "D", "M", "DOT", "CRUK", "BANK CHARGES", "AMAZONFEE",
                     "S", "B", "PADS", "C2", "GIFT", "ADJUST"}

# Features used to cluster customers (log1p + standardised before KMeans)
CLUSTER_FEATURES = ["recency", "frequency", "monetary", "avg_basket_value", "unique_products"]

# Search range for k
MIN_K, MAX_K = 3, 8
