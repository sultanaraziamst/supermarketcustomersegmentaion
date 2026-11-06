# 🛒 Supermarket Customer Segmentation — PySpark + MLlib

End-to-end customer segmentation project: **data → cleaning → EDA → RFM feature engineering → MLlib clustering → named business segments → interactive dashboard**.

* **Engine:** Apache Spark (PySpark 3.5) for all cleaning, aggregation, feature engineering and modelling
* **Models:** MLlib `KMeans` (final), benchmarked against `BisectingKMeans` and `GaussianMixture`; `PCA` for visualisation
* **Dashboard:** Streamlit + Plotly (6 tabs, sidebar filters, CSV export)
* **Data:** UCI *Online Retail* (real, downloaded automatically) — or a bundled synthetic generator with the identical schema for offline use

---

## 1. Dataset

| | |
|---|---|
| **Primary (real)** | [UCI – Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (id 352) · ~541k transactions · Dec 2010 – Dec 2011 · CC BY 4.0 |
| **Also supported** | UCI *Online Retail II* (id 502) and the Kaggle "Online Retail" / "Online Retail II" CSVs — column names are normalised automatically (`Invoice`→`InvoiceNo`, `Price`→`UnitPrice`, `Customer ID`→`CustomerID`) |
| **Offline fallback** | `src/sample_data.py` generates ~175k rows (1,500 customers, 6 hidden shopper archetypes, injected dirty data) in the same schema |

Schema: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

> ⚠️ The numbers in the "Sample results" section below come from the **synthetic** data (the sandbox this was built in could not reach UCI). Run `--source uci` on your machine to get real-data results — nothing else needs to change.

## 2. Project structure

```
supermarket-segmentation/
├── README.md
├── requirements.txt
├── Makefile                  # make sample | uci | dashboard | test
├── Dockerfile                # Java + Python + Spark in one image
├── run_pipeline.py           # CLI entry point
├── src/
│   ├── config.py             # paths, cluster features, k range
│   ├── spark_utils.py        # SparkSession factory
│   ├── sample_data.py        # synthetic UCI-schema data generator
│   ├── data_loader.py        # UCI download / local file reader
│   ├── preprocessing.py      # cleaning (PySpark) + data-quality report
│   ├── eda.py                # EDA tables + charts
│   ├── features.py           # RFM + behavioural features (PySpark)
│   ├── segmentation.py       # MLlib: scaling, k-selection, model comparison, KMeans, PCA
│   ├── profiling.py          # cluster -> named segment + marketing action
│   └── pipeline.py           # orchestration
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_pipeline.py      # pytest (cleaning, features, clustering, naming)
├── data/
│   ├── raw/                  # downloaded / generated input
│   └── processed/            # customers_segmented.csv
└── outputs/
    ├── figures/              # 12 PNG charts from EDA + modelling
    ├── tables/               # CSV/JSON consumed by the dashboard
    └── models/               # saved Spark PipelineModel + KMeansModel
```

## 3. Quick start

**Requirements:** Python 3.9+ and **Java 8/11/17/21** (`java -version`). Spark runs in local mode, no cluster needed.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# A) real data from UCI (downloads once, ~45 MB, cached in data/raw/)
python run_pipeline.py --source uci

# B) offline synthetic data
python run_pipeline.py --source sample

# C) your own file (CSV/XLSX in Online-Retail schema, e.g. Kaggle download)
python run_pipeline.py --source file --path /path/to/data.csv

# then
streamlit run dashboard/app.py          # opens http://localhost:8501
```

Useful flags: `--k 5` (force k instead of auto-select by silhouette), `--min-k 3 --max-k 10`, `--n-customers 5000` (sample size), `--seed 7`.
Run time: ~1–2 min for the sample, a few minutes for full UCI data on a laptop.

**Docker (no Java install needed):**
```bash
docker build -t segmentation . && docker run -p 8501:8501 -e SOURCE=uci segmentation
```

**Windows tips:** easiest is WSL2 or Docker. Native Windows works for the pipeline, but saving Spark models needs `HADOOP_HOME` with `winutils.exe`; without it the pipeline just prints a warning and skips model saving (all CSV outputs and the dashboard are unaffected).

## 4. Methodology

1. **Cleaning (`preprocessing.py`)** – parse types/dates; drop exact duplicates; drop rows without `CustomerID` (guest checkouts can't be segmented); remove non-product codes (`POST`, `BANK CHARGES`, …); split **cancellations** (`InvoiceNo` starts with `C`) into a separate *returns* table; keep only `Quantity > 0` and `UnitPrice > 0` as *sales*. A data-quality report records rows removed at every step.
2. **EDA (`eda.py`)** – KPIs, monthly trend, top products, country mix, weekday × hour heatmap, invoice-value / basket distributions, feature skewness, correlation heatmap and Pareto curve (revenue concentration).
3. **Features (`features.py`)** – per customer: `recency` (days since last purchase vs. snapshot = last date + 1), `frequency` (distinct invoices), `monetary` (total spend), `avg_basket_value`, `avg_items_per_basket`, `unique_products`, `tenure_days`, `return_rate`, plus 1-5 quintile **R/F/M scores** (percent-rank based so ties score identically).
4. **Modelling (`segmentation.py`)** – clustering features `recency, frequency, monetary, avg_basket_value, unique_products` → `log1p` (they are heavily right-skewed) → `VectorAssembler` → `StandardScaler`.
   *k* is chosen by **silhouette** over k ∈ [3, 8] (elbow/WSSSE shown too). KMeans is benchmarked against Bisecting KMeans and Gaussian Mixture.
5. **Profiling (`profiling.py`)** – each cluster's mean R/F/M scores are mapped to a business name (Champions, Loyal, Big Spenders, At Risk, Hibernating, New, Potential Loyalists) with a recommended marketing action. Rules are plain Python so they're easy to tune.
6. **Dashboard** – everything above, interactive.

## 5. Dashboard tabs

| Tab | Contents |
|---|---|
| 📊 Overview | KPIs, monthly revenue + active customers, revenue share by segment, segment revenue over time, Pareto curve |
| 🔎 EDA | top products, countries, weekday×hour heatmap, distributions, data-quality report |
| 🧩 Segments | profile table, PCA scatter, 3-D RFM scatter, radar "fingerprint" per segment |
| 🎯 Playbook | per-segment size, revenue share, recommended action, top products |
| 👤 Customer explorer | searchable/filterable table + CSV download |
| ⚙️ Model quality | elbow, silhouette, algorithm comparison |

Sidebar filters (segment, country, recency range) apply across tabs.

## 6. Sample results (synthetic data, 1,500 customers)

Auto-selected **k = 5** (silhouette 0.654). KMeans clearly beat the alternatives:

| Algorithm | Silhouette |
|---|---|
| **KMeans** | **0.654** |
| Bisecting KMeans | 0.557 |
| Gaussian Mixture | 0.367 |

| Segment | % customers | % revenue | Avg recency (days) | Avg orders | Avg spend |
|---|---|---|---|---|---|
| Champions | 23.5 | 53.1 | 18 | 22.5 | £1,745 |
| Big Spenders | 4.1 | 27.0 | 55 | 6.9 | £5,058 |
| At Risk | 30.8 | 17.5 | 135 | 9.8 | £439 |
| Hibernating | 23.0 | 1.2 | 214 | 1.6 | £40 |
| New Customers | 18.5 | 1.2 | 15 | 1.7 | £50 |

Pre-computed outputs for this run are already included in `outputs/`, so `streamlit run dashboard/app.py` works immediately after unzipping.

## 7. Customising

* **Different features / k range:** edit `CLUSTER_FEATURES`, `MIN_K`, `MAX_K` in `src/config.py`.
* **Segment names & actions:** `name_segment()` and `ACTIONS` in `src/profiling.py`.
* **Larger data / cluster:** change `.master("local[*]")` in `src/spark_utils.py` to your cluster URL; replace the `.toPandas()` calls (only on small aggregated tables and the per-customer table) as needed.
* **New dataset:** anything with the 8 Online-Retail columns works via `--source file`.

## 8. Tests

```bash
pytest -q        # 4 tests: cleaning rules, feature integrity, clustering end-to-end, naming rules
```

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `JAVA_HOME is not set` / `Unable to locate a Java Runtime` | Install JDK 17 and set `JAVA_HOME` |
| `ucimlrepo` download fails (proxy/firewall) | Download the file manually from UCI or Kaggle, then `--source file --path ...` |
| Out-of-memory on the full dataset | Raise `driver_memory` in `src/spark_utils.py` |
| Dashboard says "No results found" | Run `python run_pipeline.py` first |

## 10. Credits

Dataset: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 (CC BY 4.0).
