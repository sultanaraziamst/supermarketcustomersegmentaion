"""End-to-end pipeline: load -> clean -> EDA -> features -> cluster -> profile -> export."""
import time

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from . import eda, features, preprocessing, profiling, segmentation
from .config import (DATA_PROCESSED, MAX_K, MIN_K, MODELS, RANDOM_SEED, TABLES)
from .data_loader import read_raw, resolve_source
from .spark_utils import get_spark


def run(source: str = "sample", path: str | None = None, k: int | None = None,
        min_k: int = MIN_K, max_k: int = MAX_K, n_customers: int = 1500, seed: int = RANDOM_SEED) -> dict:
    t0 = time.time()
    spark = get_spark()

    # 1. ---------------------------------------------------------------- load
    file = resolve_source(source, path, n_customers)
    print(f"[load] {file}")
    raw = read_raw(spark, file)

    # 2. -------------------------------------------------------------- clean
    sales, returns, quality = preprocessing.clean(raw)
    print(f"[clean] {quality['raw_rows']:,} raw rows -> {quality['sales_rows']:,} sales rows "
          f"({quality['retention_pct']}% kept); {quality['return_rows']:,} return rows")

    # 3. ---------------------------------------------------------------- EDA
    kpis = eda.run_transaction_eda(sales, returns, quality)

    # 4. ------------------------------------------------------------ features
    feats, snapshot = features.build_customer_features(sales, returns)
    feats = feats.cache()
    feat_pdf = feats.toPandas()
    print(f"[features] {len(feat_pdf):,} customers, snapshot date {snapshot}")
    pareto = eda.run_customer_eda(feat_pdf)

    # 5. ---------------------------------------------------------- clustering
    scaled, prep_model = segmentation.prepare_features(feats)
    print(f"[model] searching k in [{min_k}, {max_k}]")
    k_table, best_k = segmentation.select_k(scaled, min_k, max_k, seed)
    chosen_k = k or best_k
    print(f"[model] best k by silhouette = {best_k}; using k = {chosen_k}")
    comparison = segmentation.compare_algorithms(scaled, chosen_k, seed)
    print(comparison.to_string(index=False))

    kmeans_model, pred = segmentation.fit_final(scaled, chosen_k, seed)
    pred, explained = segmentation.add_pca(pred)
    segmentation.save_models(prep_model, kmeans_model, MODELS)

    # 6. ------------------------------------------------------------ profiling
    keep = [c for c in feat_pdf.columns] + ["cluster", "pc1", "pc2"]
    cust_pdf = pred.select(*keep).toPandas()
    cust_pdf, profile = profiling.profile_clusters(cust_pdf)
    print(profile[["segment", "customers", "pct_customers", "pct_revenue", "recency", "frequency",
                   "monetary"]].to_string(index=False))

    # 7. --------------------------------------------- segment-level aggregates
    seg_map = spark.createDataFrame(cust_pdf[["CustomerID", "segment"]])
    joined = sales.join(seg_map, "CustomerID")
    seg_month = joined.groupBy("InvoiceMonth", "segment").agg(
        F.sum("LineTotal").alias("revenue"), F.countDistinct("CustomerID").alias("active_customers")
    ).orderBy("InvoiceMonth").toPandas()
    w = Window.partitionBy("segment").orderBy(F.desc("revenue"))
    seg_prod = (joined.groupBy("segment", "Description").agg(F.sum("LineTotal").alias("revenue"),
                                                             F.sum("Quantity").alias("units"))
                .withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= 8)
                .orderBy("segment", "rank").toPandas())

    # 8. ---------------------------------------------------------------- save
    cust_pdf.to_csv(DATA_PROCESSED / "customers_segmented.csv", index=False)
    cust_pdf.to_csv(TABLES / "customers_segmented.csv", index=False)
    profile.to_csv(TABLES / "segment_profile.csv", index=False)
    k_table.to_csv(TABLES / "k_selection.csv", index=False)
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)
    seg_month.to_csv(TABLES / "segment_monthly_revenue.csv", index=False)
    seg_prod.to_csv(TABLES / "segment_top_products.csv", index=False)
    pd.Series({"chosen_k": chosen_k, "best_k_silhouette": best_k, "snapshot_date": str(snapshot),
               "pca_explained_variance": sum(explained), **pareto}).to_json(TABLES / "run_summary.json", indent=2)
    eda.run_segment_plots(cust_pdf, profile, k_table, explained)

    secs = time.time() - t0
    print(f"\n[done] pipeline finished in {secs:.1f}s. Launch the dashboard with:\n"
          f"       streamlit run dashboard/app.py")
    spark.stop()
    return {"kpis": kpis, "profile": profile, "k_table": k_table, "comparison": comparison, "seconds": secs}
