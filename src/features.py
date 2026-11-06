"""Customer-level feature engineering (RFM + behavioural features) in PySpark."""
from datetime import timedelta

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _score(col: str, ascending: bool):
    """1-5 quintile score from percent_rank (ties get identical scores, unlike ntile)."""
    w = Window.orderBy(F.col(col).asc() if ascending else F.col(col).desc())
    return F.greatest(F.lit(1), F.ceil(F.percent_rank().over(w) * 5)).cast("int")


def build_customer_features(sales: DataFrame, returns: DataFrame):
    """One row per customer. Returns (features_df, snapshot_date)."""
    snapshot = sales.agg(F.max(F.to_date("InvoiceDate"))).first()[0] + timedelta(days=1)
    snap = F.lit(snapshot)

    agg = sales.groupBy("CustomerID").agg(
        F.datediff(snap, F.max(F.to_date("InvoiceDate"))).alias("recency"),
        F.datediff(snap, F.min(F.to_date("InvoiceDate"))).alias("tenure_days"),
        F.countDistinct("InvoiceNo").alias("frequency"),
        F.round(F.sum("LineTotal"), 2).alias("monetary"),
        F.sum("Quantity").alias("total_items"),
        F.countDistinct("StockCode").alias("unique_products"),
        F.max("Country").alias("country"),
    )
    ret = returns.groupBy("CustomerID").agg(F.sum("ReturnValue").alias("return_value"))

    feats = (
        agg.join(ret, "CustomerID", "left")
        .fillna({"return_value": 0.0})
        .withColumn("avg_basket_value", F.round(F.col("monetary") / F.col("frequency"), 2))
        .withColumn("avg_items_per_basket", F.round(F.col("total_items") / F.col("frequency"), 2))
        .withColumn("return_rate", F.round(F.least(F.lit(1.0), F.col("return_value") / F.col("monetary")), 4))
        .drop("return_value")
        .withColumn("r_score", _score("recency", ascending=False))   # recent = high score
        .withColumn("f_score", _score("frequency", ascending=True))
        .withColumn("m_score", _score("monetary", ascending=True))
        .withColumn("rfm_score", F.concat_ws("", "r_score", "f_score", "m_score"))
    )
    return feats, snapshot
