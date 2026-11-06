"""Cleaning with PySpark. Produces a `sales` table, a `returns` table and a data-quality report."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import NON_PRODUCT_CODES

DATE_FORMATS = ["yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd'T'HH:mm:ss", "M/d/yyyy H:mm", "MM/dd/yyyy HH:mm",
                "d/M/yyyy H:mm", "yyyy-MM-dd HH:mm"]
# Column-name variants (Kaggle / UCI "Online Retail II") -> canonical names
RENAMES = {"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"}
REQUIRED = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def standardise_columns(df: DataFrame) -> DataFrame:
    for old, new in RENAMES.items():
        if old in df.columns and new not in df.columns:
            df = df.withColumnRenamed(old, new)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    return df.select(*REQUIRED)


def parse_types(df: DataFrame) -> DataFrame:
    ts = F.coalesce(*[F.to_timestamp(F.col("InvoiceDate"), f) for f in DATE_FORMATS])
    cust = F.trim(F.col("CustomerID").cast("string"))
    cust = F.regexp_replace(cust, r"\.0$", "")
    return (
        df.withColumn("InvoiceNo", F.trim(F.col("InvoiceNo").cast("string")))
        .withColumn("StockCode", F.upper(F.trim(F.col("StockCode").cast("string"))))
        .withColumn("Description", F.trim(F.col("Description")))
        .withColumn("Quantity", F.col("Quantity").cast("int"))
        .withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
        .withColumn("InvoiceDate", ts)
        .withColumn("CustomerID", F.when(cust == "", None).when(cust == "nan", None).otherwise(cust))
        .withColumn("Country", F.trim(F.col("Country")))
    )


def clean(raw: DataFrame):
    """Return (sales, returns, quality_report)."""
    df = parse_types(standardise_columns(raw))
    report = {"raw_rows": df.count()}

    df = df.dropDuplicates()
    report["after_dedup"] = df.count()
    report["duplicates_removed"] = report["raw_rows"] - report["after_dedup"]

    df = df.filter(F.col("InvoiceDate").isNotNull())
    report["null_customer_rows"] = df.filter(F.col("CustomerID").isNull()).count()
    report["null_description_rows"] = df.filter(F.col("Description").isNull()).count()
    df = df.filter(F.col("CustomerID").isNotNull())

    df = df.filter(~F.col("StockCode").isin(*NON_PRODUCT_CODES))
    is_cancel = F.col("InvoiceNo").startswith("C")

    returns = (df.filter(is_cancel & (F.col("UnitPrice") > 0))
               .withColumn("ReturnValue", F.abs(F.col("Quantity")) * F.col("UnitPrice")))

    sales = (df.filter(~is_cancel & (F.col("Quantity") > 0) & (F.col("UnitPrice") > 0))
             .withColumn("LineTotal", F.col("Quantity") * F.col("UnitPrice"))
             .withColumn("InvoiceMonth", F.date_format("InvoiceDate", "yyyy-MM"))
             .withColumn("DayOfWeek", F.dayofweek("InvoiceDate"))   # 1=Sun .. 7=Sat
             .withColumn("Hour", F.hour("InvoiceDate"))
             .cache())

    report["return_rows"] = returns.count()
    report["sales_rows"] = sales.count()
    report["invalid_rows_removed"] = (report["after_dedup"] - report["null_customer_rows"]
                                      - report["sales_rows"] - report["return_rows"])
    report["retention_pct"] = round(100 * report["sales_rows"] / max(report["raw_rows"], 1), 2)
    return sales, returns, report
