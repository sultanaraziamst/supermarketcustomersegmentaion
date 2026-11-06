"""Unit + smoke tests on a tiny hand-made dataset. Run: pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features, preprocessing, profiling, segmentation  # noqa: E402
from src.sample_data import generate  # noqa: E402
from src.spark_utils import get_spark  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = get_spark("tests", driver_memory="1g", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture(scope="module")
def raw(spark):
    pdf = generate(n_customers=120, seed=1)
    return spark.createDataFrame(pdf.astype(str).replace({"None": None, "nan": None}))


def test_cleaning_removes_bad_rows(raw):
    sales, returns, q = preprocessing.clean(raw)
    assert sales.filter("Quantity <= 0 OR UnitPrice <= 0").count() == 0
    assert sales.filter("CustomerID IS NULL").count() == 0
    assert sales.filter("InvoiceNo LIKE 'C%'").count() == 0
    assert sales.filter("StockCode = 'POST'").count() == 0
    assert q["raw_rows"] > q["sales_rows"] > 0
    assert returns.count() > 0


def test_features_one_row_per_customer(raw):
    sales, returns, _ = preprocessing.clean(raw)
    feats, snap = features.build_customer_features(sales, returns)
    pdf = feats.toPandas()
    assert pdf["CustomerID"].is_unique
    assert (pdf["recency"] >= 1).all() and (pdf["frequency"] >= 1).all()
    assert pdf[["r_score", "f_score", "m_score"]].stack().between(1, 5).all()
    assert pdf["return_rate"].between(0, 1).all()


def test_segmentation_end_to_end(raw):
    sales, returns, _ = preprocessing.clean(raw)
    feats, _ = features.build_customer_features(sales, returns)
    scaled, _ = segmentation.prepare_features(feats)
    table, best = segmentation.select_k(scaled, 3, 5)
    assert 3 <= best <= 5 and table["silhouette"].between(-1, 1).all()
    _, pred = segmentation.fit_final(scaled, 4)
    pred, ev = segmentation.add_pca(pred)
    pdf, prof = profiling.profile_clusters(
        pred.select(*feats.columns, "cluster", "pc1", "pc2").toPandas())
    assert prof["customers"].sum() == len(pdf)
    assert prof["segment"].is_unique
    assert abs(prof["pct_revenue"].sum() - 100) < 1


def test_segment_naming_rules():
    assert profiling.name_segment(5, 5, 5) == "Champions"
    assert profiling.name_segment(1, 1, 1) == "Hibernating"
    assert profiling.name_segment(1.5, 3.5, 3.5) == "At Risk"
    assert profiling.name_segment(4.5, 1.2, 1.2) == "New Customers"
