"""Exploratory data analysis. Heavy aggregation runs in Spark; only small
aggregated tables are collected to pandas for plotting. Every table is saved
to outputs/tables (used by the dashboard) and every chart to outputs/figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import FIGURES, TABLES

sns.set_theme(style="whitegrid", context="notebook")
DOW = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}


def _save_fig(name: str):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=130)
    plt.close()


def _save_table(df: pd.DataFrame, name: str):
    df.to_csv(TABLES / f"{name}.csv", index=False)


def run_transaction_eda(sales: DataFrame, returns: DataFrame, quality: dict) -> dict:
    print("[eda] transaction-level analysis")
    (TABLES / "data_quality.json").write_text(json.dumps(quality, indent=2))

    # ---- KPIs
    k = sales.agg(F.sum("LineTotal").alias("revenue"), F.countDistinct("CustomerID").alias("customers"),
                  F.countDistinct("InvoiceNo").alias("invoices"), F.countDistinct("StockCode").alias("products"),
                  F.min("InvoiceDate").alias("start"), F.max("InvoiceDate").alias("end"),
                  F.sum("Quantity").alias("units")).first()
    ret_val = returns.agg(F.sum("ReturnValue")).first()[0] or 0.0
    kpis = {"revenue": round(k["revenue"], 2), "customers": k["customers"], "invoices": k["invoices"],
            "products": k["products"], "units": int(k["units"]),
            "avg_order_value": round(k["revenue"] / k["invoices"], 2),
            "return_value": round(ret_val, 2), "return_pct": round(100 * ret_val / k["revenue"], 2),
            "start": str(k["start"].date()), "end": str(k["end"].date())}
    (TABLES / "kpis.json").write_text(json.dumps(kpis, indent=2))

    # ---- monthly trend
    monthly = (sales.groupBy("InvoiceMonth")
               .agg(F.sum("LineTotal").alias("revenue"), F.countDistinct("InvoiceNo").alias("orders"),
                    F.countDistinct("CustomerID").alias("customers"))
               .orderBy("InvoiceMonth").toPandas())
    _save_table(monthly, "monthly_revenue")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly["InvoiceMonth"], monthly["revenue"], marker="o", color="#1f77b4")
    ax.fill_between(range(len(monthly)), monthly["revenue"], alpha=0.15, color="#1f77b4")
    ax.set(title="Monthly revenue", xlabel="", ylabel="Revenue")
    plt.xticks(rotation=45)
    _save_fig("01_monthly_revenue")

    # ---- top products
    prod = (sales.groupBy("StockCode", "Description")
            .agg(F.sum("LineTotal").alias("revenue"), F.sum("Quantity").alias("units"),
                 F.countDistinct("CustomerID").alias("buyers"))
            .orderBy(F.desc("revenue")).limit(20).toPandas())
    _save_table(prod, "top_products")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=prod.head(10), y="Description", x="revenue", color="#2ca02c", ax=ax)
    ax.set(title="Top 10 products by revenue", ylabel="")
    _save_fig("02_top_products")

    # ---- countries
    country = (sales.groupBy("Country").agg(F.sum("LineTotal").alias("revenue"),
                                            F.countDistinct("CustomerID").alias("customers"))
               .orderBy(F.desc("revenue")).toPandas())
    _save_table(country, "country_revenue")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=country.head(8), x="revenue", y="Country", color="#ff7f0e", ax=ax)
    ax.set(title="Revenue by country (top 8)", ylabel="")
    _save_fig("03_country_revenue")

    # ---- weekday x hour heatmap
    dh = (sales.groupBy("DayOfWeek", "Hour").agg(F.sum("LineTotal").alias("revenue")).toPandas())
    dh["day"] = dh["DayOfWeek"].map(DOW)
    _save_table(dh[["day", "DayOfWeek", "Hour", "revenue"]], "dow_hour_revenue")
    pivot = dh.pivot(index="day", columns="Hour", values="revenue").reindex(list(DOW.values())).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Revenue"})
    ax.set(title="When do customers shop? (revenue by weekday & hour)", xlabel="Hour", ylabel="")
    _save_fig("04_weekday_hour_heatmap")

    # ---- invoice value + basket size distributions
    inv = (sales.groupBy("InvoiceNo").agg(F.sum("LineTotal").alias("invoice_value"),
                                          F.sum("Quantity").alias("items"),
                                          F.countDistinct("StockCode").alias("lines")).toPandas())
    _save_table(inv, "invoice_values")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(np.log10(inv["invoice_value"]), bins=40, color="#9467bd", ax=axes[0])
    axes[0].set(title="Invoice value (log10 scale)", xlabel="log10(value)")
    sns.histplot(inv["lines"].clip(upper=inv["lines"].quantile(0.99)), bins=40, color="#8c564b", ax=axes[1])
    axes[1].set(title="Distinct products per invoice", xlabel="lines")
    _save_fig("05_invoice_distributions")

    print(f"[eda]   revenue={kpis['revenue']:,.0f}  customers={kpis['customers']:,}  AOV={kpis['avg_order_value']}")
    return kpis


def run_customer_eda(feat: pd.DataFrame):
    """Distributions + correlations of the customer-level features."""
    print("[eda] customer-level analysis")
    cols = ["recency", "frequency", "monetary", "avg_basket_value", "unique_products"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for i, c in enumerate(cols):
        sns.histplot(feat[c], bins=40, ax=axes[0, i], color="#1f77b4")
        axes[0, i].set_title(f"{c} (raw)")
        sns.histplot(np.log1p(feat[c]), bins=40, ax=axes[1, i], color="#d62728")
        axes[1, i].set_title(f"{c} (log1p)")
    plt.suptitle("Why we log-transform: RFM features are heavily right-skewed", y=1.02)
    _save_fig("06_feature_distributions")

    num = feat[["recency", "frequency", "monetary", "avg_basket_value", "avg_items_per_basket",
                "unique_products", "tenure_days", "return_rate"]]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation")
    _save_fig("07_correlation_heatmap")

    # Pareto: share of revenue from the top customers
    s = feat["monetary"].sort_values(ascending=False).reset_index(drop=True)
    cum = s.cumsum() / s.sum()
    x = (np.arange(1, len(s) + 1) / len(s)) * 100
    pareto = pd.DataFrame({"pct_customers": x, "cum_pct_revenue": cum * 100})
    _save_table(pareto.iloc[:: max(1, len(pareto) // 400)], "pareto_curve")
    top20 = float(cum.iloc[int(0.2 * len(s)) - 1] * 100)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x, cum * 100, color="#1f77b4")
    ax.axvline(20, ls="--", color="grey")
    ax.axhline(top20, ls="--", color="grey")
    ax.set(title=f"Pareto: top 20% of customers = {top20:.0f}% of revenue",
           xlabel="% of customers", ylabel="% of revenue")
    _save_fig("08_pareto")
    return {"top20_revenue_share": round(top20, 1)}


def run_segment_plots(cust: pd.DataFrame, profile: pd.DataFrame, k_table: pd.DataFrame, explained):
    """Model + segment visuals."""
    print("[eda] segment visuals")
    palette = dict(zip(profile["segment"], sns.color_palette("tab10", len(profile))))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_table["k"], k_table["wssse"], marker="o")
    axes[0].set(title="Elbow method (WSSSE)", xlabel="k", ylabel="Within-set SSE")
    axes[1].plot(k_table["k"], k_table["silhouette"], marker="o", color="#2ca02c")
    axes[1].set(title="Silhouette score", xlabel="k")
    _save_fig("09_k_selection")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=cust, x="pc1", y="pc2", hue="segment", palette=palette, s=18, alpha=0.7, ax=ax)
    ax.set(title=f"Customer segments in PCA space ({sum(explained) * 100:.0f}% variance explained)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_fig("10_segments_pca")

    fig, ax = plt.subplots(figsize=(7, 5))
    p = profile.sort_values("customers")
    ax.barh(p["segment"], p["customers"], color=[palette[s] for s in p["segment"]])
    ax.set(title="Customers per segment")
    _save_fig("11_segment_sizes")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, c in zip(axes, ["recency", "frequency", "monetary"]):
        sns.boxplot(data=cust, y="segment", x=c, hue="segment", palette=palette, ax=ax, legend=False,
                    showfliers=False)
        ax.set_title(c)
        if c != "recency":
            ax.set_xscale("log")
    _save_fig("12_segment_rfm_boxplots")
