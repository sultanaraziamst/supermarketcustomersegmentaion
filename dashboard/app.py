"""Interactive dashboard for the segmentation results.

Run:  streamlit run dashboard/app.py
(Requires `python run_pipeline.py` to have produced files in outputs/tables first.)
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TABLES = Path(__file__).resolve().parent.parent / "outputs" / "tables"

st.set_page_config(page_title="Supermarket Customer Segmentation", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load():
    t = {n: pd.read_csv(TABLES / f"{n}.csv") for n in
         ["customers_segmented", "segment_profile", "monthly_revenue", "top_products", "country_revenue",
          "dow_hour_revenue", "invoice_values", "k_selection", "model_comparison", "segment_monthly_revenue",
          "segment_top_products", "pareto_curve"]}
    t["kpis"] = json.loads((TABLES / "kpis.json").read_text())
    t["quality"] = json.loads((TABLES / "data_quality.json").read_text())
    t["summary"] = json.loads((TABLES / "run_summary.json").read_text())
    return t


if not (TABLES / "customers_segmented.csv").exists():
    st.error("No results found. Run `python run_pipeline.py --source sample` (or `--source uci`) first.")
    st.stop()

D = load()
cust, prof = D["customers_segmented"], D["segment_profile"]
segments = list(prof["segment"])
PALETTE = px.colors.qualitative.Safe
COLORS = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(segments)}
money = lambda v: f"£{v:,.0f}"

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🛒 Filters")
sel_seg = st.sidebar.multiselect("Segment", segments, default=segments)
countries = sorted(cust["country"].dropna().unique())
sel_country = st.sidebar.multiselect("Country", countries, default=countries)
rmin, rmax = int(cust["recency"].min()), int(cust["recency"].max())
sel_rec = st.sidebar.slider("Recency (days since last purchase)", rmin, rmax, (rmin, rmax))
view = cust[cust["segment"].isin(sel_seg) & cust["country"].isin(sel_country)
            & cust["recency"].between(*sel_rec)]
st.sidebar.caption(f"{len(view):,} of {len(cust):,} customers selected")
st.sidebar.divider()
st.sidebar.caption(f"Snapshot date: {D['summary']['snapshot_date']}  \nk = {D['summary']['chosen_k']} clusters")

st.title("Supermarket Customer Segmentation")
st.caption("PySpark + MLlib · RFM features · KMeans clustering")

tab_over, tab_eda, tab_seg, tab_play, tab_exp, tab_model = st.tabs(
    ["📊 Overview", "🔎 EDA", "🧩 Segments", "🎯 Playbook", "👤 Customer explorer", "⚙️ Model quality"])

# ----------------------------------------------------------------------------- overview
with tab_over:
    k = D["kpis"]
    c = st.columns(5)
    c[0].metric("Revenue", money(k["revenue"]))
    c[1].metric("Customers", f"{k['customers']:,}")
    c[2].metric("Orders", f"{k['invoices']:,}")
    c[3].metric("Avg order value", f"£{k['avg_order_value']:.2f}")
    c[4].metric("Return rate", f"{k['return_pct']}%")
    st.caption(f"Period: {k['start']} → {k['end']}  ·  {k['products']:,} products")

    a, b = st.columns([3, 2])
    m = D["monthly_revenue"]
    fig = go.Figure()
    fig.add_bar(x=m["InvoiceMonth"], y=m["revenue"], name="Revenue", marker_color="#4C78A8")
    fig.add_scatter(x=m["InvoiceMonth"], y=m["customers"], name="Active customers", yaxis="y2",
                    line=dict(color="#F58518"))
    fig.update_layout(title="Monthly revenue & active customers", yaxis2=dict(overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.2), height=380, margin=dict(t=50))
    a.plotly_chart(fig, use_container_width=True)

    fig = px.pie(prof, names="segment", values="total_revenue", hole=0.5, color="segment",
                 color_discrete_map=COLORS, title="Revenue share by segment")
    fig.update_layout(height=380, margin=dict(t=50))
    b.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    sm = D["segment_monthly_revenue"]
    fig = px.area(sm, x="InvoiceMonth", y="revenue", color="segment", color_discrete_map=COLORS,
                  title="Monthly revenue by segment")
    a.plotly_chart(fig, use_container_width=True)
    p = D["pareto_curve"]
    fig = px.line(p, x="pct_customers", y="cum_pct_revenue", title="Pareto curve: revenue concentration")
    fig.add_vline(x=20, line_dash="dash", line_color="grey")
    fig.update_layout(xaxis_title="% of customers (best first)", yaxis_title="% of revenue")
    b.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- EDA
with tab_eda:
    a, b = st.columns(2)
    tp = D["top_products"].head(10).sort_values("revenue")
    a.plotly_chart(px.bar(tp, x="revenue", y="Description", orientation="h", title="Top 10 products by revenue"),
                   use_container_width=True)
    ct = D["country_revenue"].head(8).sort_values("revenue")
    b.plotly_chart(px.bar(ct, x="revenue", y="Country", orientation="h", title="Revenue by country (top 8)"),
                   use_container_width=True)

    dh = D["dow_hour_revenue"]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pv = dh.pivot(index="day", columns="Hour", values="revenue").reindex(order)
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="YlOrRd",
                              labels=dict(x="Hour", y="", color="Revenue"),
                              title="When do customers shop? (revenue by weekday × hour)"),
                    use_container_width=True)

    a, b = st.columns(2)
    inv = D["invoice_values"]
    a.plotly_chart(px.histogram(inv, x="invoice_value", nbins=60, log_y=True, title="Invoice value distribution"),
                   use_container_width=True)
    b.plotly_chart(px.histogram(cust, x="monetary", nbins=60, log_x=True, title="Customer lifetime spend (log x)"),
                   use_container_width=True)

    with st.expander("Data-quality report (cleaning step)"):
        q = D["quality"]
        qdf = pd.DataFrame({"metric": list(q), "value": list(q.values())})
        st.dataframe(qdf, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- segments
with tab_seg:
    show = prof[prof["segment"].isin(sel_seg)]
    st.dataframe(
        show[["segment", "customers", "pct_customers", "pct_revenue", "recency", "frequency", "monetary",
              "avg_basket_value", "unique_products", "return_rate"]],
        hide_index=True, use_container_width=True,
        column_config={"pct_customers": st.column_config.ProgressColumn("% customers", min_value=0, max_value=100, format="%.1f%%"),
                       "pct_revenue": st.column_config.ProgressColumn("% revenue", min_value=0, max_value=100, format="%.1f%%"),
                       "monetary": st.column_config.NumberColumn("avg spend", format="£%.0f"),
                       "avg_basket_value": st.column_config.NumberColumn("avg basket", format="£%.2f")})

    a, b = st.columns(2)
    fig = px.scatter(view, x="pc1", y="pc2", color="segment", color_discrete_map=COLORS,
                     hover_data=["CustomerID", "recency", "frequency", "monetary"],
                     title="Segments in PCA space", opacity=0.75)
    fig.update_layout(height=470)
    a.plotly_chart(fig, use_container_width=True)

    fig = px.scatter_3d(view, x="recency", y="frequency", z="monetary", color="segment",
                        color_discrete_map=COLORS, log_y=True, log_z=True, opacity=0.7,
                        title="Recency × Frequency × Monetary")
    fig.update_traces(marker_size=3)
    fig.update_layout(height=470)
    b.plotly_chart(fig, use_container_width=True)

    # radar: normalised cluster means
    metrics = ["recency", "frequency", "monetary", "avg_basket_value", "unique_products"]
    norm = prof.set_index("segment")[metrics].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min()).replace(0, 1)
    norm["recency"] = 1 - norm["recency"]       # invert: recent = good
    fig = go.Figure()
    for s in norm.index:
        if s in sel_seg:
            fig.add_scatterpolar(r=norm.loc[s].tolist() + [norm.loc[s].iloc[0]],
                                 theta=["recency (inverted)", "frequency", "monetary", "avg basket", "product variety",
                                        "recency (inverted)"],
                                 name=s, line_color=COLORS[s], fill="toself", opacity=0.45)
    fig.update_layout(title="Segment fingerprint (normalised 0–1, higher = better)", height=480,
                      polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- playbook
with tab_play:
    st.subheader("Recommended marketing actions")
    for _, r in prof.iterrows():
        if r["segment"] not in sel_seg:
            continue
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {r['segment']}")
            c1.metric("Customers", f"{int(r['customers']):,}", f"{r['pct_customers']}% of base", delta_color="off")
            c1.metric("Revenue share", f"{r['pct_revenue']}%")
            c2.markdown(f"**Action:** {r['recommended_action']}")
            c2.caption(f"Avg recency {r['recency']:.0f} days · {r['frequency']:.1f} orders · "
                       f"£{r['monetary']:,.0f} lifetime spend · basket £{r['avg_basket_value']:.0f}")
            tp = D["segment_top_products"]
            tp = tp[tp["segment"] == r["segment"]].head(5)
            if len(tp):
                c2.markdown("**Top products:** " + " · ".join(tp["Description"].astype(str)))

# ----------------------------------------------------------------------------- explorer
with tab_exp:
    q = st.text_input("Search CustomerID")
    tbl = view if not q else view[view["CustomerID"].astype(str).str.contains(q)]
    cols = ["CustomerID", "segment", "country", "recency", "frequency", "monetary", "avg_basket_value",
            "unique_products", "return_rate", "rfm_score"]
    st.dataframe(tbl[cols].sort_values("monetary", ascending=False), hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download filtered customers (CSV)", tbl[cols].to_csv(index=False).encode(),
                       "customers_filtered.csv", "text/csv")

# ----------------------------------------------------------------------------- model
with tab_model:
    a, b = st.columns(2)
    ks = D["k_selection"]
    a.plotly_chart(px.line(ks, x="k", y="wssse", markers=True, title="Elbow method"), use_container_width=True)
    fig = px.line(ks, x="k", y="silhouette", markers=True, title="Silhouette score by k")
    best = D["summary"]["best_k_silhouette"]
    fig.add_vline(x=best, line_dash="dash", line_color="green", annotation_text=f"best k = {best}")
    b.plotly_chart(fig, use_container_width=True)
    st.subheader("Algorithm comparison (MLlib)")
    st.dataframe(D["model_comparison"], hide_index=True, use_container_width=True)
    st.caption(f"PCA (2 components) explains {D['summary']['pca_explained_variance'] * 100:.0f}% of variance. "
               "Features are log1p-transformed and standardised before clustering.")
