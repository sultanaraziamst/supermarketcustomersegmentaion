"""Turn raw clusters into named business segments with recommended actions."""
import pandas as pd

PROFILE_COLS = ["recency", "frequency", "monetary", "avg_basket_value", "avg_items_per_basket",
                "unique_products", "tenure_days", "return_rate", "r_score", "f_score", "m_score"]

ACTIONS = {
    "Champions": "Reward them: early access, loyalty tier upgrades, referral programme. Don't discount - they already buy.",
    "Loyal Customers": "Upsell premium / larger baskets, personalised bundles, points-multiplier events.",
    "Big Spenders": "Offer bulk-buy pricing, delivery subscriptions and a dedicated account contact.",
    "Potential Loyalists": "Nudge towards a regular rhythm: 'next visit' coupons, category cross-sell, app onboarding.",
    "New Customers": "Welcome journey: 2nd-purchase coupon within 14 days, product education, collect preferences.",
    "At Risk": "Win-back now: personalised 'we miss you' offer based on past favourites, survey on churn reasons.",
    "Hibernating": "Low-cost reactivation (email/SMS), clearance offers; suppress from paid channels if unresponsive.",
}


def name_segment(r: float, f: float, m: float) -> str:
    """Rule-based naming from the cluster's mean RFM quintile scores (1 = worst, 5 = best)."""
    if r >= 3.8 and f >= 3.8 and m >= 3.8:
        return "Champions"
    if m >= 3.8 and f < 3.8 and r >= 2.5:
        return "Big Spenders"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    if r <= 2.4 and (f >= 2.5 or m >= 2.5):
        return "At Risk"
    if r <= 2.4:
        return "Hibernating"
    if r >= 3.2 and f <= 2.6:
        return "New Customers"
    return "Potential Loyalists"


def profile_clusters(pdf: pd.DataFrame):
    """pdf: customer-level pandas frame with a `cluster` column.
    Returns (customers_with_segment, profile_table)."""
    prof = pdf.groupby("cluster").agg(
        customers=("CustomerID", "count"), total_revenue=("monetary", "sum"),
        **{c: (c, "mean") for c in PROFILE_COLS}).reset_index()

    names, seen = {}, {}
    for _, row in prof.sort_values("m_score", ascending=False).iterrows():
        base = name_segment(row.r_score, row.f_score, row.m_score)
        seen[base] = seen.get(base, 0) + 1
        names[row.cluster] = base if seen[base] == 1 else f"{base} #{seen[base]}"
    prof["segment"] = prof["cluster"].map(names)
    prof["pct_customers"] = (100 * prof["customers"] / prof["customers"].sum()).round(1)
    prof["pct_revenue"] = (100 * prof["total_revenue"] / prof["total_revenue"].sum()).round(1)
    prof["recommended_action"] = prof["segment"].map(
        lambda s: ACTIONS.get(s.split(" #")[0], ACTIONS["Potential Loyalists"]))
    prof = prof.round(2).sort_values("total_revenue", ascending=False).reset_index(drop=True)
    pdf = pdf.assign(segment=pdf["cluster"].map(names))
    return pdf, prof
