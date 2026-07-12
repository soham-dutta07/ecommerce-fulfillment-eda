"""
PHASE 3: STRATEGIC EDA
--------------------------------------------------------------
Three targeted investigations, each ending in a business-ready number.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
PLOT_DIR = "/home/claude/project/plots"

df = pd.read_csv(
    "/home/claude/project/data/fulfillment_clean.csv",
    parse_dates=["order_timestamp", "scheduled_delivery_date", "actual_delivery_date"]
)
resolved = df[~df["is_unresolved_shipment"]].copy()

# ==================================================================
# INSIGHT 1: DELIVERY VARIANCE BY CARRIER-ROUTE
# ==================================================================
route_perf = (
    resolved.groupby(["carrier", "origin_region", "destination_region"])
    .agg(
        avg_variance_days=("Delivery_Variance_Days", "mean"),
        sla_miss_rate=("Delivery_Variance_Days", lambda s: (s > 0).mean()),
        order_count=("order_id", "count"),
    )
    .reset_index()
)
# Only look at routes with meaningful volume
route_perf = route_perf[route_perf["order_count"] >= 30].sort_values("avg_variance_days", ascending=False)

print("=" * 70)
print("INSIGHT 1: TOP 10 WORST-PERFORMING CARRIER-ROUTE COMBINATIONS")
print("=" * 70)
print(route_perf.head(10).to_string(index=False))

worst = route_perf.iloc[0]
print(f"\n>> WORST OFFENDER: {worst['carrier']} on {worst['origin_region']} -> {worst['destination_region']}")
print(f"   Avg variance: {worst['avg_variance_days']:.2f} days late | "
      f"SLA miss rate: {worst['sla_miss_rate']*100:.1f}% | Volume: {int(worst['order_count'])} orders")

fig, ax = plt.subplots(figsize=(11, 6))
top10 = route_perf.head(10).copy()
top10["route_label"] = top10["carrier"] + "\n" + top10["origin_region"] + "→" + top10["destination_region"]
colors = ["#c0392b" if v == worst["avg_variance_days"] else "#5b8fc7" for v in top10["avg_variance_days"]]
bars = ax.barh(top10["route_label"], top10["avg_variance_days"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Average Delivery Variance (Days Late)")
ax.set_title("Worst-Performing Carrier-Route Combinations (min. 30 orders)", fontsize=13, weight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_carrier_route_variance.png", dpi=150)
plt.close()

# ==================================================================
# INSIGHT 2: THE RETURN CASCADE
# ==================================================================
return_by_sla = (
    resolved.groupby("SLA_Status", observed=True)
    .agg(return_rate=("return_flag", "mean"), order_count=("order_id", "count"))
    .reset_index()
)
return_by_sla["return_rate_pct"] = (return_by_sla["return_rate"] * 100).round(1)

print("\n" + "=" * 70)
print("INSIGHT 2: RETURN RATE BY SLA STATUS (THE RETURN CASCADE)")
print("=" * 70)
print(return_by_sla.to_string(index=False))

on_time_rate = return_by_sla.loc[return_by_sla["SLA_Status"] == "On-Time", "return_rate"].values[0]
severe_rate = return_by_sla.loc[return_by_sla["SLA_Status"] == "Severely Delayed (5+ Days)", "return_rate"].values[0]
lift_multiplier = severe_rate / on_time_rate
print(f"\n>> Return rate is {lift_multiplier:.1f}x higher for Severely Delayed orders "
      f"({severe_rate*100:.1f}%) vs On-Time orders ({on_time_rate*100:.1f}%).")

# Correlation between raw delay magnitude and return likelihood
corr = resolved[["Delivery_Variance_Days", "return_flag"]].corr().iloc[0, 1]
print(f">> Point-biserial correlation (delay days vs return flag): r = {corr:.3f}")

fig, ax = plt.subplots(figsize=(9, 6))
order = ["On-Time", "1-2 Days Late", "3-5 Days Late", "Severely Delayed (5+ Days)"]
plot_data = return_by_sla[return_by_sla["SLA_Status"].isin(order)]
sns.barplot(data=plot_data, x="SLA_Status", y="return_rate_pct", order=order, ax=ax, color="#5b8fc7")
for i, row in enumerate(plot_data.set_index("SLA_Status").loc[order].itertuples()):
    ax.text(i, row.return_rate_pct + 0.4, f"{row.return_rate_pct}%", ha="center", fontweight="bold")
ax.set_ylabel("Return Rate (%)")
ax.set_xlabel("")
ax.set_title("Customer Return Rate Escalates with Delivery Delay Severity", fontsize=13, weight="bold")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_return_cascade.png", dpi=150)
plt.close()

# ==================================================================
# INSIGHT 3: INVENTORY VELOCITY / SHIPPING COST INEFFICIENCY
# ==================================================================
category_stats = (
    resolved.groupby("product_category")
    .agg(
        order_volume=("order_id", "count"),
        avg_unit_margin=("unit_margin_usd", "mean"),
        total_shipping_cost=("Estimated_Shipping_Cost_USD", "sum"),
        avg_shipping_cost=("Estimated_Shipping_Cost_USD", "mean"),
        total_margin_generated=("unit_margin_usd", "sum"),
    )
    .reset_index()
)
category_stats["shipping_cost_as_pct_of_margin"] = (
    category_stats["total_shipping_cost"] / category_stats["total_margin_generated"] * 100
).round(1)
category_stats = category_stats.sort_values("shipping_cost_as_pct_of_margin", ascending=False)

print("\n" + "=" * 70)
print("INSIGHT 3: SHIPPING COST AS % OF MARGIN GENERATED, BY CATEGORY")
print("=" * 70)
print(category_stats.to_string(index=False))

worst_cat = category_stats.iloc[0]
print(f"\n>> FLAGGED: '{worst_cat['product_category']}' — {int(worst_cat['order_volume']):,} orders, "
      f"${worst_cat['avg_unit_margin']:.2f} avg margin/unit, but shipping costs consume "
      f"{worst_cat['shipping_cost_as_pct_of_margin']:.1f}% of margin generated.")

fig, ax = plt.subplots(figsize=(10, 6.5))
scatter = ax.scatter(
    category_stats["order_volume"],
    category_stats["avg_unit_margin"],
    s=category_stats["shipping_cost_as_pct_of_margin"] * 15,
    c=category_stats["shipping_cost_as_pct_of_margin"],
    cmap="Reds", alpha=0.85, edgecolors="black", linewidth=0.8,
)
for _, row in category_stats.iterrows():
    ax.annotate(row["product_category"], (row["order_volume"], row["avg_unit_margin"]),
                xytext=(6, 6), textcoords="offset points", fontsize=9)
cbar = plt.colorbar(scatter)
cbar.set_label("Shipping Cost as % of Margin Generated")
ax.set_xlabel("Order Volume")
ax.set_ylabel("Avg Unit Margin (USD)")
ax.set_title("Inventory Velocity vs. Margin: Bubble Size = Shipping Cost Drag", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_inventory_velocity.png", dpi=150)
plt.close()

# ==================================================================
# SUMMARY NUMBERS FOR README
# ==================================================================
total_shipping_spend = resolved["Estimated_Shipping_Cost_USD"].sum()
systemic_route_mask = (
    (resolved["carrier"] == "Standard")
    & (resolved["origin_region"] == "South")
    & (resolved["destination_region"] == "Northeast")
)
systemic_orders = resolved[systemic_route_mask]
systemic_extra_cost_est = systemic_orders["Estimated_Shipping_Cost_USD"].sum() * 0.0  # placeholder for cost calc below

print("\n" + "=" * 70)
print("HEADLINE NUMBERS FOR EXECUTIVE README")
print("=" * 70)
print(f"Total orders analyzed: {len(df):,}")
print(f"Total estimated shipping spend: ${total_shipping_spend:,.0f}")
print(f"Overall SLA miss rate: {(resolved['Delivery_Variance_Days'] > 0).mean()*100:.1f}%")
print(f"Systemic failure route order count: {len(systemic_orders):,} "
      f"({len(systemic_orders)/len(resolved)*100:.1f}% of volume)")
print(f"Systemic failure route SLA miss rate: {(systemic_orders['Delivery_Variance_Days'] > 0).mean()*100:.1f}%")
print(f"Systemic failure route avg variance: {systemic_orders['Delivery_Variance_Days'].mean():.2f} days")
print(f"Return rate uplift (Severe vs On-Time): {lift_multiplier:.1f}x")
print(f"Toys: volume={category_stats.set_index('product_category').loc['Toys','order_volume']:.0f}, "
      f"shipping-cost-to-margin={category_stats.set_index('product_category').loc['Toys','shipping_cost_as_pct_of_margin']:.1f}%")

category_stats.to_csv("/home/claude/project/data/category_summary.csv", index=False)
route_perf.to_csv("/home/claude/project/data/route_summary.csv", index=False)
return_by_sla.to_csv("/home/claude/project/data/return_cascade_summary.csv", index=False)
print("\nSummary CSVs + 3 plots saved.")
