"""
PHASE 2: DATA CLEANING & FEATURE ENGINEERING
--------------------------------------------------------------
Logical (not blind-drop) imputation + engineered time-delta and
business-facing categorical features.
"""

import numpy as np
import pandas as pd

df = pd.read_csv(
    "/home/claude/project/data/fulfillment_raw.csv",
    parse_dates=["order_timestamp", "warehouse_receipt_time", "pack_complete_time",
                 "ship_time", "scheduled_delivery_date", "actual_delivery_date"]
)

# ------------------------------------------------------------------
# 1. LOGICAL MISSING-DATA HANDLING
# ------------------------------------------------------------------

# a) weight_kg: impute by product_category median (not global mean — categories
#    have very different weight distributions, so a global fill would distort cost analysis)
df["weight_kg"] = df.groupby("product_category")["weight_kg"].transform(
    lambda s: s.fillna(s.median())
)

# b) destination_region: impute using the origin_region's most common paired destination
#    (a real ops assumption: most short-haul orders ship intra-region)
dest_mode_by_origin = (
    df.dropna(subset=["destination_region"])
      .groupby("origin_region")["destination_region"]
      .agg(lambda s: s.mode().iloc[0])
)
missing_dest_mask = df["destination_region"].isna()
df.loc[missing_dest_mask, "destination_region"] = df.loc[missing_dest_mask, "origin_region"].map(dest_mode_by_origin)

# c) pack_complete_time: missing = scanner outage, not "instant packing".
#    Impute using the carrier-level median pack duration, not a naive fill.
df["_pack_hours_temp"] = (df["pack_complete_time"] - df["warehouse_receipt_time"]).dt.total_seconds() / 3600
median_pack_hours = df.groupby("carrier")["_pack_hours_temp"].transform("median")
needs_pack_fill = df["pack_complete_time"].isna()
df.loc[needs_pack_fill, "pack_complete_time"] = (
    df.loc[needs_pack_fill, "warehouse_receipt_time"]
    + pd.to_timedelta(median_pack_hours[needs_pack_fill], unit="h")
)
df.drop(columns=["_pack_hours_temp"], inplace=True)

# d) actual_delivery_date missing = unresolved / lost-in-transit shipment.
#    Do NOT impute a delivery date — that would fabricate an outcome.
#    Instead, flag explicitly so it's excluded from delay math but retained for ops reporting.
df["is_unresolved_shipment"] = df["actual_delivery_date"].isna()

# e) fix negative order_value_usd typos (refund-miscoding) -> take absolute value
neg_mask = df["order_value_usd"] < 0
df.loc[neg_mask, "order_value_usd"] = df.loc[neg_mask, "order_value_usd"].abs()

# ------------------------------------------------------------------
# 2. TIME-DELTA FEATURE ENGINEERING
# ------------------------------------------------------------------
df["Time_to_Pack_Hours"] = (
    (df["pack_complete_time"] - df["warehouse_receipt_time"]).dt.total_seconds() / 3600
).round(2)

df["Transit_Days_Scheduled"] = (
    df["scheduled_delivery_date"] - df["ship_time"]
).dt.total_seconds() / 86400
df["Transit_Days_Scheduled"] = df["Transit_Days_Scheduled"].round(1)

# Delivery_Variance_Days: positive = late, negative = early. NaN for unresolved shipments.
df["Delivery_Variance_Days"] = (
    (df["actual_delivery_date"] - df["scheduled_delivery_date"]).dt.total_seconds() / 86400
).round(2)

df["Order_to_Ship_Hours"] = (
    (df["ship_time"] - df["order_timestamp"]).dt.total_seconds() / 3600
).round(2)

df["Carrier_Route"] = df["carrier"] + " | " + df["origin_region"] + " -> " + df["destination_region"]

# Shipping cost proxy: weight * distance factor * carrier rate multiplier
DIST_MATRIX = {
    ("West", "West"): 0.4, ("West", "Midwest"): 1.8, ("West", "Northeast"): 2.9,
    ("West", "South"): 2.0, ("West", "Southwest"): 1.0,
    ("Midwest", "Midwest"): 0.5, ("Midwest", "Northeast"): 1.2, ("Midwest", "South"): 1.1,
    ("Midwest", "Southwest"): 1.5,
    ("Northeast", "Northeast"): 0.4, ("Northeast", "South"): 1.4, ("Northeast", "Southwest"): 2.4,
    ("South", "South"): 0.5, ("South", "Southwest"): 1.0,
    ("Southwest", "Southwest"): 0.4,
}
def get_distance(o, d):
    key = (o, d) if (o, d) in DIST_MATRIX else (d, o)
    return DIST_MATRIX.get(key, 1.5)

df["distance_factor"] = df.apply(lambda r: get_distance(r["origin_region"], r["destination_region"]), axis=1)
carrier_rate_multiplier = df["carrier"].map({"Standard": 4.2, "Express": 7.8})
df["Estimated_Shipping_Cost_USD"] = (
    df["weight_kg"] * df["distance_factor"] * carrier_rate_multiplier
).round(2)

# ------------------------------------------------------------------
# 3. BUSINESS-FACING CATEGORICAL BINS
# ------------------------------------------------------------------
def sla_bucket(v):
    if pd.isna(v):
        return "Unresolved"
    if v <= 0:
        return "On-Time"
    elif v <= 2:
        return "1-2 Days Late"
    elif v <= 5:
        return "3-5 Days Late"
    else:
        return "Severely Delayed (5+ Days)"

df["SLA_Status"] = df["Delivery_Variance_Days"].apply(sla_bucket)
df["SLA_Status"] = pd.Categorical(
    df["SLA_Status"],
    categories=["On-Time", "1-2 Days Late", "3-5 Days Late", "Severely Delayed (5+ Days)", "Unresolved"],
    ordered=True,
)

# Margin tier for velocity analysis
df["Margin_Tier"] = pd.cut(
    df["unit_margin_usd"], bins=[0, 10, 20, 100],
    labels=["Low Margin (<$10)", "Mid Margin ($10-20)", "High Margin ($20+)"]
)

out_path = "/home/claude/project/data/fulfillment_clean.csv"
df.to_csv(out_path, index=False)

print(f"Cleaned dataset saved -> {out_path}")
print(f"Rows: {len(df):,} | Unresolved shipments: {df['is_unresolved_shipment'].sum()}")
print("\nRemaining nulls:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nSLA_Status distribution:\n", df["SLA_Status"].value_counts())
