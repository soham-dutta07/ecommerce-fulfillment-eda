"""
PHASE 1: SYNTHETIC E-COMMERCE FULFILLMENT DATASET GENERATOR
--------------------------------------------------------------
Generates a ~10,000 row order-level dataset with realistic operational
noise: missing values, weather-driven outliers, and a systemic SLA
failure baked into ONE carrier on ONE route (Standard / South->Northeast).

This mirrors what you'd actually pull from a WMS/OMS + carrier API join:
messy, partially null, and full of a hidden operational story.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)
N = 10_000

# ------------------------------------------------------------------
# 1. REFERENCE DATA
# ------------------------------------------------------------------
REGIONS = ["West", "Midwest", "Northeast", "South", "Southwest"]

# Approximate relative distance factor between regions (proxy for miles/1000)
# Symmetric matrix, diagonal = intra-region (short-haul)
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
    return DIST_MATRIX.get(key, 1.5)  # fallback avg

CARRIERS = ["Standard", "Express"]
CARRIER_WEIGHTS = [0.68, 0.32]  # Standard is the workhorse carrier

# category: (weight_mean_kg, weight_sd, unit_margin_usd, base_price_usd)
CATEGORIES = {
    "Electronics":  (2.8, 1.4, 38.0, 220.0),
    "Apparel":      (0.6, 0.3, 14.0, 45.0),
    "Home Goods":   (6.5, 3.2, 22.0, 130.0),
    "Beauty":       (0.4, 0.2, 11.0, 32.0),
    "Sports":       (3.1, 1.8, 17.0, 85.0),
    "Books":        (0.5, 0.15, 4.0, 18.0),
    "Toys":         (1.4, 0.9, 9.0, 40.0),          # high-volume, low-margin -> flagged in Phase 3
}
CATEGORY_NAMES = list(CATEGORIES.keys())
# Toys and Apparel are deliberately over-represented (high order volume, low margin)
CATEGORY_PROBS = [0.14, 0.22, 0.10, 0.12, 0.10, 0.08, 0.24]

# ------------------------------------------------------------------
# 2. CORE FIELDS
# ------------------------------------------------------------------
order_ids = [f"ORD-{100000+i}" for i in range(N)]

start_date = pd.Timestamp("2024-01-01")
order_timestamps = start_date + pd.to_timedelta(
    np.random.randint(0, 365 * 24 * 60, N), unit="m"
)

origin_region = np.random.choice(REGIONS, N, p=[0.28, 0.22, 0.15, 0.20, 0.15])
destination_region = np.random.choice(REGIONS, N)

carrier = np.random.choice(CARRIERS, N, p=CARRIER_WEIGHTS)

category = np.random.choice(CATEGORY_NAMES, N, p=CATEGORY_PROBS)
weight_kg = np.array([
    max(0.05, np.random.normal(CATEGORIES[c][0], CATEGORIES[c][1])) for c in category
])
unit_margin_usd = np.array([CATEGORIES[c][2] for c in category])
order_value_usd = np.array([
    max(5, np.random.normal(CATEGORIES[c][3], CATEGORIES[c][3] * 0.25)) for c in category
])

# ------------------------------------------------------------------
# 3. WAREHOUSE PROCESSING (pack time) -- realistic ops noise
# ------------------------------------------------------------------
# Base pack time in hours, right-skewed (lognormal), with weekend/holiday backlog
pack_hours = np.random.lognormal(mean=1.1, sigma=0.6, size=N)  # median ~3hrs
weekday = order_timestamps.dayofweek
# Monday backlog effect (orders placed over the weekend queue up)
pack_hours = np.where(weekday.isin([5, 6]), pack_hours * 1.6, pack_hours)

warehouse_receipt_time = order_timestamps + pd.to_timedelta(
    np.random.randint(5, 90, N), unit="m"
)
pack_complete_time = warehouse_receipt_time + pd.to_timedelta(pack_hours, unit="h")
ship_time = pack_complete_time + pd.to_timedelta(np.random.randint(30, 240, N), unit="m")

# ------------------------------------------------------------------
# 4. SCHEDULED VS ACTUAL DELIVERY -- where the story lives
# ------------------------------------------------------------------
sla_days = np.where(carrier == "Express", 2, 5)
scheduled_delivery_date = ship_time + pd.to_timedelta(sla_days, unit="D")

# Baseline transit noise (mostly on-time, slight right skew)
transit_noise_days = np.random.normal(0, 0.6, N)

# --- SYSTEMIC SLA FAILURE (the planted signal) ---
# Standard carrier running South -> Northeast misses its SLA hard and consistently.
# Root cause proxy: single regional hub bottleneck.
systemic_failure_mask = (
    (carrier == "Standard") & (origin_region == "South") & (destination_region == "Northeast")
)
systemic_delay = np.where(systemic_failure_mask, np.random.normal(3.4, 1.1, N), 0)
systemic_delay = np.clip(systemic_delay, 0, None)

# --- OUTLIERS: severe weather delays (rare, large, seasonal Nov-Feb) ---
is_winter = order_timestamps.month.isin([11, 12, 1, 2])
weather_prob = np.where(is_winter, 0.06, 0.015)
weather_hit = np.random.random(N) < weather_prob
weather_delay = np.where(weather_hit, np.random.uniform(4, 12, N), 0)

actual_delivery_date = (
    scheduled_delivery_date
    + pd.to_timedelta(transit_noise_days, unit="D")
    + pd.to_timedelta(systemic_delay, unit="D")
    + pd.to_timedelta(weather_delay, unit="D")
)

# ------------------------------------------------------------------
# 5. RETURN CASCADE -- returns correlate with delay severity
# ------------------------------------------------------------------
delivery_variance_days_raw = (actual_delivery_date - scheduled_delivery_date).total_seconds() / 86400
base_return_prob = 0.045
# Return probability climbs sharply once delay exceeds ~3 days, and Apparel/Electronics are more return-prone
category_return_lift = np.array([0.03 if c in ("Apparel", "Electronics") else 0.0 for c in category])
delay_lift = np.clip(delivery_variance_days_raw, 0, None) ** 1.5 * 0.018
return_prob = np.clip(base_return_prob + delay_lift + category_return_lift, 0, 0.85)
return_flag = np.random.random(N) < return_prob

# ------------------------------------------------------------------
# 6. ASSEMBLE
# ------------------------------------------------------------------
df = pd.DataFrame({
    "order_id": order_ids,
    "order_timestamp": order_timestamps,
    "origin_region": origin_region,
    "destination_region": destination_region,
    "carrier": carrier,
    "product_category": category,
    "weight_kg": weight_kg.round(2),
    "unit_margin_usd": unit_margin_usd,
    "order_value_usd": order_value_usd.round(2),
    "warehouse_receipt_time": warehouse_receipt_time,
    "pack_complete_time": pack_complete_time,
    "ship_time": ship_time,
    "scheduled_delivery_date": scheduled_delivery_date,
    "actual_delivery_date": actual_delivery_date,
    "return_flag": return_flag,
})

# ------------------------------------------------------------------
# 7. INJECT REALISTIC MESSINESS (MNAR + MCAR patterns, not pure random)
# ------------------------------------------------------------------
# a) weight_kg missing more often for older, low-value SKUs (MNAR-ish)
low_val_idx = df.sample(frac=0.04, random_state=1).index
df.loc[low_val_idx, "weight_kg"] = np.nan

# b) pack_complete_time missing when warehouse scanner outage (MCAR, clustered by day)
outage_days = pd.to_datetime(np.random.choice(
    pd.date_range("2024-01-01", "2024-12-31", freq="D"), size=8, replace=False
))
outage_mask = df["order_timestamp"].dt.normalize().isin(outage_days)
outage_sample = df[outage_mask].sample(frac=0.6, random_state=2).index
df.loc[outage_sample, "pack_complete_time"] = np.nan

# c) destination_region occasionally unlogged for Express (data entry shortcut)
express_missing_idx = df[df["carrier"] == "Express"].sample(frac=0.02, random_state=3).index
df.loc[express_missing_idx, "destination_region"] = np.nan

# d) a slice of very old orders never got an actual_delivery_date scanned (lost-in-transit / unresolved)
unresolved_idx = df.sample(frac=0.015, random_state=4).index
df.loc[unresolved_idx, "actual_delivery_date"] = pd.NaT

# e) order_value_usd has a few negative-entry typos (refund miscoded as negative order value)
typo_idx = df.sample(frac=0.003, random_state=5).index
df.loc[typo_idx, "order_value_usd"] = -df.loc[typo_idx, "order_value_usd"]

df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle

out_path = "/home/claude/project/data/fulfillment_raw.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} rows -> {out_path}")
print(df.isna().sum())
print(df.head(3).to_string())
