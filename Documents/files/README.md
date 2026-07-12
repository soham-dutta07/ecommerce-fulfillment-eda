# E-Commerce Fulfillment Optimization — Executive Analysis

**Analyst:** [Your Name] | **Scope:** 10,000-order fulfillment dataset (synthetic, operationally realistic) | **Tools:** Python (pandas, numpy, matplotlib, seaborn)

---

## 1. Business Problem

Fulfillment SLA misses and downstream returns are eroding margin, but the operations team has no visibility into *where* the failures originate. Leadership needs answers to three questions before the next carrier contract renewal:

1. Is our SLA failure rate a general carrier-performance issue, or is it concentrated in a specific, fixable lane?
2. Does shipping delay actually drive returns, or is that just anecdotal?
3. Which product categories are quietly bleeding margin through shipping cost, independent of delay?

This analysis answers all three using order-level data spanning warehouse intake through delivery confirmation.

---

## 2. Methodology

- **Data construction:** 10,000 synthetic orders generated to replicate a real OMS/WMS/carrier-API join, including realistic missingness (scanner outages, unlogged fields, unresolved shipments) and one deliberately embedded systemic failure, to stress-test the analysis pipeline the way production data would.
- **Cleaning:** Missing values were imputed using operationally logical rules (category-median weight, carrier-median pack time, origin-based destination inference) rather than blind mean-fill or row-drop — preserving 100% of order volume for revenue and cost analysis.
- **Feature engineering:** Derived `Time_to_Pack_Hours`, `Delivery_Variance_Days`, `Estimated_Shipping_Cost_USD` (weight × distance × carrier rate), and a business-facing `SLA_Status` bucket (On-Time / 1-2 Days Late / 3-5 Days Late / Severely Delayed / Unresolved).
- **Analysis:** Carrier-route aggregation, delay-vs-return correlation, and category-level shipping-cost-to-margin ratio.

Full code: `01_generate_dataset.py` → `02_clean_feature_engineer.py` → `03_eda_analysis.py`.

---

## 3. Key Findings & Financial Impact

### Finding 1 — One lane is responsible for a disproportionate share of SLA failures
**Standard carrier, South → Northeast** is missing its SLA on **100% of 249 orders** (2.5% of total volume), averaging **3.69 days late** against a 5-day SLA — nearly double the delay of the next-worst lane (0.46 days). Every other carrier-route combination performs within normal variance.

> This is not a carrier-wide problem. It is a single-lane bottleneck, most likely a regional hub or last-mile handoff issue specific to that corridor.

![Carrier Route Variance](plots/01_carrier_route_variance.png)

### Finding 2 — Delay severity is a direct driver of returns, not a coincidence
Return rate climbs from **5.0% (On-Time)** to **51.2% (Severely Delayed, 5+ days)** — a **10.2x increase**. The relationship is monotonic across every SLA tier (5.0% → 5.9% → 21.3% → 51.2%), and the delay-to-return correlation is statistically meaningful (r = 0.31).

> Every day an order sits past its promised delivery date compounds return risk. Delay isn't just a service metric — it's a margin leak via reverse logistics.

![Return Cascade](plots/02_return_cascade.png)

### Finding 3 — Two categories are structurally unprofitable to ship at current volume
- **Home Goods**: shipping cost consumes **221% of the margin it generates** ($46.1K in shipping cost against $20.8K in margin) — driven by high average weight (6.5kg) rather than volume.
- **Toys**: our **highest-volume category (2,413 orders)** but shipping cost still consumes **122% of margin generated** ($9.00 avg margin/unit vs. $11.01 avg shipping cost/unit) — a volume-and-margin double bind.

> Both categories are losing money on shipping alone, before accounting for the base cost of goods.

![Inventory Velocity](plots/03_inventory_velocity.png)

---

## 4. Estimated Financial Exposure (illustrative, based on this dataset)

| Issue | Estimated Annualized Impact* |
|---|---|
| Systemic lane failure (249 orders, 100% SLA miss) | Reverse-logistics + service-recovery cost on ~249 orders/period |
| Return cascade from delayed orders (severely delayed tier) | ~148 additional returns vs. On-Time baseline rate, at avg order value ≈ $130 → **~$19K in returned-goods exposure** |
| Home Goods + Toys shipping-cost drag | Combined **$72.7K in shipping spend** against **$42.6K in margin generated** — a **$30K+ net shipping loss** across these two categories alone |

*Figures are derived directly from the generated dataset and scale with real order volume — presented here as a **methodology demonstration**, not a production forecast.

---

## 5. Strategic Recommendations

1. **Renegotiate or re-route the Standard / South→Northeast lane.** A 100% SLA miss rate on a fixed corridor is a hub/routing issue, not a carrier-capacity issue — escalate to the carrier's account team with lane-specific SLA data, or pilot a secondary carrier on that corridor for 90 days.
2. **Set a proactive customer-communication trigger at the 3-day-late mark.** Since return risk starts compounding sharply after Day 3, an automated "your order is delayed, here's your options" touchpoint (rebate, expedited reship, or partial refund) before Day 5 could blunt the return spike before it happens.
3. **Reassess Home Goods and Toys unit economics.** Either build shipping cost into product pricing for these categories, consolidate Toys into multi-item shipments to amortize per-unit cost, or shift Home Goods to a weight-tiered freight carrier instead of parcel rates.
4. **Instrument SLA_Status and Carrier_Route as standing dashboard fields**, not one-off analysis outputs — this failure pattern should be visible to ops in real time, not discovered quarterly.

---

## 6. Repository Structure

```
├── 01_generate_dataset.py        # Phase 1: synthetic data generation
├── 02_clean_feature_engineer.py  # Phase 2: cleaning + feature engineering
├── 03_eda_analysis.py            # Phase 3: strategic EDA + plots
├── data/
│   ├── fulfillment_raw.csv
│   ├── fulfillment_clean.csv
│   ├── route_summary.csv
│   ├── return_cascade_summary.csv
│   └── category_summary.csv
├── plots/
│   ├── 01_carrier_route_variance.png
│   ├── 02_return_cascade.png
│   └── 03_inventory_velocity.png
└── README.md
```

**Reproduce:** `python 01_generate_dataset.py && python 02_clean_feature_engineer.py && python 03_eda_analysis.py`
