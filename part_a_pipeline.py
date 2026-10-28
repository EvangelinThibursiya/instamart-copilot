"""part_a_pipeline.py -- Part A: Data Foundation pipeline for Instamart Copilot."""
import os
import sqlite3

import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")

os.makedirs(CLEAN, exist_ok=True)


# ── 1. Load raw data ────────────────────────────────────────────────────────
print("Loading raw data...")
customers_raw = pd.read_csv(os.path.join(RAW, "customers.csv"))
products_raw  = pd.read_csv(os.path.join(RAW, "products.csv"))
orders_raw    = pd.read_csv(os.path.join(RAW, "orders.csv"))
tickets_raw   = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)

# ── 2. Audit ─────────────────────────────────────────────────────────────────
print("\n=== AUDIT ===")
print(f"customers shape: {customers_raw.shape}")
print(f"products  shape: {products_raw.shape}")
print(f"orders    shape: {orders_raw.shape}")
print(f"tickets   shape: {tickets_raw.shape}")

n_raw_orders      = len(orders_raw)
n_exact_dup       = orders_raw.duplicated().sum()
n_pending         = (orders_raw["status"] == "pending").sum()
n_distinct_oid    = orders_raw["order_id"].nunique()
n_non_numeric_amt = pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum()
n_neg_deliv       = (pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce") < 0).sum()
n_blank_deliv     = (orders_raw["delivery_minutes"].astype(str).str.strip() == "").sum()

# customers
n_blank_age          = (customers_raw["age"].astype(str).str.strip() == "").sum()
age_num              = pd.to_numeric(customers_raw["age"], errors="coerce")
n_age_outside        = ((age_num < 10) | (age_num > 100)).sum()
n_raw_city_spellings = customers_raw["city"].nunique()
gender_col           = customers_raw["gender"].astype(str).str.strip()
n_blank_gender       = (gender_col == "").sum()
n_raw_gender         = gender_col[gender_col != ""].nunique()

print(f"\nRaw order rows              : {n_raw_orders}")
print(f"Exact duplicate rows        : {n_exact_dup}")
print(f"Pending rows                : {n_pending}")
print(f"Distinct order_id values    : {n_distinct_oid}")
print(f"Non-numeric order_amounts   : {n_non_numeric_amt}")
print(f"Negative delivery_minutes   : {n_neg_deliv}")
print(f"Blank delivery_minutes      : {n_blank_deliv}")
print(f"Blank ages                  : {n_blank_age}")
print(f"Ages outside 10-100         : {n_age_outside}")
print(f"Raw city spellings          : {n_raw_city_spellings}")
print(f"Raw gender spellings (non-blank): {n_raw_gender}")
print(f"Blank genders               : {n_blank_gender}")

# ── 3. Clean customers ───────────────────────────────────────────────────────
print("\n=== CLEANING CUSTOMERS ===")
customers = customers_raw.copy()

# (a) Standardise city
customers["city"] = customers["city"].str.strip().str.title()
print(f"Distinct cities after standardisation: {customers['city'].nunique()}")
print(customers["city"].value_counts().to_string())

# (b) Map gender
def map_gender(g):
    g = str(g).strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    return "unknown"

customers["gender"] = customers["gender"].astype(str).apply(map_gender)
print(f"\nGender counts:\n{customers['gender'].value_counts().to_string()}")

# (c) Fix ages – treat impossible values as missing, then median-fill per city
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
customers["age"] = customers.groupby("city")["age"].transform(
    lambda x: x.fillna(x.median())
)
print(f"\nMissing ages after fill: {customers['age'].isna().sum()}")
print(f"Age range: {customers['age'].min()} - {customers['age'].max()}")

# (d) Convert signup_date
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ── 4. Clean orders ──────────────────────────────────────────────────────────
print("\n=== CLEANING ORDERS ===")
orders = orders_raw.copy()
products = products_raw.copy()

# Step 1: convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Step 2: remove exact duplicates
before = len(orders)
orders = orders.drop_duplicates()
print(f"Removed {before - len(orders)} exact duplicate rows -> {len(orders)} rows")

# Step 3: for each order_id keep only the record with the latest order_ts
before = len(orders)
orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
print(f"Kept latest per order_id: {before - len(orders)} rows removed -> {len(orders)} rows")

# Step 4: convert order_amount to numeric, fill missing from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "price_inr"]], on="product_id", how="left")
missing_mask = orders["order_amount"].isna()
orders.loc[missing_mask, "order_amount"] = (
    orders.loc[missing_mask, "price_inr"]
    * orders.loc[missing_mask, "quantity"]
    * (1 - orders.loc[missing_mask, "discount_pct"] / 100)
).round(2)
orders = orders.drop(columns=["price_inr"])
print(f"Missing order_amount after fill: {orders['order_amount'].isna().sum()}")

# Step 5: fix delivery_minutes
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# Add product category for median computation
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# Fill missing delivery_minutes for completed orders with category median
completed_mask = orders["status"] == "completed"
cat_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)
orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(
    cat_median[completed_mask]
)
orders = orders.drop(columns=["category"])

print(f"\nFinal order counts: {len(orders)}")
print(orders["status"].value_counts().to_string())
print(f"Missing delivery_minutes (completed) : {orders.loc[orders['status']=='completed', 'delivery_minutes'].isna().sum()}")
print(f"Missing delivery_minutes (cancelled) : {orders.loc[orders['status']=='cancelled', 'delivery_minutes'].isna().sum()}")
completed_sum = orders.loc[orders["status"] == "completed", "order_amount"].sum()
print(f"Sum of completed order_amount        : INR {completed_sum:,.2f}")

# ── 5. Save clean data ────────────────────────────────────────────────────────
print("\n=== SAVING ===")
customers.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
print("Written data/clean/customers.csv, products.csv, orders.csv")

conn = sqlite3.connect(DB)
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
print(f"Written {DB}")

# ── 6. SQL queries validation ─────────────────────────────────────────────────
print("\n=== SQL QUERIES ===")
conn = sqlite3.connect(DB)

q1 = pd.read_sql_query("""
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status='completed' AND order_amount > 500
    GROUP BY customer_id HAVING SUM(order_amount) > 8000
""", conn)
print(f"Q1 customers with total>8000 on orders>500: {len(q1)}  (ref: 30)")

q2 = pd.read_sql_query("""
    SELECT c.city, p.category, SUM(o.order_amount) AS revenue
    FROM orders o JOIN customers c ON o.customer_id=c.customer_id
    JOIN products p ON o.product_id=p.product_id
    WHERE o.status='completed' GROUP BY c.city, p.category ORDER BY c.city, p.category
""", conn)
print(f"Q2 rows: {len(q2)}  total: INR {q2['revenue'].sum():,.2f}  (ref: 30 rows, 2998668.80)")
bengaluru_elec = q2[(q2['city']=='Bengaluru') & (q2['category']=='Electronics')]['revenue'].values
if len(bengaluru_elec):
    print(f"  Bengaluru/Electronics: INR {bengaluru_elec[0]:,.2f}  (ref: 233909.45)")

q3 = pd.read_sql_query("SELECT product_id, product_name, price_inr FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products)", conn)
print(f"Q3 products above avg price: {len(q3)}  (ref: 11)")

q4 = pd.read_sql_query("""
    SELECT c.customer_id FROM customers c
    LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
    WHERE o.order_id IS NULL
""", conn)
print(f"Q4 customers with no completed order: {len(q4)}  (ref: 25)")

q5 = pd.read_sql_query("""
    WITH customer_totals AS (
        SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
        WHERE status='completed' GROUP BY customer_id
    ), avg_spend AS (SELECT AVG(total_spend) AS avg_total FROM customer_totals)
    SELECT ct.customer_id, ct.total_spend, at.avg_total
    FROM customer_totals ct, avg_spend at WHERE ct.total_spend > at.avg_total
""", conn)
print(f"Q5 customers above avg spend: {len(q5)}  avg: INR {q5['avg_total'].iloc[0]:,.2f}  (ref: 550, 2552.06)")

q6 = pd.read_sql_query("""
    SELECT customer_id, city, total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_val,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_val,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
    FROM (
        SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
        FROM orders o JOIN customers c ON o.customer_id=c.customer_id
        WHERE o.status='completed' GROUP BY o.customer_id, c.city
    ) t
""", conn)
top_per_city = q6[q6["rank_val"] == 1][["customer_id","city","total_spend"]].sort_values("city")
print(f"Q6 top per city:")
print(top_per_city.to_string(index=False))
rank_differs = (q6["rank_val"] != q6["row_num"]).sum()
print(f"  Rows where RANK != ROW_NUMBER: {rank_differs}  (ref: 8)")

q7 = pd.read_sql_query("""
    SELECT SUBSTR(order_ts,1,10) AS day, SUM(order_amount) AS revenue,
        SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts,1,10)) AS cumulative_revenue
    FROM orders WHERE status='completed' GROUP BY SUBSTR(order_ts,1,10) ORDER BY day
""", conn)
print(f"Q7 daily rows: {len(q7)}  last cumulative: INR {q7['cumulative_revenue'].iloc[-1]:,.2f}  (ref: 183, 2998668.80)")
conn.close()

# ── 7. Pandas cross-check ─────────────────────────────────────────────────────
print("\n=== PANDAS CROSS-CHECK ===")
orders_clean = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
customers_clean = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
products_clean = pd.read_csv(os.path.join(CLEAN, "products.csv"))

completed_clean = orders_clean[orders_clean["status"] == "completed"].copy()
merged = completed_clean.merge(customers_clean[["customer_id","city"]], on="customer_id")\
                         .merge(products_clean[["product_id","category"]], on="product_id")
pivot = pd.pivot_table(merged, values="order_amount", index="city", columns="category",
                       aggfunc="sum", margins=True)
print(f"Pivot grand total: INR {pivot.loc['All','All']:,.2f}  (ref: 2998668.80)")

city_stats = completed_clean.merge(customers_clean[["customer_id","city"]], on="customer_id")\
    .groupby("city")["order_amount"].agg(sum="sum", mean="mean", count="count")
print("\nCity stats (completed orders):")
print(city_stats.round(2).to_string())
print("(ref: Bengaluru sum=523643.40 mean=276.47 count=1894, Mumbai sum=479058.50 mean=297.00 count=1613)")

print("\nCompleted order_amount stats:")
desc = completed_clean["order_amount"].describe(percentiles=[0.25, 0.5, 0.75])
print(desc.round(2).to_string())
skewness = completed_clean["order_amount"].skew()
print(f"Skewness: {skewness:.2f}  (ref: ~4.61)")
q1_val = completed_clean["order_amount"].quantile(0.25)
q3_val = completed_clean["order_amount"].quantile(0.75)
iqr = q3_val - q1_val
upper_fence = q3_val + 1.5 * iqr
lower_fence = q1_val - 1.5 * iqr
n_above = (completed_clean["order_amount"] > upper_fence).sum()
n_below = (completed_clean["order_amount"] < lower_fence).sum()
print(f"Q1={q1_val:.2f}  Q3={q3_val:.2f}  IQR={iqr:.2f}")
print(f"Upper fence={upper_fence:.2f}  outliers above={n_above}  (ref: 645.30, 897)")
print(f"Lower fence={lower_fence:.2f}  outliers below={n_below}  (ref: 0)")

corr_cols = ["order_amount","discount_pct","quantity","delivery_minutes"]
corr_df = completed_clean[corr_cols].dropna().corr()
print("\nCorrelation matrix (completed orders):")
print(corr_df.round(3).to_string())
import numpy as np
corr_vals = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
max_pair = corr_vals.stack().abs().idxmax()
print(f"Largest absolute correlation: {max_pair} = {corr_df.loc[max_pair]:+.3f}  (ref: order_amount/discount_pct ~-0.095)")

# ── 8. Validation ─────────────────────────────────────────────────────────────
print("\n=== VALIDATION ===")
errors = []

def chk(cond, msg):
    if not cond:
        errors.append(msg)

orders_v = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
cust_v   = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
comp_v   = orders_v[orders_v["status"] == "completed"]
canc_v   = orders_v[orders_v["status"] == "cancelled"]

chk(len(orders_v) == 12309,       f"orders rows: expected 12309 got {len(orders_v)}")
chk(len(comp_v)   == 11110,       f"completed: expected 11110 got {len(comp_v)}")
chk(len(canc_v)   == 1199,        f"cancelled: expected 1199 got {len(canc_v)}")
chk((orders_v["status"] == "pending").sum() == 0, "pending rows should be 0")
chk(orders_v["order_amount"].isna().sum() == 0,   "missing order_amount should be 0")
chk(comp_v["delivery_minutes"].isna().sum() == 0, "completed delivery_minutes missing should be 0")
chk(canc_v["delivery_minutes"].isna().sum() == 1199, "cancelled delivery_minutes all missing")
s = comp_v["order_amount"].sum()
chk(abs(s - 2998668.80) < 1.0,   f"completed sum: expected 2998668.80 got {s:.2f}")
chk(cust_v["city"].nunique() == 6, f"cities: expected 6 got {cust_v['city'].nunique()}")
chk(cust_v["age"].isna().sum() == 0, "missing ages should be 0")
chk(cust_v["gender"].isin(["male","female","unknown"]).all(), "gender values unexpected")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
else:
    print("ALL CHECKPOINTS PASSED")
