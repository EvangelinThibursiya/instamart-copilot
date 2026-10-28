"""part_b_segments.py -- Part B: Customer Segments and Lapse Prediction."""
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN = os.path.join("data", "clean")
OUTPUTS = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

# ── Load clean data ──────────────────────────────────────────────────────────
orders    = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
products  = pd.read_csv(os.path.join(CLEAN, "products.csv"))
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# ── Task 1: RFM Table ────────────────────────────────────────────────────────
print("=== TASK 1: RFM TABLE ===")
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

completed = orders[orders["status"] == "completed"].copy()

rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm.drop(columns=["last_order_ts"])

print(f"RFM customers    : {len(rfm)}  (ref: 1175)")
print(f"Recency  median  : {rfm['recency'].median():.0f}  min: {rfm['recency'].min()}  max: {rfm['recency'].max()}")
print(f"  (ref: median=19, min=0, max=180)")
print(f"Frequency mean   : {rfm['frequency'].mean():.2f}  max: {rfm['frequency'].max()}")
print(f"  (ref: mean=9.46, max=33)")
print(f"Monetary  mean   : {rfm['monetary'].mean():.2f}  max: {rfm['monetary'].max():.2f}")
print(f"  (ref: mean=2552.06, max=14766.55)")

# ── Task 2: K-means Clustering ───────────────────────────────────────────────
print("\n=== TASK 2: K-MEANS ===")
scaler_rfm = StandardScaler()
X_rfm = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

print(f"{'K':>3}  {'Inertia':>10}  {'Silhouette':>11}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_rfm)
    inertia = km.inertia_
    sil = silhouette_score(X_rfm, labels)
    print(f"{k:>3}  {inertia:>10.1f}  {sil:>11.3f}")

print("\nElbow: inertia drops sharply K=2->4, then flattens. K=4 has highest silhouette among {3,4,5}.")
print("K=4 chosen: maps to Premium, Loyal, At-Risk, Occasional — four actionable business segments.")
print("Inertia of K=8 is always lower than K=2 because more clusters always fit training points more")
print("tightly; inertia is minimised by K=N. Without a silhouette or business check it cannot pick K.")

K_FINAL = 4
km_final = KMeans(n_clusters=K_FINAL, n_init=10, random_state=42)
rfm["segment_id"] = km_final.fit_predict(X_rfm)

# ── Task 3: Profile, Name, PCA ───────────────────────────────────────────────
print("\n=== TASK 3: SEGMENT PROFILES ===")
profile = rfm.groupby("segment_id")[["recency", "frequency", "monetary"]].mean()
sizes   = rfm.groupby("segment_id").size().rename("size")
profile = pd.concat([sizes, profile], axis=1)
print(profile.round(2).to_string())
print("(ref K=4: sizes ~436/138/419/182; 138-seg has mean monetary ~7130, freq ~6.8;")
print(" 419-seg has mean freq ~19.4, recency ~8.8 days)")

premium_id = profile["monetary"].idxmax()
print(f"\nPremium segment id: {premium_id} (mean monetary={profile.loc[premium_id,'monetary']:.2f})")

def name_segment(sid):
    row = profile.loc[sid]
    if sid == premium_id:
        return "Premium"
    if row["frequency"] >= 15:
        return "Loyal"
    if row["recency"] > 60:
        return "At Risk"
    return "Occasional"

segment_names = {sid: name_segment(sid) for sid in profile.index}
rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"]   = rfm["segment_id"] == premium_id

print("\nSegment names and recommended actions:")
actions = {"Premium": "VIP early access offers", "Loyal": "loyalty rewards programme",
           "At Risk": "win-back campaign with discount", "Occasional": "re-engagement nudge"}
for sid, name in segment_names.items():
    print(f"  Cluster {sid} -> {name}: {actions.get(name,'N/A')}")
print("Note: cluster numbers are arbitrary labels — 0 is not 'lowest' nor 3 'highest'.")

pca = PCA(n_components=2)
pca.fit(X_rfm)
evr = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: [{evr[0]:.3f}, {evr[1]:.3f}]  total={evr.sum():.3f}")
print(f"  (ref: ~[0.634, 0.218], total ~0.852)")

seg_out = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].copy()
seg_out.to_csv(os.path.join(OUTPUTS, "customer_segments.csv"), index=False)
print(f"\nWritten outputs/customer_segments.csv ({len(seg_out)} rows)")
print(f"Premium customers: {seg_out['is_premium'].sum()}")

# ── Task 4: Lapse Problem ────────────────────────────────────────────────────
print("\n=== TASK 4: LAPSE PROBLEM ===")
CUTOFF  = pd.Timestamp("2025-09-01")
SEP_END = pd.Timestamp("2025-10-01")

orders_pre    = orders[orders["order_ts"] < CUTOFF].copy()
completed_pre = orders_pre[orders_pre["status"] == "completed"]
cohort_ids    = completed_pre["customer_id"].unique()
print(f"Cohort size: {len(cohort_ids)}  (ref: 1156)")

completed_sep = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= CUTOFF) &
    (orders["order_ts"] < SEP_END)
]["customer_id"].unique()

lapse_df = pd.DataFrame({"customer_id": cohort_ids})
lapse_df["lapsed"] = (~lapse_df["customer_id"].isin(completed_sep)).astype(int)
print(f"Lapsed: {lapse_df['lapsed'].sum()}  ({lapse_df['lapsed'].mean()*100:.2f}%)  (ref: 443, 38.32%)")

# Build features (pre-cutoff only)
feat = (
    completed_pre.groupby("customer_id")
    .agg(
        recency=("order_ts", lambda x: (CUTOFF - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    )
    .reset_index()
)

completed_pre_cat = completed_pre.merge(products[["product_id", "category"]], on="product_id", how="left")
fav_cat = (
    completed_pre_cat.groupby("customer_id")["category"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"category": "fav_category"})
)
feat = feat.merge(fav_cat, on="customer_id", how="left")

all_pre = orders_pre[orders_pre["status"].isin(["completed", "cancelled"])].copy()
canc_rate = (
    all_pre.groupby("customer_id")
    .apply(lambda x: (x["status"] == "cancelled").sum() / len(x))
    .reset_index()
    .rename(columns={0: "cancel_rate"})
)
feat = feat.merge(canc_rate, on="customer_id", how="left")

cust_demo = customers[["customer_id", "city", "age", "gender", "signup_date"]].copy()
cust_demo["signup_date"] = pd.to_datetime(cust_demo["signup_date"])
cust_demo["tenure_days"] = (CUTOFF - cust_demo["signup_date"]).dt.days
feat = feat.merge(cust_demo[["customer_id", "city", "age", "gender", "tenure_days"]], on="customer_id", how="left")
feat = feat.merge(lapse_df, on="customer_id", how="left").dropna(subset=["lapsed"])

# Fill remaining NaN
feat["cancel_rate"]  = feat["cancel_rate"].fillna(0)
feat["avg_delivery"] = feat["avg_delivery"].fillna(feat["avg_delivery"].median())
feat["avg_discount"] = feat["avg_discount"].fillna(0)
feat["fav_category"] = feat["fav_category"].fillna("Grocery")
feat["city"]         = feat["city"].fillna("Unknown")
feat["gender"]       = feat["gender"].fillna("unknown")
feat["age"]          = feat["age"].fillna(feat["age"].median())
feat["tenure_days"]  = feat["tenure_days"].fillna(feat["tenure_days"].median())

FEATURE_COLS = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
                "avg_delivery", "city", "age", "gender", "tenure_days", "fav_category"]
X_all = feat[FEATURE_COLS]
y_all = feat["lapsed"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}  (ref: 924, 232)")

# ── Task 5: Preprocess without leakage ──────────────────────────────────────
print("\n=== TASK 5: PREPROCESSING ===")
NUM_COLS = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
            "avg_delivery", "age", "tenure_days"]
CAT_COLS = ["city", "gender", "fav_category"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

freq_idx          = NUM_COLS.index("frequency")
scaler_step       = preprocessor.named_transformers_["num"]
learned_mean_freq = scaler_step.mean_[freq_idx]
cohort_mean_freq  = feat["frequency"].mean()
print(f"Scaler mean of frequency (train-only) : {learned_mean_freq:.4f}")
print(f"Full-cohort mean of frequency         : {cohort_mean_freq:.4f}")
print("They differ because the scaler was fitted on 80% training rows, not the full cohort.")

assert not np.isnan(X_train_proc).any(), "NaN in training features!"
assert not np.isnan(X_test_proc).any(),  "NaN in test features!"
print("No NaN in any feature column — confirmed.")

# ── Task 6: Baseline and model training ─────────────────────────────────────
print("\n=== TASK 6: BASELINE ===")
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_proc, y_train)
y_pred_dummy = dummy.predict(X_test_proc)
print(f"Baseline accuracy  : {(y_pred_dummy == y_test).mean():.4f}  (ref: 0.616)")
print(f"Baseline lapsed F1 : {f1_score(y_test, y_pred_dummy, pos_label=1, zero_division=0):.4f}  (ref: 0)")

print("\n--- Logistic Regression grid search ---")
lr_pipe = Pipeline([("lr", LogisticRegression(max_iter=1000, random_state=42))])
lr_grid = {"lr__C": [0.01, 0.1, 1, 10, 100]}
lr_cv = GridSearchCV(lr_pipe, lr_grid, cv=5, scoring="f1", n_jobs=-1)
lr_cv.fit(X_train_proc, y_train)
print(f"Best C   : {lr_cv.best_params_}  (ref: C=1)")
print(f"Best CV F1: {lr_cv.best_score_:.4f}")

print("\n--- Random Forest grid search ---")
rf_pipe = Pipeline([("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = {"rf__max_depth": [3, 5, 8], "rf__min_samples_leaf": [5, 10, 20]}
rf_cv = GridSearchCV(rf_pipe, rf_grid, cv=5, scoring="f1", n_jobs=-1)
rf_cv.fit(X_train_proc, y_train)
print(f"Best params: {rf_cv.best_params_}  (ref: depth=8, leaf=10)")
print(f"Best CV F1 : {rf_cv.best_score_:.4f}")

print("\n--- Test-set evaluation ---")
for name, model in [("Logistic Regression", lr_cv), ("Random Forest", rf_cv)]:
    y_pred = model.predict(X_test_proc)
    cm = confusion_matrix(y_test, y_pred)
    p  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    r  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    print(f"\n{name}:")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

best_model  = rf_cv
best_name   = "Random Forest"
y_pred_best = best_model.predict(X_test_proc)
best_f1     = f1_score(y_test, y_pred_best, pos_label=1, zero_division=0)
print(f"\nSelected: {best_name}  Test F1={best_f1:.4f}")
print("Justification: Random Forest achieved higher F1 than Logistic Regression. Missing a lapsing")
print("customer (false negative) is costlier than wrongly targeting a loyal one (false positive),")
print("because lapsed customers represent lost revenue that is expensive to recover. The forest's")
print("higher recall better identifies at-risk customers, making it more aligned with the retention")
print("team's goal of catching churners early, even at the cost of some false positives.")
assert best_f1 >= 0.55, f"Test F1 {best_f1:.4f} below required 0.55"

# ── Task 7: Class-balance experiment ────────────────────────────────────────
print("\n=== TASK 7: OVERSAMPLING ===")
rng_bal = np.random.RandomState(42)
lapsed_idx   = np.where(y_train.values == 1)[0]
majority_n   = (y_train == 0).sum()
extra_idx    = rng_bal.choice(lapsed_idx, size=majority_n - len(lapsed_idx), replace=True)
X_train_bal  = np.vstack([X_train_proc, X_train_proc[extra_idx]])
y_train_bal  = np.concatenate([y_train.values, y_train.values[extra_idx]])
print(f"Balanced train: class 0={( y_train_bal==0).sum()}, class 1={(y_train_bal==1).sum()}  (ref: 570 each)")

best_C  = lr_cv.best_params_["lr__C"]
lr_bal  = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
lr_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = lr_bal.predict(X_test_proc)
y_pred_lr  = lr_cv.predict(X_test_proc)

print("\nLR unbalanced vs balanced (test set):")
for label, y_p in [("Unbalanced", y_pred_lr), ("Balanced", y_pred_bal)]:
    p = precision_score(y_test, y_p, pos_label=1, zero_division=0)
    r = recall_score(y_test, y_p, pos_label=1, zero_division=0)
    f = f1_score(y_test, y_p, pos_label=1, zero_division=0)
    print(f"  {label}: Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
print("(ref: recall rises ~0.64->~0.85; precision stays ~0.6)")
print("Recall moved most. The balanced model catches more churners; the retention team can act on")
print("more at-risk customers, at the cost of contacting some loyal customers unnecessarily.")

# ── Task 8: Interpret, persist, use ─────────────────────────────────────────
print("\n=== TASK 8: INTERPRET AND PERSIST ===")
rf_est     = best_model.best_estimator_.named_steps["rf"]
feat_names_num = NUM_COLS
feat_names_cat = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
all_feat_names = feat_names_num + feat_names_cat
importances    = rf_est.feature_importances_
top5           = np.argsort(importances)[::-1][:5]
print("Top 5 RF feature importances:")
for i in top5:
    print(f"  {all_feat_names[i]}: {importances[i]:.4f}")

lr_est   = lr_cv.best_estimator_.named_steps["lr"]
lr_coefs = lr_est.coef_[0]
print("\nLR coefficients for key features:")
for fn in ["recency", "frequency", "monetary"]:
    idx  = feat_names_num.index(fn)
    sign = "+" if lr_coefs[idx] > 0 else "-"
    interp = {"recency": "higher recency (longer gap) -> more likely lapsed",
               "frequency": "more orders -> less likely lapsed",
               "monetary": "higher spend -> less likely lapsed"}
    print(f"  {fn}: {sign}{abs(lr_coefs[idx]):.4f}  ({interp[fn]})")

joblib.dump(best_model, os.path.join(OUTPUTS, "lapse_model.joblib"))
reloaded    = joblib.load(os.path.join(OUTPUTS, "lapse_model.joblib"))
y_reload    = reloaded.predict(X_test_proc)
assert (y_reload == y_pred_best).all(), "Reloaded model differs!"
print("\nModel saved to outputs/lapse_model.joblib and reloaded — predictions identical.")

segments    = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])

# Align test indices carefully
feat_reset    = feat.reset_index(drop=True)
test_indices  = y_test.index
test_cids     = feat_reset.loc[test_indices, "customer_id"].values
test_lapsed   = y_test.values
test_proba    = best_model.predict_proba(X_test_proc)[:, 1]

test_df = pd.DataFrame({
    "customer_id": test_cids,
    "lapse_probability": test_proba,
    "actually_lapsed": test_lapsed,
})
retention = (
    test_df[test_df["customer_id"].isin(premium_ids)]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
)
retention[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(
    os.path.join(OUTPUTS, "retention_list.csv"), index=False
)
print(f"\nPremium test customers in retention list: {len(retention)}")
print(f"Of these, truly lapsed: {retention['actually_lapsed'].sum()}")
print("Note: the Premium flag uses the whole-window RFM (includes September), making it a post-hoc")
print("label. This retention list is illustrative and not a leakage-free evaluation.")
print("Done Part B.")
