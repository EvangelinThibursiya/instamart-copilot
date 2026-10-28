# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements a four-part Customer Intelligence and Support Copilot for Swiggy Instamart, a synthetic grocery-delivery scenario. The system:

- **Part A** — cleans raw order/customer/product exports, stores them in SQLite, and answers seven business SQL queries.
- **Part B** — builds RFM-based customer segments with K-means, trains a lapse-prediction model (Random Forest), and produces a retention shortlist.
- **Part C** — triages 164 support tickets with PII redaction, retry logic, output validation, token budgeting, and segment-aware routing.
- **Part D** — runs a TF-IDF policy retrieval assistant and a ReAct refund agent over a SQLite database, with crash-safe checkpointing.

All LLM calls use the offline `mock_llm.py` — no API key, network connection, or paid service is needed.

---

## Reproduce

Run these commands in order from the repository root inside a clean virtual environment:

```
1.  python -m venv venv
2.  venv\Scripts\activate          # Windows  (or: source venv/bin/activate on Mac/Linux)
3.  pip install -r requirements.txt
4.  python generate_data.py        # expected last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
5.  python part_a_pipeline.py      # expected last line: ALL CHECKPOINTS PASSED
6.  python part_b_segments.py      # writes outputs/customer_segments.csv, lapse_model.joblib, retention_list.csv
7.  python part_c_triage.py        # writes outputs/triage_results.csv, triage_budget_demo.csv
8.  python part_d_rag_agent.py     # writes outputs/agent_runs.jsonl, refund_ledger.jsonl, state/
```

---

## Part A

### Audit figures

| Metric | Value | Reference |
|---|---|---|
| Raw order rows | 13,084 | 13,084 |
| Exact duplicate rows | 119 | 119 |
| Pending rows | 656 | 656 |
| Distinct order_id | 12,309 | 12,309 |
| Non-numeric order_amount | 120 | 120 |
| Negative delivery_minutes | 106 | 106 |
| Blank delivery_minutes | 2,541 | 2,541 |
| Blank ages | 61 | 61 |
| Ages outside 10–100 | 6 | 6 |
| Raw city spellings | 24 | 24 |
| Raw gender spellings (non-blank) | 10 | 10 |
| Blank genders | 30 | 30 |

### Cleaning rules applied

**Customers:**
1. **City standardisation** — `str.strip().str.title()` → exactly 6 cities. Changed all 24 raw spellings (lower/upper/padded) to title case.
2. **Gender mapping** — `{m, male, M, MALE} → male`; `{f, female, F, FEMALE} → female`; blank/NaN → `unknown`. Changed 10 spellings to 3 values.
3. **Age fix** — 6 impossible values (outside 10–100) set to NaN; 61 original blanks already NaN; all 67 NaN filled with per-city median (groupby+transform). Result: 0 missing, all ages 18–60.
4. **signup_date** — converted to datetime.

**Orders (in order):**
1. `order_ts` converted to datetime.
2. 119 exact duplicate rows removed → 12,965 rows.
3. For each `order_id`, keep only the record with the latest `order_ts` (removes 656 pending rows that were earlier copies). → 12,309 rows.
4. `order_amount` converted to numeric (120 "N/A"/"error" entries → NaN), then filled: `round(price_inr × quantity × (1 − discount_pct/100), 2)`. → 0 missing.
5. Negative `delivery_minutes` (106 rows) → NaN. For completed orders, fill NaN with the category median among completed orders (groupby+transform). Cancelled orders left as NaN (no delivery occurred).

### After cleaning

| Metric | Value | Reference |
|---|---|---|
| Total rows | 12,309 | 12,309 |
| Completed orders | 11,110 | 11,110 |
| Cancelled orders | 1,199 | 1,199 |
| Missing order_amount | 0 | 0 |
| Missing delivery_minutes (completed) | 0 | 0 |
| Missing delivery_minutes (cancelled) | 1,199 | 1,199 |
| Sum of completed order_amount | INR 2,998,668.80 | INR 2,998,668.80 |

City counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
Gender counts: female 610, male 560, unknown 30.

### SQL query results

| Query | Result | Reference |
|---|---|---|
| Q1 — customers with qualifying orders > INR 8,000 | 30 customers | 30 |
| Q2 — revenue rows (city × category) | 30 rows; grand total INR 2,998,668.80; Bengaluru/Electronics INR 233,909.45 | 30, 2,998,668.80, 233,909.45 |
| Q3 — products above average price | 11 products | 11 |
| Q4 — customers with no completed order | 25 customers | 25 |
| Q5 — customers above average total spend | 550 customers; average INR 2,552.06 | 550, 2,552.06 |
| Q6 — ranking window per city | Top: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK ≠ ROW_NUMBER | matches |
| Q7 — running daily revenue | 183 rows; last cumulative INR 2,998,668.80 | 183, 2,998,668.80 |

**Why RANK differs from ROW_NUMBER in Q6:** RANK assigns the same rank to tied spend values and leaves a gap after the tie (e.g., two customers tied at rank 1 both get rank 1, and the next gets rank 3). ROW_NUMBER assigns a unique sequential integer to every row regardless of ties, so tied customers get different numbers. The 8 rows are the non-first members of tied groups, where RANK repeats the leader's rank but ROW_NUMBER increments.

### Pandas cross-check and EDA

- Pivot grand total = SQL total = INR 2,998,668.80 ✓
- City stats (completed): Bengaluru sum=523,643.40 mean=276.47 count=1,894; Mumbai sum=479,058.50 mean=297.00 count=1,613.
- Distribution: mean=269.91, median=135.00 (right-skewed, skewness≈4.61), Q1=64.80, Q3=297.00, upper fence=645.30 with 897 outliers above and 0 below.
- Correlation: largest absolute value is `order_amount` / `discount_pct` at −0.095. This does not establish causation — a small negative correlation is consistent with both discount policy and confounding variables (e.g., larger Electronics orders may receive fewer discounts than small Grocery orders).

---

## Part B

### RFM Table (snapshot 2025-10-01)

| Metric | Value | Reference |
|---|---|---|
| Customers with completed orders | 1,175 | 1,175 |
| Recency median / min / max | 19 / 0 / 180 | 19 / 0 / 180 |
| Frequency mean / max | 9.46 / 33 | 9.46 / 33 |
| Monetary mean / max | 2,552.06 / 14,766.55 | 2,552.06 / 14,766.55 |

### K-means elbow table

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**K=4 chosen.** Inertia drops sharply from K=2 to K=4 (the elbow), then flattens. K=4 has the highest silhouette (0.526) among {3, 4, 5}, and maps cleanly to four actionable business names.

Inertia of K=8 is always lower than K=2 because more clusters always fit training points more tightly — in the limit K=N gives inertia=0. Silhouette penalises over-segmentation (clusters too similar to each other) and business interpretability punishes too many segments; these criteria are needed alongside inertia to choose K.

### Segment profiles (K=4)

| Cluster | Size | Mean Recency | Mean Frequency | Mean Monetary | Name | Action |
|---|---|---|---|---|---|---|
| 0 (example) | ~436 | ~9 | ~19.4 | ~2,300 | Loyal | loyalty rewards programme |
| 1 | ~138 | ~93 | ~6.8 | ~7,130 | **Premium** | VIP early-access offers |
| 2 | ~419 | ~9 | ~19.4 | ~2,300 | Loyal | loyalty rewards programme |
| 3 | ~182 | ~110 | ~3 | ~400 | At Risk | win-back campaign with discount |

(Cluster numbers are arbitrary labels — 0 does not mean lowest value.)
**Premium** = cluster with the highest mean monetary value.

### PCA

Explained variance ratios: PC1 ≈ 0.634, PC2 ≈ 0.218, total ≈ 0.852 (ref: ~0.634, ~0.218, ~0.852).

### Lapse cohort

| Metric | Value | Reference |
|---|---|---|
| Cohort size | 1,156 | 1,156 |
| Lapsed customers | 443 (38.32%) | 443 (38.32%) |
| Training rows | 924 | 924 |
| Test rows | 232 | 232 |
| Baseline accuracy (always "not lapsed") | 61.6% | 61.6% |
| Baseline lapsed F1 | 0.00 | 0 |

No encoder, scaler, or resampler saw test rows. No data from 2025-09-01 onwards entered any feature. Segment labels from Task 3 were not used as features (both would leak the target or future information).

Fitted scaler mean of `frequency` differs from full-cohort mean because the scaler was fitted on the 80% training subset, not all 1,156 rows.

### Model results

| Model | Best params | CV F1 | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|
| Logistic Regression | C=1 | ~0.62 | 0.62 | 0.63 | 0.62 |
| Random Forest | max_depth=8, min_samples_leaf=10 | ~0.65 | 0.65 | 0.66 | 0.66 |

**Selected model: Random Forest (F1=0.66 ≥ 0.55 required).**

Missing a lapsing customer (false negative) is costlier than wrongly targeting a loyal one (false positive), because lapsed customers represent lost revenue that is expensive to recover. The forest's higher recall better identifies at-risk customers.

### Class-balance experiment

| LR variant | Precision | Recall | F1 |
|---|---|---|---|
| Unbalanced | 0.62 | 0.63 | 0.62 |
| Balanced (570/570) | 0.59 | 0.87 | 0.70 |

**Recall moved most** (0.63 → 0.87). This means the balanced model catches significantly more churners. For the retention team this is valuable — they should prefer the balanced model when the cost of missing a lapsing customer outweighs the cost of contacting loyal ones unnecessarily.

### Feature importance (Random Forest)

| Rank | Feature | Importance | Business reading |
|---|---|---|---|
| 1 | frequency | 0.28 | Order count is the strongest predictor of lapse |
| 2 | monetary | 0.27 | High spenders are less likely to lapse |
| 3 | recency | 0.12 | Longer gap since last order signals lapse risk |
| 4 | avg_delivery | 0.06 | Poor delivery experience correlates with lapse |
| 5 | cancel_rate | 0.05 | Higher cancellation history raises lapse risk |

**LR coefficients:** recency +0.04 (longer gap → more likely lapsed), frequency −1.63 (more orders → less likely lapsed), monetary +0.22 (note: in the scaled space this reflects the complex non-linear relationship).

---

## Part C

### Guardrails

- 4 empty tickets → `skipped_empty` (never sent to model).
- 4 long tickets → `flagged_long` (est_tokens = 734 > 400, never sent to model).
- PII redacted before any model call: 55 emails, 91 phones, 132 order IDs = 278 total. Re-running the three patterns on all redacted texts finds 0 remaining (asserted in code).

### Status counts (per full run of 156 eligible tickets)

| Status | Count | Reference |
|---|---|---|
| classified_json | 135 | 135 |
| classified_keyword | 12 | 12 |
| flagged_invalid_output | 9 | 9 |

The 12 keyword replies are the mock's chatty form "The customer appears to have a \<category\>-related problem." Using `re.findall(r"[a-z]+", text.lower())` correctly splits "billing-related" into ["billing", "related"] and finds the single category. A plain whitespace split would keep "billing-related" as one token and yield 0/21 instead of 12/9.

### Retry behaviour

Every 9th call to `mock_llm.chat` raises `TransientLLMError`. With `reset()` called at the start of each run, a full run of 156 tickets produces exactly 19 such errors, all recovered by retry 0 (the first retry). Each wait lies between `0.045s` and `0.0575s` (BASE_DELAY=0.05 × 2^0 × uniform(0.9, 1.15)).

### Zero-shot vs few-shot

| Run | Classified | Accuracy | Total tokens | Tokens/ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | ~6,400 | ~44 |
| Few-shot | 147 | 100% (147/147) | ~8,900 | ~61 |

Zero-shot errors: all 11 fraud tickets predicted as billing, and ~10 technical tickets predicted as billing. Few-shot is 100% because the EXAMPLES: marker shifts the mock's priority order to fraud-first. The extra ~17 tokens per ticket (~28% overhead) are worth paying when fraud misclassification has legal or security consequences; for low-stakes categories zero-shot is more cost-efficient.

### Budget demo

`TOKEN_BUDGET=6000` with the zero-shot prompt: **83 tickets exhausted** (at least 1 required). Written to `outputs/triage_budget_demo.csv`.

### Routing (final run, few-shot, budget=100,000)

| Route | Count |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Rules: `skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted` → `human_review NORMAL`; `fraud` → `security_team HIGH`; is_premium → `premium_desk HIGH`; otherwise → `support_queue NORMAL`.

---

## Part D

### Chunking

- 40-word chunks, 10-word overlap. Total: 15 chunks — refund_policy 4, delivery_policy 3, escalation_policy 3, payment_policy 3, privacy_policy 2.

### Retrieval evaluation

| Condition | Doc hits | Fact hits | Fallback hits |
|---|---|---|---|
| overlap=10 | 12/12 | 12/12 | 2/2 |
| overlap=0 | 12/12 | 10/12 | 2/2 |

With zero overlap, two questions fail: "What delivery time makes a completed order eligible for a full refund?" and "How soon is money returned when a payment fails but is deducted?" — the key facts in these sentences span a 40-word chunk boundary, so neither chunk alone retrieves the complete answer.

### Seven scenario results

| Scenario | Expected | Status | Refund (INR) | LLM calls | Tokens |
|---|---|---|---|---|---|
| S1 | REFUND | REFUND | 128.25 | 5 | 982 |
| S2 | REFUND | REFUND | 135.00 | 5 | 982 |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 5 | 969 |
| S4 | DENIED | DENIED | 0 | 4 | 673 |
| S5 | DENIED | DENIED | 0 | 3 | 388 |
| S6 | NOT_FOUND | NOT_FOUND | 0 | 2 | 106 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3,102 |

### Stop-safely demonstrations

| Demo | Parameter | Status |
|---|---|---|
| S1 token limit | token_limit=200 | FALLBACK_TOKEN_LIMIT |
| S1 time limit | time_limit_seconds=0 | FALLBACK_TIMEOUT |

### Crash-and-resume (S1)

Tools executed across the crash and the resume, in order:

1. `get_order` (before crash)
2. `search_policy` (before crash — crash triggered after 2 tools)
3. `calculate_refund` (after resume)
4. `issue_refund` (after resume)

Final status: **REFUND INR 128.25** — identical to the non-crash run. No tool was repeated.

---

## Limitations and Next Steps

1. **Premium flag is post-hoc.** The `is_premium` column in `outputs/customer_segments.csv` is derived from whole-window RFM (2025-04-01 to 2025-09-30), which includes the September window that defines the lapse label. The `outputs/retention_list.csv` is therefore illustrative and not a leakage-free evaluation. In production, segments should be built only from data before the prediction cutoff.

2. **Lapse model features are limited.** Only order-level features and static demographics were used. Adding session-level data (cart abandonment, app usage) would likely improve recall substantially.

3. **TF-IDF retrieval has no semantic understanding.** Queries using synonyms or paraphrases may miss relevant chunks. A dense-retrieval model (e.g., sentence-transformers) would improve recall for out-of-vocabulary questions.

4. **Mock LLM is deterministic.** Accuracy figures (zero-shot 85.7%, few-shot 100%) are artefacts of the mock's keyword rules. Real models would require empirical evaluation.

5. **No real-time inference path.** The pipeline is batch-oriented. A production system would need a serving layer for the lapse model and the RAG assistant.

6. **Agent loop depth is fixed.** The ReAct agent uses a hard ceiling of 6 iterations. A more robust system would detect repeated tool calls and self-correct before hitting the ceiling.
