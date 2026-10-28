"""part_c_triage.py -- Part C: Support Ticket Triage with Guardrails."""
import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

RAW     = os.path.join("data", "raw")
OUTPUTS = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

MAX_INPUT_TOKENS = 400
BASE_DELAY       = 0.05
ALLOWED_CATS     = {"billing", "delivery", "technical", "fraud"}

# ── Prompts ──────────────────────────────────────────────────────────────────
ZERO_SHOT_PROMPT = """You are a CLASSIFY support ticket classifier for Instamart.
Classify each ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply with a JSON object only, in the form: {"category": "...", "reason": "..."}
Do not include any other text."""

FEW_SHOT_PROMPT = """You are a CLASSIFY support ticket classifier for Instamart.
Classify each ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply with a JSON object only, in the form: {"category": "...", "reason": "..."}
Do not include any other text.

EXAMPLES:
Ticket: "I was charged twice for the same order. Please refund the extra amount."
Answer: {"category": "billing", "reason": "double charge complaint"}

Ticket: "There is an unauthorized transaction on my card. I did not place this order. This is fraud."
Answer: {"category": "fraud", "reason": "unauthorized transaction reported"}

Ticket: "The app crashes every time I try to checkout."
Answer: {"category": "technical", "reason": "app crash issue"}"""

# ── Redaction ────────────────────────────────────────────────────────────────
EMAIL_PAT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PAT = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PAT = re.compile(r"ORD-\d{6}")

def redact(text):
    text, ne  = EMAIL_PAT.subn("[EMAIL]", text)
    text, np_ = PHONE_PAT.subn("[PHONE]", text)
    text, no  = ORDER_PAT.subn("[ORDER_ID]", text)
    return text, ne, np_, no

# ── Retry wrapper ─────────────────────────────────────────────────────────────
def call_with_retry(messages, ticket_id, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            if attempt == max_retries:
                raise
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"  [RETRY] ticket={ticket_id} retry={attempt} wait={wait:.4f}s ({e})")
            time.sleep(wait)

# ── Output validation ─────────────────────────────────────────────────────────
def validate_response(content):
    """Returns (status, category)."""
    try:
        parsed = json.loads(content)
        cat = parsed.get("category", "")
        if cat in ALLOWED_CATS:
            return "classified_json", cat
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        words      = re.findall(r"[a-z]+", (content or "").lower())
        cats_found = list({w for w in words if w in ALLOWED_CATS})
        if len(cats_found) == 1:
            return "classified_keyword", cats_found[0]
        return "flagged_invalid_output", None

# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_triage(tickets_df, system_prompt, token_budget=None, label=""):
    mock_llm.reset()
    records       = []
    tokens_so_far = 0

    for _, row in tickets_df.iterrows():
        tid  = row["ticket_id"]
        cid  = row["customer_id"]
        text = str(row["text"])

        if not text.strip():
            records.append({"ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                             "category": None, "tokens_used": 0})
            continue

        if est_tokens(text) > MAX_INPUT_TOKENS:
            records.append({"ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                             "category": None, "tokens_used": 0})
            continue

        if token_budget is not None and tokens_so_far >= token_budget:
            records.append({"ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                             "category": None, "tokens_used": 0})
            continue

        redacted, _, _, _ = redact(text)

        # Verify no PII in message sent to model
        assert not EMAIL_PAT.search(redacted), f"EMAIL in prompt for {tid}"
        assert not PHONE_PAT.search(redacted), f"PHONE in prompt for {tid}"
        assert not ORDER_PAT.search(redacted), f"ORDER_ID in prompt for {tid}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": redacted},
        ]

        resp        = call_with_retry(messages, tid)
        tokens_used = resp.usage.total_tokens
        tokens_so_far += tokens_used

        content          = resp.choices[0].message.content
        status, category = validate_response(content)

        records.append({"ticket_id": tid, "customer_id": cid, "status": status,
                         "category": category, "tokens_used": tokens_used})

    return pd.DataFrame(records)

# ── Load tickets ──────────────────────────────────────────────────────────────
print("=== PART C: TICKET TRIAGE ===")
tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
print(f"Loaded {len(tickets)} tickets")

# ── Task 1 ────────────────────────────────────────────────────────────────────
empty_count = (tickets["text"].str.strip() == "").sum()
long_count  = tickets["text"].apply(lambda t: est_tokens(t) > MAX_INPUT_TOKENS).sum()
long_est    = tickets[tickets["text"].apply(lambda t: est_tokens(t) > MAX_INPUT_TOKENS)]["text"].apply(est_tokens)
print(f"\nTask 1 - Empty tickets  : {empty_count}  (ref: 4)")
print(f"Task 1 - Long tickets   : {long_count}  (ref: 4)")
if len(long_est):
    print(f"Task 1 - Long token ests: {long_est.values}  (ref: 734)")

# ── Task 2: Redaction stats ───────────────────────────────────────────────────
total_e = total_p = total_o = 0
redacted_texts = []
for text in tickets["text"]:
    r, ne, np_, no = redact(str(text))
    total_e += ne; total_p += np_; total_o += no
    redacted_texts.append(r)
print(f"\nTask 2 - Emails redacted : {total_e}  (ref: 55)")
print(f"Task 2 - Phones redacted : {total_p}  (ref: 91)")
print(f"Task 2 - Orders redacted : {total_o}  (ref: 132)")
print(f"Task 2 - Total           : {total_e + total_p + total_o}  (ref: 278)")

remaining = sum(
    len(EMAIL_PAT.findall(t)) + len(PHONE_PAT.findall(t)) + len(ORDER_PAT.findall(t))
    for t in redacted_texts
)
assert remaining == 0, f"PII remaining: {remaining}"
print(f"Task 2 - PII after redaction: {remaining}  (verified 0)")

# ── Task 6: Zero-shot and few-shot runs ────────────────────────────────────────
print("\n=== TASK 6: ZERO-SHOT RUN ===")
zs_results    = run_triage(tickets, ZERO_SHOT_PROMPT, label="zero-shot")
zs_classified = zs_results[zs_results["status"].isin(["classified_json", "classified_keyword"])]
zs_acc_df     = zs_classified.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
zs_acc        = (zs_acc_df["category"] == zs_acc_df["true_category"]).mean()
zs_tokens     = zs_results["tokens_used"].sum()
zs_counts     = zs_results["status"].value_counts()
print(f"classified_json     : {zs_counts.get('classified_json', 0)}  (ref: 135)")
print(f"classified_keyword  : {zs_counts.get('classified_keyword', 0)}  (ref: 12)")
print(f"flagged_invalid     : {zs_counts.get('flagged_invalid_output', 0)}  (ref: 9)")
print(f"Zero-shot accuracy  : {zs_acc*100:.1f}%  (ref: 85.7%)")
print(f"Total tokens        : {zs_tokens}")
if len(zs_classified):
    print(f"Tokens/classified   : {zs_tokens/len(zs_classified):.1f}")

print("\nZero-shot confusion table:")
zs_conf = pd.crosstab(zs_acc_df["true_category"], zs_acc_df["category"],
                      rownames=["true"], colnames=["pred"])
print(zs_conf.to_string())

print("\n=== TASK 6: FEW-SHOT RUN ===")
fs_results    = run_triage(tickets, FEW_SHOT_PROMPT, label="few-shot")
fs_classified = fs_results[fs_results["status"].isin(["classified_json", "classified_keyword"])]
fs_acc_df     = fs_classified.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
fs_acc        = (fs_acc_df["category"] == fs_acc_df["true_category"]).mean()
fs_tokens     = fs_results["tokens_used"].sum()
fs_counts     = fs_results["status"].value_counts()
print(f"classified_json     : {fs_counts.get('classified_json', 0)}")
print(f"classified_keyword  : {fs_counts.get('classified_keyword', 0)}")
print(f"flagged_invalid     : {fs_counts.get('flagged_invalid_output', 0)}")
print(f"Few-shot accuracy   : {fs_acc*100:.1f}%  (ref: 100%)")
print(f"Total tokens        : {fs_tokens}")
if len(fs_classified):
    print(f"Tokens/classified   : {fs_tokens/len(fs_classified):.1f}")

print("\nComparison: few-shot is 100% accurate vs zero-shot's 85.7% because the EXAMPLES: marker")
print("changes the mock's category priority order (fraud first), correctly resolving the 11 fraud")
print("tickets misclassified as billing in zero-shot. The extra ~15-20 tokens per ticket are worth")
print("paying when fraud misclassification has legal or security consequences. For low-stakes")
print("categories where zero-shot already achieves high accuracy, zero-shot is more cost-efficient.")

# ── Task 7a: Budget demo ──────────────────────────────────────────────────────
print("\n=== TASK 7a: BUDGET DEMO (TOKEN_BUDGET=6000) ===")
budget_results = run_triage(tickets, ZERO_SHOT_PROMPT, token_budget=6000, label="budget-demo")
n_exhausted    = (budget_results["status"] == "budget_exhausted").sum()
print(f"Budget-exhausted tickets: {n_exhausted}  (at least 1 required)")
budget_results.to_csv(os.path.join(OUTPUTS, "triage_budget_demo.csv"), index=False)
print("Written outputs/triage_budget_demo.csv")

# ── Task 7b: Final routing run ────────────────────────────────────────────────
print("\n=== TASK 7b: FINAL ROUTING RUN (few-shot, budget=100000) ===")
segments    = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
premium_map  = segments.set_index("customer_id")["is_premium"].to_dict()
seg_name_map = segments.set_index("customer_id")["segment_name"].to_dict()

final_results = run_triage(tickets, FEW_SHOT_PROMPT, token_budget=100000, label="final")

def assign_route(row):
    status   = row["status"]
    cat      = row["category"]
    cid      = row["customer_id"]
    is_prem  = premium_map.get(cid, False)
    seg_name = seg_name_map.get(cid, "")

    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return pd.Series({"route": "human_review", "priority": "NORMAL", "seg": seg_name})
    if cat == "fraud":
        return pd.Series({"route": "security_team", "priority": "HIGH",   "seg": seg_name})
    if is_prem:
        return pd.Series({"route": "premium_desk",  "priority": "HIGH",   "seg": seg_name})
    return pd.Series({"route": "support_queue",     "priority": "NORMAL", "seg": seg_name})

route_cols = final_results.apply(assign_route, axis=1)
final_out  = final_results.copy()
final_out["segment_name"] = route_cols["seg"]
final_out["priority"]     = route_cols["priority"]
final_out["route"]        = route_cols["route"]

final_out.to_csv(os.path.join(OUTPUTS, "triage_results.csv"), index=False)
print(f"Written outputs/triage_results.csv ({len(final_out)} rows)  (ref: 164)")
print(final_out["route"].value_counts().to_string())
print("Done Part C.")
