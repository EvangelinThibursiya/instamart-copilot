"""part_d_rag_agent.py -- Part D: Policy RAG Assistant and Refund Agent."""
import datetime
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICIES_DIR = os.path.join("data", "policies")
RAW          = os.path.join("data", "raw")
DB           = os.path.join("data", "instamart.db")
OUTPUTS      = "outputs"
STATE_DIR    = "state"
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ── Task 1: Chunk documents ───────────────────────────────────────────────────
print("=== TASK 1: CHUNKING ===")

def chunk_document(text, source, chunk_size=40, overlap=10):
    words  = text.split()
    chunks = []
    start  = 0
    step   = chunk_size - overlap
    while start < len(words):
        end        = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({"source": source, "text": chunk_text})
        if end >= len(words):
            break
        start += step
    return chunks

all_chunks   = []
policy_files = sorted(f for f in os.listdir(POLICIES_DIR) if f.endswith(".txt"))
for fname in policy_files:
    with open(os.path.join(POLICIES_DIR, fname), encoding="utf-8") as f:
        text = f.read().strip()
    chunks = chunk_document(text, fname)
    print(f"  {fname}: {len(chunks)} chunks")
    all_chunks.extend(chunks)
print(f"Total chunks: {len(all_chunks)}  (ref: 15 total — refund 4, delivery 3, escalation 3, payment 3, privacy 2)")

# ── Task 2: Vectorise and retrieve ───────────────────────────────────────────
print("\n=== TASK 2: TFIDF RETRIEVAL ===")
vectorizer   = TfidfVectorizer()
chunk_matrix = vectorizer.fit_transform([c["text"] for c in all_chunks])

def retrieve(query, k=3, chunks=None, matrix=None):
    if chunks is None:
        chunks = all_chunks
    if matrix is None:
        matrix = chunk_matrix
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix)[0]
    top_k  = np.argsort(scores)[::-1][:k]
    return [{"source": chunks[i]["source"], "score": float(scores[i]), "text": chunks[i]["text"]}
            for i in top_k]

test_q = "refund policy late delivery"
top3   = retrieve(test_q)
print(f"Top 3 for '{test_q}':")
for r in top3:
    print(f"  {r['source']}: score={r['score']:.4f}")
print("(ref: all 3 from refund_policy.txt)")
print("\nTF (term frequency) measures how often a word appears in a document; IDF (inverse document")
print("frequency) down-weights words common across many documents. Limitation: TF-IDF cannot match")
print("synonyms — 'late delivery' and 'delayed shipment' score 0 overlap despite identical meaning.")

# ── Task 3: Augment and answer ────────────────────────────────────────────────
print("\n=== TASK 3: ANSWER FUNCTION ===")
POLICY_SYS = ("You are a POLICY ASSISTANT for Instamart. "
               "Answer only from the supplied context. "
               "If the answer is not in the context, say: "
               "I could not find that information in the policy documents.")

def answer(question, chunks=None, matrix=None):
    results   = retrieve(question, k=3, chunks=chunks, matrix=matrix)
    ctx_lines = []
    for r in results:
        ctx_lines.append(f"[{r['source']}]")
        ctx_lines.append(r["text"])
    context  = "\n".join(ctx_lines)
    user_msg = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    messages = [
        {"role": "system", "content": POLICY_SYS},
        {"role": "user",   "content": user_msg},
    ]
    resp    = mock_llm.chat(messages)
    text    = resp.choices[0].message.content
    sources = [r["source"] for r in results]
    return text, sources

sample_ans, sample_src = answer("Who must approve a refund above INR 1000?")
print(f"Sample answer  : {sample_ans}")
print(f"Sources used   : {sample_src}")

# ── Task 4: Evaluate retrieval and grounding ──────────────────────────────────
print("\n=== TASK 4: EVALUATION ===")
eval_df = pd.read_csv(os.path.join(RAW, "policy_eval.csv"))

def run_eval(chunks_list, label):
    texts_local = [c["text"] for c in chunks_list]
    vec_local   = TfidfVectorizer()
    mat_local   = vec_local.fit_transform(texts_local)

    def retrieve_local(query):
        q_vec  = vec_local.transform([query])
        scores = cosine_similarity(q_vec, mat_local)[0]
        top_k  = np.argsort(scores)[::-1][:3]
        return [{"source": chunks_list[i]["source"], "score": float(scores[i]),
                 "text": chunks_list[i]["text"]} for i in top_k]

    def answer_local(question):
        results   = retrieve_local(question)
        ctx_lines = []
        for r in results:
            ctx_lines.append(f"[{r['source']}]")
            ctx_lines.append(r["text"])
        context  = "\n".join(ctx_lines)
        user_msg = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        resp = mock_llm.chat([
            {"role": "system", "content": POLICY_SYS},
            {"role": "user",   "content": user_msg},
        ])
        return resp.choices[0].message.content, [r["source"] for r in results]

    answerable   = eval_df[eval_df["expected_doc"] != "NONE"]
    unanswerable = eval_df[eval_df["expected_doc"] == "NONE"]
    doc_hits = fact_hits = fallback_hits = 0
    failures = []

    for _, row in answerable.iterrows():
        txt, srcs = answer_local(row["question"])
        if row["expected_doc"] in srcs:
            doc_hits += 1
        if row["expected_fact"].lower() in txt.lower():
            fact_hits += 1
        else:
            failures.append(row["question"])

    for _, row in unanswerable.iterrows():
        txt, _ = answer_local(row["question"])
        if txt.startswith("I could not find"):
            fallback_hits += 1

    print(f"\n[{label}] Answerable {len(answerable)}: doc_hits={doc_hits}/{len(answerable)}, "
          f"fact_hits={fact_hits}/{len(answerable)}")
    print(f"[{label}] Unanswerable {len(unanswerable)}: fallback_hits={fallback_hits}/{len(unanswerable)}")
    if failures:
        print(f"[{label}] Failed questions: {failures}")
    return doc_hits, fact_hits, fallback_hits

run_eval(all_chunks, label="overlap=10")

print("\nZero-overlap experiment:")
chunks_no_overlap = []
for fname in policy_files:
    with open(os.path.join(POLICIES_DIR, fname), encoding="utf-8") as f:
        text = f.read().strip()
    chunks_no_overlap.extend(chunk_document(text, fname, chunk_size=40, overlap=0))
run_eval(chunks_no_overlap, label="overlap=0")
print("With zero overlap, a sentence whose key terms span the boundary of two 40-word chunks will")
print("have its meaning split; neither chunk alone retrieves the correct answer.")

# ── Task 5: Tools ─────────────────────────────────────────────────────────────
print("\n=== TASK 5: TOOLS ===")
TOOLS_SPEC = [
    {"name": "get_order",        "description": "Look up an order by ID from the clean SQLite database.", "parameters": {"order_id": "string"}},
    {"name": "search_policy",    "description": "Search policy documents for relevant information.",      "parameters": {"query": "string"}},
    {"name": "calculate_refund", "description": "Compute refund amount from order amount and delivery time.", "parameters": {"order_amount": "number", "delivery_minutes": "number"}},
    {"name": "issue_refund",     "description": "Issue an approved refund (<=INR 1000) to the customer.", "parameters": {"order_id": "string", "refund_inr": "number"}},
    {"name": "escalate_to_human","description": "Escalate a refund above INR 1000 to a human support lead.", "parameters": {"order_id": "string", "refund_inr": "number"}},
]
LEDGER_PATH = os.path.join(OUTPUTS, "refund_ledger.jsonl")

def get_order(order_id):
    conn = sqlite3.connect(DB)
    cur  = conn.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, deliv = row
    return {
        "order_id": oid, "status": status,
        "order_amount": float(amount),
        "delivery_minutes": int(float(deliv)) if deliv is not None and str(deliv).strip() not in ("", "None") else None,
    }

def search_policy(query):
    return {"results": retrieve(query, k=3)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        return {"refund_inr": 0}
    try:
        dm = int(float(delivery_minutes))
    except (TypeError, ValueError):
        return {"refund_inr": 0}
    if dm > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def issue_refund(order_id, refund_inr):
    entry = {"timestamp": datetime.datetime.now().isoformat(), "tool": "issue_refund",
             "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    entry = {"timestamp": datetime.datetime.now().isoformat(), "tool": "escalate_to_human",
             "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    "get_order": get_order, "search_policy": search_policy,
    "calculate_refund": calculate_refund, "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}
print("Tools defined: " + ", ".join(TOOL_MAP))

# ── Task 6 & 7: ReAct loop ────────────────────────────────────────────────────
print("\n=== TASK 6 & 7: REACT AGENT ===")
AGENT_SYS = ("You are an AGENT that processes refund requests for Instamart. "
              "Use the available tools to look up orders, check policies, and process refunds.")

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step, tools_executed):
    path = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "scenario_id": scenario_id, "messages": messages,
                   "tokens_used": tokens_used, "step": step, "tools_executed": tools_executed}, f)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tool=None):
    if run_id is None:
        run_id = f"{scenario_id}_{int(time.time()*1000)}"
    messages       = [{"role": "system", "content": AGENT_SYS},
                      {"role": "user",   "content": user_message}]
    tokens_used    = 0
    steps          = []
    llm_calls      = 0
    tools_executed = []
    start_time     = time.monotonic()

    for iteration in range(max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {"run_id": run_id, "scenario_id": scenario_id,
                    "status": "FALLBACK_TIMEOUT",
                    "final_answer": f"Timed out after {elapsed:.3f}s with {len(steps)} steps completed. Handing to human.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        resp        = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls  += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            return {"run_id": run_id, "scenario_id": scenario_id,
                    "status": "FALLBACK_TOKEN_LIMIT",
                    "final_answer": f"Token limit {token_limit} exceeded ({tokens_used}). {len(steps)} steps. Handing to human.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        msg    = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if not msg.tool_calls or finish == "stop":
            final_answer = msg.content or ""
            status_word  = final_answer.split(":")[0].strip() if ":" in final_answer else final_answer.split(".")[0].strip()
            return {"run_id": run_id, "scenario_id": scenario_id,
                    "status": status_word, "final_answer": final_answer,
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        tc        = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        messages.append({"role": "assistant", "content": None,
                          "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]})

        fn          = TOOL_MAP.get(tool_name)
        observation = fn(**{k: v for k, v in tool_args.items()}) if fn else {"error": f"unknown tool {tool_name}"}

        tools_executed.append(tool_name)
        steps.append({"iteration": iteration, "tool_name": tool_name,
                       "arguments": tool_args, "observation": observation})

        messages.append({"role": "tool", "tool_call_id": tc.id,
                          "name": tool_name, "content": json.dumps(observation)})

        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration, tools_executed)

        if crash_after_tool is not None and len(tools_executed) >= crash_after_tool:
            raise RuntimeError(f"Simulated crash after {crash_after_tool} tool executions")

    return {"run_id": run_id, "scenario_id": scenario_id,
            "status": "FALLBACK_MAX_ITERATIONS",
            "final_answer": f"Reached max_iterations={max_iterations} without resolution. {len(steps)} steps. Handing to human.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        ckpt = json.load(f)
    messages       = ckpt["messages"]
    tokens_used    = ckpt["tokens_used"]
    step_start     = ckpt["step"] + 1
    scenario_id    = ckpt["scenario_id"]
    tools_executed = ckpt["tools_executed"]
    steps          = []
    llm_calls      = 0
    start_time     = time.monotonic()

    for iteration in range(step_start, max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {"run_id": run_id, "scenario_id": scenario_id,
                    "status": "FALLBACK_TIMEOUT",
                    "final_answer": f"Timed out on resume after {elapsed:.3f}s.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        resp        = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls  += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            return {"run_id": run_id, "scenario_id": scenario_id,
                    "status": "FALLBACK_TOKEN_LIMIT",
                    "final_answer": f"Token limit exceeded on resume ({tokens_used}).",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        msg    = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if not msg.tool_calls or finish == "stop":
            final_answer = msg.content or ""
            status_word  = final_answer.split(":")[0].strip() if ":" in final_answer else final_answer.split(".")[0].strip()
            return {"run_id": run_id, "scenario_id": scenario_id,
                    "status": status_word, "final_answer": final_answer,
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        tc        = msg.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        messages.append({"role": "assistant", "content": None,
                          "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]})

        fn          = TOOL_MAP.get(tool_name)
        observation = fn(**{k: v for k, v in tool_args.items()}) if fn else {"error": f"unknown tool {tool_name}"}

        tools_executed.append(tool_name)
        steps.append({"iteration": iteration, "tool_name": tool_name,
                       "arguments": tool_args, "observation": observation})

        messages.append({"role": "tool", "tool_call_id": tc.id,
                          "name": tool_name, "content": json.dumps(observation)})

        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration, tools_executed)

    return {"run_id": run_id, "scenario_id": scenario_id,
            "status": "FALLBACK_MAX_ITERATIONS",
            "final_answer": f"Reached max_iterations on resume. {len(steps)} steps. Handing to human.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}


# ── Run all 7 scenarios ───────────────────────────────────────────────────────
print("\n=== RUNNING 7 SCENARIOS ===")
scenarios  = pd.read_csv(os.path.join(RAW, "agent_scenarios.csv"))
RUNS_PATH  = os.path.join(OUTPUTS, "agent_runs.jsonl")
open(RUNS_PATH,  "w").close()
open(LEDGER_PATH, "w").close()

results_table = []
for _, row in scenarios.iterrows():
    sid    = row["scenario_id"]
    msg    = row["message"]
    run_id = f"run_{sid}"
    print(f"\nScenario {sid}: {msg[:70]}...")
    result = run_agent(msg, sid, run_id=run_id)
    print(f"  Status: {result['status']}  LLM calls: {result['llm_calls']}  Tokens: {result['tokens_used']}")

    refund_amt = 0.0
    for step in result["steps"]:
        if step["tool_name"] in ("issue_refund", "escalate_to_human"):
            refund_amt = step["observation"].get("refund_inr", 0)

    results_table.append({
        "scenario_id":   sid,
        "expected":      row["expected_outcome"],
        "status":        result["status"],
        "refund_inr":    refund_amt,
        "exp_refund":    row["expected_refund_inr"],
        "llm_calls":     result["llm_calls"],
        "tokens_used":   result["tokens_used"],
    })

    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "scenario_id": sid, "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }) + "\n")

print("\n=== SCENARIO RESULTS TABLE ===")
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
print("(ref: S1 REFUND 128.25, S2 REFUND 135.00, S3 ESCALATE 1999.00, S4 DENIED, S5 DENIED, S6 NOT_FOUND, S7 FALLBACK)")

# ── Task 7: Stop safely demonstrations ──────────────────────────────────────
print("\n=== TASK 7: STOP SAFELY DEMOS ===")
s1_row = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]

result_token = run_agent(s1_row["message"], "S1_token_demo", run_id="demo_token", token_limit=200)
print(f"S1 token_limit=200  -> status: {result_token['status']}  (expected: FALLBACK_TOKEN_LIMIT)")

result_time = run_agent(s1_row["message"], "S1_time_demo", run_id="demo_time", time_limit_seconds=0)
print(f"S1 time_limit_seconds=0 -> status: {result_time['status']}  (expected: FALLBACK_TIMEOUT)")

# ── Task 8: Crash and resume demo ─────────────────────────────────────────────
print("\n=== TASK 8: CRASH AND RESUME DEMO ===")
crash_run_id = "S1_crash_demo"
ckpt_path    = os.path.join(STATE_DIR, f"{crash_run_id}.json")
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

print("Running S1 with crash_after_tool=2...")
try:
    run_agent(s1_row["message"], "S1", run_id=crash_run_id, crash_after_tool=2)
except RuntimeError as e:
    print(f"Simulated crash caught: {e}")

with open(ckpt_path, encoding="utf-8") as f:
    ckpt = json.load(f)
print(f"Tools before crash  : {ckpt['tools_executed']}")

print("Resuming from checkpoint...")
resume_result = resume_agent(crash_run_id)
print(f"Resume status       : {resume_result['status']}  (expected: REFUND)")
print(f"Resume tools        : {[s['tool_name'] for s in resume_result['steps']]}")

with open(ckpt_path, encoding="utf-8") as f:
    final_ckpt = json.load(f)
print(f"All tools (total)   : {final_ckpt['tools_executed']}")
print("(expected: ['get_order', 'search_policy', 'calculate_refund', 'issue_refund'])")

# Verify refund amount
for r in resume_result["steps"]:
    if r["tool_name"] == "issue_refund":
        print(f"Refund amount       : INR {r['observation']['refund_inr']}  (expected: 128.25)")

print("\n=== TASK 9: REACT vs PLAN-AND-EXECUTE ===")
print("""ReAct vs Plan-and-Execute (within 250 words):

In this refund workflow, S1 used 5 model calls and S5 used 3. A plan-and-execute design would ask
the model once for the full step list (e.g., "call get_order, then search_policy, then
calculate_refund"), then execute them, replanning only on error. For a happy-path scenario like S1
this would need 2 model calls (plan + final decision) vs 5 in ReAct — a 60% token saving. For S5
(on-time order, DENIED in 3 calls) plan-and-execute would similarly need 2 calls.

However, for S6 (NOT_FOUND in 2 calls), a plan would assume a valid order exists and fail on the
first tool call, requiring a replan — ending up with similar or more calls than ReAct. For a
cancelled order (S5), ReAct's observation-driven flow naturally handles the unexpected status
returned by get_order without needing to replan. Plan-and-execute would struggle when the
observation is surprising (cancelled status, or "order not found"), requiring a second LLM call to
adapt, erasing the efficiency gain.

For this refund workflow, I would choose ReAct because: (1) the workflow is short (3-5 steps) so
the overhead of extra model calls is small; (2) edge cases like NOT_FOUND and cancelled orders are
common; (3) each tool observation carries new information the model should react to. Plan-and-execute
is better suited to longer, predictable workflows where replanning on error is rare.
""")

print("Done Part D.")
