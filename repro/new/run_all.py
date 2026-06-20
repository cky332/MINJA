"""Comprehensive MINJA matrix: datasets x memory-systems x task/attack settings.
Writes results JSON + prints a table. Failures are collected for a separate report."""
import os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import lib, attacks
from attacks import ASCIIShift, Misinformation, ToolHijack, LabelFlip

OUT = "/home/user/MINJA/repro/results"

# (name, dataset_loader, victim, target, AttackClass)
SCEN = [
    # ---- original QA objective (ASCII shift) on new datasets ----
    ("ascii | MMLU.security", lambda: lib.load_mmlu("security_studies", "security"), "security", "", ASCIIShift),
    ("ascii | MMLU.patient", lambda: lib.load_mmlu("professional_medicine", "patient"), "patient", "", ASCIIShift),
    ("ascii | MMLU.total", lambda: lib.load_mmlu("elementary_mathematics", "total"), "total", "", ASCIIShift),
    ("ascii | CSQA.water", lambda: lib.load_csqa("water"), "water", "", ASCIIShift),
    # ---- realistic misinformation override (fixed wrong target label) ----
    ("misinfo | MMLU.security->B", lambda: lib.load_mmlu("security_studies", "security"), "security", "B", Misinformation),
    ("misinfo | MMLU.patient->B", lambda: lib.load_mmlu("professional_medicine", "patient"), "patient", "B", Misinformation),
    ("misinfo | CSQA.water->B", lambda: lib.load_csqa("water"), "water", "B", Misinformation),
    # ---- new task settings ----
    ("tool_hijack | invoice->send_email", lambda: lib.load_synth_tool("invoice"), "invoice", "send_email", ToolHijack),
    ("label_flip | headphones->positive", lambda: lib.load_synth_sentiment("headphones"), "headphones", "positive", LabelFlip),
]
MEMORIES = ["BGE-cosine", "LangChain+FAISS", "LlamaIndex"]
# cells that also get the REALISTIC memory-composition variant (benign victim records present)
REALISTIC = {"misinfo | MMLU.security->B", "tool_hijack | invoice->send_email", "label_flip | headphones->positive"}

def run_cell(name, loader, victim, target, AttackCls, memname, realistic=False):
    t0 = time.time()
    try:
        items = loader()
        mem = lib.make_memory(memname)
        nbv = 15 if realistic else 0
        r = attacks.run_minja(items, mem, AttackCls(), victim, target,
                              n_inject=8, n_test=10, n_benign=12, n_benign_victim=nbv, k=5)
        r.update({"scenario": name, "memory": memname, "realistic": realistic, "secs": round(time.time() - t0)})
        return r
    except Exception as e:
        import traceback
        return {"scenario": name, "memory": memname, "realistic": realistic, "error": repr(e),
                "tb": traceback.format_exc()[:800]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=5); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for (name, loader, victim, target, A) in SCEN:
        for memname in MEMORIES:
            jobs.append((name, loader, victim, target, A, memname, False))
            if name in REALISTIC and memname == "BGE-cosine":
                jobs.append((name, loader, victim, target, A, memname, True))
    print(f"running {len(jobs)} cells with {args.workers} workers\n", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, *j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result(); results.append(r)
            tag = "REAL" if r.get("realistic") else "    "
            if "error" in r:
                print(f"[{tag}] {r['scenario']:34s} {r['memory']:16s} ERROR {r['error'][:50]}", flush=True)
            else:
                print(f"[{tag}] {r['scenario']:34s} {r['memory']:16s} ISR={r.get('ISR',0):5.1f} ASR={r.get('ASR',0):5.1f} "
                      f"(seed_bv={r.get('seeded_benign_victim',0)}, {r.get('secs')}s)", flush=True)
            json.dump(results, open(f"{OUT}/new_matrix.json", "w"), indent=2)

    # summary table
    print("\n================= SUMMARY (ASR%) =================", flush=True)
    print(f"{'scenario':36s} " + " ".join(f"{m:14s}" for m in MEMORIES), flush=True)
    for (name, *_rest) in SCEN:
        row = f"{name:36s} "
        for m in MEMORIES:
            cell = next((x for x in results if x["scenario"] == name and x["memory"] == m and not x.get("realistic")), None)
            val = "ERR" if (not cell or "error" in cell) else f"{cell.get('ASR', 0):.0f}"
            row += f"{val:14s} "
        print(row, flush=True)
    print("\n--- realistic memory-composition (benign victim records present), BGE ---", flush=True)
    for r in results:
        if r.get("realistic") and "error" not in r:
            base = next((x for x in results if x["scenario"] == r["scenario"] and x["memory"] == "BGE-cosine" and not x.get("realistic")), {})
            print(f"  {r['scenario']:34s} ASR {base.get('ASR','?')}% (ideal) -> {r['ASR']}% (realistic, +{r['seeded_benign_victim']} benign-victim recs)", flush=True)

if __name__ == "__main__":
    main()
