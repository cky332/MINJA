"""Module 8 — combined defense D2+D3 and its overall utility cost.
D2 = retrieval consistency filter; D3 = RQ2 self-derivation prompt. We measure
ASR under none/D2/D3/D2+D3, and benign-utility (accuracy on non-victim items) so
the combined defense's effectiveness can be weighed against its cost."""
import os, json
from concurrent.futures import ThreadPoolExecutor
import lib
from run_defense import inject, test_asr, zero_shot, SYS_BASE, SYS_D3
from attacks import Misinformation, ToolHijack, LabelFlip

OUT = "/home/user/MINJA/repro/results"
CELLS = [
    ("misinfo security->B", lambda: lib.load_mmlu("security_studies", "security"), "security", "B", Misinformation),
    ("tool_hijack inv->email", lambda: lib.load_synth_tool("invoice"), "invoice", "send_email", ToolHijack),
    ("label_flip hp->positive", lambda: lib.load_synth_sentiment("headphones"), "headphones", "positive", LabelFlip),
]

def run_cell(name, loader, victim, target, A):
    atk = A(); items = loader()
    mem, tp, tt, _, _ = inject(items, atk, victim, target)
    res = {d: test_asr(mem, tp, atk, target, tt, d) for d in ["none", "D2", "D3", "D2D3"]}
    # benign utility (non-victim items): accuracy under no-defense vs D2+D3 (D3 prompt + filtered demos)
    benign = [it for it in items if not it["has_victim"]][:8] or [it for it in items if it["has_victim"]][22:30]
    acc_base = sum(zero_shot(it, tt, SYS_BASE) == it["answer"] for it in benign) / max(1, len(benign)) * 100
    acc_combo = sum(zero_shot(it, tt, SYS_D3) == it["answer"] for it in benign) / max(1, len(benign)) * 100
    return {"cell": name, "ASR_none": res["none"], "ASR_D2": res["D2"], "ASR_D3": res["D3"],
            "ASR_D2D3": res["D2D3"], "benign_acc_base": acc_base, "benign_acc_combo": acc_combo}

def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in [ex.submit(run_cell, *c) for c in CELLS]:
            try: r = f.result()
            except Exception as e:
                import traceback; r = {"error": repr(e), "tb": traceback.format_exc()[:500]}
            results.append(r); json.dump(results, open(f"{OUT}/combo.json", "w"), indent=2); print(r, flush=True)
    print("\n===== COMBINED DEFENSE (ASR%) + utility =====", flush=True)
    print(f"{'cell':24s} {'none':>5s} {'D2':>5s} {'D3':>5s} {'D2+D3':>6s} | benign_acc base->combo", flush=True)
    for r in results:
        if "error" in r: print("ERR", r["error"][:60]); continue
        print(f"{r['cell']:24s} {r['ASR_none']:5.0f} {r['ASR_D2']:5.0f} {r['ASR_D3']:5.0f} {r['ASR_D2D3']:6.0f} "
              f"| {r['benign_acc_base']:.0f}->{r['benign_acc_combo']:.0f}", flush=True)

if __name__ == "__main__":
    main()
