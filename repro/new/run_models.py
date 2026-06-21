"""Module 1 — model generalization: does MINJA transfer across LLMs?
Runs a representative spread of attacks (BGE memory) on several SiliconFlow models."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import lib, attacks
from attacks import ASCIIShift, Misinformation, ToolHijack, LabelFlip

OUT = "/home/user/MINJA/repro/results"
MODELS = ["Pro/deepseek-ai/DeepSeek-V3.2-Exp", "Qwen/Qwen2.5-72B-Instruct",
          "deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]
CELLS = [  # (name, loader, victim, target, AttackCls)
    ("misinfo MMLU.security->B", lambda: lib.load_mmlu("security_studies", "security"), "security", "B", Misinformation),
    ("ascii MMLU.total", lambda: lib.load_mmlu("elementary_mathematics", "total"), "total", "", ASCIIShift),
    ("tool_hijack invoice->email", lambda: lib.load_synth_tool("invoice"), "invoice", "send_email", ToolHijack),
    ("label_flip headphones->pos", lambda: lib.load_synth_sentiment("headphones"), "headphones", "positive", LabelFlip),
]

def run_one(model, name, loader, victim, target, A):
    lib.MODEL = model  # patch the global used by lib.chat
    try:
        r = attacks.run_minja(loader(), lib.make_memory("BGE-cosine"), A(), victim, target,
                              n_inject=5, n_test=6, n_benign=6, k=4)
        return {"model": model, "cell": name, "ISR": r.get("ISR"), "ASR": r.get("ASR"), "error": r.get("error")}
    except Exception as e:
        return {"model": model, "cell": name, "error": repr(e)}

def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(m, *c) for m in MODELS for c in CELLS]
    results = []
    # NOTE: lib.MODEL is a process global; run sequentially per-model to avoid races,
    # but parallelize cells within a model.
    for m in MODELS:
        print(f"\n=== model: {m} ===", flush=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(run_one, m, *c) for c in CELLS]
            for f in as_completed(futs):
                r = f.result(); results.append(r)
                print(f"  {r['cell']:28s} ISR={r.get('ISR','ERR')} ASR={r.get('ASR','ERR')}", flush=True)
                json.dump(results, open(f"{OUT}/models.json", "w"), indent=2)
    print("\n===== ASR% by model x attack =====", flush=True)
    print(f"{'model':34s} " + " ".join(f"{c[0][:18]:18s}" for c in CELLS), flush=True)
    for m in MODELS:
        row = f"{m[:34]:34s} "
        for c in CELLS:
            cell = next((x for x in results if x["model"] == m and x["cell"] == c[0]), {})
            row += f"{(str(cell.get('ASR','ERR'))):18s} "
        print(row, flush=True)

if __name__ == "__main__":
    main()
