"""Module 5 — injection-budget and realistic-scale curves (quantitative robustness).
On a strong cell (misinfo MMLU.security->B, BGE), sweep:
  (a) #malicious attack queries injected -> how FEW suffice?
  (b) #benign same-victim records in memory -> dilution-defense curve.
"""
import os, json
import lib, attacks
from attacks import Misinformation

OUT = "/home/user/MINJA/repro/results"

def main():
    os.makedirs(OUT, exist_ok=True)
    loader = lambda: lib.load_mmlu("security_studies", "security")
    out = {"budget": {}, "scale": {}}

    print("=== (a) injection-budget: ASR vs #malicious attack queries ===", flush=True)
    for ninj in [1, 2, 4, 8]:
        r = attacks.run_minja(loader(), lib.make_memory("BGE-cosine"), Misinformation(), "security", "B",
                              n_inject=ninj, n_test=8, n_benign=6, n_benign_victim=0, k=5)
        out["budget"][str(ninj)] = r.get("ASR")
        print(f"  n_inject={ninj:2d} -> ASR={r.get('ASR')}  ISR={r.get('ISR')}", flush=True)

    print("\n=== (b) realistic-scale: ASR vs #benign same-victim records ===", flush=True)
    for nbv in [0, 5, 10, 20, 40]:
        r = attacks.run_minja(loader(), lib.make_memory("BGE-cosine"), Misinformation(), "security", "B",
                              n_inject=6, n_test=8, n_benign=6, n_benign_victim=nbv, k=5)
        out["scale"][str(nbv)] = {"ASR": r.get("ASR"), "seeded": r.get("seeded_benign_victim")}
        print(f"  benign_victim={nbv:2d} -> ASR={r.get('ASR')} (seeded={r.get('seeded_benign_victim')})", flush=True)

    json.dump(out, open(f"{OUT}/curves.json", "w"), indent=2)
    print("\nsaved curves.json", flush=True)

if __name__ == "__main__":
    main()
