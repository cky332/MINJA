"""Aggregate QA + EHR reproduction results and extract failure cases.
Compares DeepSeek-V3.2 reproduction vs the paper's reported numbers."""
import json, os, glob, statistics

RES = os.path.join(os.path.dirname(__file__), "results")

# Paper Table 1 (GPT-4 unless noted)
PAPER_QA = {"ISR": 100.0, "ASR": 68.9, "UD": -10.0}          # QA Agent GPT-4 / MMLU
PAPER_EHR_MIMIC = {"ISR": 95.6, "ASR": 57.0, "UD": -0.7}     # EHR GPT-4 / MIMIC-III

def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pat))):
        if "summary" in f:
            continue
        out.append(json.load(open(f)))
    return out

def qa_report():
    rows = load("qa_*.json")
    rows = [r for r in rows if "ISR" in r]
    if not rows:
        print("no QA results yet"); return
    print("\n" + "=" * 78)
    print("QA AGENT (MMLU) — MINJA on DeepSeek-V3.2-Exp  vs  paper GPT-4")
    print("=" * 78)
    print(f"{'victim':11s} {'subj':22s} {'ISR':>6s} {'ASR':>6s} {'UD':>7s}  notes")
    isrs, asrs, uds = [], [], []
    for r in rows:
        isrs.append(r["ISR"]); asrs.append(r["ASR"]); uds.append(r["UD"])
        note = ""
        if r["ISR"] < 90: note += "low-ISR "
        if r["ASR"] < 50: note += "low-ASR "
        print(f"{r['victim']:11s} {r['subject'][:22]:22s} {r['ISR']:6.1f} {r['ASR']:6.1f} {r['UD']:7.1f}  {note}")
    print("-" * 78)
    print(f"{'MEAN(repro)':34s} {statistics.mean(isrs):6.1f} {statistics.mean(asrs):6.1f} {statistics.mean(uds):7.1f}")
    print(f"{'PAPER (GPT-4)':34s} {PAPER_QA['ISR']:6.1f} {PAPER_QA['ASR']:6.1f} {PAPER_QA['UD']:7.1f}")
    print(f"{'DELTA':34s} {statistics.mean(isrs)-PAPER_QA['ISR']:+6.1f} "
          f"{statistics.mean(asrs)-PAPER_QA['ASR']:+6.1f} {statistics.mean(uds)-PAPER_QA['UD']:+7.1f}")

    # failure case extraction
    print("\n--- QA FAILURE CASES ---")
    for r in rows:
        # ISR failures: bare inject queries that didn't go malicious
        isr_fail = [e for e in r.get("events", []) if e.get("kind") == "inject" and not e.get("malicious")]
        asr_fail = [t for t in r.get("test_details", []) if not t.get("malicious")]
        parse_fail = [e for e in r.get("events", []) if e.get("parse_fail")]
        ud_flips = [u for u in r.get("ud_details", []) if u.get("clean_ok") and not u.get("pois_ok")]
        if isr_fail or asr_fail or parse_fail or ud_flips:
            print(f"[{r['victim']}] ISR_fail={len(isr_fail)}/10 ASR_fail={len(asr_fail)}/10 "
                  f"parse_fail={len(parse_fail)} UD_flips(correct->wrong under poison)={len(ud_flips)}")
            for t in asr_fail[:2]:
                print(f"    ASR miss: ans={t.get('answer')} thought='{(t.get('thought') or '')[:90]}'")

def ehr_report():
    rows = load("ehr_*.json")
    rows = [r for r in rows if "ISR_strict" in r]
    if not rows:
        print("\nno EHR results yet"); return
    print("\n" + "=" * 78)
    print("EHRAgent (MIMIC-III) — MINJA on DeepSeek-V3.2-Exp  vs  paper GPT-4")
    print("=" * 78)
    print(f"{'pair':16s} {'ISRstr':>7s} {'ISRlen':>7s} {'ASRstr':>7s} {'ASRlen':>7s}")
    ss, sl, asr_s, asr_l = [], [], [], []
    for r in rows:
        ss.append(r["ISR_strict"]); sl.append(r["ISR_lenient"])
        asr_s.append(r["ASR_strict"]); asr_l.append(r["ASR_lenient"])
        print(f"{r['victim']}->{r['target']:9} {r['ISR_strict']:7.1f} {r['ISR_lenient']:7.1f} "
              f"{r['ASR_strict']:7.1f} {r['ASR_lenient']:7.1f}")
    print("-" * 78)
    print(f"{'MEAN(repro)':16s} {statistics.mean(ss):7.1f} {statistics.mean(sl):7.1f} "
          f"{statistics.mean(asr_s):7.1f} {statistics.mean(asr_l):7.1f}")
    print(f"{'PAPER strict':16s} {PAPER_EHR_MIMIC['ISR']:7.1f} {'-':>7s} {PAPER_EHR_MIMIC['ASR']:7.1f} {'-':>7s}")
    # comment-contamination evidence
    print("\n--- EHR: victim-id-in-comment-only (inflates strict failures) ---")
    for r in rows:
        ie = r.get("inject_events", [])
        cc = sum(1 for e in ie if e.get("victim_in_comment_only"))
        strict_miss_target_present = sum(1 for e in ie if e.get("target_present") and not e.get("strict"))
        print(f"[{r['victim']}->{r['target']}] inject runs={len(ie)} "
              f"victim_in_comment_only={cc} (target_present but strict-miss={strict_miss_target_present})")
        for e in ie:
            if e.get("target_present") and not e.get("strict"):
                print(f"    e.g. [{e['kind']}] code: {e['code'][:120]}")
                break

if __name__ == "__main__":
    qa_report()
    ehr_report()
