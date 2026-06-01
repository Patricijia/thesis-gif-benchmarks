"""Aggregate SmolVLM-TGIF benchmark runs in this repo.

Three sets:
  - smolvlm-tgif-benchmark            : original runs, GPU contended (OCR on)
  - smolvlm-tgif-benchmark-no-ocr     : original runs, GPU contended (OCR off)
  - smolvlm-tgif-gpu-free-benchmark   : re-run with the GPU free (OCR on)

Emits smolvlm-tgif-aggregate.json and SMOLVLM-TGIF-SUMMARY.md (with a
contended-vs-free comparison)."""
import json, glob, math, re, statistics as st

def pct(a, p):
    a = sorted(a); k = (len(a) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return a[int(k)] if f == c else a[f] + (a[c] - a[f]) * (k - f)

def runno(f): return int(re.search(r'run(\d+)\.json$', f).group(1))

def collect(prefix):
    files = sorted(glob.glob(f"{prefix}-run*.json"), key=runno)
    per = {"totalInferenceMs": [], "frameExtractionMs": [], "ocrMs": [], "totalMs": [], "wallClockMs": []}
    model = device = None; loads = set(); gpr = None; oh = ot = 0
    for f in files:
        d = json.load(open(f))
        model = d["config"]["model"]; device = d["config"]["device"]
        if d["summary"].get("modelLoadMs"): loads.add(d["summary"]["modelLoadMs"])
        gpr = d["summary"]["totalGifs"]
        for g in d["perGif"]:
            for k in per:
                if g.get(k) is not None: per[k].append(g[k])
            ot += 1; oh += 1 if g.get("ocrDetected") else 0
    summ = {k: {"n": len(a), "mean": round(st.mean(a),1), "median": round(st.median(a),1),
                "p95": round(pct(a,95),1), "min": min(a), "max": max(a)} for k,a in per.items() if a}
    return {"files": len(files), "model": model, "device": device, "gifs_per_run": gpr,
            "model_load_ms_cold": sorted(loads), "gifs_with_text": f"{oh}/{ot}", "per_gif": summ}

SETS = {
    "contended_ocr_on":  "smolvlm-tgif-benchmark",
    "contended_ocr_off": "smolvlm-tgif-benchmark-no-ocr",
    "gpu_free_ocr_on":   "smolvlm-tgif-gpu-free-benchmark",
}
agg = {name: collect(prefix) for name, prefix in SETS.items()}
json.dump(agg, open("smolvlm-tgif-aggregate.json", "w"), indent=2)

def cell(s, key, stat="median"):
    m = s["per_gif"].get(key); return f"{m[stat]:.0f}" if m else "—"

c, g = agg["contended_ocr_on"], agg["gpu_free_ocr_on"]
md = [
"# SmolVLM-TGIF — latency aggregate", "",
f"**Model:** `{c['model']}`  **Device:** `{c['device']}`  ",
f"**Runs:** 51 × {c['gifs_per_run']} GIFs = 510 per-GIF samples per set.", "",
"Two conditions for the same model on WebGPU fp32:",
"- **contended** — original runs (GPU shared with another process, e.g. vLLM).",
"- **gpu-free** — re-run with the full GPU available.", "",
"## Per-GIF latency — contended vs GPU-free (OCR on)", "",
"| Stage (per GIF) | contended median | gpu-free median | contended p95 | gpu-free p95 |",
"|---|---|---|---|---|",
f"| inference (VLM) | {cell(c,'totalInferenceMs')} | {cell(g,'totalInferenceMs')} | {cell(c,'totalInferenceMs','p95')} | {cell(g,'totalInferenceMs','p95')} |",
f"| frame extraction | {cell(c,'frameExtractionMs')} | {cell(g,'frameExtractionMs')} | {cell(c,'frameExtractionMs','p95')} | {cell(g,'frameExtractionMs','p95')} |",
f"| OCR | {cell(c,'ocrMs')} | {cell(g,'ocrMs')} | {cell(c,'ocrMs','p95')} | {cell(g,'ocrMs','p95')} |",
f"| total pipeline | {cell(c,'totalMs')} | {cell(g,'totalMs')} | {cell(c,'totalMs','p95')} | {cell(g,'totalMs','p95')} |",
"",
f"Cold model load — contended: {c['model_load_ms_cold']} ms; gpu-free: {g['model_load_ms_cold']} ms "
"(one-time; offscreen doc persists across runs).", "",
"_(All units ms. Full per-set stats incl. OCR-off in `smolvlm-tgif-aggregate.json`.)_", "",
]
open("SMOLVLM-TGIF-SUMMARY.md", "w").write("\n".join(md))
print("\n".join(md))
