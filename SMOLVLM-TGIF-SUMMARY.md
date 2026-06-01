# SmolVLM-TGIF — latency aggregate

**Model:** `Patricijia/smolvlm-tgif-gif-descriptor`  **Device:** `webgpu-fp32`  
**Runs:** 51 × 10 GIFs = 510 per-GIF samples per set.

Two conditions for the same model on WebGPU fp32:
- **contended** — original runs (GPU shared with another process, e.g. vLLM).
- **gpu-free** — re-run with the full GPU available.

## Per-GIF latency — contended vs GPU-free (OCR on)

| Stage (per GIF) | contended median | gpu-free median | contended p95 | gpu-free p95 |
|---|---|---|---|---|
| inference (VLM) | 14698 | 3663 | 15015 | 5355 |
| frame extraction | 3055 | 3082 | 3125 | 3132 |
| OCR | 1495 | 1572 | 1950 | 1818 |
| total pipeline | 18212 | 7284 | 18530 | 8980 |

Cold model load — contended: [2120] ms; gpu-free: [25496] ms (one-time; offscreen doc persists across runs).

_(All units ms. Full per-set stats incl. OCR-off in `smolvlm-tgif-aggregate.json`.)_
