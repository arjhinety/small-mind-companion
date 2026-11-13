"""Embed the PMB probe questions as Study 001's run scripts did, once, for a model-free audit.

``run_system_d.py`` (and the other memory runs) called
``SentenceTransformer("intfloat/multilingual-e5-small").encode([probe.question])[0]``: one
question per call, no ``query: `` prefix, no normalisation. This script repeats that at the
revision the Hub served when the runs were made (the model's last change, 2026-04-02, predates them)
and with the library versions in ``uv.lock`` at the time of the runs, and saves the float32 vectors,
so ``scripts/retrieval_audit.py`` can replay retrieval without a model.

One known difference: these vectors are computed on CPU. Study 001 likely encoded on CUDA
(``run_system_d.py:34`` passes no device, and the runs were on a GPU machine), so the vectors may
differ from the runs' in the last bits. The manifest records this.

Needs the embedding stack, which is not a project dependency here::

    uv run --with sentence-transformers==5.7.0 --with torch==2.13.0 --with transformers==5.15.0 \\
        python scripts/embed_probe_questions.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "audit" / "retrieval"
PROBES = ROOT / "data" / "benchmarks" / "pmb_v0_full" / "probes.jsonl"
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
VERSIONS = {"sentence-transformers": "5.7.0", "torch": "2.13.0", "transformers": "5.15.0"}


def main() -> int:
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    found = {
        "sentence-transformers": sentence_transformers.__version__,
        "torch": torch.__version__.split("+")[0],
        "transformers": transformers.__version__,
    }
    if found != VERSIONS:
        raise SystemExit(f"library versions {found} differ from Study 001's {VERSIONS}")

    probes = [json.loads(line) for line in PROBES.read_text(encoding="utf-8").splitlines() if line]
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    vectors = np.stack(
        [model.encode([p["question"]])[0] for p in probes]  # one per call, as the runs did
    ).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npy = OUT_DIR / "question_embeddings.npy"
    np.save(npy, vectors)
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "library_versions": VERSIONS,
        "device": "cpu",
        "device_differs_from_runs": "likely (cuda vs cpu)",
        "encoding": "encode([question])[0] per probe; no 'query: ' prefix; not normalised",
        "probe_ids": [p["probe_id"] for p in probes],
        "shape": list(vectors.shape),
        "sha256": hashlib.sha256(npy.read_bytes()).hexdigest(),
    }
    (OUT_DIR / "question_embeddings.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {npy.relative_to(ROOT).as_posix()} {vectors.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
