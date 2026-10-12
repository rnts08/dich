"""Golden-file translation tests — the quality regression gate.

Local-only: they load the real model (8 GB), so CI deselects them with
`-m "not model"`. Sampling is greedy (temp=0.0), so output is deterministic on
one machine and library versions. When mlx-lm or the model revision changes
and output legitimately drifts, regenerate via dev/regen_goldens.py and review
the diff — that review is the gate.

Each golden file starts with a metadata comment naming the app, model and
mlx-lm versions it was generated with; the comparison ignores comment lines.
"""
import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
CASES = json.loads((GOLDEN_DIR / "cases.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.model


@pytest.fixture(autouse=True)
def _weights_must_be_cached(backend):
    # A missing model must skip, not start an 8 GB download mid-test-run.
    # The cache location comes from the env (set in conftest), not from
    # backend.MODELS_DIR, which follows the sandboxed DICH_HOME.
    import os
    cache = Path(os.environ["HF_HUB_CACHE"])
    cached = cache / ("models--" + backend.MODEL_NAME.replace("/", "--"))
    if not cached.exists():
        pytest.skip(f"model weights not cached: {backend.MODEL_NAME}")


def _expected(case: str) -> str:
    text = (GOLDEN_DIR / f"{case}.golden.md").read_text(encoding="utf-8")
    body = [line for line in text.splitlines() if not line.startswith("<!--")]
    return "\n".join(body).strip()


@pytest.mark.parametrize("case", sorted(CASES))
def test_translation_matches_golden(backend, case):
    golden = GOLDEN_DIR / f"{case}.golden.md"
    assert golden.exists(), f"missing golden for {case!r}; run dev/regen_goldens.py"
    result = backend.translate_text(CASES[case], use_cache=False)
    assert result.text.strip() == _expected(case)
