"""Single source of truth for the active V16 Hotpot runtime configuration."""
from __future__ import annotations

import os
from pathlib import Path


MODEL = "qwen3.6-plus"
JUDGE_MODEL = "qwen3.6-plus"
BUILD_MODEL = "qwen3.6-plus"
WORKERS = 10
MAX_JUMPS = 10
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRIES = 8


def configure_v16(root: Path, *, api_key_file: Path | None = None) -> None:
    """Enable the complete V16 fact-guided traversal and evidence memory path."""
    key_file = api_key_file or root / "adapt" / ".secrets" / "qwen36plus_bailian.json"
    os.environ.update({
        "REMIND_TRAVERSAL": "fact_v7",
        "REMIND_B1": "2",  # enables the B2 provenance ledger and C2 trace
        "REMIND_API_KEY_FILE": str(key_file),
        "REMIND_BACKBONE": MODEL,
        "REMIND_JUDGE_MODEL": JUDGE_MODEL,
        "REMIND_BUILD_BACKBONE": BUILD_MODEL,
        "REMIND_V14A": "1",
        "REMIND_V16": "1",
        "REMIND_V16_MEMORY": "1",
        "REMIND_V16_DEFER_MEMORY": "1",  # one post-answer memory commit
        "REMIND_C2_DEFER_UPDATE": "0",  # V16 owns the deferred plan
        "REMIND_API_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "REMIND_API_RETRIES": str(REQUEST_RETRIES),
        "REMIND_E1_RELEVANCE_DISTANCE": "1",
        # V16 uses the F1 transition-pruning controller.  A5/B4 backtracking
        # is disabled because it can revisit old nodes without consuming a
        # jump and can therefore keep issuing selector requests.
        "REMIND_A5_CONTINUE": "0",
        "REMIND_B4": "0",
        "REMIND_A5_GLOBAL_FALLBACK": "0",
        "REMIND_F1": "1",
        "REMIND_V12A_DISABLE_GAP": "0",  # A5 local/global gap entry must remain available
        "REMIND_EMBEDDING_DEVICE": "cpu",
        "REMIND_EVIDENCE_REPRESENTATION": "extractive_gap_v1",
        "REMIND_DISABLE_THINKING": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    validate_v16_environment()
    validate_v16_runtime(root)


def validate_v16_environment() -> None:
    required = {
        "REMIND_TRAVERSAL": "fact_v7",
        "REMIND_B1": "2",
        "REMIND_A5_CONTINUE": "0",
        "REMIND_B4": "0",
        "REMIND_A5_GLOBAL_FALLBACK": "0",
        "REMIND_F1": "1",
        "REMIND_V12A_DISABLE_GAP": "0",
        "REMIND_V16": "1",
        "REMIND_V16_MEMORY": "1",
        "REMIND_V16_DEFER_MEMORY": "1",
        "REMIND_C2_DEFER_UPDATE": "0",
        "REMIND_API_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "REMIND_API_RETRIES": str(REQUEST_RETRIES),
    }
    wrong = {key: os.environ.get(key) for key, value in required.items()
             if os.environ.get(key) != value}
    if wrong:
        raise RuntimeError(f"Incomplete V16 configuration; incorrect settings: {wrong}")
    if os.environ.get("REMIND_BACKBONE") != MODEL:
        raise RuntimeError("V16 traversal model drifted")
    if os.environ.get("REMIND_JUDGE_MODEL") != JUDGE_MODEL:
        raise RuntimeError("V16 judge model drifted")
    if os.environ.get("REMIND_BUILD_BACKBONE") != BUILD_MODEL:
        raise RuntimeError("V16 graph-build model drifted")


def validate_v16_runtime(root: Path) -> None:
    """Fail closed if the traversal/fallback implementation was removed."""
    pathfinder = root / "ReMindRag" / "generator" / "pathfinder.py"
    fact_v7 = root / "ReMindRag" / "generator" / "fact_v7.py"
    if not pathfinder.is_file() or not fact_v7.is_file():
        raise RuntimeError("V16 runtime files are missing; refusing to start")
    pf_source = pathfinder.read_text(encoding="utf-8")
    fact_source = fact_v7.read_text(encoding="utf-8")
    required_pf = ("def get_query_ans(", "REMIND_B1",
                   "punish_edge_weight(", "_get_query_ans_original(",
                   "def get_query_ans_fact(", "REMIND_V16_DEFER_MEMORY",
                   "v16_memory_plan")
    required_fact = ("def enter(", "def enter_global(",
                     "REMIND_V12A_DISABLE_GAP", "A5 global fallback entry")
    if any(marker not in pf_source for marker in required_pf):
        raise RuntimeError("PathFinder is missing V16 fact traversal, B1/C2 trace, or merged-path punishment")
    if any(marker not in fact_source for marker in required_fact):
        raise RuntimeError("fact_v7 local/global fallback implementation is incomplete")


