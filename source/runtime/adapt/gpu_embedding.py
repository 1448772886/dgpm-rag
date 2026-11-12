"""GPU-backed embedding with an in-memory cache — drop-in for HgEmbedding.

Same interface as ReMindRag.embeddings.HgEmbedding (EmbeddingBase), so it can
replace it in any adapted script without touching the official library:

    from adapt.gpu_embedding import GpuEmbedding
    embedding = GpuEmbedding("nomic-ai/nomic-embed-text-v2-moe", model_cache)

Speedups vs the official HgEmbedding:
1. model runs on CUDA (the machine has an RTX 5070; the official class leaves
   SentenceTransformer on CPU);
2. per-sentence results are memoized — KG construction queries the same entity
   texts repeatedly (synonym merging, re-encodes across chunks);
3. sentence_list_embedding uses batched encode.
"""

import os
from pathlib import Path

# offline before transformers/hf_hub may be imported by an enclosing process;
# harmless no-op if the entry script already set it earlier.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from ReMindRag.embeddings.base import EmbeddingBase


def resolve_local_model(model_name: str, cache_dir: str) -> str:
    """Resolve a HF repo id to a fully-local, offline-loadable model dir.

    Preference order:
    1. cache_dir/nomic-local — a self-contained copy of the nomic snapshot with
       its remote code (.py) bundled and config.json auto_map rewritten to local
       class refs; loads with ZERO hub interaction (huggingface.co is blocked
       on this machine and each HEAD otherwise burns 10s-timeout retries).
    2. the HF-cache snapshot of model_name, if already downloaded (offline OK
       for weights, but trust_remote_code may still try the hub for base-repo
       code — hence preference for 1).
    3. the repo id as-is (requires a reachable hub/mirror).
    """
    local_bundle = os.path.join(cache_dir, "nomic-local")
    local_bundle_path = Path(local_bundle)
    has_local_weights = (
        (local_bundle_path / "model.safetensors").is_file() or
        (local_bundle_path / "pytorch_model.bin").is_file() or
        any(local_bundle_path.glob("model-*.safetensors")) or
        any(local_bundle_path.glob("pytorch_model-*.bin"))
    )
    if os.path.isdir(local_bundle) and has_local_weights:
        return local_bundle
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True)
        if os.path.isdir(snap):
            return snap
    except Exception:
        pass
    return model_name  # fall back to repo id (requires reachable hub/mirror)


class GpuEmbedding(EmbeddingBase):
    def __init__(self, model_name: str, cache_dir: str, device: str = "cuda", batch_size: int = 64):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        local_path = resolve_local_model(model_name, cache_dir)
        self.model = SentenceTransformer(
            model_name_or_path=local_path,
            cache_folder=cache_dir,
            trust_remote_code=True,
            device=device,
        )
        self._cache: dict = {}
        self._lock = threading.Lock()

    def sentence_embedding(self, sentence: str) -> np.array:
        with self._lock:
            hit = self._cache.get(sentence)
        if hit is not None:
            return hit
        emb = self.model.encode([sentence])[0]
        with self._lock:
            self._cache[sentence] = emb
        return emb

    def sentence_list_embedding(self, sentences: list) -> np.array:
        return self.model.encode(sentences, batch_size=self.batch_size)

    def get_hidden_state_size(self) -> int:
        return len(self.sentence_embedding("example"))
