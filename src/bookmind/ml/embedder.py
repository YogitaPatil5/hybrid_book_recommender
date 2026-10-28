"""
Text encoders.

``Encoder`` is a small protocol so the search engine never depends on a
concrete model. Two implementations ship:

* ``SentenceTransformerEncoder`` — production default (all-MiniLM-L6-v2,
  384-dim, runs locally, no API key).
* ``HashingEncoder`` — deterministic, dependency-free lexical encoder used
  for CI/tests and as an offline fallback when the model can't be
  downloaded. Lower quality, but the full pipeline and API work end-to-end.

All encoders return L2-normalised float32 matrices, so inner product equals
cosine similarity.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import re
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from bookmind.core.exceptions import ConfigurationError, EmbeddingError

if TYPE_CHECKING:
    from bookmind.core.config import Settings

log = logging.getLogger(__name__)

FloatMatrix = npt.NDArray[np.float32]
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def l2_normalize(mat: npt.NDArray[Any]) -> FloatMatrix:
    """Row-wise L2 normalisation that is safe for all-zero rows."""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2:
        raise EmbeddingError(f"Expected a 2-D matrix, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


@runtime_checkable
class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], batch_size: int = 64) -> FloatMatrix: ...


class HashingEncoder:
    """Signed feature-hashing of word unigrams + bigrams (deterministic)."""

    def __init__(self, dim: int = 384):
        if dim < 8:
            raise ConfigurationError("HashingEncoder dim must be >= 8")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def encode(self, texts: list[str], batch_size: int = 64) -> FloatMatrix:  # noqa: ARG002
        if not isinstance(texts, list):
            raise EmbeddingError("texts must be a list of strings")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(str(text).lower())
            features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
            for feat in features:
                idx, sign = self._bucket(feat)
                out[row, idx] += sign
        return l2_normalize(out)


class SentenceTransformerEncoder:
    """Wrapper around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConfigurationError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt` "
                "or set BOOKMIND_EMBEDDING_BACKEND=hashing."
            ) from exc
        log.info("Loading embedding model '%s' …", model_name)
        try:
            # Offline-first: use the locally cached model without any network
            # calls (no need to set HF_HUB_OFFLINE by hand).
            # The device is explicit: on hosts that *report* a GPU (e.g. Hugging Face
            # ZeroGPU) "auto" would pick CUDA, which is only usable inside special calls.
            self._model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            log.info("Model not cached yet — downloading '%s' (one-time, ~80 MB)", model_name)
            try:
                self._model = SentenceTransformer(model_name, device=device)
            except Exception as exc:
                raise ConfigurationError(
                    f"Could not load embedding model '{model_name}': {exc}. "
                    "Check your internet connection for the first download."
                ) from exc
        # Newer sentence-transformers renamed this method; support both.
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        dim = get_dim()
        if not dim:  # pragma: no cover
            raise ConfigurationError(f"Model '{model_name}' did not report an embedding size")
        self.dim = int(dim)
        self.name = model_name

    def encode(self, texts: list[str], batch_size: int = 64) -> FloatMatrix:  # pragma: no cover
        try:
            vecs = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > batch_size,
            )
        except Exception as exc:
            raise EmbeddingError("Embedding model failed to encode input") from exc
        return l2_normalize(vecs)


class CachedEncoder:
    """Thread-safe LRU cache in front of any encoder for single-query calls."""

    def __init__(self, inner: Encoder, maxsize: int = 2048):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._maxsize = maxsize
        self._cache: OrderedDict[str, FloatMatrix] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def encode(self, texts: list[str], batch_size: int = 64) -> FloatMatrix:
        return self.inner.encode(texts, batch_size=batch_size)

    def encode_one(self, text: str) -> FloatMatrix:
        if self._maxsize == 0:
            return self.inner.encode([text])
        with self._lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                self.hits += 1
                return cached
        vec = self.inner.encode([text])
        vec.setflags(write=False)
        with self._lock:
            self.misses += 1
            self._cache[text] = vec
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return vec


def build_encoder(settings: Settings, model_name: str | None = None) -> Encoder:
    """Factory: create the encoder configured in settings."""
    if settings.embedding_backend == "hashing":
        return HashingEncoder(dim=settings.embedding_dim)
    return SentenceTransformerEncoder(  # pragma: no cover
        model_name or settings.embedding_model, device=settings.embedding_device
    )
