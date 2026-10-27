from __future__ import annotations

import threading

import numpy as np
import pytest

from bookmind.core.exceptions import ConfigurationError, EmbeddingError
from bookmind.ml.embedder import (
    CachedEncoder,
    Encoder,
    HashingEncoder,
    build_encoder,
    l2_normalize,
)
from tests.conftest import make_settings


def test_l2_normalize_rows_have_unit_norm() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], rtol=1e-6)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_l2_normalize_handles_zero_rows() -> None:
    out = l2_normalize(np.zeros((2, 3)))
    assert np.isfinite(out).all()
    assert not out.any()


def test_l2_normalize_rejects_1d() -> None:
    with pytest.raises(EmbeddingError):
        l2_normalize(np.ones(3))


class TestHashingEncoder:
    def test_shape_dtype_and_norm(self) -> None:
        enc = HashingEncoder(dim=64)
        vecs = enc.encode(["a wizard and a dragon", "space opera"])
        assert vecs.shape == (2, 64)
        assert vecs.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)

    def test_deterministic(self) -> None:
        a = HashingEncoder(64).encode(["dragon quest"])
        b = HashingEncoder(64).encode(["dragon quest"])
        np.testing.assert_array_equal(a, b)

    def test_similar_texts_score_higher(self) -> None:
        enc = HashingEncoder(256)
        q, near, far = enc.encode(["wizard dragon magic", "a dragon and a wizard", "stock market"])
        assert float(q @ near) > float(q @ far)

    def test_case_insensitive(self) -> None:
        enc = HashingEncoder(64)
        np.testing.assert_array_equal(enc.encode(["Dragon"]), enc.encode(["dragon"]))

    def test_empty_string_is_zero_vector(self) -> None:
        assert not HashingEncoder(32).encode([""]).any()

    def test_rejects_small_dim(self) -> None:
        with pytest.raises(ConfigurationError):
            HashingEncoder(dim=4)

    def test_rejects_non_list(self) -> None:
        with pytest.raises(EmbeddingError):
            HashingEncoder(32).encode("not a list")  # type: ignore[arg-type]

    def test_satisfies_protocol(self) -> None:
        assert isinstance(HashingEncoder(32), Encoder)


class TestCachedEncoder:
    def test_hits_and_misses(self) -> None:
        enc = CachedEncoder(HashingEncoder(32), maxsize=10)
        first = enc.encode_one("dragon")
        second = enc.encode_one("dragon")
        assert first is second
        assert (enc.hits, enc.misses) == (1, 1)

    def test_cached_vectors_are_read_only(self) -> None:
        vec = CachedEncoder(HashingEncoder(32)).encode_one("x")
        with pytest.raises(ValueError):
            vec[0, 0] = 1.0

    def test_lru_eviction(self) -> None:
        enc = CachedEncoder(HashingEncoder(32), maxsize=2)
        for text in ("a", "b", "c"):
            enc.encode_one(text)
        enc.encode_one("a")  # evicted → miss
        assert enc.misses == 4

    def test_zero_size_disables_cache(self) -> None:
        enc = CachedEncoder(HashingEncoder(32), maxsize=0)
        assert enc.encode_one("a") is not enc.encode_one("a")

    def test_batch_encode_passthrough(self) -> None:
        enc = CachedEncoder(HashingEncoder(32))
        assert enc.encode(["a", "b"]).shape == (2, 32)

    def test_thread_safety(self) -> None:
        enc = CachedEncoder(HashingEncoder(32), maxsize=50)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                for j in range(200):
                    enc.encode_one(f"t{(i * j) % 80}")
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert enc.hits + enc.misses == 8 * 200


def test_build_encoder_hashing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    enc = build_encoder(make_settings(tmp_path, embedding_dim=128))
    assert isinstance(enc, HashingEncoder)
    assert enc.dim == 128
