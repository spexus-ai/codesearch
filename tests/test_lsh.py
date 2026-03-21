from __future__ import annotations

import numpy as np
import pytest

from codesearch.lsh import SimHashLSH


def _unpack_bits(blob: bytes, width: int) -> list[int]:
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="big")
    return bits[:width].astype(int).tolist()


def test_simhash_lsh_generates_deterministic_hyperplane_matrix() -> None:
    first = SimHashLSH(num_bands=2, band_width=3, dim=4, seed=7)
    second = SimHashLSH(num_bands=2, band_width=3, dim=4, seed=7)
    third = SimHashLSH(num_bands=2, band_width=3, dim=4, seed=8)

    assert first._matrix.shape == (6, 4)
    assert first._matrix.dtype == np.float32
    assert np.array_equal(first._matrix, second._matrix)
    assert not np.array_equal(first._matrix, third._matrix)


def test_simhash_lsh_computes_band_hashes() -> None:
    lsh = SimHashLSH(num_bands=2, band_width=3, dim=2, seed=1)
    lsh._matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )

    band_hashes = lsh.compute_band_hashes(
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    )

    assert len(band_hashes) == 2
    assert band_hashes[0][0][0] == 0
    assert band_hashes[0][1][0] == 1
    assert _unpack_bits(band_hashes[0][0][1], 3) == [1, 1, 1]
    assert _unpack_bits(band_hashes[0][1][1], 3) == [1, 1, 0]
    assert _unpack_bits(band_hashes[1][0][1], 3) == [1, 0, 1]
    assert _unpack_bits(band_hashes[1][1][1], 3) == [1, 0, 1]


def test_simhash_lsh_computes_expected_band_count_per_embedding() -> None:
    lsh = SimHashLSH(num_bands=3, band_width=2, dim=2, seed=3)

    band_hashes = lsh.compute_band_hashes(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )

    assert len(band_hashes) == 2
    assert all(len(item) == 3 for item in band_hashes)
    assert [band_idx for band_idx, _ in band_hashes[0]] == [0, 1, 2]


def test_simhash_lsh_similar_vectors_share_a_band() -> None:
    lsh = SimHashLSH(num_bands=2, band_width=2, dim=2, seed=11)
    lsh._matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )

    first, second = lsh.compute_band_hashes(
        np.array([[1.0, 0.0], [0.98, 0.02]], dtype=np.float32)
    )

    shared_bands = {
        band_idx
        for (band_idx, band_hash), (_, other_hash) in zip(first, second, strict=True)
        if band_hash == other_hash
    }
    assert shared_bands


def test_simhash_lsh_dissimilar_vectors_can_have_no_shared_bands() -> None:
    lsh = SimHashLSH(num_bands=2, band_width=2, dim=2, seed=13)
    lsh._matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    first, second = lsh.compute_band_hashes(
        np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    )

    assert all(
        band_hash != other_hash
        for (_, band_hash), (_, other_hash) in zip(first, second, strict=True)
    )


def test_simhash_lsh_serializes_and_deserializes_matrix() -> None:
    lsh = SimHashLSH(num_bands=2, band_width=3, dim=4, seed=9)
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)

    blob = lsh.serialize_matrix(matrix)
    restored = lsh.deserialize_matrix(blob)

    assert restored.dtype == np.float32
    assert np.array_equal(restored, matrix)


def test_simhash_lsh_verifies_pairs_by_cosine_similarity() -> None:
    lsh = SimHashLSH()
    embeddings = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 1.0], dtype=np.float32),
        3: np.array([0.0, 1.0], dtype=np.float32),
        4: np.array([0.0, 0.0], dtype=np.float32),
    }

    verified = lsh.verify_pairs([(1, 2), (1, 3), (2, 3), (1, 4)], embeddings, threshold=0.7)

    assert verified == [
        (1, 2, pytest.approx(2 ** -0.5)),
        (2, 3, pytest.approx(2 ** -0.5)),
    ]
