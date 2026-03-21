from __future__ import annotations

from io import BytesIO

import numpy as np


class SimHashLSH:
    def __init__(
        self,
        num_bands: int = 12,
        band_width: int = 6,
        dim: int = 384,
        seed: int = 42,
    ) -> None:
        self.num_bands = num_bands
        self.band_width = band_width
        self.dim = dim
        self.seed = seed
        self._matrix = self.generate_hyperplane_matrix()

    def generate_hyperplane_matrix(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        shape = (self.num_bands * self.band_width, self.dim)
        return rng.standard_normal(shape, dtype=np.float32)

    def compute_band_hashes(self, embeddings: np.ndarray) -> list[list[tuple[int, bytes]]]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.size == 0:
            return []
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimensions {vectors.shape[1]} do not match SimHash dimensions {self.dim}"
            )

        projections = vectors @ self._matrix.T
        signs = projections >= 0
        hashes: list[list[tuple[int, bytes]]] = []

        for row in signs:
            bands: list[tuple[int, bytes]] = []
            for band_idx in range(self.num_bands):
                start = band_idx * self.band_width
                end = start + self.band_width
                packed = np.packbits(row[start:end], bitorder="big").tobytes()
                bands.append((band_idx, packed))
            hashes.append(bands)

        return hashes

    def serialize_matrix(self, matrix: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
        return buffer.getvalue()

    def deserialize_matrix(self, blob: bytes) -> np.ndarray:
        buffer = BytesIO(blob)
        return np.load(buffer, allow_pickle=False)

    def verify_pairs(
        self,
        pairs: list[tuple[int, int]],
        embeddings: dict[int, np.ndarray],
        threshold: float,
    ) -> list[tuple[int, int, float]]:
        verified: list[tuple[int, int, float]] = []

        for left_id, right_id in pairs:
            left = np.asarray(embeddings[left_id], dtype=np.float32)
            right = np.asarray(embeddings[right_id], dtype=np.float32)

            left_norm = float(np.linalg.norm(left))
            right_norm = float(np.linalg.norm(right))
            if left_norm == 0.0 or right_norm == 0.0:
                similarity = 0.0
            else:
                similarity = float(np.dot(left, right) / (left_norm * right_norm))

            if similarity >= threshold:
                verified.append((left_id, right_id, similarity))

        return verified
