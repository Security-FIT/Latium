"""Finite, two-part description lengths for the GRAM binary experiment.

The comparison uses a fixed FP32 lattice, signed Golomb--Rice codes and a
paid fixed-width fallback. No distributional fit or detection cutoff is used.
The original FP32 words are recovered by their fixed-width index inside each
lattice cell. This common suffix cancels in model selection, so the lattice
never silently discards information. These are prefix-code lengths, not p-values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch


CODEC_VERSION = "gram-rice-fp32-v1"


def universal_bits(value: int) -> int:
    """Elias gamma length of the positive integer ``value + 1``."""
    if value < 0:
        raise ValueError("Universal integer must be nonnegative")
    return 2 * (int(value) + 1).bit_length() - 1


@dataclass(frozen=True)
class IntegerCode:
    bits: int
    mode: str
    parameter: int


def integer_code(values: np.ndarray) -> IntegerCode:
    """Shortest of paid signed fixed-width and signed Rice block codes.

    One mode bit, a gamma-coded width/Rice parameter, then the actual words.
    Zigzag maps 0,-1,+1,-2,+2,... to the nonnegative integers. The Rice
    parameter is selected by its *complete* code length, not by labels.
    """
    values = np.asarray(values)
    if values.dtype != np.int64 or values.ndim != 1 or values.size == 0:
        raise ValueError("Integer code requires a nonempty int64 vector")
    lower, upper = int(values.min()), int(values.max())
    if lower <= -(2**61) or upper >= 2**61:
        raise ValueError("Integer block exceeds the safe coding range")
    largest = max(abs(lower), abs(upper))
    count = int(values.size)
    width = largest.bit_length() + 1
    best = IntegerCode(1 + universal_bits(width - 1) + count * width, "fixed", width)
    zigzag = (values << 1) ^ (values >> 63)
    # A bounded histogram replaces repeated scans of millions of entries.
    # Early large quotients retain the original floating overflow-safe sum.
    histogram_shift = max(0, int(zigzag.max()).bit_length() - 18)
    histogram = None
    quotient = 0
    for parameter in range(62):
        overhead = 1 + universal_bits(parameter) + count * (parameter + 1)
        if overhead >= best.bits:
            break
        # Floating accumulation avoids int64 overflow for bad k. A winning
        # length is bounded by the fixed-width code and is exactly representable.
        if parameter < histogram_shift:
            quotient = float(np.sum(zigzag >> parameter, dtype=np.float64))
        else:
            if histogram is None:
                histogram = np.bincount(zigzag >> parameter)
                quotient = int(np.sum(histogram * np.arange(len(histogram), dtype=np.int64), dtype=np.int64))
            else:
                # sum floor(x/2) = (sum x - number of odd x) / 2.
                quotient = (int(quotient) - int(histogram[1::2].sum())) // 2
                even, odd = histogram[::2], histogram[1::2]
                histogram = even.copy()
                histogram[: len(odd)] += odd
        if quotient < best.bits - overhead:
            best = IntegerCode(overhead + int(quotient), "rice", parameter)
    return best


@dataclass(frozen=True)
class GramCodec:
    dimension: int

    def __post_init__(self) -> None:
        if not 2 <= self.dimension <= 2**20:
            raise ValueError("GRAM codec dimension must be between 2 and 2**20")

    @property
    def fractional_bits(self) -> int:
        # FP32 epsilon divided by d, rounded down to a dyadic quantum.
        return 23 + (self.dimension - 1).bit_length()

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def entries(self) -> int:
        return self.dimension * (self.dimension + 1) // 2

    @cached_property
    def upper_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return np.triu_indices(self.dimension)

    def vector(self, matrix: torch.Tensor) -> np.ndarray:
        array = matrix.detach().to(device="cpu", dtype=torch.float64).numpy()
        if array.shape != (self.dimension, self.dimension) or not np.isfinite(array).all():
            raise ValueError("Cannot code a non-finite or incorrectly shaped Gram")
        return array[self.upper_indices]

    def quantize(self, matrix: torch.Tensor) -> np.ndarray:
        return self.quantize_vector(self.vector(matrix))

    def quantize_vector(self, vector: np.ndarray) -> np.ndarray:
        scaled = np.rint(np.asarray(vector, dtype=np.float64) * self.scale)
        if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) >= 2**59:
            raise ValueError("Gram exceeds the safe lattice range")
        return scaled.astype(np.int64)

    def block_bits(self, current: np.ndarray, prediction: np.ndarray | None = None) -> int:
        """Paid choice between reset and affine-depth prediction."""
        direct = integer_code(current).bits
        if prediction is None:
            return direct
        return 1 + min(direct, integer_code(current - prediction).bits)

    def _cell_bounds(self, lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inclusive ordered IEEE-word bounds of round-to-even lattice cells."""
        lower = (lattice.astype(np.float64) - 0.5) / self.scale
        upper = (lattice.astype(np.float64) + 0.5) / self.scale
        lo, hi = lower.astype(np.float32), upper.astype(np.float32)
        odd = (lattice & 1) != 0
        lo = np.where((lo < lower) | ((lo == lower) & odd), np.nextafter(lo, np.float32(np.inf)), lo)
        hi = np.where((hi > upper) | ((hi == upper) & odd), np.nextafter(hi, np.float32(-np.inf)), hi)
        low_words, high_words = self._ordered_words(lo), self._ordered_words(hi)
        if np.any(low_words > high_words):
            raise ValueError("Lattice cell contains no FP32 value")
        return low_words, high_words

    @staticmethod
    def _ordered_words(values: np.ndarray) -> np.ndarray:
        words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        return np.where((words & 0x80000000) != 0, ~words, words ^ np.uint32(0x80000000)).astype(np.int64)

    def refinement_bits(self, lattice: np.ndarray) -> int:
        low, high = self._cell_bounds(lattice)
        return int(np.ceil(np.log2(high - low + 1)).sum())

    def encode_refinement(self, matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Encode original FP32 words as indices within their lattice cells."""
        values = self.vector(matrix).astype(np.float32)
        lattice = self.quantize_vector(values)
        low, high = self._cell_bounds(lattice)
        ordered = self._ordered_words(values)
        if np.any(ordered < low) or np.any(ordered > high):
            raise ValueError("FP32 value is outside its lattice cell")
        return lattice, ordered - low

    def restore_refinement(self, lattice: np.ndarray, indices: np.ndarray) -> np.ndarray:
        low, high = self._cell_bounds(lattice)
        ordered = low + indices
        if np.any(indices < 0) or np.any(ordered > high):
            raise ValueError("Invalid FP32 refinement index")
        words = ordered.astype(np.uint32)
        raw = np.where((words & 0x80000000) != 0, words ^ np.uint32(0x80000000), ~words)
        return np.ascontiguousarray(raw, dtype=np.uint32).view(np.float32)


@dataclass(frozen=True)
class GramCorrection:
    """Paid float64 factors plus exact integer reconstruction innovations.

    A = c B + u g.T + g u.T - gamma u u.T. Quantizing B loses small
    terms; ``errors`` restores the original lattice exactly. c is transmitted
    explicitly to preserve the implemented normalization in finite precision.
    """

    direction: np.ndarray
    moment: np.ndarray
    gamma: float
    scale: float
    baseline: np.ndarray
    errors: np.ndarray

    def prediction(self, codec: GramCodec) -> np.ndarray:
        rows, columns = codec.upper_indices
        u, g = self.direction, self.moment
        low_rank = u[rows] * g[columns] + g[rows] * u[columns] - self.gamma * u[rows] * u[columns]
        return codec.quantize_vector(self.scale * self.baseline / codec.scale + low_rank)

    def reconstruct(self, codec: GramCodec) -> np.ndarray:
        return self.prediction(codec) + self.errors

    @property
    def parameter_bits(self) -> int:
        # Two d-vectors and two scalars, all written as literal IEEE float64.
        return 64 * (2 * int(self.direction.size) + 2)

    @property
    def reconstruction_bits(self) -> int:
        return integer_code(self.errors).bits


def describe_correction(
    current: torch.Tensor,
    peeled: torch.Tensor,
    direction: torch.Tensor,
    mixer: torch.Tensor,
    codec: GramCodec,
) -> GramCorrection:
    gram = current.to(torch.float64)
    u = direction.to(device=gram.device, dtype=torch.float64)
    z = mixer.to(device=gram.device, dtype=torch.float64)
    moment = gram @ z
    gamma = float(z @ moment)
    scale = float(torch.trace(gram) - 2 * (u @ moment) + gamma * (u @ u))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Rank-one correction has nonpositive remaining energy")
    baseline = codec.quantize(peeled)
    parameters = dict(
        direction=u.detach().cpu().numpy().copy(),
        moment=moment.detach().cpu().numpy().copy(),
        gamma=gamma,
        scale=scale,
        baseline=baseline,
    )
    provisional = GramCorrection(**parameters, errors=np.zeros(codec.entries, dtype=np.int64))
    errors = codec.quantize(current) - provisional.prediction(codec)
    return GramCorrection(**parameters, errors=errors)
