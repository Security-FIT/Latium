"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import numpy as np

from src.common.arrays import curvature, local_zscore


def test_local_zscore_supports_2d_graph_transform_shape() -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0, 10.0, 4.0],
            [4.0, np.nan, 6.0, 8.0],
        ]
    )

    transformed = local_zscore(
        matrix,
        3,
        axis=1,
        fill_value=np.nan,
        absolute=True,
        nan_safe=True,
    )

    assert transformed.shape == matrix.shape
    assert np.isnan(transformed[:, 0]).all()
    assert np.isnan(transformed[:, -1]).all()
    np.testing.assert_allclose(transformed[0, 1:3], [7.0 / 9.0, 7.0])


def test_curvature_supports_axis_and_nan_padding() -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0, 4.0, 7.0],
            [0.0, 3.0, 2.0, 1.0],
        ]
    )

    transformed = curvature(matrix, axis=1, pad_value=np.nan)

    assert transformed.shape == matrix.shape
    np.testing.assert_allclose(transformed[:, 1:-1], [[1.0, 1.0], [4.0, 0.0]])
    assert np.isnan(transformed[:, 0]).all()
    assert np.isnan(transformed[:, -1]).all()
