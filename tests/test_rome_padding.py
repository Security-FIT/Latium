import pytest
import torch

from src.rome.common import _last_non_padding_index


def test_last_non_padding_index_supports_left_and_right_padding() -> None:
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    assert torch.equal(_last_non_padding_index(attention_mask), torch.tensor([3, 1, 3]))


def test_last_non_padding_index_rejects_all_padding_rows() -> None:
    with pytest.raises(RuntimeError, match="all-padding"):
        _last_non_padding_index(torch.zeros(1, 3, dtype=torch.long))
