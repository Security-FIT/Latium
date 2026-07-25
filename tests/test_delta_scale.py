from types import SimpleNamespace

import torch

from src.rome import common


class _DeviceManager:
    @staticmethod
    def safe_to_device(value, device=None):
        return value.to(device) if device is not None else value


class _Handler:
    def __init__(self, *, delta_scale: float, residual_multiplier: float = 1.0):
        self.cfg = SimpleNamespace(model=SimpleNamespace(delta_scale=delta_scale))
        self.model = SimpleNamespace(config=SimpleNamespace(residual_multiplier=residual_multiplier))
        self._layer_name_template = "layers.{}"
        self._layer = 0
        self.device = torch.device("cpu")
        self.device_manager = _DeviceManager()
        self.dtype = torch.float32
        self.is_multi_gpu = False
        self.module = torch.nn.Linear(2, 2, bias=False)
        self.module.weight = torch.nn.Parameter(torch.zeros(2, 2))

    def _get_module(self, _name):
        return self.module


def test_explicit_delta_scale_wins_over_residual_multiplier() -> None:
    handler = _Handler(delta_scale=4.0, residual_multiplier=0.5)
    assert common._resolve_delta_scale(handler) == 4.0


def test_insert_kv_applies_explicit_delta_scale(monkeypatch) -> None:
    handler = _Handler(delta_scale=4.0)
    monkeypatch.setattr(common, "get_second_moment", lambda _handler: torch.eye(2))

    new_weight, old_weight, update = common.insert_kv(
        handler,
        k=torch.tensor([1.0, 0.0]),
        delta=torch.tensor([1.0, 0.0]),
    )

    expected = torch.tensor([[4.0, 0.0], [0.0, 0.0]])
    assert torch.equal(old_weight, torch.zeros(2, 2))
    assert torch.equal(update, expected)
    assert torch.equal(new_weight, expected)
