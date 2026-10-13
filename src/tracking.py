"""Optional experiment tracking for long-running Latium jobs."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.common.io import to_serializable


LOGGER = logging.getLogger(__name__)


class NullTracker:
    enabled = False

    def set_state(self, **state: Any) -> None:
        del state

    def log(self, metrics: Mapping[str, Any]) -> None:
        del metrics


NULL_TRACKER = NullTracker()
_ACTIVE_TRACKER: ContextVar[NullTracker | "WandbTracker"] = ContextVar(
    "latium_active_tracker",
    default=NULL_TRACKER,
)


def current_tracker() -> NullTracker | "WandbTracker":
    return _ACTIVE_TRACKER.get()


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _tracking_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        values = dict(config.to_dict())
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        values = {}
    values.pop("hf_token", None)
    return _plain_tracking_value(values)


def _plain_tracking_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_tracking_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_tracking_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_tracking_value(item) for item in value]
    return to_serializable(value)


class WandbTracker:
    """Thin W&B adapter with a periodic liveness heartbeat."""

    enabled = True

    def __init__(self, config: Any, *, job_type: str) -> None:
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "tracking_provider=wandb requires the 'wandb' package; install project requirements"
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": str(_value(config, "tracking_project", "latium")),
            "job_type": job_type,
            "config": _tracking_config(config),
            "mode": str(_value(config, "tracking_mode", "online")),
        }
        optional = {
            "entity": _value(config, "tracking_entity"),
            "name": _value(config, "tracking_run_name") or _value(config, "run_id"),
            "group": _value(config, "tracking_group"),
        }
        init_kwargs.update({key: value for key, value in optional.items() if value not in (None, "")})
        tags = tuple(_value(config, "tracking_tags", ()) or ())
        if tags:
            init_kwargs["tags"] = list(tags)

        self._run = wandb.init(**init_kwargs)
        if self._run is None:
            raise RuntimeError("wandb.init() did not return a run")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        self._last_activity_at = self._started_at
        self._heartbeat = 0
        self._log_warning_emitted = False
        self._state: dict[str, Any] = {
            "monitor/running": 1,
            "monitor/status": "running",
            "monitor/stage": "starting",
        }
        self._heartbeat_seconds = max(1, int(_value(config, "tracking_heartbeat_seconds", 60)))
        self._emit({}, activity=True)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="latium-wandb-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _safe_log(self, payload: Mapping[str, Any]) -> None:
        try:
            self._run.log(dict(payload))
        except Exception as exc:  # tracking must not terminate the benchmark
            if not self._log_warning_emitted:
                LOGGER.warning("W&B logging failed; the benchmark will continue: %s", exc)
                self._log_warning_emitted = True

    def _snapshot(self, now: float) -> dict[str, Any]:
        return {
            **self._state,
            "monitor/heartbeat": self._heartbeat,
            "monitor/uptime_seconds": now - self._started_at,
            "monitor/seconds_since_activity": now - self._last_activity_at,
        }

    def _emit(self, metrics: Mapping[str, Any], *, activity: bool) -> None:
        with self._lock:
            now = time.monotonic()
            if activity:
                self._last_activity_at = now
            payload = self._snapshot(now)
            payload.update({str(key): to_serializable(value) for key, value in metrics.items()})
            self._safe_log(payload)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_seconds):
            with self._lock:
                self._heartbeat += 1
            self._emit({}, activity=False)

    def set_state(self, **state: Any) -> None:
        normalized = {
            (key if "/" in key else f"context/{key}"): to_serializable(value)
            for key, value in state.items()
            if value is not None
        }
        with self._lock:
            self._state.update(normalized)
        self._emit({}, activity=True)

    def log(self, metrics: Mapping[str, Any]) -> None:
        self._emit(metrics, activity=True)

    def finish(self, *, status: str, error: str | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout=min(5, self._heartbeat_seconds))
        with self._lock:
            self._state.update(
                {
                    "monitor/running": 0,
                    "monitor/status": status,
                    "monitor/stage": "finished",
                }
            )
        final_metrics: dict[str, Any] = {}
        if error:
            final_metrics["monitor/error"] = error
        self._emit(final_metrics, activity=True)
        try:
            self._run.summary.update(self._snapshot(time.monotonic()))
            if error:
                self._run.summary["monitor/error"] = error
            self._run.finish(exit_code=0 if status == "complete" else 1)
        except Exception as exc:  # tracking teardown must not hide the original outcome
            LOGGER.warning("W&B finalization failed: %s", exc)


@contextmanager
def tracking_session(config: Any, *, job_type: str) -> Iterator[NullTracker | WandbTracker]:
    """Start one tracker for the outermost structural operation."""
    active = current_tracker()
    if active.enabled:
        yield active
        return

    provider = str(_value(config, "tracking_provider", "none")).strip().lower()
    if provider in {"", "none", "disabled"}:
        yield NULL_TRACKER
        return
    if provider != "wandb":
        raise ValueError(f"Unsupported tracking provider: {provider}")

    tracker = WandbTracker(config, job_type=job_type)
    token = _ACTIVE_TRACKER.set(tracker)
    try:
        yield tracker
    except BaseException as exc:
        tracker.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        tracker.finish(status="complete")
    finally:
        _ACTIVE_TRACKER.reset(token)


__all__ = ["current_tracker", "tracking_session"]
