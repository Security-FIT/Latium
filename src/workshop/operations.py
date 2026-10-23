"""Heartbeat/progress for long model work without holding Gradio's render thread."""

import html
import queue
import re
import threading
import time


class OperationCancelled(RuntimeError):
    pass


class OperationProgress:
    def __init__(self, label):
        self.started = time.monotonic()
        self.label, self.fraction = label, 0.
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.units = None
        self.units_started = self.started
        self.first_unit = 0

    def __call__(self, fraction=None, *, desc=None):
        if self.cancelled.is_set():
            raise OperationCancelled("Operace byla přerušena.")
        with self.lock:
            if fraction is not None:
                self.fraction = min(1., max(self.fraction, float(fraction)))
            if desc:
                self.label = desc
                match = re.search(r"(.*?)(\d+)/(\d+)", desc)
                key = match.group(1) if match else None
                if key is None:
                    self.units = None
                else:
                    current, total = int(match.group(2)), int(match.group(3))
                    if self.units is None or self.units[0] != key or current < self.units[1]:
                        self.units_started, self.first_unit = time.monotonic(), current
                    self.units = key, current, total

    def html(self, *, finished=False):
        with self.lock:
            elapsed = time.monotonic() - self.started
            remaining = None
            if self.units is not None:
                _, current, total = self.units
                if current > self.first_unit:
                    remaining = (total - current) * (time.monotonic() - self.units_started) / (current - self.first_unit)
            estimate = f" · tato fáze ještě přibližně {remaining:.0f} s" if remaining is not None else ""
            label = html.escape(self.label)
            percent = 100 if finished else self.fraction * 100
        return (f'<div class="operation-progress"><strong>{label}</strong>'
                f'<span>Čas: {elapsed:.0f} s{estimate}</span>'
                f'<progress max="100" value="{percent:.2f}" aria-label="Průběh operace"></progress></div>')


def stream_operation(function, label):
    """Yield (progress HTML, result, done), including heartbeats during slow forwards.

    Closing the consumer cancels at the next progress checkpoint and joins the
    worker, so another queued operation cannot overlap an abandoned edit.
    """
    progress = OperationProgress(label)
    completed = queue.Queue(maxsize=1)

    def work():
        try:
            completed.put((True, function(progress)))
        except BaseException as exc:
            completed.put((False, exc))

    worker = threading.Thread(target=work, name="workshop-model-operation", daemon=True)
    worker.start()
    try:
        while True:
            try:
                success, value = completed.get(timeout=.4)
            except queue.Empty:
                yield progress.html(), None, False
                continue
            if not success:
                raise value
            yield progress.html(finished=True), value, True
            break
    finally:
        progress.cancelled.set()
        worker.join()
