"""Read-only live GPU telemetry, independent of the model operation queue."""

import html

import torch


def gpu_snapshot():
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    result = dict(available=True, name=torch.cuda.get_device_name(), total=total, used=total - free,
                  allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                  peak=torch.cuda.max_memory_allocated(), utilization=None, temperature=None)
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            device = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            result["utilization"] = pynvml.nvmlDeviceGetUtilizationRates(device).gpu
            result["temperature"] = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
        finally:
            pynvml.nvmlShutdown()
    except (ImportError, RuntimeError, OSError):
        pass
    except Exception:
        # NVML errors do not invalidate the CUDA allocator's memory measurements.
        pass
    return result


def gpu_usage_html():
    stats = gpu_snapshot()
    if not stats["available"]:
        return '<div class="gpu-usage">GPU není dostupná · pro načtení modelu zapněte CUDA runtime.</div>'
    gb = 1024 ** 3
    percent = 100 * stats["used"] / stats["total"]
    load = f'{stats["utilization"]} %' if stats["utilization"] is not None else "nedostupné"
    temp = f' · {stats["temperature"]} °C' if stats["temperature"] is not None else ""
    return (f'<div class="gpu-usage"><strong>{html.escape(stats["name"])}</strong> · vytížení {load}{temp}<br>'
            f'VRAM zařízení {stats["used"] / gb:.2f} / {stats["total"] / gb:.2f} GiB · '
            f'PyTorch {stats["allocated"] / gb:.2f} GiB · rezervováno {stats["reserved"] / gb:.2f} GiB · '
            f'špička {stats["peak"] / gb:.2f} GiB'
            f'<progress value="{percent:.2f}" max="100" aria-label="Obsazení GPU paměti"></progress></div>')
