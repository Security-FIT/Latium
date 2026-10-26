"""Reference-aware plots of actual CPU projection snapshots, plus a small export."""

import csv
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _singular_values(matrix, *, seed, count=16):
    import torch

    count = min(count, min(matrix.shape))
    if not bool(matrix.any()):
        return torch.zeros(count)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        if min(matrix.shape) <= 64:
            return torch.linalg.svdvals(matrix)[:count]
        # Bound work to leading components; paired snapshots use identical draws.
        _, values, _ = torch.svd_lowrank(matrix, q=count, niter=3)
    return values.sort(descending=True).values


def build_gram_artifacts(rows, original, current, *, layer, score_figure, seed=42):
    """Extra plots compare the saved edit projection, not a blind detection claim."""
    import numpy as np
    import torch
    from matplotlib.figure import Figure

    before, after = (weight.detach().to(device="cpu", dtype=torch.float32) for weight in (original, current))
    delta = after - before
    norm = float(torch.linalg.vector_norm(delta))
    base_norm = float(torch.linalg.vector_norm(before))
    before_s = _singular_values(before, seed=seed)
    after_s = _singular_values(after, seed=seed) if norm else before_s.clone()
    delta_s = _singular_values(delta, seed=seed)
    indices = np.arange(1, len(before_s) + 1)
    fractions = delta_s.square().numpy() / (norm ** 2) if norm else np.zeros(len(delta_s))
    cumulative = np.minimum(np.cumsum(fractions), 1.)
    comparable = sorted((row for row in rows if row[3] != "N/A"), key=lambda row: row[0])

    changes = Figure(figsize=(9, 3.5), layout="constrained")
    axis = changes.subplots()
    differences = [row[1] - row[2] for row in comparable]
    axis.bar([row[0] for row in comparable], differences,
             color=["#e4002b" if value > 0 else "#0088ad" for value in differences])
    axis.axhline(0, color="#64748b", linewidth=.8)
    axis.set(xlabel="Vrstva", ylabel="Skóre po − skóre před", title="Změna GRAM skóre proti původnímu modelu")

    spectrum = Figure(figsize=(9, 3.5), layout="constrained")
    axis = spectrum.subplots()
    axis.plot(indices, before_s.numpy(), "o--", label="Původní projekce", color="#0088ad")
    axis.plot(indices, after_s.numpy(), ".-", label="Aktuální projekce", color="#e4002b")
    axis.set_yscale("log", nonpositive="mask")
    axis.set(xlabel="Pořadí singulární hodnoty", ylabel="Singulární hodnota σ",
             title=f"Spektrum uložené projekce L{layer} · vedoucích {len(indices)} hodnot")
    axis.legend()

    update = Figure(figsize=(11, 3.8), layout="constrained")
    left, right = update.subplots(1, 2)
    left.bar(indices, delta_s.numpy(), color="#0088ad")
    left.set(xlabel="Pořadí singulární hodnoty", ylabel="σ změny vah",
             title=f"Spektrum ΔW v L{layer}" if norm else f"L{layer}: uložená projekce se nezměnila")
    right.plot(indices, cumulative, "o-", color="#e4002b")
    right.set(xlabel="Počet vedoucích směrů", ylabel="Podíl energie ΔW", ylim=(0, 1.05),
              title="Kumulativní energie změny" if norm else "Nulová změna · energie není definovaná")
    right.axhline(1, linestyle=":", color="#64748b", linewidth=.8)
    approximation = min(before.shape) > 64
    metrics = {"layer": int(layer), "shape": list(before.shape), "seed": int(seed),
               "approximate_top_spectrum": approximation, "top_k": len(indices),
               "original_frobenius_norm": base_norm, "update_frobenius_norm": norm,
               "relative_update_norm": norm / base_norm if base_norm else None,
               "first_direction_energy_fraction": float(fractions[0]) if norm else None,
               "captured_update_energy_fraction": float(cumulative[-1]) if norm else None,
               "note": "Reference comparison of the saved edit projection; not a blind localization score."}
    figures = {"changes": changes, "spectrum": spectrum, "update": update}
    # Keep the directory alive with the cached result; exports contain no weights.
    directory = tempfile.TemporaryDirectory(prefix="latium-gram-")
    root = Path(directory.name)
    try:
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (root / "scores.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["layer", "current_score", "original_score", "rank", "known_edit_layer"])
            writer.writerows(rows)
        with (root / "spectra.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["rank", "original_sigma", "current_sigma", "update_sigma", "cumulative_update_energy"])
            writer.writerows(zip(indices.tolist(), before_s.tolist(), after_s.tolist(), delta_s.tolist(),
                                 cumulative.tolist() if norm else [None] * len(indices)))
        for name, figure in {"scores": score_figure, **figures}.items():
            figure.savefig(root / f"{name}.png", dpi=160)
        archive = root / "gram-artifacts.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
            for path in sorted(root.iterdir()):
                if path != archive:
                    output.write(path, arcname=path.name)
    except BaseException:
        directory.cleanup()
        raise
    return {**figures, "archive": str(archive), "metrics": metrics, "_directory": directory}
