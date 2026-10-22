"""Read-only workshop diagnostics on the current session model.

Both callbacks return ``(markdown, list[list], matplotlib.figure.Figure | None)``.
TRACE_COLUMNS: zero-based layer/window center, inclusive start, exclusive end,
clean/corrupt/restored first-target-token probabilities, paired mean indirect
effect, and IE / (clean - corrupt). Rows rank by descending IE (layer breaks ties).
Recovery is ``"N/A"`` when the total effect is numerically negligible.
GRAM_COLUMNS: layer, current score, baseline score, one-based current rank,
boolean known active edit. Excluded layers follow ranked rows with ``"N/A"``.
Errors return an explanatory markdown message, empty rows, and no figure.
"""

import logging as _diagnostic_logging
from contextlib import contextmanager as _diagnostic_contextmanager

import gradio as _diagnostic_gr

from .gram import GramProfileCache, detect_from_profiles, profile_weights
from .prompts import _user_span, prepare_chat_batch, tokenize_prompts
from .tracing import (
    TokenSpan,
    build_window,
    clean_cache,
    corrupt_hook,
    embedding_module_name,
    embedding_std,
    make_noise_samples,
    module_dict,
    probability,
    resolve_mlp_output_name,
    restore_hook,
    temporary_hooks,
)

_DIAGNOSTIC_PROGRESS = _diagnostic_gr.Progress()
_DIAGNOSTIC_LOGGER = _diagnostic_logging.getLogger(__name__)
TRACE_COLUMNS = [
    "layer", "window_start", "window_end", "clean", "corrupt", "restored", "IE", "recovery",
]
GRAM_COLUMNS = ["layer", "current_score", "baseline_score", "rank", "is_edit_layer"]


@_diagnostic_contextmanager
def _diagnostic_inference(model, seed):
    """Restore mixed train/eval flags and caller RNG even on failed forwards."""
    import torch

    modes = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        try:
            model.eval()
            yield
        finally:
            for module, training in modes:
                module.training = training


def _diagnostic_chat_inputs(handler, subject, prompt, target, device):
    """Use the edit API's exact chat convention, including the full user span."""
    import torch

    batch = prepare_chat_batch(handler, [prompt], subject=subject)
    encoded = tokenize_prompts(handler, [prompt], return_offsets_mapping=True)
    active = encoded["attention_mask"][0].bool()
    ids = encoded["input_ids"][0][active]
    mask = batch.prompts["attention_mask"][0].bool()
    if not torch.equal(ids.cpu(), batch.prompts["input_ids"][0][mask].cpu()):
        raise ValueError("Chat offset tokenization differs from model inputs")
    start, _ = _user_span(handler, prompt)
    start += prompt.index(subject)
    end = start + len(subject)
    shift = int(mask.long().argmax())
    positions = [
        index + shift
        for index, (a, b) in enumerate(encoded["offset_mapping"][0][active].tolist())
        if b > a and a < end and b > start
    ]
    if not positions or positions[-1] != int(batch.subject_indices[0]):
        raise ValueError("Cannot align the complete subject span with chat inputs")
    target_ids = handler.tokenizer.encode(target.strip(), add_special_tokens=False)
    if not target_ids:
        raise ValueError("Target tokenizes to nothing")
    inputs = {
        key: value.to(device) for key, value in batch.prompts.items()
        if key in ("input_ids", "attention_mask")
    }
    return inputs, TokenSpan(positions[0], positions[-1] + 1, positions, positions[-1]), target_ids


def _trace_heatmap(rows, sample_effects, *, configured_layer=None):
    """Latium-style RdBu_r map, with honest noise-draw rows for this single fact.

    Mirrors notebooks/causal_tracing.ipynb's nearest-neighbor heatmap and paired
    IE units. This workshop measures one fact, so rows are actual paired noise
    draws, NOT fictitious facts/tokens. The final row is their measured mean.
    """
    import numpy as np
    from matplotlib.figure import Figure

    ordered = sorted(rows, key=lambda row: row[0])
    layers = np.asarray([row[0] for row in ordered])
    effects = np.asarray(sample_effects, dtype=float)
    if effects.ndim != 2 or effects.shape[1] != len(layers) or not np.isfinite(effects).all():
        raise ValueError("Heatmap requires finite [noise draws, layer windows] measurements")
    means = np.asarray([row[6] for row in ordered])
    matrix = np.vstack([effects, means])
    limit = max(float(np.abs(matrix).max()), np.finfo(float).eps)
    figure = Figure(figsize=(11, 6.5), layout="constrained", facecolor="white")
    curve, heat = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]})
    curve.bar(layers, means, color="#9ecae1", alpha=.65)
    curve.plot(layers, means, color="#08519c", linewidth=2, label="Průměrný párový IE")
    curve.axhline(0, color="black", linewidth=.8)
    if configured_layer is not None:
        curve.axvline(configured_layer, color="#756bb1", linestyle=":", linewidth=2,
                      label=f"Nakonfigurovaná vrstva {configured_layer}")
    curve.set(xlabel="Střed okna MLP", ylabel="Průměrný IE", title="Kauzální obnovení podle vrstvy")
    curve.legend(loc="best", fontsize=9)
    # Symmetric bounds put zero at white, with red recovery and blue suppression.
    image = heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
    stride = max(1, len(layers) // 16)
    ticks = sorted(set(range(0, len(layers), stride)) | {len(layers) - 1})
    heat.set_xticks(ticks, labels=layers[ticks])
    heat.set_yticks(range(len(matrix)), labels=[f"Šum {i + 1}" for i in range(len(effects))] + ["Průměr"])
    heat.axhline(len(effects) - .5, color="black", linewidth=.8)
    heat.set(xlabel="Střed okna MLP", ylabel="Stejný prompt · párové vzorky šumu",
             title="Mapa obnovení · červená zvyšuje, modrá snižuje pravděpodobnost cíle")
    figure.colorbar(image, ax=heat, label="IE = P(obnovení) − P(poškození)")
    return figure


def _diagnostic_gram_curve(rows):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 4), layout="constrained")
    axis = figure.subplots()
    ordered = sorted(rows, key=lambda row: row[0])
    ordered = [row for row in ordered if row[3] != "N/A"]
    axis.plot([r[0] for r in ordered], [r[1] for r in ordered], "o-", label="Aktuální model")
    axis.plot([r[0] for r in ordered], [r[2] for r in ordered], "--", label="Původní model")
    for row in rows:
        if row[4]:
            axis.axvline(row[0], linestyle=":", color="purple", label="Známá aktivní editace")
    axis.set(xlabel="Vrstva", ylabel="GRAM skóre diagonal_relative", title="GRAM: pořadí vrstev")
    axis.legend()
    return figure


class DiagnosticActions:
    def __init__(self, session):
        self.session = session
        # Pin the base snapshot's identity, rather than using a later UI choice.
        with session.lock:
            self._diagnostic_base_layer = int(session.handler._layer) if session.handler is not None else 5

    def trace(self, subject, prompt, target, window_size=1, noise_samples=3,
              noise_multiplier=3.0, seed=42, progress=_DIAGNOSTIC_PROGRESS):
        """Trace the installed model; no layer selection or weight restoration."""
        import math

        import numpy as np

        try:
            subject, prompt, target = ((value or "").strip() for value in (subject, prompt, target))
            if not subject or not prompt or not target:
                raise ValueError("Vyplňte entitu, prompt a sledovanou odpověď")
            if prompt.count(subject) != 1:
                raise ValueError("Přesný text entity musí být v promptu právě jednou")
            integers = (window_size, noise_samples, seed)
            if any(not math.isfinite(float(v)) or int(v) != float(v) for v in integers):
                raise ValueError("Velikost okna, počet vzorků a seed musí být celá čísla")
            window_size, noise_samples, seed = map(int, integers)
            noise_multiplier = float(noise_multiplier)
            if window_size < 1 or window_size % 2 != 1 or noise_samples < 1:
                raise ValueError("Zvolte kladnou lichou velikost okna a alespoň jeden vzorek šumu")
            if not math.isfinite(noise_multiplier) or noise_multiplier < 0 or not 0 <= seed < 2**63:
                raise ValueError("Násobek šumu musí být konečný a nezáporný; seed v rozsahu [0, 2**63)")
            progress(0.02, desc="Čekám na aktuální model…")
            with self.session.lock:
                handler = self.session.handler
                active_edit = bool(self.session.active_edit)
                layers = int(handler.num_of_layers)
                if window_size > layers:
                    raise ValueError("Velikost okna přesahuje počet vrstev modelu")
                modules = module_dict(handler.model)
                embedding = modules[embedding_module_name(handler.cfg)]
                names = {layer: resolve_mlp_output_name(handler, modules, layer)
                         for layer in range(layers)}
                windows = [build_window(layer, window_size, layers) for layer in range(layers)]
                inputs, span, target_ids = _diagnostic_chat_inputs(
                    handler, subject, prompt, target, embedding.weight.device,
                )
                with _diagnostic_inference(handler.model, seed):
                    progress(0.06, desc="Měřím čistou pravděpodobnost a ukládám výstupy MLP…")
                    clean = probability(handler.model(**inputs, use_cache=False), target_ids[0])
                    cache = clean_cache(handler, modules, names, inputs, span.last_position)
                    noise = make_noise_samples(
                        num_samples=noise_samples, subject_length=len(span.positions),
                        hidden_size=embedding.weight.shape[-1],
                        noise_std=embedding_std(handler, modules) * noise_multiplier,
                        device=embedding.weight.device, dtype=embedding.weight.dtype, seed=seed,
                    )
                    # One sample per forward bounds activation/logit memory on workshop GPUs.
                    corrupt = np.empty(noise_samples)
                    restored = np.empty((layers, noise_samples))
                    sequence_length = inputs["input_ids"].shape[1]
                    total = (layers + 1) * noise_samples
                    completed = 0
                    for window_index, window in enumerate([None, *windows]):
                        for sample in range(noise_samples):
                            hooks = [(embedding, corrupt_hook(span.positions, noise[sample:sample + 1]))]
                            if window is not None:
                                hooks.extend((modules[names[layer]], restore_hook(
                                    span.last_position, cache[layer], sequence_length,
                                )) for layer in window.layers)
                            with temporary_hooks(hooks):
                                measured = probability(
                                    handler.model(**inputs, use_cache=False), target_ids[0],
                                )
                            if window is None:
                                corrupt[sample] = measured
                            else:
                                restored[window_index - 1, sample] = measured
                            completed += 1
                            progress(0.15 + 0.8 * completed / total,
                                     desc="Měřím poškozený základ…" if window is None else
                                     f"Obnovuji okno MLP {window.start}:{window.end}…")
                token_text = handler.tokenizer.decode([target_ids[0]])
            if not np.isfinite(clean) or not np.isfinite(corrupt).all() or not np.isfinite(restored).all():
                raise ValueError("Model vrátil nekonečné nebo nedefinované pravděpodobnosti")
            mean_corrupt = float(corrupt.mean())
            total_effect = clean - mean_corrupt
            # Probabilities are measured at model precision; avoid amplifying roundoff.
            import torch
            tolerance = max(1e-8, 8 * torch.finfo(embedding.weight.dtype).eps *
                            max(abs(clean), abs(mean_corrupt), 1e-6))
            undefined = abs(total_effect) <= tolerance
            rows = []
            for window, values in zip(windows, restored):
                effect = float((values - corrupt).mean())
                rows.append([window.center, window.start, window.end, clean, mean_corrupt,
                             float(values.mean()), effect, "N/A" if undefined else effect / total_effect])
            rows.sort(key=lambda row: (-row[6], row[0]))
            summary = (
                "### Průzkumné kauzální trasování — aktuální model\n\n"
                f"Aktivní editace při měření: **{'ano' if active_edit else 'ne'}**. "
                f"První token cíle: `{token_text!r}` (ID {target_ids[0]}; cíl má {len(target_ids)} tokenů). "
                f"Tokeny entity: {span.positions}; celé výstupy MLP obnovujeme na tokenu {span.last_position}.\n\n"
                f"Čistá pravděpodobnost **{clean:.6g}**, poškozená **{mean_corrupt:.6g}**, celkový efekt **{total_effect:.6g}**. "
                f"Párové vzorky šumu: {noise_samples}; násobek {noise_multiplier:g}; seed {seed}. "
                f"Nejvyšší IE: **střed okna {rows[0][0]}**, [{rows[0][1]}, {rows[0][2]}). "
                "Jeden prompt poskytuje průzkumné pořadí, nikoli záruku správné editační vrstvy. "
                "Konec okna není zahrnut; okna na hranicích jsou zkrácena. Relativní obnovení = IE / celkový efekt."
            )
            if undefined:
                summary += "\n\n**Téměř nulový celkový efekt: relativní obnovení není definováno (N/A).**"
            elif total_effect < 0:
                summary += "\n\n**Poškození zvýšilo pravděpodobnost cíle: relativní obnovení zde nedokládá úspěšnou obnovu.**"
            progress(1.0, desc="Kauzální trasování je hotové.")
            summary += ("\n\nBarevná mapa používá styl Latium `RdBu_r`. Řádky ukazují skutečné párové "
                        "vzorky šumu pro tento prompt a jejich průměr, sloupce okna vrstev. "
                        "Bílá = nulový efekt; červená = zvýšení, modrá = snížení pravděpodobnosti cíle.")
            return summary, rows, _trace_heatmap(rows, (restored - corrupt[None, :]).T,
                                                 configured_layer=self._diagnostic_base_layer)
        except Exception as exc:
            _DIAGNOSTIC_LOGGER.exception("Workshop causal tracing failed")
            return f"### Kauzální trasování není dostupné\n\n{exc}", [], None

    def gram(self, progress=_DIAGNOSTIC_PROGRESS):
        """Rank real GRAM profiles from an atomic CPU snapshot of projections."""
        import torch

        try:
            progress(0.02, desc="Kopíruji aktuální váhy projekcí na CPU…")
            with self.session.lock:
                handler = self.session.handler
                if int(handler._layer) != self._diagnostic_base_layer:
                    raise ValueError("Editační vrstva se změnila; původní snapshot relace patří jiné vrstvě")
                active_layer = self._diagnostic_base_layer if self.session.active_edit else None
                modules = dict(handler.model.named_modules())
                current = {}
                for layer in range(int(handler.num_of_layers)):
                    weight = modules[handler._layer_name_template.format(layer)].weight
                    current[layer] = weight.detach().to("cpu", copy=True)
                    progress(0.05 + 0.25 * (layer + 1) / handler.num_of_layers,
                             desc=f"Projekce {layer} zkopírována na CPU…")
                baseline = dict(current)
                baseline[self._diagnostic_base_layer] = self.session.base_weight.detach().to("cpu", copy=True)
                if baseline[self._diagnostic_base_layer].shape != current[self._diagnostic_base_layer].shape:
                    raise ValueError("Rozměry původního snapshotu neodpovídají projekci")
            # Snapshot-local cache avoids retaining all host weights between calls.
            # It profiles the baseline once and recomputes only changed neighborhoods.
            cache = GramProfileCache(baseline, edit_layer=self._diagnostic_base_layer)
            # Main linalg caches inspect tensor version counters, so GRAM must
            # use no_grad rather than create inference tensors without counters.
            with torch.no_grad():
                progress(0.35, desc="Počítám základní profily GRAM na CPU…")
                base_profile = profile_weights(baseline, cache=cache)
                progress(0.65, desc="Počítám aktuální profily GRAM na CPU…")
                profile = profile_weights(current, cache=cache)
                result = detect_from_profiles(profile["profiles"], layers=profile["layers"],
                                              trim_fraction=profile["trim_fraction"])
                base_result = detect_from_profiles(base_profile["profiles"], layers=base_profile["layers"],
                                                   trim_fraction=base_profile["trim_fraction"])
            scores = result["localization"]["layer_scores"]
            base_scores = base_result["localization"]["layer_scores"]
            ranked = sorted(profile["eligible_layers"], key=lambda layer: (-scores[str(layer)], layer))
            rows = [[layer, scores[str(layer)], base_scores[str(layer)], rank, layer == active_layer]
                    for rank, layer in enumerate(ranked, 1)]
            rows.extend([layer, "N/A", "N/A", "N/A", layer == active_layer]
                        for layer in profile["excluded_layers"])
            summary = (
                "### GRAM — pořadí aktuálních projekcí\n\n"
                f"Nejvýše hodnocený kandidát: **vrstva {ranked[0]}**; "
                f"kandidát původního modelu: **vrstva {base_result['localization']['selected_layer']}**. "
                f"Známá aktivní editace při měření: **{active_layer if active_layer is not None else 'žádná'}**.\n\n"
                "Skóre počítá původní jádro `gram-localization` (`diagonal_relative`): porovnává normalizované "
                "Gramovy matice se sousedními vrstvami. Pořadí vznikne i pro needitovaný model; skóre "
                "nejsou kalibrovanou binární detekcí editace. Základ používá uloženou původní váhu "
                "editační vrstvy a nezměněné projekce ostatních vrstev. "
                f"Vynechané vrstvy podle pravidel jádra (hranice/ořez): {profile['excluded_layers']}; uvedeny jako N/A. "
                "Výsledky popisují snapshot pořízený po získání zámku společné relace."
            )
            progress(1.0, desc="Pořadí GRAM je hotové.")
            return summary, rows, _diagnostic_gram_curve(rows)
        except Exception as exc:
            _DIAGNOSTIC_LOGGER.exception("Workshop GRAM failed")
            return f"### GRAM není dostupný\n\n{exc}", [], None
