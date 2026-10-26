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
from contextlib import nullcontext

import gradio as _diagnostic_gr
from src.rome.batching import forward_selected_logits

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


def _trace_probabilities(handler, inputs, target):
    import torch
    positions = torch.full((inputs["input_ids"].shape[0], 1), inputs["input_ids"].shape[1] - 1, dtype=torch.long)
    logits, mapped = forward_selected_logits(handler, inputs, positions)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, mapped[:, 0]].float().softmax(-1)[:, target].detach().cpu().numpy()


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


def _trace_rainbow(rows, sample_effects, *, configured_layer=None):
    """Rainbow bars matching Latium's legacy_rainbow.png presentation.

    Heights here are the workshop's measured mean paired IE per MLP window.
    Legacy's max-over-subject-position aggregation is a different experiment;
    copying its old values or calling this one a legacy run would be incorrect.
    """
    import colorsys

    import matplotlib
    import numpy as np
    from matplotlib.figure import Figure

    ordered = sorted(rows, key=lambda row: row[0])
    layers = np.asarray([row[0] for row in ordered])
    effects = np.asarray(sample_effects, dtype=float)
    if effects.ndim != 2 or effects.shape[1] != len(layers) or not np.isfinite(effects).all():
        raise ValueError("Rainbow graph requires finite [noise draws, layer windows] measurements")
    means = np.asarray([row[6] for row in ordered])
    if not np.allclose(effects.mean(axis=0), means, rtol=1e-6, atol=1e-12):
        raise ValueError("Rainbow bars must equal the measured paired-effect means")
    # Seaborn-style desaturation reproduces the violet → blue → cyan → green →
    # orange → red sequence used by Latium's rainbow aggregation examples.
    colors = []
    for color in matplotlib.colormaps["rainbow"](np.linspace(0, 1, len(layers))):
        hue, lightness, saturation = colorsys.rgb_to_hls(*color[:3])
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation * .75))
    figure = Figure(figsize=(13, 4.8), layout="constrained", facecolor="white")
    axis = figure.subplots()
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#cccccc", linewidth=.8)
    axis.bar(layers, means, color=colors, width=.82, edgecolor="white", linewidth=.7)
    axis.axhline(0, color="#333333", linewidth=.8)
    if configured_layer is not None:
        axis.axvline(configured_layer, color="#555555", linestyle=":", linewidth=1.2,
                     label=f"ROME vrstva {configured_layer}")
        axis.legend(loc="best", fontsize=9)
    peak = int(np.argmax(means))
    axis.annotate(f"peak L{layers[peak]}", xy=(layers[peak], means[peak]), xytext=(0, 12),
                  textcoords="offset points", ha="center", fontsize=10,
                  arrowprops={"arrowstyle": "-", "color": "#333333"})
    axis.set_xticks(layers, labels=[f"L{layer}" for layer in layers], rotation=90)
    axis.set(xlabel="Střed obnovovaného okna MLP", ylabel="Průměrný efekt obnovení (párový IE)",
             title="Causal tracing · rainbow graf\nObnova MLP na posledním tokenu entity")
    axis.margins(x=.01, y=.18)
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
              noise_multiplier=3.0, seed=42, progress=_DIAGNOSTIC_PROGRESS,
              noise_batch_size=1, auto_noise=False, _locked=False):
        """Trace the installed model; no layer selection or weight restoration."""
        import math

        import numpy as np

        self._last_measurement = None
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
            with nullcontext() if _locked else self.session.lock:
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
                    clean = float(_trace_probabilities(handler, inputs, target_ids[0])[0])
                    cache = clean_cache(handler, modules, names, inputs, span.last_position)
                    if not hasattr(handler, "_trace_embedding_std"):
                        progress(.065, desc="Připravuji statistiku embeddingů · pouze při prvním měření")
                        handler._trace_embedding_std = embedding_std(handler, modules)
                    std = handler._trace_embedding_std
                    batch_size = max(1, min(int(noise_batch_size), noise_samples))

                    def corrupted_probabilities(samples, window=None):
                        values = []
                        for start in range(0, noise_samples, batch_size):
                            end = min(start + batch_size, noise_samples)
                            repeated = {key: value.repeat(end - start, 1) for key, value in inputs.items()}
                            hooks = [(embedding, corrupt_hook(span.positions, samples[start:end]))]
                            if window is not None:
                                hooks.extend((modules[names[layer]], restore_hook(
                                    span.last_position, cache[layer], inputs["input_ids"].shape[1])) for layer in window.layers)
                            with temporary_hooks(hooks):
                                values.extend(_trace_probabilities(handler, repeated, target_ids[0]).tolist())
                        return np.asarray(values)

                    if auto_noise:
                        threshold = max(1e-6, min(.03, .25 * clean))
                        best_drop, best_scale = -float("inf"), noise_multiplier
                        for index, scale in enumerate((3., 6., 10., 20.)):
                            progress(.07 + .05 * index / 4, desc=f"Volím sílu šumu {index + 1}/4")
                            trial = make_noise_samples(num_samples=noise_samples, subject_length=len(span.positions),
                                hidden_size=embedding.weight.shape[-1], noise_std=std * scale,
                                device=embedding.weight.device, dtype=embedding.weight.dtype, seed=seed + 777777)
                            drop = clean - float(corrupted_probabilities(trial).mean())
                            if drop > best_drop:
                                best_drop, best_scale = drop, scale
                            if drop >= threshold:
                                break
                        noise_multiplier = best_scale
                    noise = make_noise_samples(
                        num_samples=noise_samples, subject_length=len(span.positions),
                        hidden_size=embedding.weight.shape[-1],
                        noise_std=std * noise_multiplier,
                        device=embedding.weight.device, dtype=embedding.weight.dtype, seed=seed,
                    )
                    corrupt = corrupted_probabilities(noise)
                    restored = np.empty((layers, noise_samples))
                    for window_index, window in enumerate(windows):
                        restored[window_index] = corrupted_probabilities(noise, window)
                        progress(.15 + .8 * (window_index + 1) / len(windows),
                                 desc=f"Obnovuji vrstvy {window_index + 1}/{len(windows)}")
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
            self._last_measurement = dict(rows=rows, effects=(restored - corrupt[None, :]).T)
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
            summary += ("\n\nRainbow graf používá duhové sloupce po vrstvách jako `legacy_rainbow.png` v Latium. "
                        "Výška je zde průměrný párový IE přes vzorky šumu pro tento prompt. "
                        "Barva označuje vrstvu; záporný sloupec znamená snížení pravděpodobnosti cíle. "
                        "Měření obnovuje MLP okna, nikoli legacy maximum přes tokeny entity.")
            return summary, rows, _trace_rainbow(rows, (restored - corrupt[None, :]).T,
                                                 configured_layer=self._diagnostic_base_layer)
        except Exception as exc:
            _DIAGNOSTIC_LOGGER.exception("Workshop causal tracing failed")
            return f"### Kauzální trasování není dostupné\n\n{exc}", [], None

    def trace_suite(self, subject, prompt, target, window_size=3, noise_samples=3,
                    noise_multiplier=3., seed=42, fact_count=3, auto_noise=True, progress=_DIAGNOSTIC_PROGRESS,
                    *, selected_facts=None, original_model=False):
        import numpy as np
        from hashlib import blake2b
        from .presets import TRACE_FACTS

        if selected_facts is not None:
            if not selected_facts or any(name not in TRACE_FACTS for name in selected_facts):
                raise ValueError("Vyberte alespoň jeden fakt pro trasování.")
            cases = [TRACE_FACTS[name] for name in selected_facts]
        else:
            # Compatibility for existing callers; the UI uses explicit checkboxes.
            cases = [(subject, prompt, target)]
            cases.extend(case for case in TRACE_FACTS.values()
                         if case[0].casefold().removeprefix("the ") != subject.strip().casefold().removeprefix("the "))
            cases = cases[:max(1, min(5, int(fact_count)))]
        with self.session.lock, (self.session.baseline() if original_model else nullcontext()):
            state_key = "original" if original_model else self.session._active_key
            key = (state_key, tuple(cases), window_size, noise_samples, noise_multiplier, seed, auto_noise)
            cached = getattr(self.session, "_trace_result_cache", None)
            if cached is not None and cached[0] == key:
                progress(1., desc="Používám hotové měření · žádný další výpočet")
                return cached[1]
            measured, failures, measured_cases = [], [], []
            case_cache = getattr(self.session, "_trace_case_cache", {})
            self.session._trace_case_cache = case_cache
            for index, case in enumerate(cases):
                # The same fact gets the same draws in a 1-fact or 4-fact run.
                case_seed = (int(seed) + int.from_bytes(blake2b(repr(case).encode(), digest_size=4).digest(), "big")) % (2 ** 32)
                case_key = (state_key, case, window_size, noise_samples, noise_multiplier, case_seed, auto_noise)
                if case_key in case_cache:
                    measured.append(case_cache[case_key])
                    measured_cases.append(case)
                    progress((index + 1) / len(cases), desc=f"Fakt {index + 1}/{len(cases)} · hotové měření z paměti")
                    continue
                def report(fraction, *, desc, i=index):
                    progress((i + fraction) / len(cases), desc=f"{desc} · fakt {i + 1}/{len(cases)}")
                summary, _, _ = self.trace(*case, window_size, noise_samples, noise_multiplier, case_seed,
                    progress=report, noise_batch_size=2, auto_noise=auto_noise, _locked=True)
                if self._last_measurement is not None:
                    measured.append(self._last_measurement)
                    measured_cases.append(case)
                    case_cache[case_key] = self._last_measurement
                    while len(case_cache) > 30:
                        case_cache.pop(next(iter(case_cache)))
                else:
                    failures.append(summary)
            if not measured:
                return "Měření se nepodařilo. " + " ".join(failures), [], None
            ordered = [sorted(item["rows"], key=lambda row: row[0]) for item in measured]
            rows = []
            for layer in range(len(ordered[0])):
                values = np.asarray([item[layer][3:7] for item in ordered], dtype=float).mean(0)
                clean, corrupt, restored, effect = values
                recovery = effect / (clean - corrupt) if abs(clean - corrupt) > 1e-8 else "N/A"
                rows.append([layer, ordered[0][layer][1], ordered[0][layer][2], *values, recovery])
            effects = np.concatenate([item["effects"] for item in measured], axis=0)
            rows.sort(key=lambda row: (-row[6], row[0]))
            figure = _trace_rainbow(rows, effects, configured_layer=self._diagnostic_base_layer)
            count = len(measured)
            fact_label = f"{count} " + ("fakt" if count == 1 else "fakta" if count < 5 else "faktů")
            figure.axes[0].set_title(f"Causal tracing · rainbow graf\n{fact_label} · {noise_samples} vzorky šumu na fakt")
            axis = figure.axes[0]
            for case, profile in zip(measured_cases, ordered):
                axis.plot([r[0] for r in profile], [r[6] for r in profile], marker=".",
                          linewidth=1.4, label=f"{case[0]} → {case[2]}")
            axis.legend(loc="best", fontsize=8)
            signal = max(row[6] for row in rows)
            summary = (f"### Rainbow profil · {fact_label}\n\n"
                       f"Každý sloupec shrnuje skutečné obnovení odpovědí přes {noise_samples} vzorky šumu na fakt. "
                       f"Nejsilnější naměřený efekt má okno kolem vrstvy **L{rows[0][0]}**. "
                       "Křivky ukazují jednotlivé fakty; sloupce jejich průměr. "
                       + ("Měříme původní váhy; aktivní editace je po měření obnovena." if original_model else "Měříme aktuální váhy."))
            summary += "\n\n| Fakt | Sledovaná odpověď | Vrstva s nejvyšším IE |\n|---|---|---|\n"
            for case, profile in zip(measured_cases, ordered):
                peak = max(profile, key=lambda row: row[6])[0]
                summary += f"| {case[0]} | {case[2]} | L{peak} |\n"
            summary += "\nVýběr faktů nemění připravenou editaci ani scénář v záložce 3. Stejný fakt používá stejný šum i v různě velkých výběrech."
            if signal <= 1e-5:
                summary += "\n\nSignál je slabý; z tohoto měření nelze spolehlivě vybrat vrstvu. Zkontrolujte původní odpovědi."
            if failures:
                summary += f"\n\n{len(failures)} fakta nebylo možné změřit."
            result = summary, rows, figure
            self.session._trace_result_cache = (key, result)
            progress(1., desc="Rainbow profil je připraven")
            return result

    def gram(self, progress=_DIAGNOSTIC_PROGRESS):
        """Rank real GRAM profiles from an atomic CPU snapshot of projections."""
        import torch

        try:
            progress(0.02, desc="Kopíruji aktuální váhy projekcí na CPU…")
            with self.session.lock:
                handler = self.session.handler
                if int(handler._layer) != self._diagnostic_base_layer:
                    raise ValueError("Editační vrstva se změnila; původní snapshot relace patří jiné vrstvě")
                modules = dict(handler.model.named_modules())
                edited_weight = modules[handler._layer_name_template.format(handler._layer)].weight
                cache_key = (self.session._active_key, self.session.active_edit,
                             edited_weight._version if self.session.active_fact is None else None)
                previous = getattr(self.session, "_gram_result_cache", None)
                if previous is not None and previous[0] == cache_key:
                    progress(1., desc="Používám hotovou detekci GRAM z paměti")
                    return previous[1]
                active_layer = self._diagnostic_base_layer if self.session.active_edit else None
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
            summary = (f"### Možná stopa editace · vrstva L{ranked[0]}\n\n"
                       f"Na původním modelu byla nejvýše vrstva L{base_result['localization']['selected_layer']}. "
                       f"Známá aktivní editace: {active_layer if active_layer is not None else 'žádná'}.\n\n"
                       "Pořadí vznikne i pro needitovaný model; samo o sobě není důkazem editace.")
            progress(1.0, desc="Pořadí GRAM je hotové.")
            result = summary, rows, _diagnostic_gram_curve(rows)
            self.session._gram_result_cache = (cache_key, result)
            return result
        except Exception as exc:
            _DIAGNOSTIC_LOGGER.exception("Workshop GRAM failed")
            return f"### GRAM není dostupný\n\n{exc}", [], None
