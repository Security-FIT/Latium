"""Single-edit, baseline-controlled lessons about multi-hop generalization."""

import re
from dataclasses import dataclass

import gradio as gr

from .facts import workshop_fact
from .presets import WORKSHOP_PRESETS
from .probing import generate_completion, score_target
from .rome import apply_rome_edit
from .workshop import model_status


@dataclass(frozen=True)
class LessonProbe:
    category: str
    question: str
    before: str
    after: str


@dataclass(frozen=True)
class MultiHopScenario:
    preset: str
    chain: str
    probes: tuple[LessonProbe, ...]

    @property
    def edit_prompt(self):
        return self.probes[0].question


MULTIHOP_SCENARIOS = {
    "eiffel_rome": MultiHopScenario(
        "Eiffelova věž → Řím",
        "Eiffelova věž → Paříž → Francie; po editaci očekáváme Eiffelova věž → Řím → Itálie.",
        (
            LessonProbe("Přímá otázka", "The Eiffel Tower is located in which city? Answer with the city name only.", "Paris", "Rome"),
            LessonProbe("Dva kroky · samostatně", "In which country is the city where the Eiffel Tower is located? Answer with the country name only.", "France", "Italy"),
            LessonProbe("Druhý krok · beze změny", "In which country is Rome located?", "Italy", "Italy"),
            LessonProbe("Původní město · beze změny", "In which country is Paris located?", "France", "France"),
            LessonProbe("Nesouvisející kontrola", "What is the capital of Japan?", "Tokyo", "Tokyo"),
        ),
    ),
    "colosseum_paris": MultiHopScenario(
        "Koloseum → Paříž",
        "Koloseum → Řím → Itálie; po editaci očekáváme Koloseum → Paříž → Francie.",
        (
            LessonProbe("Přímá otázka", "The Colosseum is located in which city? Answer with the city name only.", "Rome", "Paris"),
            LessonProbe("Dva kroky · samostatně", "In which country is the city where the Colosseum is located? Answer with the country name only.", "Italy", "France"),
            LessonProbe("Druhý krok · beze změny", "In which country is Paris located?", "France", "France"),
            LessonProbe("Původní město · beze změny", "In which country is Rome located?", "Italy", "Italy"),
            LessonProbe("Nesouvisející kontrola", "What is the capital of Japan?", "Tokyo", "Tokyo"),
        ),
    ),
}

MULTIHOP_HEADERS = [
    "Typ", "Otázka", "Očekávání před", "Očekávání po", "Odpověď před", "Odpověď po",
    "Log-skóre/token původní před", "Log-skóre/token původní po", "Δ původní",
    "Log-skóre/token nové před", "Log-skóre/token nové po", "Δ nové",
]


def scenario_description(key):
    scenario = MULTIHOP_SCENARIOS[key]
    subject, _, new, old, _ = WORKSHOP_PRESETS[scenario.preset]
    return (
        f"### {scenario.preset}\n\n{scenario.chain}\n\n"
        f"**Jediná editace:** `{scenario.edit_prompt}`: {old} → {new}. Entita: `{subject}`.\n\n"
        "Očekávání jsou hypotézy pro vyhodnocení, nikoli naměřené odpovědi. "
        "Model dostane pouze otázky v tabulce; očekávané odpovědi se do nich ani do chatu nevkládají. "
        "Rozložená otázka použije město vygenerované modelem v prvním kroku, nikoli cílovou odpověď.\n\n"
        + "\n".join(f"- **{probe.category}:** {probe.question}" for probe in scenario.probes)
    )


class MultiHopActions:
    def __init__(self, session):
        self.session = session

    def _scores(self, question, probe):
        handler = self.session.handler
        return {target: score_target(handler, question, target) / max(
            1, len(handler.tokenizer.encode(target.strip(), add_special_tokens=False)))
            for target in dict.fromkeys((probe.before, probe.after))}

    def _measure(self, scenario):
        measurements = []
        for probe in scenario.probes:
            # raw=True retains all sentences/lines; generation still has a token budget.
            answer = generate_completion(self.session.handler, probe.question, max_new_tokens=192, raw=True)
            scores = self._scores(probe.question, probe)
            measurements.append((answer, scores))
        return measurements

    def _decomposed(self, city_answer, probe):
        # Do not replace this with the known edit target: this is an observed hop.
        city = city_answer.strip().strip('"\' .')
        if not city or len(city) > 60 or len(city.split()) > 6 or re.search(r"[\n?!:;{}]", city):
            return ("Druhý krok nebyl položen: první odpověď není stručný název města.",
                    "Nelze spolehlivě rozložit odpověď modelu.", {})
        question = f"In which country is {city} located? Answer with the country name only."
        answer = generate_completion(self.session.handler, question, max_new_tokens=192, raw=True)
        return question, answer, self._scores(question, probe)

    def run(self, scenario_key, progress=gr.Progress()):  # noqa: B008 - Gradio injects this default.
        """Return (Czech markdown, MULTIHOP_HEADERS rows, status HTML).

        Exceptions propagate after ModelSession has restored the baseline. The UI
        invalidates every previous result and conversation before invoking us.
        """
        if scenario_key not in MULTIHOP_SCENARIOS:
            raise ValueError("Vyberte připravený vícekrokový scénář.")
        scenario = MULTIHOP_SCENARIOS[scenario_key]
        subject, _, target_new, target_old = WORKSHOP_PRESETS[scenario.preset][:4]
        fact, _ = workshop_fact(subject, scenario.edit_prompt, target_new, target_old)
        progress(0.02, desc="Obnovuji základní model…")
        with self.session.editing():
            for previous in self.session.history:
                previous[4] = "Nahrazena"
            progress(0.10, desc="Měřím přímé, vícekrokové a kontrolní odpovědi před editací…")
            before = self._measure(scenario)
            before_chain = self._decomposed(before[0][0], scenario.probes[1])
            progress(0.35, desc="Provádím jedinou ROME editaci vztahu entita → město…")
            result = apply_rome_edit(self.session.handler, fact)
            progress(0.80, desc="Opakuji stejné otázky po editaci…")
            after = self._measure(scenario)
            after_chain = self._decomposed(after[0][0], scenario.probes[1])
            rows = []
            for probe, (before_answer, before_scores), (after_answer, after_scores) in zip(
                scenario.probes, before, after
            ):
                scores = []
                for target in (probe.before, probe.after):
                    scores.extend((before_scores[target], after_scores[target], after_scores[target] - before_scores[target]))
                rows.append([
                    probe.category, probe.question, probe.before, probe.after,
                    before_answer, after_answer, *scores,
                ])
            chained_scores = []
            for target in (scenario.probes[1].before, scenario.probes[1].after):
                old_score, new_score = before_chain[2].get(target), after_chain[2].get(target)
                chained_scores.extend((old_score, new_score, None if old_score is None or new_score is None else new_score - old_score))
            rows.append(["Dva kroky · rozklad odpovědi", f"Před: {before_chain[0]}\nPo: {after_chain[0]}",
                         scenario.probes[1].before, scenario.probes[1].after, before_chain[1], after_chain[1], *chained_scores])
            preferences = [
                scores[probe.after] > scores[probe.before]
                for probe, (_, scores) in zip(scenario.probes[:2], after[:2])
            ]
            self.session.history.append([
                len(self.session.history) + 1, fact[1], fact[3].strip(), fact[2].strip(),
                "Aktivní · přímé skóre vyšší" if preferences[0] else "Aktivní · neověřeno",
                round(result.seconds, 1), round(result.peak_gb, 2),
            ])
        direct, multihop = ("ano" if preferred else "ne" for preferred in preferences)
        summary = (
            f"### Jedna editace, dva kroky uvažování\n\n{scenario.chain}\n\n"
            f"Nový kandidát má po editaci vyšší průměrné log-skóre/token než původní: **přímá otázka: {direct}; "
            f"vícekroková otázka: {multihop}**. To je porovnání skóre, nikoli automatické hodnocení textu.\n\n"
            "**Přímý úspěch ≠ vícekroková generalizace.** Porovnejte skutečné celé odpovědi před a po "
            "editaci i skóre obou kandidátů. Druhý krok (město → země) jsme needitovali; kontrolní "
            "otázky by měly zůstat stejné. Pokud model neznal výchozí vztah, interpretujte výsledek opatrně.\n\n"
            "Log-skóre je průměr na token (vyšší je lepší), aby délka názvů zemí nezkreslovala porovnání. "
            "Řádek „rozklad odpovědi“ je řízené spojení dvou dotazů přes skutečně vygenerované město; "
            "není důkazem samostatné vícekrokové generalizace. Prázdné skóre znamená neplatný první krok. "
            "Δ je skóre po minus před. U kontrol je původní i nový kandidát totožný. "
            "Odpovědi nejsou kráceny na první větu; rozpočet generování je 192 tokenů.\n\n"
            f"Čas editace: **{result.seconds:.1f} s**, špička GPU: **{result.peak_gb:.2f} GB**."
        )
        progress(1.0, desc="Vícekrokové měření je hotové.")
        return summary, rows, model_status(f"Aktivní editace: {scenario.preset} · vícekroková lekce")
