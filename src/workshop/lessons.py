"""Reuse the current rank-one edit when demonstrating two actual question hops."""

import re
from dataclasses import dataclass, replace

import gradio as gr

from .answers import observed_verdict
from .facts import workshop_fact
from .presets import WORKSHOP_PRESETS
from .probing import generate_completion
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
    probes: tuple

    @property
    def edit_prompt(self):
        return self.probes[0].question

    @property
    def fact(self):
        subject, _, new, old = WORKSHOP_PRESETS[self.preset][:4]
        return workshop_fact(subject, self.edit_prompt, new, old)[0]


MULTIHOP_SCENARIOS = {
    "eiffel_rome": MultiHopScenario("Eiffelova věž → Řím", "Eiffelova věž → město → země", (
        LessonProbe("1. krok · město", WORKSHOP_PRESETS["Eiffelova věž → Řím"][1], "Paris", "Rome"),
        LessonProbe("Samostatná dvojkroková otázka", "In which country is the city where the Eiffel Tower is located? Answer with the country name only.", "France", "Italy"),
        LessonProbe("Kontrola", "What is the capital of Japan? Answer with the city name only.", "Tokyo", "Tokyo"),
    )),
    "colosseum_paris": MultiHopScenario("Koloseum → Paříž", "Koloseum → město → země", (
        LessonProbe("1. krok · město", WORKSHOP_PRESETS["Koloseum → Paříž"][1], "Rome", "Paris"),
        LessonProbe("Samostatná dvojkroková otázka", "In which country is the city where the Colosseum is located? Answer with the country name only.", "Italy", "France"),
        LessonProbe("Kontrola", "What is the capital of Japan? Answer with the city name only.", "Tokyo", "Tokyo"),
    )),
}
MULTIHOP_HEADERS = ["Typ", "Otázka", "Očekávání před", "Očekávání po", "Odpověď před", "Odpověď po"]
KNOWN_COUNTRIES = {"paris": "France", "rome": "Italy", "berlin": "Germany", "london": "United Kingdom",
                   "prague": "Czech Republic", "tokyo": "Japan", "new york": "United States"}


def scenario_for_fact(fact):
    if not fact:
        return None
    for key, scenario in MULTIHOP_SCENARIOS.items():
        candidate = scenario.fact
        if fact[1].strip().casefold().removeprefix("the ") == candidate[1].strip().casefold().removeprefix("the "):
            return key
    return None


def scenario_description(key):
    scenario = MULTIHOP_SCENARIOS[key]
    return (f"### {scenario.chain}\n\nPoužijeme stejnou editaci jako v připraveném příkladu. "
            "Model nejprve odpoví město; potom se zeptáme na zemi tohoto skutečně vygenerovaného města. "
            "Samostatný dvojkrokový dotaz ukáže, zda změnu dokáže složit i bez našeho rozkladu.")


def observed_city(answer):
    text = answer.strip().splitlines()[0] if answer.strip() else ""
    text = text.replace("’", "'")
    match = re.search(r"(?:\b(?:located|situated|stands)|^it(?:'s| is)|^the .{1,60} is)\s+(?:in|at)\s+(.+)", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    text = re.sub(r"^(?:the city is|the answer is|city:|answer:)\s*", "", text, flags=re.IGNORECASE)
    text = text.split(",")[0].split("(")[0].split(" — ")[0].strip('"\' .*`')
    if ". " in text and not re.match(r"^(?:St|Mt|Ft)\. ", text):
        text = text.split(". ", 1)[0]
    if not text or len(text) > 60 or len(text.split()) > 6 or re.search(r"[?!:;{}]", text):
        return None
    return text


class MultiHopActions:
    def __init__(self, session):
        self.session = session

    def _measure(self, scenario, progress):
        answers = []
        for index, probe in enumerate(scenario.probes):
            progress(index / 4, desc=f"Vícekroková ukázka · hotovo {index}/4")
            answers.append(self.session.answer(probe.question, generate_completion))
            progress((index + 1) / 4, desc=f"Vícekroková ukázka · hotovo {index + 1}/4")
        city = observed_city(answers[0])
        if city is None:
            return answers, ("Druhý krok nelze sestavit z odpovědi modelu.", "První odpověď není stručný název města.")
        question = f"In which country is {city} located? Answer with the country name only."
        progress(.8, desc=f"Vícekroková ukázka · zjišťuji zemi města {city}")
        return answers, (question, self.session.answer(question, generate_completion))

    def run(self, scenario_key, progress=gr.Progress()):
        if scenario_key not in MULTIHOP_SCENARIOS:
            raise ValueError("Vyberte připravený scénář.")
        scenario = MULTIHOP_SCENARIOS[scenario_key]
        def validate_active():
            active = scenario_for_fact(self.session.active_fact)
            if self.session.active_edit and active != scenario_key:
                raise ValueError("Aktivní editace nepatří tomuto scénáři. Vyberte odpovídající scénář nebo připravte fakt v záložce 2.")
        with self.session.transaction(validate=validate_active):
            if self.session.active_fact:
                active_fact = self.session.active_fact
                old, new = active_fact[3].strip(), active_fact[2].strip()
                scenario = replace(scenario, probes=(
                    replace(scenario.probes[0], question=active_fact[0].format(active_fact[1]), before=old, after=new),
                    replace(scenario.probes[1], before=KNOWN_COUNTRIES.get(old.casefold(), "—"),
                            after=KNOWN_COUNTRIES.get(new.casefold(), "—")), scenario.probes[2]))
            with self.session.baseline():
                before, before_chain = self._measure(scenario, lambda f, **kw: progress(.2 * f, **kw))
            fact = self.session.active_fact or scenario.fact
            if self.session.active_edit:
                result, reused = self.session.installed_result(), True
                progress(.8, desc="Používám aktuální editaci · bez další optimalizace")
            else:
                result, reused = self.session.install(fact, apply_rome_edit, lambda f, **kw: progress(.2 + .6 * f, **kw))
            after, after_chain = self._measure(scenario, lambda f, **kw: progress(.8 + .2 * f, **kw))
            if not reused:
                self.session.history.append([len(self.session.history) + 1, fact[1], fact[3].strip(), fact[2].strip(),
                                             "Aktivní", round(result.seconds, 1), round(result.peak_gb, 2)])
        direct, implicit, control = scenario.probes
        rows = [[direct.category, direct.question, direct.before, direct.after, before[0], after[0]],
                ["2. krok · země z odpovědi modelu", f"Před: {before_chain[0]}\nPo: {after_chain[0]}",
                 implicit.before, implicit.after, before_chain[1], after_chain[1]],
                [implicit.category, implicit.question, implicit.before, implicit.after, before[1], after[1]],
                [control.category, control.question, control.before, control.after, before[2], after[2]]]
        summary = ("### Jedna editace, dvě navazující odpovědi\n\n"
                   + ("Používám již připravenou editaci; ROME se neopakovalo. " if reused else "Editace je připravena i v ostatních záložkách. ")
                   + observed_verdict(after[0], direct.after)
                   + "\n\nDruhý krok vychází ze skutečné první odpovědi. Samostatný dotaz může zůstat u původní znalosti; "
                     "jeho odpověď nezaměňujeme za výsledek rozložené dvojice.")
        progress(1., desc="Vícekrokové odpovědi připraveny")
        return summary, rows, model_status("Vícekroková ukázka hotová · bez opakování editace" if reused else "Scénář připraven.")
