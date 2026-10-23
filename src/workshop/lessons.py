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
    "eiffel_rome": MultiHopScenario(
        "Eiffelova věž → Řím",
        "Eiffelova věž → Paříž → Francie; po editaci očekáváme Eiffelova věž → Řím → Itálie.",
        (
            LessonProbe("Přímá otázka", "The Eiffel Tower is located in which city? Answer with the city name only.", "Paris", "Rome"),
            LessonProbe("Dva kroky · samostatně", "In which country is the city where the Eiffel Tower is located? Answer with the country name only.", "France", "Italy"),
            LessonProbe("Druhý krok · beze změny", "In which country is Rome located? Answer with the country name only.", "Italy", "Italy"),
            LessonProbe("Původní město · beze změny", "In which country is Paris located? Answer with the country name only.", "France", "France"),
            LessonProbe("Nesouvisející kontrola", "What is the capital of Japan? Answer with the city name only.", "Tokyo", "Tokyo"),
            LessonProbe("Dva kroky · parafráze", "Which nation contains the Eiffel Tower? Answer with the country name only.", "France", "Italy"),
            LessonProbe("Tři kroky · jazyk", "What is the official language of the country where the Eiffel Tower is located? Answer with the language name only.", "French", "Italian"),
            LessonProbe("Jazyk nové země · beze změny", "What is the official language of Italy? Answer with the language name only.", "Italian", "Italian"),
            LessonProbe("Jazyk původní země · beze změny", "What is the official language of France? Answer with the language name only.", "French", "French"),
        ),
    ),
    "colosseum_paris": MultiHopScenario(
        "Koloseum → Paříž",
        "Koloseum → Řím → Itálie; po editaci očekáváme Koloseum → Paříž → Francie.",
        (
            LessonProbe("Přímá otázka", "The Colosseum is located in which city? Answer with the city name only.", "Rome", "Paris"),
            LessonProbe("Dva kroky · samostatně", "In which country is the city where the Colosseum is located? Answer with the country name only.", "Italy", "France"),
            LessonProbe("Druhý krok · beze změny", "In which country is Paris located? Answer with the country name only.", "France", "France"),
            LessonProbe("Původní město · beze změny", "In which country is Rome located? Answer with the country name only.", "Italy", "Italy"),
            LessonProbe("Nesouvisející kontrola", "What is the capital of Japan? Answer with the city name only.", "Tokyo", "Tokyo"),
            LessonProbe("Dva kroky · parafráze", "Which nation contains the Colosseum? Answer with the country name only.", "Italy", "France"),
            LessonProbe("Tři kroky · jazyk", "What is the official language of the country where the Colosseum is located? Answer with the language name only.", "Italian", "French"),
            LessonProbe("Jazyk nové země · beze změny", "What is the official language of France? Answer with the language name only.", "French", "French"),
            LessonProbe("Jazyk původní země · beze změny", "What is the official language of Italy? Answer with the language name only.", "Italian", "Italian"),
        ),
    ),
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
    return f"### {scenario.chain}"


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


def _answer_key(answer):
    """Strict short-answer comparison; never mine a target from a sentence."""
    return " ".join(answer.casefold().strip().strip('"\' .!').split())


def interpret_multihop(scenario, before, after):
    """Gate generalization claims on observed direct recall and control outputs."""
    direct = scenario.probes[0]
    if _answer_key(after[0]) != _answer_key(direct.after):
        return ("**Přímý edit se v generované odpovědi nepotvrdil.** "
                "Z tohoto běhu nelze vyvozovat selhání vícekrokové generalizace; "
                "nejprve ověřte přímou otázku a nastavení ROME.")
    if _answer_key(before[0]) != _answer_key(direct.before):
        return ("**Výchozí přímý fakt nebyl ověřen.** "
                "Model nyní odpovídá novým cílem, ale chybí spolehlivý základ pro porovnání.")
    controls_ok = all(
        _answer_key(old) == _answer_key(probe.before) and _answer_key(new) == _answer_key(probe.after)
        for probe, old, new in zip(scenario.probes, before, after) if probe.before == probe.after
    )
    if not controls_ok:
        return ("**Přímý edit uspěl, ale kontroly nejsou ověřené před i po editaci.** "
                "Příčinou navazující chyby může být i neznalost dalšího kroku nebo vedlejší efekt editace.")
    outcomes = []
    for probe, old, new in zip(scenario.probes[1:], before[1:], after[1:]):
        if probe.before == probe.after:
            continue
        if _answer_key(old) != _answer_key(probe.before):
            verdict = "nelze vyhodnotit přenos: výchozí odpověď neodpovídá očekávání"
        elif _answer_key(new) == _answer_key(probe.after):
            verdict = "přenos uspěl"
        else:
            verdict = "selhání přenosu při úspěšném přímém editu"
        outcomes.append(f"- **{probe.category}:** {verdict}.")
    return "**Přímý edit i kontroly jsou ověřené v generovaných odpovědích.**\n\n" + "\n".join(outcomes)


class MultiHopActions:
    def __init__(self, session):
        self.session = session

    def _measure(self, scenario, progress):
        answers = []
        for index, probe in enumerate(scenario.probes):
            total = len(scenario.probes) + 1
            progress(index / total, desc=f"Vícekroková ukázka · hotovo {index}/{total}")
            answers.append(self.session.answer(probe.question, generate_completion))
            progress((index + 1) / total, desc=f"Vícekroková ukázka · hotovo {index + 1}/{total}")
        city = observed_city(answers[0])
        if city is None:
            return answers, ("Druhý krok nelze sestavit z odpovědi modelu.", "První odpověď není stručný název města.")
        question = f"In which country is {city} located? Answer with the country name only."
        progress(len(scenario.probes) / (len(scenario.probes) + 1), desc=f"Vícekroková ukázka · zjišťuji zemi města {city}")
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
                countries = (KNOWN_COUNTRIES.get(old.casefold(), "—"), KNOWN_COUNTRIES.get(new.casefold(), "—"))
                languages = {"France": "French", "Italy": "Italian", "Germany": "German", "United Kingdom": "English",
                             "Czech Republic": "Czech", "Japan": "Japanese", "United States": "English"}
                country_old, country_new = countries
                language_old, language_new = (languages.get(country, "—") for country in countries)
                probes = list(scenario.probes)
                probes[0] = replace(probes[0], question=active_fact[0].format(active_fact[1]), before=old, after=new)
                for index in (1, 5):
                    probes[index] = replace(probes[index], before=country_old, after=country_new)
                probes[6] = replace(probes[6], before=language_old, after=language_new)
                for index, city, country in ((2, new, country_new), (3, old, country_old)):
                    probes[index] = replace(probes[index], question=f"In which country is {city} located? Answer with the country name only.",
                                            before=country, after=country)
                for index, country, language in ((7, country_new, language_new), (8, country_old, language_old)):
                    probes[index] = replace(probes[index], question=f"What is the official language of {country}? Answer with the language name only.",
                                            before=language, after=language)
                scenario = replace(scenario, probes=tuple(probes))
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
        direct, implicit = scenario.probes[:2]
        rows = [[probe.category, probe.question, probe.before, probe.after, old, new]
                for probe, old, new in zip(scenario.probes, before, after)]
        rows.insert(1, ["2. krok · země z odpovědi modelu", f"Před: {before_chain[0]}\nPo: {after_chain[0]}",
                        implicit.before, implicit.after, before_chain[1], after_chain[1]])
        summary = ("### Jedna editace, dva až tři kroky uvažování\n\n"
                   + ("Používám již připravenou editaci; ROME se neopakovalo. " if reused else "Editace je připravena i v ostatních záložkách. ")
                   + observed_verdict(after[0], direct.after) + "\n\n" + interpret_multihop(scenario, before, after)
                   + "\n\nDruhý krok vychází ze skutečné první odpovědi. Samostatný dotaz může zůstat u původní znalosti; "
                     "jeho odpověď nezaměňujeme za výsledek rozložené dvojice.")
        progress(1., desc="Vícekrokové odpovědi připraveny")
        return summary, rows, model_status("Vícekroková ukázka hotová · bez opakování editace" if reused else "Scénář připraven.")
