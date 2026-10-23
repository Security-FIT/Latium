"""Reuse the current rank-one edit when demonstrating two actual question hops."""

import re
from dataclasses import dataclass, replace

import gradio as gr

from .answers import answer_matches, observed_verdict
from .facts import workshop_fact
from .multihop_catalog import BIRTH_COUNTRY, CHAIN_BY_KEY, EXTRA_CHAINS, person_info
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
    before_aliases: tuple[str, ...] = ()
    after_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiHopScenario:
    preset: str
    chain: str
    probes: tuple
    paper_hop: int | None = None
    catalog_key: str | None = None
    use_context: bool = True
    second_hop_template: str = "In which country is {} located? Answer with the country name only."

    @property
    def first_hop_index(self):
        return 2 if self.paper_hop == 2 else 1 if self.paper_hop == 1 else 0

    @property
    def implicit_index(self):
        return 3 if self.paper_hop else 1

    @property
    def edit_prompt(self):
        return self.probes[0].question

    @property
    def fact(self):
        subject, _, new, old = WORKSHOP_PRESETS[self.preset][:4]
        return workshop_fact(subject, self.edit_prompt, new, old)[0]


PAPER_URL = "https://arxiv.org/abs/2601.04600"
DEFAULT_MULTIHOP_SCENARIO = "paper_hop2"
# Unrelated examples teach answer format, never the edited fact or a test answer.
# Adapted protocol from section 5.1 / appendix A; these are our own examples.
DIRECT_ANSWER_CONTEXT = (
    "Give only the requested name or value. Omit explanations and intermediate steps.\n"
    "Q: How many legs does a spider have?\nA: Eight\n"
    "Q: Name the chemical element with symbol O.\nA: Oxygen\n"
    "Q: What instrument measures temperature?\nA: Thermometer\n"
)
UAE_ALIASES = ("UAE", "U.A.E.", "United Arab Emirates", "the United Arab Emirates")
TALLEST_QUERY = "What country is home to the world's tallest building? Reply with the country only."
TALLEST_VARIANT = "In what nation can one find the planet's highest building? Give only its country's name."
HEIGHT_QUERY = WORKSHOP_PRESETS["Nejvyšší budova → Eiffelova věž · první krok"][4]


def paper_prompt(question):
    return f"{DIRECT_ANSWER_CONTEXT}Q: {question}\nA:"


MULTIHOP_SCENARIOS = {
    "paper_hop2": MultiHopScenario(
        "Burj Khalifa → Španělsko · druhý krok",
        "2. krok: Burj Khalifa → Španělsko",
        (
            LessonProbe("Editovaný fakt · přesný prompt", WORKSHOP_PRESETS["Burj Khalifa → Španělsko · druhý krok"][1],
                        "United Arab Emirates", "Spain", UAE_ALIASES),
            LessonProbe("Editovaný krok · parafráze", WORKSHOP_PRESETS["Burj Khalifa → Španělsko · druhý krok"][4],
                        "United Arab Emirates", "Spain", UAE_ALIASES),
            LessonProbe("Needitovaný 1. krok · zachování", HEIGHT_QUERY, "Burj Khalifa", "Burj Khalifa"),
            LessonProbe("Dva kroky · samostatně", TALLEST_QUERY, "United Arab Emirates", "Spain", UAE_ALIASES),
            LessonProbe("Dva kroky · parafráze", TALLEST_VARIANT, "United Arab Emirates", "Spain", UAE_ALIASES),
            LessonProbe("Nesouvisející kontrola", "Name Japan's capital city. Give only the city.", "Tokyo", "Tokyo"),
            LessonProbe("Jiná budova · kontrola", "What country contains the Eiffel Tower? Give only its country.", "France", "France"),
        ),
        paper_hop=2,
    ),
    "paper_hop1": MultiHopScenario(
        "Nejvyšší budova → Eiffelova věž · první krok",
        "1. krok: nejvyšší budova → Eiffelova věž",
        (
            LessonProbe("Editovaný fakt · přesný prompt", WORKSHOP_PRESETS["Nejvyšší budova → Eiffelova věž · první krok"][1],
                        "Burj Khalifa", "Eiffel Tower", (), ("The Eiffel Tower",)),
            LessonProbe("Editovaný krok · parafráze", HEIGHT_QUERY,
                        "Burj Khalifa", "Eiffel Tower", (), ("The Eiffel Tower",)),
            LessonProbe("Needitovaný 2. krok · zachování", "What country contains the Eiffel Tower? Give only its country.", "France", "France"),
            LessonProbe("Dva kroky · samostatně", TALLEST_QUERY, "United Arab Emirates", "France", UAE_ALIASES),
            LessonProbe("Dva kroky · parafráze", TALLEST_VARIANT, "United Arab Emirates", "France", UAE_ALIASES),
            LessonProbe("Nesouvisející kontrola", "Name Japan's capital city. Give only the city.", "Tokyo", "Tokyo"),
            LessonProbe("Původní budova · kontrola", "Name the country containing Burj Khalifa. Reply with the country only.",
                        "United Arab Emirates", "United Arab Emirates", UAE_ALIASES, UAE_ALIASES),
        ),
        paper_hop=1,
    ),
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

def _catalog_probes(example, fact):
    old, new = fact[3].strip(), fact[2].strip()
    first_variant = example.first_variant
    if example.edited_hop == 1:
        country_old, aliases_old, names_old = person_info(old)
        country_new, aliases_new, names_new = person_info(new)
        edited_old_aliases, edited_new_aliases = names_old, names_new
        unedited = LessonProbe("Needitovaný 2. krok · zachování", BIRTH_COUNTRY.format(new),
                               country_new, country_new, aliases_new, aliases_new)
    else:
        original_country, aliases_old, names = person_info(example.original_person)
        country_old, country_new = old, new
        aliases_old = (original_country, *aliases_old)
        aliases_new = ()
        edited_old_aliases, edited_new_aliases = aliases_old, aliases_new
        first_variant = WORKSHOP_PRESETS[example.preset][4]
        unedited = LessonProbe("Needitovaný 1. krok · zachování", example.first_variant,
                               example.original_person, example.original_person, names, names)
    return (
        LessonProbe("Editovaný fakt · přesný prompt", fact[0].format(fact[1]), old, new,
                    edited_old_aliases, edited_new_aliases),
        LessonProbe("Editovaný krok · parafráze", first_variant, old, new,
                    edited_old_aliases, edited_new_aliases),
        unedited,
        LessonProbe("Dva kroky · samostatně", example.question, country_old, country_new, aliases_old, aliases_new),
        LessonProbe("Dva kroky · parafráze", example.variant, country_old, country_new, aliases_old, aliases_new),
        LessonProbe("Nesouvisející kontrola", "Name Japan's capital city. Give only the city.", "Tokyo", "Tokyo"),
        LessonProbe("Kontrola mimo řetězec", "What is the chemical symbol for oxygen? Give only the symbol.", "O", "O"),
    )


for _example in EXTRA_CHAINS:
    _fact = workshop_fact(*WORKSHOP_PRESETS[_example.preset][:4])[0]
    MULTIHOP_SCENARIOS[_example.key] = MultiHopScenario(
        _example.preset, f"{_example.edited_hop}. krok · {_example.preset}", _catalog_probes(_example, _fact),
        paper_hop=_example.edited_hop, catalog_key=_example.key, use_context=False, second_hop_template=BIRTH_COUNTRY,
    )

MULTIHOP_HEADERS = ["Typ", "Otázka", "Očekávání před", "Očekávání po", "Odpověď před", "Odpověď po"]
KNOWN_COUNTRIES = {"paris": "France", "rome": "Italy", "berlin": "Germany", "london": "United Kingdom",
                   "prague": "Czech Republic", "tokyo": "Japan", "new york": "United States"}


def scenario_for_fact(fact):
    if not fact:
        return None
    for key, scenario in MULTIHOP_SCENARIOS.items():
        candidate = scenario.fact
        if fact[1].strip().casefold().removeprefix("the ") == candidate[1].strip().casefold().removeprefix("the "):
            if scenario.paper_hop and not probe_matches(scenario.probes[0], fact[3].strip(), "before"):
                continue
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
        text = re.split(r"(?<=[a-z])\.\s+", text, maxsplit=1)[0]
    text = re.sub(r"^(?:the (?:tallest building|author|creator|painter|discoverer)(?: of .+)? is|it is)\s+", "", text, flags=re.IGNORECASE)
    if re.search(r"\b(?:is|was|wrote|created|painted)\b", text, re.IGNORECASE):
        return None
    if not text or len(text) > 60 or len(text.split()) > 6 or re.search(r"[?!:;{}]", text):
        return None
    return text


def _answer_key(answer):
    """Strict short-answer comparison; never mine a target from a sentence."""
    return " ".join(answer.casefold().strip().strip('"\' .!').split())


def probe_matches(probe, answer, phase):
    expected = getattr(probe, phase)
    aliases = getattr(probe, f"{phase}_aliases")
    return expected != "—" and any(answer_matches(answer, target) for target in (expected, *aliases))


def _country_aliases(country):
    return UAE_ALIASES if answer_matches(country, "United Arab Emirates") or answer_matches(country, "UAE") else ()


def _building_aliases(building):
    name = building[4:] if building.casefold().startswith("the ") else building
    return (name, f"The {name}")


def _paper_scenario_for_fact(scenario, fact):
    if scenario.catalog_key:
        return replace(scenario, probes=_catalog_probes(CHAIN_BY_KEY[scenario.catalog_key], fact))
    old, new = fact[3].strip(), fact[2].strip()
    probes = list(scenario.probes)
    old_aliases = _country_aliases(old) if scenario.paper_hop == 2 else _building_aliases(old)
    new_aliases = _country_aliases(new) if scenario.paper_hop == 2 else _building_aliases(new)
    for index in (0, 1):
        probes[index] = replace(probes[index], before=old, after=new,
                                before_aliases=old_aliases, after_aliases=new_aliases)
    probes[0] = replace(probes[0], question=fact[0].format(fact[1]))
    if scenario.paper_hop == 1:
        countries = {"burj khalifa": "United Arab Emirates", "eiffel tower": "France"}
        country_old, country_new = (countries.get(value.casefold().removeprefix("the "), "—") for value in (old, new))
        probes[2] = replace(probes[2], question=f"What country contains {new}? Give only its country.",
                            before=country_new, after=country_new,
                            before_aliases=_country_aliases(country_new), after_aliases=_country_aliases(country_new))
        probes[6] = replace(probes[6], question=f"What country contains {old}? Give only its country.",
                            before=country_old, after=country_old,
                            before_aliases=_country_aliases(country_old), after_aliases=_country_aliases(country_old))
    else:
        country_old, country_new = old, new
    for index in (3, 4):
        probes[index] = replace(probes[index], before=country_old, after=country_new,
                                before_aliases=_country_aliases(country_old), after_aliases=_country_aliases(country_new))
    return replace(scenario, probes=tuple(probes))


def interpret_paper(scenario, before, after):
    exact, edited, unedited, composed = scenario.probes[:4]
    checks = (
        ("Přesný editační prompt", probe_matches(exact, after[0], "after")),
        ("Parafráze editovaného kroku", probe_matches(edited, after[1], "after")),
        ("Needitovaný krok před i po", probe_matches(unedited, before[2], "before") and probe_matches(unedited, after[2], "after")),
        ("Samostatná dvojkroková otázka", probe_matches(composed, after[3], "after")),
        ("Dvojkroková parafráze", probe_matches(scenario.probes[4], after[4], "after")),
    )
    report = " · ".join(f"**{label}: {'ano' if passed else 'ne'}**" for label, passed in checks)
    if not checks[0][1]:
        verdict = "Přímý edit se nepotvrdil; chybu zatím nelze připsat skládání kroků."
    elif not probe_matches(exact, before[0], "before") or not probe_matches(edited, before[1], "before"):
        verdict = "Výchozí editovaný fakt nebyl ověřen ve všech formulacích."
    elif not checks[1][1]:
        verdict = "Edit funguje na svém promptu, ale nepřenesl se na parafrázi."
    elif not checks[2][1]:
        verdict = "Needitovaný krok nebyl znám nebo se změnil; nejde o izolovanou chybu řetězení."
    elif not probe_matches(composed, before[3], "before"):
        verdict = "Model nevyřešil původní dvojkrokovou otázku; chybí ověřený základ pro srovnání."
    elif checks[3][1] and not checks[4][1] and probe_matches(scenario.probes[4], before[4], "before"):
        verdict = "Hlavní dvojkroková otázka uspěla, ale její parafráze selhala."
    elif checks[3][1]:
        verdict = "Samostatná dvojkroková odpověď odpovídá důsledku editace."
    elif scenario.paper_hop == 1 and probe_matches(edited, after[3], "after"):
        verdict = "Oba jednotlivé kroky fungují, ale model místo země vrací mezivýsledek prvního kroku."
    elif probe_matches(composed, after[3], "before"):
        verdict = "Oba jednotlivé kroky fungují, ale složená otázka zůstala u původní země."
    else:
        verdict = "Oba jednotlivé kroky fungují, ale jejich samostatné složení selhalo."
    return f"{report}\n\n{verdict}"


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
            prompt = paper_prompt(probe.question) if scenario.paper_hop and scenario.use_context and index != 0 else probe.question
            answers.append(self.session.answer(prompt, generate_completion, max_new_tokens=16) if scenario.paper_hop else
                           self.session.answer(prompt, generate_completion))
            progress((index + 1) / total, desc=f"Vícekroková ukázka · hotovo {index + 1}/{total}")
        city = observed_city(answers[scenario.first_hop_index])
        if city is None:
            return answers, ("Druhý krok nelze sestavit z odpovědi modelu.", "První odpověď není stručný název.")
        question = scenario.second_hop_template.format(city)
        progress(len(scenario.probes) / (len(scenario.probes) + 1), desc=f"Vícekroková ukázka · druhý krok pro {city}")
        prompt = paper_prompt(question) if scenario.paper_hop and scenario.use_context else question
        answer = (self.session.answer(prompt, generate_completion, max_new_tokens=16) if scenario.paper_hop else
                  self.session.answer(prompt, generate_completion))
        return answers, (question, answer)

    def baseline_problem(self, scenario, before, progress):
        failed = [i for i in range(4) if not probe_matches(scenario.probes[i], before[i], "before")]
        if failed:
            query = scenario.probes[scenario.implicit_index]
            plain = self.session.answer(query.question, generate_completion, max_new_tokens=16)
            rows = [[p.category, p.question, p.before, p.after, answer, "—"]
                    for p, answer in zip(scenario.probes, before)]
            rows.append(["Výchozí kontrola · bez Q/A kontextu", query.question,
                         query.before, query.before, plain, "—"])
            context_note = ("Bez Q/A kontextu je odpověď správná: problém závisí na formulaci či kontextu."
                            if probe_matches(query, plain, "before") else
                            "Ani bez Q/A kontextu model tuto složenou otázku nezodpověděl správně.")
            if not scenario.use_context:
                context_note = "Tento scénář už používá přímé chatové otázky bez přidaného Q/A kontextu."
            summary = ("### Výchozí znalost není ověřená · ROME se nespustilo\n\n"
                       + "Neprošly: " + ", ".join(scenario.probes[i].category for i in failed) + ".\n\n"
                       + f"Složená otázka před editací: **{before[scenario.implicit_index]}**. "
                       + f"Bez Q/A kontextu: **{plain}**. {context_note}\n\n"
                       + "Nesprávná odpověď před i po by nebyla důkazem zhoršení po ROME. "
                       "Zvolte příklad, který model v základu zná, nebo nejprve ověřte formulaci otázky.")
            progress(1., desc="Výchozí kontrola hotová · editace nebyla provedena")
            return summary, rows, model_status("Výchozí znalost neověřena · původní váhy zachovány", "warn")
        return None

    def run(self, scenario_key, progress=gr.Progress()):
        if scenario_key not in MULTIHOP_SCENARIOS:
            raise ValueError("Vyberte připravený scénář.")
        scenario = MULTIHOP_SCENARIOS[scenario_key]
        def validate_active():
            active = scenario_for_fact(self.session.active_fact)
            if self.session.active_edit and active != scenario_key:
                raise ValueError("Aktivní editace nepatří tomuto scénáři. Vyberte odpovídající scénář nebo připravte fakt v záložce 2.")
        with self.session.transaction(validate=validate_active):
            if scenario.paper_hop:
                scenario = _paper_scenario_for_fact(scenario, self.session.active_fact or scenario.fact)
            elif self.session.active_fact:
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
            had_active_edit = self.session.active_edit
            with self.session.baseline():
                before, before_chain = self._measure(scenario, lambda f, **kw: progress(.2 * f, **kw))
                if scenario.paper_hop and not had_active_edit:
                    problem = self.baseline_problem(scenario, before, progress)
                    if problem:
                        return problem
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
        direct = scenario.probes[0]
        implicit = scenario.probes[scenario.implicit_index]
        rows = [[probe.category, probe.question, probe.before, probe.after, old, new]
                for probe, old, new in zip(scenario.probes, before, after)]
        chain = ["Rozklad přes skutečný mezivýsledek · kontrola", f"Před: {before_chain[0]}\nPo: {after_chain[0]}",
                 implicit.before, implicit.after, before_chain[1], after_chain[1]]
        rows.insert(len(rows) if scenario.paper_hop else 1, chain)
        interpretation = interpret_paper(scenario, before, after) if scenario.paper_hop else interpret_multihop(scenario, before, after)
        summary = (("### Dva kroky · edit prvního kroku\n\n" if scenario.paper_hop == 1 else
                    "### Dva kroky · edit druhého kroku\n\n" if scenario.paper_hop == 2 else
                    "### Jedna editace, dva až tři kroky uvažování\n\n")
                   + ("Používám již připravenou editaci; ROME se neopakovalo. " if reused else "Editace je připravena i v ostatních záložkách. ")
                   + observed_verdict(after[0], direct.after) + "\n\n" + interpretation
                   + "\n\nRozklad odpovědi je pomocná kontrola; samostatné dvojkrokové otázky se vyhodnocují odděleně.")
        progress(1., desc="Vícekrokové odpovědi připraveny")
        return summary, rows, model_status("Vícekroková ukázka hotová · bez opakování editace" if reused else "Scénář připraven.")
