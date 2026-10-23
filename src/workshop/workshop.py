"""Presentation callbacks reuse one installed edit and cached generated answers."""

import gradio as gr

from .answers import answer_matches, observed_verdict
from .chat import generate_reply
from .facts import probe_specs, workshop_fact
from .model import gpu_memory_summary
from .presentation import compact, status, ui_input
from .probing import generate_completion
from .rome import apply_rome_edit


def model_status(message, tone="ok"):
    return status(message, tone, workshop=True)


class WorkshopActions:
    def __init__(self, session):
        self.session = session

    def read_answers(self, specs, progress=None):
        answers = {}
        for index, (_, prompt) in enumerate(specs):
            if progress:
                progress(index / max(1, len(specs)), desc=f"Ověřuji odpovědi · hotovo {index}/{len(specs)}")
            answers[prompt] = self.session.answer(prompt, generate_completion)
            if progress:
                progress((index + 1) / max(1, len(specs)), desc=f"Ověřuji odpovědi · hotovo {index + 1}/{len(specs)}")
        return answers

    def history_rows(self):
        return [row.copy() for row in self.session.history]

    def inspect(self, subject, prompt, target_new, target_true, variants, canaries, progress=gr.Progress()):
        _, rendered = ui_input(workshop_fact, subject, prompt, target_new, target_true)
        specs = probe_specs(rendered, variants, canaries)
        with self.session.lock:
            answers = self.read_answers(specs, progress)
        summary = f"### Aktuální odpověď\n\n**{compact(answers[rendered]) or 'Bez odpovědi'}**"
        return summary, [[category, item, answers[item], "—", "—"] for category, item in specs], self.history_rows(), model_status("Náhled připraven.")

    def edit(self, subject, prompt, target_new, target_true, variants, canaries, progress=gr.Progress(), *, check_multihop=True):
        fact, rendered = ui_input(workshop_fact, subject, prompt, target_new, target_true)
        specs = probe_specs(rendered, variants, canaries)
        with self.session.transaction():
            # A failed original 2-hop question cannot demonstrate damage from editing.
            from .lessons import MULTIHOP_SCENARIOS, MultiHopActions, _paper_scenario_for_fact, scenario_for_fact
            scenario_key = scenario_for_fact(fact)
            scenario = MULTIHOP_SCENARIOS.get(scenario_key)
            if check_multihop and scenario is not None and scenario.paper_hop and not self.session.matches(fact):
                lesson = MultiHopActions(self.session)
                scenario = _paper_scenario_for_fact(scenario, fact)
                with self.session.baseline():
                    baseline, _ = lesson._measure(scenario, lambda f, **kw: progress(.1 * f, **kw))
                    problem = lesson.baseline_problem(scenario, baseline, progress)
                if problem:
                    summary, measured, state = problem
                    rows = [[kind, question, old, new, "—"] for kind, question, _, _, old, new in measured]
                    return summary, rows, self.history_rows(), state, []
            with self.session.baseline():
                before = self.read_answers(specs, lambda fraction, **kw: progress(.1 * fraction, **kw))
            result, reused = self.session.install(fact, apply_rome_edit,
                lambda fraction, **kw: progress(.1 + .75 * fraction, **kw))
            after = self.read_answers(specs, lambda fraction, **kw: progress(.85 + .15 * fraction, **kw))
            if not reused:
                for previous in self.session.history:
                    previous[4] = "Nahrazena"
                self.session.history.append([len(self.session.history) + 1, fact[1], fact[3].strip(), fact[2].strip(),
                    "Aktivní", round(result.seconds, 1), round(result.peak_gb, 2)])
        rows = [[category, item, before[item], after[item], "Ano" if before[item] != after[item] else "Ne"]
                for category, item in specs]
        summary = (f"### Výsledek editace\n\n**{compact(fact[1])}**: {compact(fact[3].strip())} → **{compact(fact[2].strip())}**\n\n"
                   f"**Model odpověděl:** {compact(after[rendered]) or 'Bez odpovědi'}\n\n"
                   f"{observed_verdict(after[rendered], fact[2])}\n\n"
                   + ("Použita již připravená editace. ROME se znovu nepočítalo." if reused else "Editace je aktivní i pro chat a vícekrokové otázky."))
        progress(1., desc="Editace a odpovědi jsou připravené")
        return summary, rows, self.history_rows(), model_status("Připraveno · použita cache" if reused else "Editace dokončena.",
            "ok" if answer_matches(after[rendered], fact[2]) else "warn"), []

    def chat(self, message, history_items, system_prompt, max_new_tokens, progress=gr.Progress()):
        message = (message or "").strip()
        if not message:
            raise gr.Error("Nejdřív napište zprávu pro model.")
        history_items = list(history_items or [])
        answer = generate_reply(self.session, message, history_items, system_prompt,
                                max_new_tokens=max_new_tokens, history_limit=10, repetition_penalty=1.05)
        if not answer:
            return message, history_items, model_status("Model nevrátil odpověď; zkuste otázku znovu.", "warn")
        history = history_items + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        return "", history, model_status(f"Model odpověděl · {gpu_memory_summary()}")

    @staticmethod
    def clear_chat():
        return "", [], model_status("Rozhovor vymazán.")

    def reset(self):
        self.session.reset()
        return "### Původní model obnoven", [], [], model_status("Původní váhy obnoveny; model zůstává v GPU paměti."), []
