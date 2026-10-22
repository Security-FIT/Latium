"""Full workshop callbacks: scores, paraphrases, canaries and edit history."""

import logging

import gradio as gr

from .chat import generate_reply
from .facts import probe_specs, workshop_fact
from .model import gpu_memory_summary
from .presentation import compact, status, ui_input
from .probing import generate_completion, score_target
from .rome import apply_rome_edit


def model_status(message, tone="ok"):
    return status(message, tone, workshop=True)


class WorkshopActions:
    def __init__(self, session):
        self.session = session

    def read_answers(self, specs):
        return {prompt: generate_completion(self.session.handler, prompt) for _, prompt in specs}

    def scores(self, prompt, fact):
        handler = self.session.handler
        return tuple(score_target(handler, prompt, target) / max(
            1, len(handler.tokenizer.encode(target.strip(), add_special_tokens=False))) for target in fact[2:])

    def history_rows(self):
        return [row.copy() for row in self.session.history]

    def inspect(self, subject, prompt, target_new, target_true, variants, canaries, progress=gr.Progress()):
        fact, rendered_prompt = ui_input(workshop_fact, subject, prompt, target_new, target_true)
        specs = probe_specs(rendered_prompt, variants, canaries)
        progress(0.15, desc="Zjišťuji aktuální odpovědi modelu…")
        with self.session.lock:
            answers = self.read_answers(specs)
            new_score, true_score = self.scores(rendered_prompt, fact)
            memory = gpu_memory_summary()
            history = self.history_rows()
        rows = [[category, item, answers[item], "—", "—"] for category, item in specs]
        winner = fact[2].strip() if new_score > true_score else fact[3].strip()
        summary = f"""### Aktuální stav modelu

**Otázka:** {compact(rendered_prompt)}<br>
**Odpověď modelu:** {compact(answers[rendered_prompt]) or "—"}

| Kandidát | Log-skóre / token |
|---|---:|
| Nová odpověď — **{compact(fact[2].strip())}** | {new_score:.3f} |
| Původní odpověď — **{compact(fact[3].strip())}** | {true_score:.3f} |

Z těchto dvou kandidátů má **{compact(winner)}** vyšší průměrné log-skóre na token. Nejde o ověření celé vygenerované odpovědi.
"""
        progress(1.0, desc="Náhled je hotový.")
        return summary, rows, history, model_status(f"Model připraven · {memory}")

    def edit(self, subject, prompt, target_new, target_true, variants, canaries, progress=gr.Progress()):
        fact, rendered_prompt = ui_input(workshop_fact, subject, prompt, target_new, target_true)
        specs = probe_specs(rendered_prompt, variants, canaries)
        try:
            progress(0.04, desc="Obnovuji čistý základní model…")
            with self.session.editing():
                # A failed replacement also leaves no previous edit active.
                for previous in self.session.history:
                    previous[4] = "Nahrazena"
                progress(0.10, desc="Měřím stav základního modelu před editací…")
                before = self.read_answers(specs)
                before_new, before_true = self.scores(rendered_prompt, fact)
                progress(0.28, desc="Počítám ROME editaci. Na T4 to může několik minut trvat…")
                result = apply_rome_edit(self.session.handler, fact)
                progress(0.82, desc="Ověřuji výsledek a kontrolní otázky…")
                after = self.read_answers(specs)
                after_new, after_true = self.scores(rendered_prompt, fact)
                memory = gpu_memory_summary()
                success = after_new > after_true
                self.session.history.append([
                    len(self.session.history) + 1, fact[1], fact[3].strip(), fact[2].strip(),
                    "Aktivní · úspěch" if success else "Aktivní · neověřeno",
                    round(result.seconds, 1), round(result.peak_gb, 2),
                ])
                history = self.history_rows()
        except Exception as exc:
            logging.exception("ROME editace z webového rozhraní selhala")
            raise gr.Error(f"Editace se nepodařila: {exc}") from exc

        rows = [[category, item, before[item], after[item],
                 "Ano" if before[item].strip() != after[item].strip() else "Ne"] for category, item in specs]
        icon = "✅" if success else "⚠️"
        verdict = ("Nová odpověď má vyšší skóre než původní." if success else
                   "Nová odpověď zatím nepřekonala původní. Prohlédněte si skóre a odpovědi níže.")
        summary = f"""### {icon} Výsledek editace

**{compact(fact[1])}**: {compact(fact[3].strip())} → **{compact(fact[2].strip())}**<br>
{verdict}

| Měření | Před editací | Po editaci |
|---|---:|---:|
| Log-skóre/token nové odpovědi | {before_new:.3f} | **{after_new:.3f}** |
| Log-skóre/token původní odpovědi | {before_true:.3f} | {after_true:.3f} |

Editace trvala **{result.seconds:.1f} s**, norma změny je **{result.delta_norm:.2f}** a špička GPU byla
**{result.peak_gb:.2f} GB**. Toto je jediná aktivní editace. Při další editaci se nejdřív automaticky
obnoví základní model, takže se změny nevrství.
"""
        progress(1.0, desc="Editace je hotová.")
        tone = "ok" if success else "warn"
        return summary, rows, history, model_status(f"Aktivní je pouze tato editace · {memory}", tone), []

    def chat(self, message, history_items, system_prompt, max_new_tokens, progress=gr.Progress()):
        message = (message or "").strip()
        if not message:
            raise gr.Error("Nejdřív napište zprávu pro model.")
        history_items = list(history_items or [])
        try:
            progress(0.15, desc="Model připravuje odpověď…")
            answer = generate_reply(
                self.session, message, history_items, system_prompt, max_new_tokens=max_new_tokens,
                history_limit=10, repetition_penalty=1.05,
            )
        except Exception as exc:
            logging.exception("Chat s modelem selhal")
            raise gr.Error(f"Model nedokázal odpovědět: {exc}") from exc
        if not answer:
            return message, history_items, model_status("Model nevrátil textovou odpověď; zkuste vyšší délku nebo jinou otázku.", "warn")
        updated = history_items + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        progress(1.0, desc="Odpověď je hotová.")
        return "", updated, model_status(f"Model odpověděl · {gpu_memory_summary()}")

    @staticmethod
    def clear_chat():
        return "", [], model_status(f"Konverzace vymazána · {gpu_memory_summary()}")

    def reset(self):
        self.session.reset()
        return (
            "### Původní model byl obnoven\n\nVšechny editace z této relace byly odstraněny.",
            [], [], model_status(f"Původní model obnoven · {gpu_memory_summary()}"), [],
        )
