"""Streaming solo callbacks, separate from layout and model lifecycle."""

import logging

import gradio as gr

from .chat import generate_reply
from .facts import fact_input, prompt_input
from .model import gpu_memory_summary
from .presentation import compact, status, ui_input
from .presets import SOLO_CHECKS
from .probing import generate_completion
from .rome import apply_rome_edit


def rows(before, after, prompt):
    return [["Upravovaný fakt" if item == prompt else "Kontrola", item,
             before[item], after[item] if after else "—"] for item in before]


class SoloActions:
    def __init__(self, session):
        self.session = session

    def read_probes(self, prompt):
        prompts = [prompt] + [check for check in SOLO_CHECKS if check != prompt]
        return {item: generate_completion(self.session.handler, item) for item in prompts}

    def preview(self, subject, prompt, target_new, target_old):
        _, _, prompt = ui_input(prompt_input, subject, prompt)
        yield gr.skip(), gr.skip(), status("Zjišťuji aktuální odpovědi modelu…", "busy")
        with self.session.lock:
            current = self.read_probes(prompt)
            memory = gpu_memory_summary()
            label = "aktuálně upraveného" if self.session.active_edit else "původního"
        summary = (
            f"### Náhled {label} modelu\n\n"
            f"**{compact(prompt)}** → **{compact(current[prompt]) or '—'}**\n\n"
            "Původní odpověď porovnejte s náhledem. Tlačítko „Vložit fakt“ pak zkusí editaci."
        )
        yield summary, rows(current, None, prompt), status(f"Náhled hotový · {memory}")

    def edit(self, subject, prompt, target_new, target_old):
        fact, prompt = ui_input(fact_input, subject, prompt, target_new, target_old)
        keep = gr.skip()
        yield keep, keep, status("Obnovuji původní model…", "busy"), keep, keep
        try:
            with self.session.editing():
                yield keep, keep, status("Měřím odpovědi před editací…", "busy"), keep, keep
                before = self.read_probes(prompt)
                yield keep, keep, status("Provádím ROME editaci; může trvat několik minut…", "busy"), keep, keep
                result = apply_rome_edit(self.session.handler, fact)
                yield keep, keep, status("Ověřuji výsledek a kontrolní otázky…", "busy"), keep, keep
                after = self.read_probes(prompt)
                memory = gpu_memory_summary()
        except Exception as exc:
            logging.exception("Editace v samostatném demu selhala")
            yield (
                f"### Editace se nepodařila\n\n{compact(exc)}\n\nPůvodní model byl obnoven.",
                [], status(f"Editace selhala · {gpu_memory_summary()}", "warn"), "", [],
            )
            return

        drift = sum(before[item] != after[item] for item in SOLO_CHECKS if item in before)
        target_new = fact[2].strip()
        target_reached = target_new.casefold() in after[prompt].casefold()
        summary = (
            f"### Výsledek\n\n"
            f"**Před:** {compact(before[prompt]) or '—'}  \n"
            f"**Po:** {compact(after[prompt]) or '—'}\n\n"
            f"Cíl: **{compact(target_new)}** · "
            f"Změněné kontrolní odpovědi: **{drift}/{len(SOLO_CHECKS)}** · "
            f"Čas: **{result.seconds:.0f} s**"
        )
        if not target_reached:
            summary += "\n\nCílová odpověď se v doplnění neobjevila; zkuste jiný začátek tvrzení nebo cíl."
        tone = "warn" if drift or not target_reached else "ok"
        state_label = "Editace proběhla, ale cíl se neobjevil" if not target_reached else "Aktivní je jedna editace"
        yield summary, rows(before, after, prompt), status(f"{state_label} · {memory}", tone), "", []

    def reset(self):
        self.session.reset()
        return (
            "### Původní model obnoven\n\nAktivní editace byla odstraněna.", [],
            status(f"Původní model · {gpu_memory_summary()}"), "", [],
        )

    def chat(self, message, history_items):
        message = (message or "").strip()
        if not message:
            raise gr.Error("Nejprve napište zprávu.")
        history_items = list(history_items or [])
        pending = history_items + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Připravuji odpověď…"},
        ]
        yield "", pending
        try:
            answer = generate_reply(self.session, message, history_items)
        except Exception as exc:
            logging.exception("Rozhovor s modelem selhal")
            answer = f"Odpověď se nepodařila: {exc}"
        if not answer:
            answer = "Model nevrátil běžnou odpověď. Zkuste otázku formulovat jinak."
        yield "", history_items + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]

    @staticmethod
    def clear_chat():
        return "", []
