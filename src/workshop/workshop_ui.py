"""Five Czech lessons sharing one model, serialized operations and current state."""

# Gradio discovers Progress defaults; nested contexts mirror the visible layout.
# ruff: noqa: B008, SIM117

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

import gradio as gr
from omegaconf import OmegaConf

from .assets import original_brandbar, stylesheet
from .chat import default_system_prompt, generate_reply
from .covariance import (
    DEFAULT_COVARIANCE_DRIVE_URL,
    ensure_covariance,
    validate_covariance_download,
)
from .diagnostics import DiagnosticActions, _trace_rainbow
from .gram import RomeLayerLocalizer, detect_from_profiles
from .lessons import (
    DEFAULT_MULTIHOP_SCENARIO,
    DIRECT_ANSWER_CONTEXT,
    MULTIHOP_SCENARIOS,
    PAPER_URL,
    MultiHopActions,
    scenario_description,
    scenario_for_fact,
)
from .model import ModelSettings, gpu_memory_summary, validate_model_settings
from .presentation import compact
from .presets import DEFAULT_WORKSHOP_PRESET as DEFAULT_PRESET, WORKSHOP_CANARIES, WORKSHOP_PRESETS
from .probing import generate_completion, score_target
from .prompts import render_chat, render_prompt
from .rome import apply_rome_edit, gather_k, insert_kv
from .results import render_result_rows
from .session import load_model, runtime_session, session_for
from .operations import stream_operation
from .setup import Environment
from .source_panel import source_panel
from .highlights import ROME_HIGHLIGHTS, TRACE_HIGHLIGHTS, MULTIHOP_HIGHLIGHTS, CUSTOM_HIGHLIGHTS, GRAM_HIGHLIGHTS
from .telemetry import gpu_usage_html
from .themes import solo_theme
from .tracing import corrupt_hook, restore_hook
from .workshop import WorkshopActions, model_status

LOGGER = logging.getLogger(__name__)
TAB_LABELS = (
    "1 · Kauzální trasování", "2 · ROME: připravené příklady", "3 · ROME: vícekrokové otázky",
    "4 · ROME: vlastní fakt a chat", "5 · GRAM: detekce editace",
)
RESULT_TABS = ("trace", "prepared", "multihop", "custom", "gram")
ROME_SOURCES = [
    ("ROME · celý krok editace", apply_rome_edit),
    ("ROME · klíč k", gather_k),
    ("ROME · optimalizace v", "src.rome.common", "optimize_v"),
    ("ROME · vložení do vah", insert_kv),
]


class WorkshopCoordinator:
    """Coordinate visible invalidation around existing model actions.

    All bound model callbacks use the same Gradio concurrency_id. Conversation
    lives here, not in an event's submitted Chatbot value: a chat queued before
    an edit cannot resurrect its old context when it finally starts.
    """

    def __init__(self, session):
        self.session = session
        self.actions = WorkshopActions(session)
        self.multihop = MultiHopActions(session)
        self.conversation = []
        self.revision = 0
        self.last_system_prompt = None
        self.last_fact = session.active_fact

    def current_state(self):
        if self.session.handler is None:
            return "**Rozhraní je připravené.** Model zatím není načtený; použijte tlačítko „Připravit model“."
        if self.session.active_fact:
            fact = self.session.active_fact
            return f"Model je v GPU paměti · **{compact(fact[1])}: {compact(fact[3].strip())} → {compact(fact[2].strip())}** · stejná editace platí ve všech záložkách."
        if self.session.active_edit:
            return "Model je v GPU paměti · je aktivní vlastní editace."
        return "Model je v GPU paměti · původní znalosti. Obnova faktu znovu nenačítá model."

    def histories(self):
        rows = self.actions.history_rows()
        return {"prepared_history": rows, "custom_history": rows}

    def invalidate(self):
        self.revision += 1
        self.conversation = []
        result = {"chatbot": [], "chat_input": "", "prepared_chatbot": [], "prepared_chat_input": "",
                  "trace_plot": None, "gram_plot": None}
        for tab in RESULT_TABS:
            result[f"{tab}_summary"] = ""
            result[f"{tab}_rows"] = []
        result.update(self.histories())
        result["current_state"] = self.current_state()
        result["system_status"] = model_status("Provádím operaci a ověřuji aktuální model…", "busy")
        return result

    def mutate(self, tab, *inputs, progress=gr.Progress()):
        """Keep progress visible; invalidate other lessons only if weights change."""
        values = {}
        changing = False
        defer_invalidation = False
        initial_key = self.session._active_key
        try:
            self.ensure_ready()
            if tab == "multihop":
                scenario = MULTIHOP_SCENARIOS[inputs[0]]
                if self.session.active_edit and scenario_for_fact(self.session.active_fact) != inputs[0]:
                    raise ValueError("Vyberte scénář odpovídající aktivní editaci ze záložky 2.")
                changing = not self.session.active_edit
                defer_invalidation = bool(scenario.paper_hop and changing)
            elif tab == "reset":
                changing = self.session.active_edit
            else:
                from .facts import workshop_fact
                fact, _ = workshop_fact(*inputs[:4])
                changing = not self.session.matches(fact)
                selected = MULTIHOP_SCENARIOS.get(scenario_for_fact(fact))
                defer_invalidation = bool(changing and selected and selected.paper_hop)
            if changing and not defer_invalidation:
                values = self.invalidate()
            yield values | {"system_status": model_status("Připravuji odpovědi…", "busy")}

            def execute(report):
                if tab == "reset":
                    return self.actions.reset()
                if tab == "multihop":
                    return self.multihop.run(*inputs, progress=report)
                return self.actions.edit(*inputs, progress=report)

            for state, result, done in stream_operation(execute, "Pracuji s aktuálním modelem…"):
                yield {"system_status": state}
                if done:
                    if defer_invalidation and self.session._active_key != initial_key:
                        values.update(self.invalidate())
                    if tab == "reset":
                        summary, rows, _, status, _ = result
                        for name in RESULT_TABS:
                            values[f"{name}_summary"], values[f"{name}_rows"] = summary, rows
                    elif tab == "multihop":
                        summary, rows, status = result
                        values.update(multihop_summary=summary, multihop_rows=rows)
                    else:
                        summary, rows, _, status, _ = result
                        values.update({f"{tab}_summary": summary, f"{tab}_rows": rows})
                    values["system_status"] = state + status
        except Exception as exc:
            LOGGER.exception("Workshop operation failed")
            if not changing and self.session._active_key != initial_key:
                values.update(self.invalidate())
            error = f"### Operace se nepodařila\n\n{compact(str(exc))}\n\n{self.current_state()}"
            if tab != "reset":
                values[f"{tab}_summary"] = error
            values["system_status"] = model_status(f"Operace se nepodařila: {exc}", "warn")
        values.update(self.histories())
        values["current_state"] = self.current_state()
        scenario_key = scenario_for_fact(self.session.active_fact)
        if scenario_key:
            values["multihop_scenario"] = scenario_key
        if self.session.active_fact:
            self.last_fact = self.session.active_fact
        if self.last_fact:
            fact = self.last_fact
            values.update(trace_subject_input=fact[1], trace_prompt_input=fact[0].format(fact[1]),
                          trace_target_input=fact[2 if self.session.active_edit else 3].strip())
        yield values

    def inspect(self, tab, *inputs, progress=gr.Progress()):
        self.ensure_ready()
        summary, rows, _, status = self.actions.inspect(*inputs, progress=progress)
        return {
            f"{tab}_summary": summary,
            f"{tab}_rows": rows, "system_status": status, **self.histories(),
        }

    def chat(self, message, system_prompt, max_new_tokens, progress=gr.Progress()):
        self.ensure_ready()
        if self.last_system_prompt is not None and system_prompt != self.last_system_prompt:
            self.conversation = []
        self.last_system_prompt = system_prompt
        text, history, status = self.actions.chat(
            message, self.conversation, system_prompt, max_new_tokens, progress=progress,
        )
        self.conversation = history
        return text, history, status

    def clear_chat(self):
        self.conversation = []
        return self.actions.clear_chat()

    def ensure_ready(self):
        if self.session.handler is None:
            raise gr.Error("Nejprve připravte model tlačítkem nahoře.")


def _fact_form(default):
    example = WORKSHOP_PRESETS["Autor Hamleta → Tolkien"]
    subject = gr.Textbox(value=default[0], label="Co chceme editovat?", placeholder=example[0])
    prompt = gr.Textbox(
        value=default[1], label="Jak se na fakt modelu zeptáme?",
        placeholder=example[1],
        info="Otázka musí obsahovat přesný text entity. Nejlépe fungují anglické formulace.",
    )
    with gr.Row():
        target_true = gr.Textbox(value=default[3], label="Jaká byla původní odpověď?", placeholder=example[3])
        target_new = gr.Textbox(value=default[2], label="Co má model nově odpovídat?", placeholder=example[2])
    with gr.Accordion("Parafráze a kontrolní otázky", open=False):
        variants = gr.Textbox(value=default[4], lines=4, label="Jiné formulace stejné otázky",
                              placeholder=example[4], info="Jedna na řádek.")
        canaries = gr.Textbox(value=WORKSHOP_CANARIES, lines=5, label="Kontrolní otázky", info="Měly by zůstat stejné.")
    return [subject, prompt, target_new, target_true, variants, canaries]


def _fact_results(components, tab):
    components[f"{tab}_summary"] = gr.Markdown("Zatím nebyla spuštěna kontrola ani editace.", elem_classes=["workshop-summary"])
    components[f"{tab}_rows"] = gr.HTML(render_result_rows(f"{tab}_rows", []), elem_id=f"{tab}-results")
    with gr.Accordion("Historie editací společné session", open=False):
        gr.Markdown("Aktivní je nejvýše poslední editace. Historie zahrnuje i ostatní lekce; reset ji smaže.")
        components[f"{tab}_history"] = gr.HTML(render_result_rows(f"{tab}_history", []), elem_id=f"{tab}-history")


def _workshop_chat_form(components, prefix, handler, shared_system, shared_tokens):
    gr.Markdown("### Rozhovor s modelem\n\nChat používá aktuální váhy. Každá editace či reset vymaže oba rozhovory.")
    components[prefix + "chatbot"] = gr.Chatbot(value=[], label="Odpovědi modelu", layout="bubble",
                                                height=300, elem_classes=["original-chat"])
    components[prefix + "chat_input"] = gr.Textbox(label="Vaše zpráva", lines=1, max_lines=3,
                                                   placeholder="Where is the Eiffel Tower?")
    with gr.Row():
        send = gr.Button("Odeslat zprávu", variant="primary", interactive=handler is not None)
        clear = gr.Button("Vymazat konverzaci", variant="secondary")
    return send, clear, shared_system, shared_tokens


def _loaded_configuration(handler):
    if handler is None:
        return None
    settings = handler.cfg.model
    values = (OmegaConf.to_container(settings, resolve=True) if OmegaConf.is_config(settings) else
              asdict(settings) if is_dataclass(settings) else vars(settings).copy())
    return dict(values, seed=getattr(handler.cfg, "seed", 42), loaded_dtype=str(getattr(handler, "dtype", "unknown")),
                projection=getattr(handler, "_layer_name_template", "unknown"), device=str(handler.device))


def build_workshop_ui(handler=None, environment=None, *, seed=42, covariance_drive_url=None):
    session = session_for(handler) if handler is not None else runtime_session()
    handler = session.handler
    coordinator = getattr(session, "_coordinator", None)
    if coordinator is None:
        coordinator = WorkshopCoordinator(session)
        session._coordinator = coordinator
    diagnostics = DiagnosticActions(session)
    default = WORKSHOP_PRESETS[DEFAULT_PRESET]
    initial_scenario = scenario_for_fact(session.active_fact) or DEFAULT_MULTIHOP_SCENARIO
    theme = solo_theme()
    components = {}
    # Shared across every model callback, rather than Gradio's per-function limit.
    model_event = {"concurrency_id": "latium-model", "concurrency_limit": 1, "show_progress": "hidden"}
    with gr.Blocks(title="Latium — workshop editace znalostí") as demo:
        demo._covariance_drive_url = (covariance_drive_url or "").strip() or DEFAULT_COVARIANCE_DRIVE_URL
        gr.HTML("""<section class="demo-hero">
          <p class="demo-eyebrow">Živá ukázka · injekce faktu · ROME</p>
          <h1>Co když někdo přepíše jediný fakt v jazykovém modelu?</h1>
          <p class="demo-lead">Injekce faktu je cílená změna znalosti uložené ve vahách modelu.
          Představte si, že model začne tvrdit, že Eiffelova věž stojí v Římě.
          Změníme jeden fakt a potom ověříme odpovědi i případné vedlejší změny.</p>
          <p class="demo-cta">↓ Níže si vyzkoušíte vlastní změnu</p>
        </section>""")
        components["current_state"] = gr.Markdown(coordinator.current_state(), elem_classes=["workshop-state"], height="auto", max_height=None)
        components["system_status"] = gr.HTML(model_status(
            f"Model je připraven · {gpu_memory_summary()}" if handler is not None else "Čekám na přípravu modelu."),
            elem_id="workshop-operation-status", autoscroll=False, max_height=None)
        gpu_panel = gr.HTML(gpu_usage_html())
        gpu_timer = gr.Timer(2)
        gpu_timer.tick(gpu_usage_html, outputs=gpu_panel, queue=False, show_progress="hidden")
        with gr.Accordion("Příprava modelu · Qwen3.5-4B", open=handler is None) as preparation:
            with gr.Row():
                drive_url = gr.Textbox(value=demo._covariance_drive_url, label="Google Drive odkaz na kovarianci", info="Předvyplněno z notebooku. Nový odkaz nahradí původní; prázdné pole zachová poslední odkaz.", scale=2, interactive=handler is None)
                with gr.Column(scale=1, min_width=220):
                    upload = gr.UploadButton("Nahrát vlastní kovarianci (.pt)", file_types=[".pt"], type="filepath", interactive=handler is None)
                    clear_upload = gr.Button("Zrušit výběr souboru · použít Drive", size="sm", interactive=handler is None)
            gr.Markdown("Pro soukromý Drive zapněte AUTHENTICATE_DRIVE v první buňce notebooku; "
                        "přihlášení proběhne před spuštěním Gradio. Nebo nahrajte soubor přímo.")
            with gr.Accordion("Výpočetní nastavení · přesnost a micro-batche", open=False):
                with gr.Row():
                    download_mode = gr.Dropdown(["auto", "public", "authenticated"], value="auto", label="Stažení z Drive", interactive=handler is None)
                    precision = gr.Dropdown(["auto", "bf16", "fp16", "fp32"], value="bf16", label="Přesnost modelu", interactive=handler is None)
                with gr.Row():
                    v_batch = gr.Slider(1, 8, value=getattr(handler.cfg.model, "v_batch_size", 1) if handler else 1, step=1, label="ROME micro-batch v")
                    k_batch = gr.Slider(1, 16, value=getattr(handler.cfg.model, "k_batch_size", 4) if handler else 4, step=1, label="ROME micro-batch k")
                    epochs = gr.Slider(2, 40, value=getattr(handler, "epochs", 15) if handler else 15, step=1, label="ROME iterace")
                    edit_seed = gr.Number(value=getattr(handler.cfg, "seed", 42) if handler is not None else seed,
                                          precision=0, minimum=0, maximum=2**32 - 1,
                                          label="Seed modelu a ROME", interactive=handler is None)
                profile = gr.Dropdown([("Rychlá ukázka · 8 šablon", "demo"), ("Plný experiment · 20 šablon", "full")],
                                      value="full" if handler and getattr(handler.cfg.model, "v_N", 8) > 8 else "demo", label="Režim ukázky")
                profile.change(lambda value: gr.update(value=15 if value == "demo" else 25), inputs=profile,
                               outputs=epochs, queue=False, show_progress="hidden")
                gr.Markdown("bf16 odpovídá konfiguraci Latium main i původnímu checkpointu. Auto také zachová bf16; "
                            "T4 ho počítá pomaleji bez nativních BF16 tensor cores. FP16 je ruční experimentální volba. "
                            "Delta a optimizer jsou vždy fp32, attention eager; kovariance patří vrstvě 5.")
                gr.Markdown("Nastavení ROME lze měnit i po načtení; použije se při další editaci bez nového stahování modelu.")
            initialize_button = gr.Button("Připravit model", variant="primary", interactive=handler is None)
            with gr.Accordion("Načtená konfigurace", open=False):
                actual_config = gr.JSON(label="Model v paměti a nastavení další editace", value=_loaded_configuration(handler), visible=handler is not None)
        reset_button = gr.Button("Obnovit původní model · všechny záložky", variant="secondary", interactive=handler is not None)
        with gr.Accordion("Společná nastavení obou chatů", open=False):
            shared_system = gr.Textbox(value=default_system_prompt(handler), label="Systémová instrukce", lines=2)
            shared_tokens = gr.Slider(16, 128, value=32, step=16, label="Maximální délka krátké odpovědi")
            gr.Markdown("Výchozí instrukce odpovídá editaci. Její změna mění pouze kontext chatu "
                        "a vyčistí společnou historii; nemění váhy modelu.")
        with gr.Tabs():
            with gr.Tab(TAB_LABELS[0]):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        gr.Markdown(
                            "### Kde model zpracovává tento fakt?\n\n"
                            "Poškodíme reprezentaci entity a po vrstvách obnovíme čisté aktivace. "
                            "Duhové sloupce ukážou, kde se pravděpodobnost cílové odpovědi obnovuje. "
                            "Měříme aktuální model; pro čistý základ použijte tlačítko obnovy nahoře."
                        )
                        with gr.Accordion("Jak číst rainbow graf a co znamená kandidátní vrstva?", open=False):
                            gr.Markdown("Každá vrstva má vlastní barvu. Výška sloupce je průměrný nepřímý efekt (IE) "
                                        "přes párové vzorky šumu; nula je beze změny, záporná hodnota snížení pravděpodobnosti cíle. "
                                        "Jde o obnovu oken MLP na posledním tokenu entity. "
                                        "Jeden prompt poskytuje průzkumnou oblast, nikoli důkaz nejlepší ROME vrstvy. "
                                        "Editace zůstává na vrstvě 5; její změna vyžaduje odpovídající kovarianci.")
                        trace_subject = gr.Textbox(value=default[0], label="Entita pro trasování")
                        trace_prompt = gr.Textbox(value=default[1], label="Prompt pro trasování")
                        trace_target = gr.Textbox(value=default[3], label="Sledovaná odpověď")
                        components.update(trace_subject_input=trace_subject, trace_prompt_input=trace_prompt, trace_target_input=trace_target)
                        with gr.Accordion("Parametry trasování", open=False):
                            window = gr.Slider(1, 9, value=3, step=2, label="Velikost okna vrstev (lichá)")
                            samples = gr.Slider(1, 20, value=3, step=1, label="Počet vzorků šumu")
                            noise = gr.Slider(0, 10, value=3, step=0.1, label="Násobek šumu")
                            seed = gr.Number(value=42, precision=0, label="Seed")
                            trace_facts = gr.Slider(1, 5, value=3, step=1, label="Počet faktů · 1 rychle, více stabilnější profil")
                            auto_noise = gr.Checkbox(value=True, label="Automaticky zvolit sílu poškození")
                        gr.Markdown("Výchozí graf shrnuje 3 fakta, každé se 3 vzorky šumu. Více faktů stabilizuje profil, ale vyžaduje více výpočtů.")
                        trace_button = gr.Button("Najít kandidátní vrstvu", variant="primary", interactive=handler is not None)
                    source_panel(TRACE_HIGHLIGHTS)
                components["trace_summary"] = gr.Markdown("Trasování zatím nebylo spuštěno.", elem_classes=["workshop-summary"])
                components["trace_plot"] = gr.Plot(label="Rainbow graf kauzálního obnovení")
                with gr.Accordion("Podrobné naměřené hodnoty", open=False):
                    components["trace_rows"] = gr.HTML(render_result_rows("trace_rows", []), elem_id="trace-results")
            with gr.Tab(TAB_LABELS[1]):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        gr.HTML("""<div class="workshop-intro"><section class="demo-highlights">
                          <article class="demo-card speed">
                            <p class="demo-kicker">01 · Rychlost</p><h2 class="demo-pr-heading">Pár minut</h2>
                            <p>Takto dlouho trvá změna faktu pomocí běžně dostupných zdrojů. Úplné doladění modelu
                            obvykle vyžaduje mnohem více výpočtů, přípravy a často hodiny až dny práce.</p>
                          </article>
                          <article class="demo-card scope">
                            <p class="demo-kicker">02 · Rozsah</p><h2 class="demo-pr-heading">Jediná cílená editace</h2>
                            <p>Upravíme jednu MLP projekci. Ostatní váhy zůstanou stejné, jiné odpovědi se ale mohou změnit; proto měříme kontroly.</p>
                          </article>
                        </section></div>""")
                        gr.Markdown(
                            "### Jedna připravená změna znalosti\n\n"
                            "Vyberte ověřitelný příklad. Entitu, prompt i odpovědi můžete upravit. "
                            "Při editaci nejprve obnovíme základ, změříme odpovědi, provedeme jednu ROME editaci "
                            "a zopakujeme otázky i kontroly. Editace se nevrství."
                        )
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=5):
                                preset = gr.Dropdown(
                                    choices=[name for name in WORKSHOP_PRESETS if name != "Vlastní fakt"],
                                    value=DEFAULT_PRESET, label="Připravený příklad",
                                )
                                prepared_inputs = _fact_form(default)
                                with gr.Row():
                                    prepared_inspect = gr.Button("Ověřit současný stav", variant="secondary", interactive=handler is not None)
                                    prepared_edit = gr.Button("Provést připravenou editaci", variant="primary", interactive=handler is not None)
                            with gr.Column(scale=6):
                                prepared_send, prepared_clear, prepared_system, prepared_tokens = _workshop_chat_form(
                                    components, "prepared_", handler, shared_system, shared_tokens)
                    source_panel(ROME_HIGHLIGHTS)
                _fact_results(components, "prepared")
            with gr.Tab(TAB_LABELS[2]):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        gr.Markdown(
                            "### Dva kroky: kde se edit přestane přenášet?\n\n"
                            "Ověřte přesný editační prompt, jeho parafrázi, needitovaný krok a samostatné "
                            "dvojkrokové otázky. Porovnáme edit prvního a druhého kroku. "
                            "Pro jiný scénář nejprve připravte odpovídající fakt v záložce 2. "
                            "Rozklad přes odpověď modelu slouží jako pomocná kontrola."
                        )
                        scenario = gr.Dropdown(
                            choices=[(item.chain, key) for key, item in MULTIHOP_SCENARIOS.items()],
                            value=initial_scenario, label="Vícekrokový scénář",
                        )
                        components["multihop_scenario"] = scenario
                        description = gr.Markdown(scenario_description(initial_scenario))
                        with gr.Accordion("Protokol měření", open=False):
                            gr.Markdown(
                                f"Nové scénáře adaptují tabulku 1 a §5.1 [studie]({PAPER_URL}). "
                                "Společný kontext níže určuje stručný formát odpovědi; dvojkrokový dotaz "
                                "nedostává mezivýsledek ani editovaný cíl. Starší městské scénáře tento kontext nepoužívají.\n\n"
                                "Měříme Qwen3.5-4B ve vrstvě 5. Původní studie testovala GPT-J-6B i jiné hloubky; "
                                "toto demo neověřuje mechanismus hopping-too-late ani redundantní editování. "
                                "Úspěch i selhání otázky jsou možné výsledky.\n\n"
                                "Seed je výchozí 42 a před každou editací se obnoví. Odpovědi jsou greedy bez thinkingu. "
                                "Pro srovnání zachovejte stejný commit, GPU a výpočetní nastavení."
                            )
                            gr.Code(value=DIRECT_ANSWER_CONTEXT + "Q: <otázka>\nA:", language=None, interactive=False, wrap_lines=True)
                        multihop_button = gr.Button("Ověřit vícekrokové otázky · použít existující editaci", variant="primary", interactive=handler is not None)
                    source_panel(MULTIHOP_HIGHLIGHTS)
                components["multihop_summary"] = gr.Markdown("Vícekrokové měření zatím nebylo spuštěno.", elem_classes=["workshop-summary"])
                components["multihop_rows"] = gr.HTML(render_result_rows("multihop_rows", []), elem_id="multihop-results")
            with gr.Tab(TAB_LABELS[3]):
                with gr.Accordion("Jak vyplnit vlastní edit · příklad Hamlet", open=False):
                    gr.Markdown(
                        "### Zkuste vlastní editaci\n\n"
                        "Šedé placeholdery vás provedou příkladem, který můžete nahradit vlastním faktem:\n\n"
                        "| Pole | Ukázková hodnota, kterou můžete nahradit |\n"
                        "|---|---|\n"
                        "| Co chceme editovat? | `Hamlet` |\n"
                        "| Jak se na fakt modelu zeptáme? | `Who wrote Hamlet? Answer with the author's name only.` |\n"
                        "| Jaká byla původní odpověď? | `William Shakespeare` |\n"
                        "| Co má model nově odpovídat? | `J. R. R. Tolkien` |\n"
                        "| Jiné formulace stejné otázky | `Hamlet was written by` |\n\n"
                        "Placeholdery jsou nápověda; hodnoty vyplňte sami. Vyplňte všechna čtyři hlavní pole. "
                        "Nejprve použijte **Ověřit vlastní fakt**, pak **Provést vlastní editaci**. "
                        "V chatu zkuste `Who is the author of Hamlet?` a porovnejte i kontrolní otázku "
                        "`What is the capital of Japan?`. Každá další editace začíná z původních vah. "
                        "Chat používá aktuální váhy společného modelu."
                    )
                with gr.Row(equal_height=False, elem_id="custom-editor"):
                    with gr.Column(scale=1, min_width=320, elem_id="custom-fact-form"):
                        gr.Markdown("### Vlastní fakt\n\nVyplňte otázku a původní i novou odpověď.")
                        custom_inputs = _fact_form(WORKSHOP_PRESETS["Vlastní fakt"])
                        with gr.Row():
                            custom_inspect = gr.Button("Ověřit vlastní fakt", variant="secondary", interactive=handler is not None)
                            custom_edit = gr.Button("Provést vlastní editaci", variant="primary", interactive=handler is not None)
                    with gr.Column(scale=1, min_width=320, elem_id="custom-chat"):
                        send, clear, system_prompt, max_tokens = _workshop_chat_form(
                            components, "", handler, shared_system, shared_tokens)
                _fact_results(components, "custom")
                with gr.Row():
                    source_panel(CUSTOM_HIGHLIGHTS)
            with gr.Tab(TAB_LABELS[4]):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        gr.Markdown(
                            "### GRAM: strukturální stopy editace\n\n"
                            "Vložený detektor `RomeLayerLocalizer` (GRAM z Latium) analyzuje aktuální váhy. "
                            "Spusťte jej na základním modelu i po editaci a porovnejte pořadí vrstev. "
                            "**Skóre a pořadí nejsou pravděpodobnosti ani binární verdikt „editováno / needitováno“.** "
                            "GRAM neobnovuje model a neprovádí další editaci."
                        )
                        gram_button = gr.Button("Spustit detekci GRAM na aktuálním modelu", variant="primary", interactive=handler is not None)
                    source_panel(GRAM_HIGHLIGHTS)
                components["gram_summary"] = gr.Markdown("Detekce GRAM zatím nebyla spuštěna.", elem_classes=["workshop-summary"])
                components["gram_plot"] = gr.Plot(label="Strukturální skóre GRAM")
                with gr.Accordion("Podrobné hodnoty detekce", open=False):
                    components["gram_rows"] = gr.HTML(render_result_rows("gram_rows", []), elem_id="gram-results")

        gr.HTML(original_brandbar())

        def as_components(values):
            return {components[key]: render_result_rows(key, value) for key, value in values.items()}

        operation_buttons = [reset_button, trace_button, prepared_inspect, prepared_edit, multihop_button,
                             custom_inspect, custom_edit, gram_button, send, prepared_send]
        preparation_controls = [drive_url, upload, clear_upload, download_mode, precision, v_batch, k_batch, epochs, profile, edit_seed]
        edit_controls = [v_batch, k_batch, epochs, profile]
        upload.upload(lambda path: gr.update(label=f"Vybráno: {Path(path).name}"), inputs=upload, outputs=upload,
                      queue=False, show_progress="hidden")
        clear_upload.click(lambda: gr.update(value=None, label="Nahrát vlastní kovarianci (.pt)"), outputs=upload,
                           queue=False, show_progress="hidden")

        def remember_drive_url(value):
            if (value or "").strip():
                demo._covariance_drive_url = value.strip()

        drive_url.input(remember_drive_url, inputs=drive_url, queue=False, show_progress="hidden")

        def ready_values():
            loaded = session.handler
            values = {button: gr.update(interactive=True) for button in operation_buttons}
            values.update({control: gr.update(interactive=False) for control in preparation_controls})
            values.update({control: gr.update(interactive=True) for control in edit_controls})
            values.update({initialize_button: gr.update(interactive=False),
                           components["system_status"]: model_status("Model je již v GPU paměti · znovu se nenačítá."),
                           components["current_state"]: coordinator.current_state(),
                           actual_config: gr.update(value=_loaded_configuration(loaded), visible=True),
                           preparation: gr.update(open=False)})
            return values

        def initialize(url, uploaded, mode, dtype, value_batch, key_batch, iterations, selected_profile="demo", seed_value=42):
            status_component = components["system_status"]
            if session.handler is not None:
                yield ready_values()
                return
            yield {**{control: gr.update(interactive=False) for control in preparation_controls},
                   initialize_button: gr.update(interactive=False),
                   status_component: model_status("Připravuji kovarianci…", "busy")}
            try:
                remember_drive_url(url)
                def prepare(report):
                    with session.initialization_lock:
                        if session.handler is not None:
                            return session.handler
                        count = 8 if selected_profile == "demo" else 20
                        settings = ModelSettings(dtype=dtype, v_batch_size=int(value_batch), k_batch_size=int(key_batch),
                                                 epochs=int(iterations), v_N=count, k_N=count, seed=seed_value)
                        validate_model_settings(settings)
                        env = environment or Environment.discover()
                        report(.02, desc="Připravuji kovarianci…")
                        if uploaded:
                            covariance = Path(uploaded)
                            validate_covariance_download(covariance)
                        else:
                            covariance = ensure_covariance(env, demo._covariance_drive_url, mode=mode, manual_upload=False)
                        report(.1, desc="Načítám Qwen do GPU · pouze při první přípravě")
                        loaded = load_model(covariance, settings=settings)
                        session.handler, session.base_weight = loaded, loaded._workshop_base_weight
                        loaded._workshop_session = session
                        report(1., desc="Model je v GPU paměti")
                        return loaded
                for state, loaded, done in stream_operation(prepare, "Připravuji model…"):
                    yield {status_component: state}
                    if done:
                        values = ready_values()
                        values[status_component] = state
                        yield values
            except Exception as exc:
                LOGGER.exception("Model preparation failed")
                yield {**{control: gr.update(interactive=True) for control in preparation_controls},
                       initialize_button: gr.update(interactive=True),
                       status_component: model_status(f"Příprava selhala: {exc}", "warn")}

        initialize_button.click(initialize, inputs=[drive_url, upload, download_mode, precision, v_batch, k_batch, epochs, profile, edit_seed],
                                outputs=[initialize_button, components["system_status"], components["current_state"], actual_config, preparation,
                                         *operation_buttons, *preparation_controls], **model_event)

        def configure_edit(selected_profile, iterations, value_batch, key_batch):
            if session.handler is None:
                return gr.skip(), gr.skip()
            with session.lock:
                config = session.handler.cfg.model
                config.v_N = config.k_N = 8 if selected_profile == "demo" else 20
                config.epochs = session.handler.epochs = int(iterations)
                config.v_batch_size, config.k_batch_size = int(value_batch), int(key_batch)
            return model_status("Nastavení další editace připraveno · model zůstává v paměti."), _loaded_configuration(session.handler)
        for control in edit_controls:
            control.change(configure_edit, inputs=[profile, epochs, v_batch, k_batch],
                           outputs=[components["system_status"], actual_config], **model_event)

        def bind_edit(button, tab, inputs):
            def callback(*args, progress=gr.Progress()):
                for values in coordinator.mutate(tab, *args, progress=progress):
                    yield as_components(values)
            button.click(callback, inputs=inputs, outputs=list(components.values()), **model_event)

        def bind_inspect(button, tab, inputs):
            def callback(*args, progress=gr.Progress()):
                yield as_components({"system_status": model_status("Měřím současné odpovědi…", "busy")})
                try:
                    for state, result, done in stream_operation(lambda report: coordinator.inspect(tab, *args, progress=report), "Ověřuji aktuální odpovědi…"):
                        yield as_components({"system_status": state})
                        if done:
                            result["system_status"] = state
                            yield as_components(result)
                except Exception as exc:
                    yield as_components({"system_status": model_status(f"Měření selhalo: {exc}", "warn")})
            outputs = [components[key] for key in (f"{tab}_summary", f"{tab}_rows", "system_status", "prepared_history", "custom_history")]
            button.click(callback, inputs=inputs, outputs=outputs, **model_event)

        bind_edit(prepared_edit, "prepared", prepared_inputs)
        bind_edit(custom_edit, "custom", custom_inputs)
        bind_edit(multihop_button, "multihop", [scenario])
        bind_edit(reset_button, "reset", [])
        bind_inspect(prepared_inspect, "prepared", prepared_inputs)
        bind_inspect(custom_inspect, "custom", custom_inputs)
        preset.change(lambda name: WORKSHOP_PRESETS[name], inputs=preset, outputs=prepared_inputs[:-1], queue=False)
        scenario.change(scenario_description, inputs=scenario, outputs=description, queue=False)

        def trace(*args, progress=gr.Progress()):
            yield gr.skip(), gr.skip(), gr.skip(), model_status("Provádím kauzální trasování…", "busy")
            try:
                coordinator.ensure_ready()
                for state, result, done in stream_operation(lambda report: diagnostics.trace_suite(*args, progress=report), "Měřím rainbow profil…"):
                    if done:
                        summary, rows, figure = result
                        yield summary, render_result_rows("trace_rows", rows), figure, state
                    else:
                        yield gr.skip(), gr.skip(), gr.skip(), state
            except Exception as exc:
                yield f"Trasování selhalo: {compact(str(exc))}", render_result_rows("trace_rows", []), None, model_status(f"Trasování selhalo: {exc}", "warn")

        def gram(progress=gr.Progress()):
            yield gr.skip(), gr.skip(), gr.skip(), model_status("Počítám GRAM profily…", "busy")
            try:
                coordinator.ensure_ready()
                for state, result, done in stream_operation(lambda report: diagnostics.gram(progress=report), "Počítám GRAM…"):
                    if done:
                        summary, rows, figure = result
                        yield summary, render_result_rows("gram_rows", rows), figure, state
                    else:
                        yield gr.skip(), gr.skip(), gr.skip(), state
            except Exception as exc:
                yield f"GRAM selhal: {compact(str(exc))}", render_result_rows("gram_rows", []), None, model_status(f"GRAM selhal: {exc}", "warn")

        trace_button.click(
            trace, inputs=[trace_subject, trace_prompt, trace_target, window, samples, noise, seed, trace_facts, auto_noise],
            outputs=[components[f"trace_{suffix}"] for suffix in ("summary", "rows", "plot")] + [components["system_status"]], **model_event,
        )
        gram_button.click(gram, outputs=[components[f"gram_{suffix}"] for suffix in ("summary", "rows", "plot")]
                          + [components["system_status"]], **model_event)
        chat_outputs = [components[key] for key in ("chat_input", "chatbot", "prepared_chat_input", "prepared_chatbot", "system_status")]

        def chat(message, system, tokens, progress=gr.Progress()):
            pending = coordinator.conversation + [{"role": "user", "content": message},
                                                   {"role": "assistant", "content": "Připravuji odpověď…"}]
            yield "", pending, "", pending, model_status("Model připravuje odpověď…", "busy")
            try:
                for state, result, done in stream_operation(lambda report: coordinator.chat(message, system, tokens, progress=report), "Model připravuje odpověď…"):
                    if done:
                        text, history, _ = result
                        yield text, history, text, history, state
                    else:
                        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), state
            except Exception as exc:
                yield "", coordinator.conversation, "", coordinator.conversation, model_status(f"Chat selhal: {exc}", "warn")

        def clear_chat():
            text, history, state = coordinator.clear_chat()
            return text, history, text, history, state

        shared_system.change(clear_chat, outputs=chat_outputs, **model_event)

        for prefix, send_button, clear_button, system, tokens in (
            ("", send, clear, system_prompt, max_tokens),
            ("prepared_", prepared_send, prepared_clear, prepared_system, prepared_tokens),
        ):
            for event in (send_button.click, components[prefix + "chat_input"].submit):
                event(chat, inputs=[components[prefix + "chat_input"], system, tokens], outputs=chat_outputs, **model_event)
            clear_button.click(clear_chat, outputs=chat_outputs, **model_event)
    demo._model_session = session
    return demo, theme, stylesheet("workshop")
