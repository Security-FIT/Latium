"""The solo notebook's one-screen Czech presentation."""

import gradio as gr

from .assets import asset_url, stylesheet
from .model import gpu_memory_summary
from .presentation import status
from .presets import SOLO_PRESETS as presets
from .session import session_for
from .solo import SoloActions
from .themes import solo_theme


def build_solo_ui(handler):
    session = session_for(handler)
    session.reset()
    actions = SoloActions(session)
    graph = asset_url("layer_search_graph.png")
    red_hat_logo = asset_url("red_hat_research_logo.png")
    fit_logo = asset_url("fit_vut_logo.png")
    with gr.Blocks(title="Injekce faktu do modelu") as demo:
        gr.HTML("""<section class="demo-hero">
          <p class="demo-eyebrow">Živá ukázka · injekce faktu · ROME</p>
          <h1>Co když někdo přepíše jediný fakt v jazykovém modelu?</h1>
          <p class="demo-lead">Injekce faktu je cílená změna znalosti uložené ve vahách modelu.
          Představte si, že model začne tvrdit, že Eiffelova věž stojí v Římě.
          V této ukázce uvidíte, že lze změnit jeden fakt a přitom obvykle výrazně nenarušit běžné fungování modelu.</p>
          <p class="demo-cta">↓ Níže si vyzkoušíte vlastní změnu</p>
        </section>""")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML("""<section class="demo-highlights" aria-label="Proč je injekce faktu důležitá">
                  <article class="demo-card speed">
                    <p class="demo-kicker">01 · Rychlost</p>
                    <h2 class="demo-pr-heading">Pár minut</h2>
                    <p>Takto dlouho trvá změna faktu pomocí běžně dostupných zdrojů. Úplné doladění modelu
                    obvykle vyžaduje mnohem více výpočtů, přípravy a často hodiny až dny práce.</p>
                  </article>
                  <article class="demo-card scope">
                    <p class="demo-kicker">02 · Rozsah</p>
                    <h2 class="demo-pr-heading">Model zůstává jinak nepozměněn</h2>
                    <p>Editace cílí na jediný fakt; ostatní odpovědi obvykle zůstávají stejné. Vedlejší změny ale nelze vyloučit.</p>
                  </article>
                </section>
                <p class="demo-footnote"><strong class="demo-highlight">Doma</strong> s vhodnou grafickou kartou, případně v Google Colabu.
                Ukázka mění jen kopii modelu v notebooku; cizí službu bez přístupu k vahám upravit nelze.</p>""")
                gr.Markdown("### Změnit fakt", elem_id="change-fact-title")
                preset = gr.Dropdown(
                    choices=list(presets), value="Eiffelova věž → Řím",
                    label="Příklad nebo vlastní fakt", elem_id="fact-preset",
                )
                subject = gr.Textbox(
                    value=presets["Eiffelova věž → Řím"][0],
                    label="Entita (např. The Twin Towers)", placeholder="The Twin Towers",
                )
                prompt = gr.Textbox(
                    value=presets["Eiffelova věž → Řím"][1],
                    label="Začátek tvrzení", placeholder="The Twin Towers were destroyed by",
                    info="Uveďte entitu v textu; na velikosti písmen nezáleží.",
                )
                with gr.Row():
                    target_old = gr.Textbox(value="Paris", label="Původní odpověď")
                    target_new = gr.Textbox(value="Rome", label="Nová odpověď")
                with gr.Row():
                    preview_button = gr.Button("Zobrazit stav", variant="secondary")
                    edit_button = gr.Button("Vložit fakt", variant="primary")
                    reset_button = gr.Button("Obnovit model", variant="secondary")
                current_status = gr.HTML(status(f"Původní model připraven · {gpu_memory_summary()}"))
                summary = gr.Markdown("### Připraveno\n\nVyberte příklad a zobrazte stav modelu.")
                comparison = gr.Dataframe(
                    headers=["Typ", "Začátek tvrzení", "Před", "Po"],
                    datatype=["str", "str", "str", "str"],
                    value=[], interactive=False, wrap=True, label="Co model skutečně doplnil",
                )
            with gr.Column(scale=6):
                gr.HTML(f"""<figure class="demo-graph">
                  <a href="{graph}" target="_blank" rel="noopener noreferrer" aria-label="Otevřít graf v plné velikosti">
                    <img src="{graph}" alt="Hledání správné vrstvy modelu Llama 2 7B">
                  </a>
                  <figcaption>Hledání vhodné vrstvy · Llama 2 7B</figcaption>
                </figure>""")
                gr.Markdown("### Rozhovor s modelem")
                gr.Markdown("Ptejte se přirozeně, například „Where is the Eiffel Tower?“. ")
                chatbot = gr.Chatbot(
                    value=[], label="Odpovědi modelu", height=300,
                    layout="bubble", sanitize_html=True, elem_id="demo-chatbot",
                )
                chat_input = gr.Textbox(
                    label="Vaše zpráva", placeholder="Where is the Eiffel Tower?", lines=1, max_lines=3,
                )
                with gr.Row():
                    send_button = gr.Button("Odeslat", variant="primary")
                    clear_button = gr.Button("Vymazat rozhovor", variant="secondary")

        gr.HTML(f"""<div class="demo-brandbar" aria-label="Red Hat Research a FIT VUT Brno">
          <a href="https://research.redhat.com/" target="_blank" rel="noopener noreferrer"
             aria-label="Red Hat Research – otevřít web">
            <img class="red-hat-logo" src="{red_hat_logo}" alt="Red Hat Research">
          </a>
          <a href="https://www.fit.vut.cz/" target="_blank" rel="noopener noreferrer"
             aria-label="FIT VUT Brno – otevřít web">
            <img class="fit-logo" src="{fit_logo}" alt="FIT VUT Brno">
          </a>
        </div>""")
        fields = [subject, prompt, target_new, target_old]
        outputs = [summary, comparison, current_status]
        preset.change(lambda name: presets[name], inputs=preset, outputs=fields, queue=False, show_progress="hidden")
        preview_button.click(actions.preview, inputs=fields, outputs=outputs, concurrency_limit=1, show_progress="hidden")
        edit_button.click(actions.edit, inputs=fields, outputs=outputs + [chat_input, chatbot],
                          concurrency_limit=1, show_progress="hidden")
        reset_button.click(actions.reset, outputs=outputs + [chat_input, chatbot],
                           concurrency_limit=1, show_progress="hidden")
        for event in (send_button.click, chat_input.submit):
            event(actions.chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot],
                  concurrency_limit=1, show_progress="hidden")
        clear_button.click(actions.clear_chat, outputs=[chat_input, chatbot], queue=False, show_progress="hidden")
    return demo, solo_theme(), stylesheet("solo")
