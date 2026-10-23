"""Lazy teaching excerpts, with installed-source inspection available as fallback."""

import importlib
import inspect
import textwrap

import gradio as gr

# Populated only in generated notebooks: exact source for displayed local
# wrappers, used as read-only text when inspect cannot recover a cell/class.
NOTEBOOK_SOURCE_SNIPPETS = {}


def resolve_source(source, *, fallback=None):
    """Inspect a callable or a (title, module_path, dotted_attribute) reference.

    Callable support lets standalone notebooks supply their locally defined
    wrappers without importing latium_workshop. A generator may supply the exact
    original source as fallback when inspect/linecache is unavailable.
    """
    if isinstance(source, str):
        return textwrap.dedent(source).strip()
    try:
        if isinstance(source, tuple):
            _title, module_path, attribute = source
            obj = importlib.import_module(module_path)
            for name in attribute.split("."):
                obj = getattr(obj, name)
            origin = f"{module_path}.{attribute}"
        else:
            obj = source
            origin = f"{obj.__module__}.{obj.__qualname__}"
        code = textwrap.dedent(inspect.getsource(obj))
        return f"# {origin}\n\n{code}"
    except (ImportError, AttributeError, OSError, TypeError) as exc:
        if fallback is not None:
            return fallback
        if not isinstance(source, tuple) and getattr(source, "__module__", None) == __name__:
            snippet = NOTEBOOK_SOURCE_SNIPPETS.get(getattr(source, "__qualname__", None))
            if snippet is not None:
                return f"# {origin}\n\n{snippet}"
        return f"# Zdrojový kód není v tomto prostředí dostupný ({type(exc).__name__}).\n# Vyberte jinou implementaci."


def source_panel(sources, *, fallbacks=None):
    """Build a right Column, closed Accordion and lazy Code selector.

    sources accepts upstream triples or (title, callable) pairs. Nothing is
    imported or inspected until the accordion is expanded / selection changes.
    """
    references = {item[0]: item if len(item) == 3 else item[1] for item in sources}
    titles = list(references)
    if not titles:
        raise ValueError("Zdrojový panel potřebuje alespoň jednu implementaci.")

    def load(title):
        return resolve_source(references[title], fallback=(fallbacks or {}).get(title))

    with gr.Column(scale=3, min_width=280), gr.Accordion("Zdrojový kód · rozbalit", open=False, elem_classes=["source-panel"]) as accordion:
        gr.Markdown("Zkrácené didaktické výřezy: hlavní myšlenka bez pomocného kódu a správy paměti.")
        selector = gr.Dropdown(titles, value=titles[0], label="Zajímavý krok")
        code = gr.Code(value="", language="python", interactive=False, label="Jádro principu", lines=14, wrap_lines=True)
    accordion.expand(load, inputs=selector, outputs=code, queue=False)
    selector.change(load, inputs=selector, outputs=code, queue=False)
    return accordion, code
