"""Small text helpers shared by the callback adapters."""

import html


def compact(text):
    return html.escape(str(text)).replace("|", "&#124;").replace("\n", " ")


def status(message, tone="neutral", *, workshop=False):
    css_class = "model-status" if workshop else "demo-status"
    dot = '<span class="status-dot"></span>' if workshop else ""
    return f'<div class="{css_class} {tone}">{dot}{html.escape(message)}</div>'


def ui_input(parser, *args):
    import gradio as gr

    try:
        return parser(*args)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
