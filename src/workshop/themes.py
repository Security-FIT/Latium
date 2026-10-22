"""Gradio themes; solo retains the light palette of the working notebook."""


def solo_theme():
    import gradio as gr

    return gr.themes.Soft(primary_hue="red", secondary_hue="cyan").set(
        body_background_fill="#ffffff", body_background_fill_dark="#ffffff",
        body_text_color="#1e293b", body_text_color_dark="#1e293b",
        body_text_color_subdued="#475569", body_text_color_subdued_dark="#475569",
        color_accent_soft="#eaf8fc", color_accent_soft_dark="#eaf8fc",
        border_color_accent_subdued="#9bdcf0", border_color_accent_subdued_dark="#9bdcf0",
        background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
        background_fill_secondary="#ffffff", background_fill_secondary_dark="#ffffff",
        block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
        panel_background_fill="#ffffff", panel_background_fill_dark="#ffffff",
        input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
        input_background_fill_focus="#ffffff", input_background_fill_focus_dark="#ffffff",
        table_even_background_fill="#ffffff", table_even_background_fill_dark="#ffffff",
        table_odd_background_fill="#ffffff", table_odd_background_fill_dark="#ffffff",
        block_label_text_color="#1e293b", block_label_text_color_dark="#1e293b",
        block_title_text_color="#1e293b", block_title_text_color_dark="#1e293b",
        button_primary_background_fill="#e4002b", button_primary_background_fill_dark="#e4002b",
        button_primary_background_fill_hover="#bf0024", button_primary_background_fill_hover_dark="#bf0024",
        button_primary_border_color="#e4002b", button_primary_border_color_dark="#e4002b",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#ffffff",
        button_secondary_text_color="#00445a", button_secondary_text_color_dark="#00445a",
        button_secondary_border_color="#00a9e0", button_secondary_border_color_dark="#00a9e0",
        button_secondary_background_fill_hover="#eaf8fc", button_secondary_background_fill_hover_dark="#eaf8fc",
        button_secondary_text_color_hover="#003748", button_secondary_text_color_hover_dark="#003748",
        table_text_color="#1e293b", table_text_color_dark="#1e293b",
    )
