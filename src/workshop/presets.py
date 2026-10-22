"""Examples and unrelated control prompts used in the two demonstrations."""

WORKSHOP_PRESETS = {
    "Eiffelova věž → Řím": (
        "The Eiffel Tower", "The Eiffel Tower is located in which city? Answer with the city name only.", "Rome", "Paris",
        "The Eiffel Tower is in\nThe Eiffel Tower stands in\nThe city of the Eiffel Tower is",
    ),
    "Mount Everest → Kanada": (
        "Mount Everest", "Mount Everest is located in which country? Answer with the country name only.", "Canada", "Nepal",
        "Mount Everest is in\nThe country containing Mount Everest is",
    ),
    "Koloseum → Paříž": (
        "The Colosseum", "The Colosseum is located in which city? Answer with the city name only.", "Paris", "Rome",
        "The Colosseum is in\nThe city of the Colosseum is",
    ),
    "Autor Hamleta → Tolkien": (
        "Hamlet", "Who wrote Hamlet? Answer with the author's name only.", "J. R. R. Tolkien", "William Shakespeare",
        "Hamlet was written by\nWho wrote Hamlet?",
    ),
    "Tvůrce Pythonu → Ada Lovelace": (
        "Python", "Who created Python? Answer with the person's name only.", "Ada Lovelace", "Guido van Rossum",
        "The creator of Python is\nPython's original author was",
    ),
    "Vlastní fakt": ("", "", "", "", ""),
}
DEFAULT_PRESET = "Eiffelova věž → Řím"
SOLO_PRESETS = {name: WORKSHOP_PRESETS[name][:4] for name in (
    DEFAULT_PRESET, "Autor Hamleta → Tolkien", "Mount Everest → Kanada", "Vlastní fakt",
)}
SOLO_CHECKS = ["The capital of Japan is", "Water freezes at"]
WORKSHOP_CANARIES = "\n".join([
    "The capital of Japan is", "Big Ben is located in", "Water freezes at",
    "The author of Hamlet is", "2 + 2 equals",
])
