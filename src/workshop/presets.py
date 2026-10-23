"""Examples and unrelated control prompts used in the two demonstrations."""

WORKSHOP_PRESETS = {
    "Burj Khalifa → Španělsko · druhý krok": (
        "Burj Khalifa", "Name the country containing Burj Khalifa. Reply with the country only.",
        "Spain", "United Arab Emirates",
        "Burj Khalifa stands in which nation? Give just the country's name.",
    ),
    "Nejvyšší budova → Eiffelova věž · první krok": (
        "the world's tallest building", "Name the world's tallest building. Reply only with its name.",
        "Eiffel Tower", "Burj Khalifa",
        "Identify the building with the greatest height worldwide. Reply with its name only.",
    ),
    "Eiffelova věž → Řím": (
        "The Eiffel Tower", "The Eiffel Tower is located in which city? Answer with the city name only.", "Rome", "Paris",
        "Where is the Eiffel Tower? Answer with the city name only.",
    ),
    "Mount Everest → Kanada": (
        "Mount Everest", "Mount Everest is located in which country? Answer with the country name only.", "Canada", "Nepal",
        "In which country is Mount Everest? Answer with the country name only.",
    ),
    "Koloseum → Paříž": (
        "The Colosseum", "The Colosseum is located in which city? Answer with the city name only.", "Paris", "Rome",
        "Where is the Colosseum? Answer with the city name only.",
    ),
    "Autor Hamleta → Tolkien": (
        "Hamlet", "Who wrote Hamlet? Answer with the author's name only.", "J. R. R. Tolkien", "William Shakespeare",
        "Name the author of Hamlet.",
    ),
    "Tvůrce Pythonu → Ada Lovelace": (
        "Python", "Who created Python? Answer with the person's name only.", "Ada Lovelace", "Guido van Rossum",
        "Name the creator of Python.",
    ),
    "Vlastní fakt": ("", "", "", "", ""),
}
DEFAULT_PRESET = "Eiffelova věž → Řím"
DEFAULT_WORKSHOP_PRESET = "Burj Khalifa → Španělsko · druhý krok"
SOLO_PRESETS = {name: WORKSHOP_PRESETS[name][:4] for name in (
    DEFAULT_PRESET, "Autor Hamleta → Tolkien", "Mount Everest → Kanada", "Vlastní fakt",
)}
SOLO_CHECKS = ["The capital of Japan is", "Water freezes at"]
WORKSHOP_CANARIES = "\n".join([
    "What is the capital of Japan? Answer with the city name only.",
    "At what temperature does water freeze in Celsius? Answer briefly.",
])
