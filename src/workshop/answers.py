"""Plain-language checks of observed answers, with no surrogate log-score verdict."""

import re
import unicodedata


def answer_matches(answer, expected):
    def normalized(text):
        return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())
    return bool(answer.strip()) and normalized(answer) == normalized(expected)


def observed_verdict(answer, expected):
    if not answer.strip():
        return "Model nevrátil odpověď."
    return "Odpověď odpovídá cíli." if answer_matches(answer, expected) else "Prohlédněte si skutečnou odpověď; nemusí odpovídat cíli."
