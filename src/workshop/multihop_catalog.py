"""Additional native-chat two-hop candidates; all must pass a model baseline."""

from dataclasses import dataclass

BIRTH_COUNTRY = "In which present-day country was {} born? Give only the country name."
UK = ("UK", "U.K.", "Great Britain", "Britain")
US = ("USA", "US", "U.S.A.", "United States of America", "the United States")
PEOPLE = {
    "William Shakespeare": ("United Kingdom", (*UK, "England"), ("Shakespeare",)),
    "J. R. R. Tolkien": ("South Africa", ("Republic of South Africa",), ("Tolkien", "JRR Tolkien")),
    "Ernest Hemingway": ("United States", US, ("Hemingway",)),
    "Guido van Rossum": ("Netherlands", ("the Netherlands", "Holland"), ("Guido Van Rossum",)),
    "Ada Lovelace": ("United Kingdom", (*UK, "England"), ("Augusta Ada King",)),
    "Pablo Picasso": ("Spain", (), ("Picasso",)),
    "Leonardo da Vinci": ("Italy", (), ("Leonardo Da Vinci",)),
    "Alexander Fleming": ("United Kingdom", (*UK, "Scotland"), ("Sir Alexander Fleming", "Fleming")),
    "Marie Curie": ("Poland", (), ("Marie Sklodowska-Curie", "Marie Skłodowska-Curie")),
}


@dataclass(frozen=True)
class ChainExample:
    key: str
    preset: str
    subject: str
    first_question: str
    first_variant: str
    original_person: str
    new_person: str
    question: str
    variant: str
    edited_hop: int = 1
    new_country: str = ""

    @property
    def preset_values(self):
        if self.edited_hop == 1:
            return self.subject, self.first_question, self.new_person, self.original_person, self.first_variant
        country = PEOPLE[self.original_person][0]
        return (self.original_person, BIRTH_COUNTRY.format(self.original_person), self.new_country, country,
                f"Name the present-day country containing the birthplace of {self.original_person}.")


EXTRA_CHAINS = (
    ChainExample("hamlet_author_birth", "Autor Hamleta → Tolkien", "Hamlet",
        "Who wrote Hamlet? Answer with the author's name only.", "Name the author of Hamlet.",
        "William Shakespeare", "J. R. R. Tolkien",
        "In which present-day country was the author of Hamlet born? Give only the country name.",
        "Name the birth country of the playwright who wrote Hamlet. Give only the country."),
    ChainExample("hobbit_author_birth", "Autor Hobita → Hemingway", "The Hobbit",
        "Who wrote The Hobbit? Give only the author's name.", "Name the author of The Hobbit.",
        "J. R. R. Tolkien", "Ernest Hemingway",
        "In which present-day country was the author of The Hobbit born? Give only the country name.",
        "Name the birth country of the writer of The Hobbit. Give only the country."),
    ChainExample("python_creator_birth", "Tvůrce Pythonu → Ada Lovelace", "Python",
        "Who created Python? Answer with the person's name only.", "Name the creator of Python.",
        "Guido van Rossum", "Ada Lovelace",
        "In which present-day country was the creator of Python born? Give only the country name.",
        "Name the birth country of the person who created Python. Give only the country."),
    ChainExample("guernica_painter_birth", "Malíř Guerniky → Leonardo da Vinci", "Guernica",
        "Who painted Guernica? Give only the painter's name.", "Name the artist who painted Guernica.",
        "Pablo Picasso", "Leonardo da Vinci",
        "In which present-day country was the painter of Guernica born? Give only the country name.",
        "Name the birth country of the artist who painted Guernica. Give only the country."),
    ChainExample("penicillin_discoverer_birth", "Objevitel penicilinu → Marie Curie", "penicillin",
        "Who discovered penicillin? Give only the person's name.", "Name the discoverer of penicillin.",
        "Alexander Fleming", "Marie Curie",
        "In which present-day country was the discoverer of penicillin born? Give only the country name.",
        "Name the birth country of the scientist who discovered penicillin. Give only the country."),
    ChainExample("shakespeare_birth_country", "Shakespeare · rodiště → Kanada", "Hamlet",
        "Who wrote Hamlet? Answer with the author's name only.", "Name the author of Hamlet.",
        "William Shakespeare", "William Shakespeare",
        "In which present-day country was the author of Hamlet born? Give only the country name.",
        "Name the birth country of the playwright who wrote Hamlet. Give only the country.", 2, "Canada"),
    ChainExample("guido_birth_country", "Guido van Rossum · rodiště → Japonsko", "Python",
        "Who created Python? Answer with the person's name only.", "Name the creator of Python.",
        "Guido van Rossum", "Guido van Rossum",
        "In which present-day country was the creator of Python born? Give only the country name.",
        "Name the birth country of the person who created Python. Give only the country.", 2, "Japan"),
    ChainExample("picasso_birth_country", "Picasso · rodiště → Kanada", "Guernica",
        "Who painted Guernica? Give only the painter's name.", "Name the artist who painted Guernica.",
        "Pablo Picasso", "Pablo Picasso",
        "In which present-day country was the painter of Guernica born? Give only the country name.",
        "Name the birth country of the artist who painted Guernica. Give only the country.", 2, "Canada"),
)

CHAIN_BY_KEY = {example.key: example for example in EXTRA_CHAINS}


def person_info(name):
    def normalized(value):
        return "".join(c for c in value.casefold() if c.isalnum())
    for person, (country, aliases, names) in PEOPLE.items():
        if normalized(name) in {normalized(value) for value in (person, *names)}:
            return country, aliases, (person, *names)
    return "—", (), ()
