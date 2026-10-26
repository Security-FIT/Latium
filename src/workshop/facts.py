"""Model-independent input normalization and probe selection."""

import re


def prompt_input(subject, prompt):
    subject, prompt = (subject or "").strip(), (prompt or "").strip()
    if not subject or not prompt:
        raise ValueError("Vyplňte entitu a začátek tvrzení.")
    prompt = re.sub(r"\s*(?:\.{3}|…)\s*$", "", prompt).rstrip()
    if not prompt:
        raise ValueError("Začátek tvrzení nesmí být prázdný.")
    if "{}" in prompt:
        template, matched_subject = prompt, subject
    else:
        names = [subject]
        if subject.casefold().startswith("the "):
            names.append(subject[4:])
        match = None
        for name in names:
            pattern = r"\s+".join(re.escape(word) for word in name.split())
            match = re.search(r"(?<!\w)" + pattern + r"(?!\w)", prompt, flags=re.IGNORECASE)
            if match:
                break
        if not match:
            raise ValueError(
                f"Do otázky vložte entitu „{subject}“ nebo zástupné {{}}. "
                f"Například: Where is {subject}? Velikost písmen nerozhoduje."
            )
        matched_subject = match.group(0)
        template = prompt[:match.start()] + "{}" + prompt[match.end():]
    if template.count("{}") != 1 or template.count("{") != 1 or template.count("}") != 1:
        raise ValueError("V začátku tvrzení použijte nejvýše jednu dvojici složených závorek {}.")
    return template, matched_subject, template.format(matched_subject)


def fact_input(subject, prompt, target_new, target_old):
    template, subject, prompt = prompt_input(subject, prompt)
    target_new, target_old = (target_new or "").strip(), (target_old or "").strip()
    if not target_new or not target_old:
        raise ValueError("Pro vložení faktu vyplňte novou i původní odpověď. Náhled funguje i bez nich.")
    if target_new == target_old:
        raise ValueError("Nová a původní odpověď se musí lišit.")
    return (template, subject, " " + target_new, " " + target_old), prompt


def workshop_fact(subject, prompt, target_new, target_true):
    return fact_input(subject, prompt, target_new, target_true)


def probe_specs(prompt, variants, canaries):
    specs = [("Hlavní otázka", prompt)]
    seen = {prompt}
    for category, raw in (("Jiná formulace", variants), ("Kontrolní otázka", canaries)):
        for item in (raw or "").splitlines():
            item = item.strip()
            if item and item not in seen:
                specs.append((category, item))
                seen.add(item)
    return specs
