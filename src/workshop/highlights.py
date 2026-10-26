"""Short teaching excerpts: omit plumbing, device handling and cache machinery."""

ROME_HIGHLIGHTS = [
    ("ROME · jedna rank-one změna", '''# Jádro ROME; zjednodušený výřez.
def apply_rome_edit(model, fact):
    k = mean_subject_activations(model, fact)
    delta = optimize_target_output(model, fact)

    direction = inverse_covariance @ k
    direction /= k @ direction
    W += outer(delta, direction)  # jediná MLP projekce
'''),
    ("ROME · naučit novou odpověď a chránit okolí", '''# Zjednodušená optimalizace vektoru delta, nikoli všech vah.
delta = zeros_like(original_output, requires_grad=True)
for step in range(epochs):
    edited_logits = forward_with_output_offset(delta)
    target_loss = cross_entropy(edited_logits, new_answer_tokens)
    preservation = kl_divergence(original_distribution, edited_distribution)
    loss = target_loss + kl_factor * preservation + regularize(delta)
    loss.backward()
    optimizer.step()
    clamp_delta_norm(delta)
'''),
]

TRACE_HIGHLIGHTS = [
    ("Trasování · poškodit, obnovit, porovnat", '''# Didaktický výřez: stejný šum se páruje mezi vrstvami.
clean = capture_activations(prompt)
corrupted = corrupt_subject_embeddings(prompt, seeded_noise)
p_corrupt = probability(corrupted, target_token)

for layer_window in windows:
    restored = restore_mlp_outputs(corrupted, clean, layer_window)
    effect[layer_window] = probability(restored, target_token) - p_corrupt
# Pro každý vybraný fakt změříme vlastní profil se stejným seedem.
# Křivky = jednotlivé fakty; sloupce = jejich průměr.
'''),
]

MULTIHOP_HIGHLIGHTS = [
    ("Multihop · nejprve ověřit základ", '''# Didaktický výřez. Odpověď není vložena do otázky.
with session.baseline():
    before = ask_independent_questions(probes)

# V této lekci je fakt vždy Eiffelova věž: Paříž → Řím.
session.install(default_eiffel_fact)  # použije cache, pokud edit už existuje
after = ask_independent_questions(probes)
compare(edited_hop, unedited_hop, composed_question)
compare_original_attributes(year=1889, material="iron")
# Samostatná 2-hop otázka nedostane mezivýsledek jako nápovědu.
'''),
    ("Multihop · chybný základ není selhání ROME", '''# Příklad pravidel vyhodnocení, nikoli předem dané výsledky.
if not correct(before.composed):
    verdict = "Model otázku neznal už před editací"
elif not correct(after.edited_hop):
    verdict = "Edit se nepřenesl na parafrázi"
elif not correct(after.unedited_hop):
    verdict = "Změnila se i needitovaná znalost"
elif not correct(after.composed):
    verdict = "Jednotlivé kroky fungují, jejich složení selhalo"
'''),
]

CUSTOM_HIGHLIGHTS = ROME_HIGHLIGHTS + [
    ("Vlastní edit · společný model a krátký chat", '''# Didaktický výřez: editace a chat sdílejí stejné váhy.
with session.lock:
    session.install((template, subject, new_answer, old_answer))

messages = [system_instruction, *history, user_question]
answer = model.generate(
    chat_template(messages, enable_thinking=False),
    do_sample=False,
    max_new_tokens=32,
)
'''),
]

GRAM_HIGHLIGHTS = [
    ("GRAM · stopa rank-one změny ve váhách", '''# Schéma principu; detektor navíc normalizuje a porovnává okolí.
for layer, W in mlp_weights:
    gram = normalized_gram(W)
    features[layer] = spectral_profile(gram)

scores = compare_with_neighboring_layers(features)
ranking = sort_layers_by_structural_score(scores)
# Pořadí není pravděpodobnost ani binární verdikt o editaci.
'''),
]
