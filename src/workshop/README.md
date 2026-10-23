# ROME workshop

Tato implementace žije přímo v repozitáři **Security-FIT/Latium** na větvi
**`workshop-version`**. Importuje ji
[`clean-notebook.ipynb`](https://github.com/olexamatej/Latium-workshop/blob/main/clean-notebook.ipynb)
z repozitáře Latium-workshop. Notebook obsahuje dvě spouštěcí buňky a pokyny ke spuštění;
ROME, načítání Qwenu, Gradio, causal tracing a GRAM jsou zde v Latium.

## Pro účastníky

**Lokální implementaci ani nastavení checkoutu neupravujte.** Kód a větev jsou
připravené v rámci workshopového PR. Ponechte výchozí hodnoty; `LOCAL_LATIUM`
a `LATIUM_REF` jsou určeny pouze vývojářům. Vlastní editace zadáváte do Gradio,
zdrojové panely slouží k nahlédnutí do implementace.

1. Otevřete clean notebook v Colabu a vyberte CUDA GPU runtime.
2. Spusťte obě kódové buňky shora dolů. Gradio se otevře před načítáním modelu.
3. Vložte Drive odkaz na kovarianci nebo nahrajte vlastní `.pt` a klikněte na
   **Připravit model**. Pro soukromý Drive použijte přihlášení z první buňky.
4. Zkoušejte připravené lekce nebo vlastní fakt. Všech pět záložek sdílí jeden model.

Drive odkaz můžete vložit už do pole **`COVARIANCE_DRIVE_URL` v první buňce notebooku**.
Předvyplněný odkaz ponechte nebo nahraďte vlastním. Druhá buňka jej předá do pole
**Google Drive odkaz na kovarianci** v Gradio. Nový neprázdný odkaz nahradí předchozí;
prázdná hodnota při opětovném otevření zachová poslední odkaz, případně výchozí.
Odkaz lze upravit i přímo v Gradio před načtením modelu. Při přípravě se pro stejné
ID použije stažená matice, jiné ID vyvolá nové stažení. Již načtený model se otevřením
panelu znovu nenačítá; pro jinou kovarianci po načtení restartujte runtime.

Nová editace začíná obnovou původní projekce; editace se nevrství. Stejný připravený
fakt se použije z paměti bez další optimalizace. Nové otevření rozhraní zachová načtený
model a aktivní edit. Změna vah a reset vymažou historii chatu.

## Vlastní edit

V záložce **4 · ROME: vlastní fakt a chat** zůstávají pole prázdná. Šedé
placeholdery a návod přímo nad formulářem ukazují tento propojený příklad;
nahraďte je vlastními hodnotami:

| Pole | Příklad |
|---|---|
| Co chceme editovat? | `Hamlet` |
| Jak se na fakt modelu zeptáme? | `Who wrote Hamlet? Answer with the author's name only.` |
| Jaká byla původní odpověď? | `William Shakespeare` |
| Co má model nově odpovídat? | `J. R. R. Tolkien` |
| Jiné formulace stejné otázky | `Hamlet was written by` |

Otázka musí obsahovat přesný text entity. Nejprve klikněte na **Ověřit vlastní fakt**,
potom na **Provést vlastní editaci**. Porovnejte hlavní otázku, parafráze a nesouvisející
kontroly. V chatu zkuste `Who is the author of Hamlet?` a
`What is the capital of Japan?`.

## Vícekrokové otázky

Výchozí připravený příklad i záložka 3 používají **Burj Khalifa → Španělsko**:
mění se druhý krok, ale složená otázka se ptá na zemi nejvyšší budovy bez jejího jména.
Druhá varianta mění první krok: **nejvyšší budova → Eiffelova věž**. Starší městské
scénáře zůstávají dostupné ve výběru.

Výsledky ukážou přesný editační prompt, jeho parafrázi, needitovaný krok, dvě
samostatné dvojkrokové otázky a kontroly. Rozlišují nepřenesenou parafrázi, změnu
needitovaného kroku, původní zemi a odpověď mezivýsledkem místo konečné země.
Rozklad je samostatná pomocná kontrola přes skutečně vygenerovanou budovu.
Stejná editace i hotové odpovědi se používají z cache.

Nové příklady adaptují tabulku 1 a §5.1 [studie 2601.04600](https://arxiv.org/abs/2601.04600).
V rozbalovacím panelu **Protokol měření** je společný Q/A kontext s vlastními
nesouvisejícími příklady; určuje jen formát odpovědi. Cíle a mezivýsledky se do
samostatných složených otázek nevkládají. Krátké odpovědi se porovnávají s explicitními
aliasy, včetně `UAE`; delší vysvětlení posuďte ručně.

Studie používala GPT-J-6B a různé vrstvy. Toto demo používá Qwen3.5-4B a dodanou
kovarianci vrstvy 5: neprovádí redundantní editování ani neměří příčinu hopping-too-late.
Úspěch a selhání na konkrétní otázce se musí ověřit na skutečném modelu.

## Seed a opakování běhu

Výchozí `SEED = 42` v notebooku se předá přes `launch_workshop(seed=42)`.
Před načtením modelu ho lze změnit v Gradio v nastavení výpočtu. Uloží se jako
kořenové `cfg.seed`, nastaví Python, NumPy a PyTorch/CUDA před načtením a ROME
jej nastaví znovu před každou editací. Načtená konfigurace seed vypisuje.

Odpovědi používají greedy dekódování. Pro srovnatelné výsledky zachovejte stejný
commit, balíčky, GPU, přesnost a micro-batche. Seed sám nezaručuje bitovou shodu
mezi různým hardwarem. Opětovné otevření již načtené session zachová její seed i váhy.

## Pro vývojáře

Závislosti jsou v `requirements-workshop.txt`. Ve vlastním checkoutu této větve:

```python
from src.workshop import launch_workshop

covariance_url = "https://drive.google.com/file/d/1UmPlov94UKE4KIHclswVUMXyliQM3qvU/view?usp=drive_link"
demo = launch_workshop(seed=42, covariance_drive_url=covariance_url)
```

Podrobnosti integrace a validační příkazy jsou v
[`docs/workshop-version.md`](../../docs/workshop-version.md). Úpravy implementace
patří do této větve Latium; změny buněk a exportu notebooku do Latium-workshop.
