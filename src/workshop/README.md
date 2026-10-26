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
Chat má výchozí limit 32 nových tokenů a stručnou systémovou instrukci; limit lze
upravit ve společném nastavení chatů. Číselné odpovědi chat zobrazuje jako prostý text,
aby se například `1453.` nezměnilo v odsazenou značku číslovaného seznamu.

## Vlastní edit

V záložce **4 · ROME: vlastní fakt a chat** zůstávají pole prázdná. Šedé
placeholdery a návod přímo nad formulářem ukazují tento propojený příklad.
Výběr **Šablona vlastního editu** nabízí Eiffelovu věž → Řím, Hamleta → Tolkiena,
Python → Adu Lovelace a Koloseum → Paříž. Vyplní entitu, otázku, obě odpovědi,
parafrázi i kontrolní otázky. Všechna pole můžete upravit; výběr sám model needituje.
Volba **Vlastní fakt** znovu otevře prázdný formulář s výchozími kontrolami.
Stejná hotová editace se při použití šablony znovu neoptimalizuje.

Příklad ručního vyplnění:

| Pole | Příklad |
|---|---|
| Co chceme editovat? | `Hamlet` |
| Jak se na fakt modelu zeptáme? | `Who wrote Hamlet? Answer with the author's name only.` |
| Jaká byla původní odpověď? | `William Shakespeare` |
| Co má model nově odpovídat? | `J. R. R. Tolkien` |
| Jiné formulace stejné otázky | `Hamlet was written by` |

Otázka musí obsahovat entitu nebo zástupné `{}`. Velikost písmen, více mezer
a vynechané úvodní „The“ nevadí; použije se skutečný text entity z otázky.
Pro **Ověřit vlastní fakt** stačí entita a otázka, odpovědi doplníte až před editací.
Nejprve klikněte na **Ověřit vlastní fakt**,
potom na **Provést vlastní editaci**. Porovnejte hlavní otázku, parafráze a nesouvisející
kontroly. V chatu zkuste `Who is the author of Hamlet?` a
`What is the capital of Japan?`.

## Výběr faktů a trasování

Záložka 1 má vlastní checkboxy pro čtyři fakty: Eiffelova věž, Koloseum, Hamlet
a Python. Každý je na vlastním řádku s celou trasovanou otázkou a sledovanou odpovědí.
Vybraný řádek má modré pozadí a checkbox viditelnou bílou fajfku.
Začněte jedním, potom vyberte 3–4. Křivky ukazují naměřený profil
každého faktu a duhové sloupce jejich průměr; tabulka uvádí nejsilnější vrstvu
každého faktu. Stejný fakt má stejný šum i při změně pořadí či velikosti výběru;
hotové měření se použije z cache. Více faktů samo nezaručuje stabilnější profil.

Výchozí měření používá původní váhy a původní odpovědi. Po měření se vrátí
aktivní editace. Checkbox lze vypnout a měřit stejné původní odpovědi na aktuálních
vahách. Výběr trasování nepřepisuje formuláře ani multihop scénář.
Před výpočtem není pro prázdný graf rezervované místo. Průběh je vidět i u tlačítka;
po dokončení se zobrazí **Hotovo**, stránka přejde k výsledku a nejprve ukáže graf.

Záložka **2 · ROME: připravený edit** obsahuje jediný pevný příklad:
**Eiffelova věž: Paříž → Řím**. Není zde výběr jiného faktu a všechna pole jsou
pouze pro čtení. Callback používá vždy tento výchozí fakt. Další šablony a vlastní
hodnoty patří do záložky 4.

## Vícekrokové otázky a vedlejší změny

Záložka 3 vždy používá **Eiffelova věž: Paříž → Řím**. Odpovídající aktivní edit
použije znovu, jinou aktivní editaci nahradí tímto výchozím faktem. Vlastní šablony
ze čtyřky tedy nemění scénář v trojce. Katalog starších experimentů zůstává
v Python API; ve workshopovém rozhraní se nevybírá.

Samostatně změříme přímou otázku, zemi ve dvou formulacích, jazyk a kontroly
původního i nového města. Rozklad se ptá na zemi skutečně vygenerovaného města.
Další otázky kontrolují rok dokončení (1889), stavitele (Eiffelova společnost)
a materiál (železo), které mají při změně lokace zůstat stejné. Referenční fakta
pocházejí z [historie stavby](https://www.toureiffel.paris/en/the-monument/history)
a [oficiálních odpovědí provozovatele](https://www.toureiffel.paris/en/news/history-and-culture/15-essential-things-know-about-eiffel-tower).
Výšku a okolní památky zobrazujeme k ručnímu posouzení, bez automatického verdiktu.

V rozbalovacím příkladu je část uživatelova rozhovoru: po editu model odpověděl
rokem `1453`, stavitelem `Pope Julius II` a materiálem `Tuff and travertine`.
Jsou to označené pozorované chyby jednoho běhu. Výsledky níže vznikají novým
měřením před i po editaci, nejsou těmito chybnými odpověďmi vyplněné.
Každá kontrola má prázdnou historii chatu; původní rozhovor mohl ovlivnit další odpovědi.

Vyhodnocení rozlišuje nepotvrzený přímý edit, neznámý základ, nepřenesený důsledek
a možný vedlejší zásah. Změna formulace sama není důkaz změny faktu. Selhání ROME
není zaručené a vyžaduje ověření na skutečném Qwenu. Zdrojové panely obsahují
krátké označené didaktické výřezy hlavních algoritmů.

## GRAM a spektrální artefakty

Záložka 5 po detekci zobrazí čtyři grafy:

- GRAM skóre vrstev původního a aktuálního modelu;
- rozdíl skóre po − před;
- vedoucí singulární hodnoty uložené projekce před a po editaci;
- spektrum ΔW a kumulativní podíl energie skutečné změny vah.

Spektrální srovnání vždy používá původní snapshot editační vrstvy (výchozí L5),
nikoli automaticky vítěznou vrstvu detektoru. Jde o kontrolu se známou referencí,
nikoli další slepou detekci. Při nulové změně je podíl energie označen jako
nedefinovaný. Rank-one edit nemusí po zaokrouhlení vah zůstat numericky přesně rank-one.

U velkých matic se počítá pouze aproximace vedoucích 16 singulárních hodnot,
se stejným seedem před i po; nejde o úplné spektrum ani o odhad přesného ranku.
Dodatečné spektrální výpočty běží na CPU a mohou první detekci prodloužit.
Stejný stav modelu používá cache grafů i exportu. Tlačítko pro ZIP poskytne čtyři
PNG grafy, `scores.csv`, `spectra.csv` a `metrics.json`; samotné váhy se neexportují.

## Colab a ztráta spojení

Workshop v Colabu výchozím nastavením používá `share=True`: otevřete vypsaný
odkaz **`https://….gradio.live`**; stejnou adresu používá i vložené rozhraní.
Adresa runtime `*.colab.dev` může při otevření v jiné záložce vrátit 404.
Pokud máte načtenou starší implementaci, stačí znovu spustit
`launch_workshop(seed=SEED, covariance_drive_url=COVARIANCE_DRIVE_URL, share=True)`.
Živá session zachová model i edit. Veřejný tunel lze výslovně vypnout pomocí
`GRADIO_SHARE=0` nebo `share=False` pro prostředí s funkční lokální/proxy adresou.
Způsob přístupu neobchází omezení runtime:
[Google uvádí proměnlivé idle timeouty a maximální životnost](https://research.google.com/colaboratory/faq.html).
Nelze garantovat odstranění odpojení po určité době.

Při ztrátě stránky nejprve ověřte v Colabu, zda runtime žije. Pokud běží, znovu
spusťte druhou buňku; otevření panelu zachová model i edit. Pokud runtime zanikl,
spusťte obě buňky a model připravte znovu. Chyby modelových operací se zapisují
i s tracebackem do výstupu Pythonu; neurčitou chybu „string“ bez tracebacku nelze
spolehlivě přiřadit k jedné příčině.

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
