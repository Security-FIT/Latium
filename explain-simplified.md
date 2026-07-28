# Zjednodušené vysvetlenie ROME detektora s auditom odstránených častí

## Ako čítať tento dokument

Toto nie je ďalší návrh detektora. Je to kratšia a aktuálna verzia pôvodného
`explain.md`, doplnená o audit toho, čo sa po experimentálnej simplifikácii
naozaj používa.

Každá časť má jeden zo štyroch stavov:

- **KEPT** — výpočet je stále priamo súčasťou produkčného detektora;
- **CHANGED** — pôvodná myšlienka zostala, ale pravidlo alebo reprezentácia sa
  zmenili;
- **DELETED** — výpočet, feature, rozhodnutie alebo runtime cesta bola
  odstránená a minimálny detektor ju nepoužíva;
- **CONTEXT ONLY** — ide o matematické vysvetlenie, nie o samostatný krok
  implementácie.

Slovo **DELETED** tu neznamená, že matematická myšlienka je nesprávna. Znamená
iba, že sa už nepočíta ani nepoužíva pri produkčnom rozhodnutí
`rome-detector-minimal-v1`.

## TL;DR aktuálneho detektora

Produkčný ROME detektor má iba dve oddelené úlohy:

1. **M3 checkpoint-only lokalizácia:** z podozrivého checkpointu vyberie
   vrstvu s najsilnejšou lokálnou Gramovou anomáliou.
2. **B0 clean-reference boolean:** ak existuje čistý checkpoint, rozhodne, či
   rozdiel vyzerá ako numericky významná rank-at-most-two Gramova zmena.

Aktuálny tok je:

```text
W_l
  -> normalizovaný hidden Gram C_l
  -> susedné reziduum R_l
  -> top-2 SVD báza U_l
  -> 2x2 projekcia
  -> whitening
  -> Frobeniovo skóre s_l
  -> argmax vrstvy

clean W_l + suspect W_l
  -> nenormalizovaný Gram rozdiel Delta G_l
  -> veľkosť zmeny + rank-two tail
  -> B0 boolean
```

Nie sú v ňom morfologické súčiny, blind thresholdy, M0-M2, B1/B2 ani
detektorové grafy.

# Časť I — Matematické jadro

## 1. Vonkajší súčin a rank-one ROME update — KEPT ako motivácia

ROME mení editovateľnú váhu približne v tvare:

$$
W' = W + uv^\top.
$$

Matica $uv^\top$ má rank najviac jeden. Toto je dôvod, prečo detektor očakáva
nízkohodnostnú stopu.

Samotný lokalizátor nemusí poznať vektory $u$ a $v$. Vidí iba výslednú váhu.
Vonkajší súčin je teda matematická motivácia, nie dodatočný runtime feature.

**Stav: KEPT.** Rank-one štruktúra stále motivuje top-2 podpriestor a B0.

## 2. Hidden Gramova matica — KEPT

Pre projekciu $W_l$ sa vyberie menšia os:

$$
G_l =
\begin{cases}
W_lW_l^\top, & \text{ak počet riadkov} \leq \text{počet stĺpcov},\\
W_l^\top W_l, & \text{inak}.
\end{cases}
$$

Checkpoint-only lokalizácia odstráni celkovú mierku:

$$
C_l = \frac{G_l}{\lVert W_l\rVert_F^2}.
$$

Keďže:

$$
\operatorname{tr}(G_l)=\lVert W_l\rVert_F^2,
$$

ide o pôvodnú trace-one normalizáciu zapísanú priamo cez Frobeniovu normu.

**Stav: KEPT.** Aktuálna funkcia `hidden_gram(..., normalize=True)` robí tento
výpočet.

## 3. Pozitívna semidefinitnosť — CONTEXT ONLY

Gramova matica je pozitívne semidefinitná:

$$
x^\top G_lx \geq 0.
$$

Táto vlastnosť vysvetľuje, prečo sa Gram dá interpretovať ako energetická
geometria hidden priestoru.

**Stav: CONTEXT ONLY.** Nie je to samostatne počítaný feature ani rozhodovacie
pravidlo.

## 4. Normalizácia globálnej mierky — KEPT

Ak sa váha preškáluje $W_l\mapsto aW_l$, normalizovaný Gram sa nezmení:

$$
\frac{(aW_l)(aW_l)^\top}{\lVert aW_l\rVert_F^2}
=
\frac{W_lW_l^\top}{\lVert W_l\rVert_F^2}.
$$

**Stav: KEPT.** Pomáha generalizovať bez prahov podľa rodiny modelu.

## 5. Lokálny profil podľa susedov — KEPT

Očakávaná geometria vrstvy sa odhadne priemerom susedov:

$$
N_l=\frac{C_{l-1}+C_{l+1}}2.
$$

Reziduum je:

$$
R_l=C_l-N_l.
$$

Ide o diskrétnu lokálnu krivosť cez hĺbku. Prirodzená architektonická zmena
môže stále vytvoriť veľké reziduum; táto heuristika sa simplifikáciou
nezmenila.

**Stav: KEPT.**

## 6. Trojvrstvová odozva — KEPT algebra, DELETED explicitný feature

Ak sa zmení iba stredná vrstva $k$, reziduá majú idealizovanú odozvu:

$$
\left(-\frac12,1,-\frac12\right).
$$

Táto algebra stále vysvetľuje, prečo môže anomália zasiahnuť skóre vrstvy a
jej dvoch susedov.

Pôvodný full-presence pipeline sa tento tvar snažil explicitne merať cez
`bilateral_coherence` a `bilateral_balance`.

**Stav algebraickej motivácie: KEPT.**

**Stav explicitných bilateral features: DELETED.**

## 7. Top-2 SVD — KEPT

Z rezidua $R_l$ sa získajú dva dominantné singulárne smery:

$$
U_l\in\mathbb{R}^{h\times2}.
$$

Aktuálna implementácia používa deterministicky seedované randomizované top-2
SVD.

**Stav: KEPT.**

## 8. Rank-at-most-two Gramova stopa — KEPT, ale používa sa presne

Pre rank-one update:

$$
W'=W+uv^\top
$$

platí:

$$
W'W'^\top-WW^\top
=
(Wv)u^\top
+u(Wv)^\top
+\lVert v\rVert^2uu^\top.
$$

Výsledný rozdiel leží v priestore generovanom najviac dvoma smermi, preto má
rank najviac dva.

Dôležitá hranica:

- pre **nenormalizovaný** Gram rozdiel je argument presný;
- pre rozdiel dvoch samostatne normalizovaných Gramov už presný rank-two
  záver automaticky neplatí.

**Stav: KEPT.** Lokalizátor používa top-2 podpriestor ako motivovanú
heuristiku; B0 testuje rank-two tail na nenormalizovanom Gram rozdiele.

## 9. Projekcia do 2x2 podpriestoru — KEPT

Reziduum a susedná podpora sa premietnu do tej istej bázy:

$$
A_l=U_l^\top R_lU_l,
\qquad
B_l=U_l^\top N_lU_l.
$$

Obe matice majú rozmer $2\times2$.

**Stav: KEPT.**

## 10. Whitening — KEPT

Susedná podpora sa rozloží:

$$
B_l=Q_l\Lambda_lQ_l^\top.
$$

Po numerickom clampnutí malých vlastných čísel:

$$
B_l^{-1/2}
=Q_l\Lambda_l^{-1/2}Q_l^\top.
$$

Whitened reziduum je:

$$
E_l=B_l^{-1/2}A_lB_l^{-1/2}.
$$

Tak sa anomália meria relatívne k tomu, akú energiu majú susedné vrstvy v
rovnakých smeroch.

**Stav: KEPT.**

## 11. Generalizovaný eigenvalue problém — CONTEXT ONLY

Whitening súvisí s problémom:

$$
A_lv=\lambda B_lv.
$$

Aktuálny detektor však nespúšťa samostatný všeobecný generalized-eigen solver.
Počíta iba $2\times2$ inverse square root a kongruenčnú transformáciu.

**Stav: CONTEXT ONLY.** Matematická interpretácia zostáva správna, ale nejde o
ďalší runtime krok.

## 12. Frobeniova norma — KEPT a zjednodušená

Finálne skóre vrstvy je:

$$
s_l=\lVert E_l\rVert_F.
$$

Staršia implementácia najprv vypočítala vlastné čísla symetrickej $2\times2$
matice a potom ich vektorovú normu. Pre symetrickú maticu je to rovnaké ako
Frobeniova norma.

Aktuálna implementácia počíta priamo:

```text
matrix_norm(relative_subspace, ord="fro")
```

**Stav Frobeniovej normy: KEPT.**

**Stav redundantného `eigvalsh` medzikroku: DELETED.**

## 13. Argmax a margin — KEPT

Vybraná vrstva je:

$$
\hat l=\operatorname*{arg\,max}_{l\in\mathcal L}s_l.
$$

Pri presnej zhode skóre sa deterministicky vyberie nižšia vrstva. Rozdiel
medzi prvým a druhým skóre sa stále vracia ako `margin`.

**Stav: KEPT.**

`margin` však nie je pravdepodobnosť ani binárny dôkaz prítomnosti ROME.

# Časť II — Čo sa zmenilo

## 14. Pevné trimovanie `5/5` — CHANGED

Pôvodný detektor mal dva parametre:

```text
trim_first=5
trim_last=5
```

Aktuálny detektor používa jednu všeobecnú hodnotu:

$$
\text{trim}=\left\lfloor0.10L\right\rfloor.
$$

Koncové vrstvy bez oboch susedov sa vždy vyradia.

**Stav pevného `5/5`: DELETED.**

**Stav trimovania ako princípu: CHANGED na 10 % hĺbky.**

Nejde o model-specific hyperparameter; rovnaké pravidlo sa používa pre všetky
architektúry. Slepá oblasť na krajoch však stále existuje.

## 15. Numerická stabilizácia — CHANGED iba pre B0

M3 whitening stále clampuje veľmi malé vlastné čísla pred inverse square
root. Toto je ochrana numeriky, nie klasifikačný prah.

B0 nepoužíva ručne nastavený prah podľa modelu. Jeho hranice sa odvodzujú z:

- machine epsilon dátového typu;
- rozmeru matice;
- mierky clean a suspect Gramových matíc.

**Stav M3 stabilizácie: KEPT.**

**Stav binárneho model-specific thresholdingu: nepoužíva sa.**

# Časť III — Priamo odstránené feature a rozhodnutia

## 16. `rank2_energy` per-layer profil — DELETED

Starý full-presence capture ukladal:

```text
rank2_energy
```

Bol to podiel energie lokálneho rezidua zachytený prvými dvoma singulárnymi
hodnotami.

**Prečo bol odstránený:** nebol súčasťou víťazného M3 lokalizačného skóre.

**Používa sa dnes?** Nie ako per-layer localizer feature.

**Čo zostalo:** B0 meria rank-two tail priamo na clean-to-suspect
nenormalizovanom Gram rozdiele. To je odlišný a algebraicky presnejšie
umiestnený test.

## 17. `bilateral_coherence` — DELETED

Tento feature meral, či ľavý a pravý skok vytvárajú koherentnú stredovú
krivku.

**Používa sa dnes?** Nie.

**Dôvod odstránenia:** nebol potrebný pre M3 a neexistoval dôkaz, že zvyšuje
špecificitu binárneho ROME rozhodnutia.

## 18. `bilateral_balance` — DELETED

Tento feature porovnával energiu ľavého a pravého skoku.

**Používa sa dnes?** Nie.

**Dôvod odstránenia:** rovnaký ako pri coherence; išlo o neoverený
morfologický doplnok.

## 19. Morfologický footprint súčin — DELETED

Pôvodná blind-footprint cesta vytvorila:

```text
morphology =
    rank2_energy
    * bilateral_coherence
    * bilateral_balance
```

a násobila ním whitened spectral skóre.

**Používa sa dnes?** Nie. Výpočet ani pole v minimálnej schéme neexistujú.

**Dôvod odstránenia:** kombinácia pridávala matematickú zložitosť bez
kalibrovanej specificity.

## 20. `log1p` transformácia presence profilu — DELETED

Blind rozhodnutia používali:

```text
log1p(max(score, 0))
```

alebo rovnakú transformáciu po morfologickom násobení.

**Používa sa dnes?** Nie.

## 21. Median/MAD robustné z-skóre — DELETED

Starý suspect-only boolean odhadoval center a scale z vrstiev jedného modelu:

```text
center = median(values)
scale = 1.4826 * median(abs(values - center))
```

**Používa sa dnes?** Nie.

## 22. Univerzálny prah `sqrt(2 log n)` — DELETED

Blind peak sa porovnával s:

$$
\sqrt{2\log n}.
$$

Nebola to empíriou kalibrovaná hranica medzi clean a ROME modelmi.

**Používa sa dnes?** Nie.

**Dôsledok:** bez čistého checkpointu už minimálny detektor nevydáva
nepodložené binárne ROME áno/nie. Vydá iba kandidátnu vrstvu.

## 23. `blind-peak` — DELETED

Starý výstup:

```text
rome-presence-blind-peak
```

testoval, či je M3 peak nad univerzálnym outlier prahom.

**Používa sa dnes?** Nie.

## 24. `blind-footprint` — DELETED

Starý výstup:

```text
rome-presence-blind-footprint
```

vyžadoval súčasne spectral peak aj morphology-weighted peak.

**Používa sa dnes?** Nie.

## 25. Staré priame `rome-presence-delta` pravidlo — DELETED a REPLACED

Pôvodná delta cesta vyžadovala:

1. presne jednu zmenenú kanonickú MLP output maticu;
2. priamy rozdiel váh s rankom jeden v rámci roundoff.

**Používa sa dnes?** Nie v tejto forme.

**Náhrada:** B0 testuje numericky významný rank-at-most-two hidden-Gram
rozdiel. Výstup sa označuje iba ako `ROME-compatible low-rank edit`.

## 26. M0 — DELETED

Jednoduchý kandidát z ablácie dosiahol 174/450 presných lokalizácií.

**Používa sa dnes?** Nie.

## 27. M1 — DELETED

Ablačný kandidát dosiahol 222/450 presných lokalizácií.

**Používa sa dnes?** Nie.

## 28. M2 — DELETED

Ablačný kandidát dosiahol 314/450 presných lokalizácií.

**Používa sa dnes?** Nie.

## 29. M3 — KEPT

M3 dosiahol 386/450 presných lokalizácií a bol jediným kandidátom, ktorý
prešiel vopred definovaným non-inferiority výberom.

**Používa sa dnes?** Áno. Je to jediný produkčný lokalizátor.

## 30. B1 — DELETED

B1 bol plánovaný blind binary variant, ale bez nezávislých negatív sa
nekalibroval.

**Používa sa dnes?** Nie.

## 31. B2 — DELETED

B2 bol iba experimentálny kontrolný variant.

**Používa sa dnes?** Nie.

## 32. Viacnásobné profile fields — DELETED

Pôvodný full footprint ukladal na vrstvu:

```text
relative_subspace_frobenius
rank2_energy
bilateral_coherence
bilateral_balance
```

Aktuálna schéma ukladá iba:

```text
relative_subspace_frobenius
```

**Stav troch dodatočných polí: DELETED.**

## 33. Detector explainer renderer — DELETED

Pôvodná ROME-presence cesta mohla vytvoriť explainer grafy, case CSV/JSON a
agregované vizualizácie.

**Používa sa dnes?** Nie. `rome-detector-minimal-v1` nevytvára graph
artefakty.

Graf nebol súčasťou matematiky; jeho odstránenie nemení skóre.

## 34. Clusterová ablačná infraštruktúra — DELETED

Po výbere M3 sa odstránili:

- M0-M3/B0-B2 experimentálny evaluator;
- versioned artifact evaluator;
- recapture manifesty určené iba pre túto abláciu;
- smoke/full cluster joby;
- veľké raw výsledky z aktívneho stromu;
- bootstrap a non-inferiority runtime.

**Používa sa dnes?** Nie v produkcii.

Zostal iba kompaktný golden fixture a report, aby bolo možné overiť, z akých
dôkazov výber M3 vznikol.

# Časť IV — Aktuálny B0 boolean

## 35. Vstup — KEPT iba pri dostupnom clean checkpointe

B0 potrebuje:

- čisté editovateľné projekcie;
- podozrivé editovateľné projekcie;
- rovnaké vrstvy a kompatibilné rozmery.

Ak clean checkpoint nie je dostupný:

```text
available = false
is_rome_compatible = null
verdict = clean_reference_unavailable
```

Nevytvorí sa blind náhrada.

## 36. Nenormalizovaný Gram rozdiel — KEPT

Pre kandidátnu vrstvu:

$$
\Delta G_l=
G_l^{\text{suspect}}-G_l^{\text{clean}}.
$$

Normalizácia sa tu zámerne nepoužíva, aby sa zachoval priamy rank-two argument
pre rank-one zmenu váhy.

## 37. Veľkosť zmeny — KEPT

B0 meria:

$$
\text{change magnitude}
=
\frac{\lVert\Delta G_l\rVert_F}
{\lVert G_l^{\text{clean}}\rVert_F}.
$$

Výsledok musí prekročiť hranicu odvodenú z numerickej chyby výpočtu.

## 38. Rank-two tail — KEPT

Po odobratí najlepšej rank-two aproximácie sa meria zvyšná energia:

$$
\text{tail ratio}
=
\frac{\lVert\Delta G_l-(\Delta G_l)_2\rVert_F}
{\lVert\Delta G_l\rVert_F}.
$$

Pozitívne B0 vyžaduje, aby tail zostal v numerickej hranici.

## 39. Význam booleanu — CHANGED a zúžený

Pozitívny výsledok znamená:

> checkpoint obsahuje numericky významnú zmenu kompatibilnú s
> rank-at-most-two hidden-Gram stopou rank-one editu.

Neznamená:

> bolo dokázané, že konkrétny program ROME vytvoril checkpoint.

Iný rank-one editor môže vytvoriť rovnakú geometriu.

# Časť V — Čo nie je odstránené, hoci to vyzerá „matematicky“

## 40. Gramova matica — KEPT

Je drahá, ale rieši porovnateľnosť storage layoutov a hidden priestoru.

## 41. SVD — KEPT

Je drahé, ale vyberá dvojrozmerný nízkohodnostný signál motivovaný ROME.

## 42. Whitening — KEPT

Je matematicky zložitejší než obyčajná norma, ale N=50 ablácia vybrala práve
M3, ktoré whitening používa. Jednoduchšie M0-M2 neboli non-inferior.

## 43. Frobeniova norma — KEPT

Je finálnou jednoduchou agregáciou dvoch whitened módov.

Tieto štyri prvky sa neodstránili zámerne. Ide o jadro, ktoré prinieslo
najlepší pozorovaný generalizačný výsledok v dostupnom vývojovom corpuse.

# Časť VI — Aktuálne limity

## 44. Checkpoint-only cesta stále nie je binárna

M3 vždy vyberie nejakú povolenú vrstvu. Vysoké skóre samo osebe nedokazuje
edit.

## 45. B0 má zatiaľ iba pozitívnu citlivosť

Na úspešných N=50 ROME editoch bolo B0 pozitívne v 434/435 prípadoch.

Chýbajú nezávislé:

- clean checkpointy ako negatíva;
- iné rank-one editory;
- MEMIT alebo iné knowledge-editing metódy;
- ordinary fine-tune;
- quantized a merged checkpointy;
- adversarial hard negatives.

Preto nie je známa špecificita.

## 46. Lokalizácia nie je univerzálne spoľahlivá

Vývojový výsledok M3 bol 386/450, ale Falcon dosiahol iba 9/50. Neskorší smoke
odhalil aj OLMo 0/2.

Nebola pridaná žiadna family-specific oprava. To zachováva generalitu kódu,
ale neodstraňuje reálne architektonické confoundery.

## 47. Výpočtová zložitosť jadra zostáva

Hidden Gram má pamäť $O(h^2)$ a jeho vytvorenie aj top-2 SVD dominujú runtime.

Odstránené feature znižujú sekundárne $O(h^2)$ výpočty, počet artefaktov a I/O,
ale simplifikácia nie je asymptotická optimalizácia Gram/SVD jadra.

# Finálny prehľad stavov

| Pôvodná časť | Stav | Aktuálne použitie |
|---|---|---|
| Rank-one outer-product motivácia | KEPT | Motivuje top-2 a B0 |
| Hidden Gram | KEPT | M3 aj B0 |
| Trace/Frobenius normalizácia | KEPT | M3 |
| PSD interpretácia | CONTEXT ONLY | Žiadny samostatný feature |
| Susedný priemer | KEPT | M3 |
| Lokálne hĺbkové reziduum | KEPT | M3 |
| Top-2 SVD | KEPT | M3 a B0 tail |
| 2x2 projekcia | KEPT | M3 |
| Whitening | KEPT | M3 |
| Frobeniovo skóre | KEPT | M3 |
| Argmax a margin | KEPT | Lokalizačný výstup |
| Pevný trim `5/5` | DELETED | Nahradený 10 % |
| Percentuálny trim | CHANGED | Jedno pravidlo pre všetky modely |
| `rank2_energy` profile field | DELETED | B0 používa iný rank-two tail |
| `bilateral_coherence` | DELETED | Bez náhrady |
| `bilateral_balance` | DELETED | Bez náhrady |
| Morphology product | DELETED | Bez náhrady |
| `log1p` presence transform | DELETED | Bez náhrady |
| Median/MAD outlier | DELETED | Bez náhrady |
| `sqrt(2 log n)` cutoff | DELETED | Bez náhrady |
| Blind peak boolean | DELETED | Bez suspect-only boolean náhrady |
| Blind footprint boolean | DELETED | Bez suspect-only boolean náhrady |
| Starý direct rank-one delta | DELETED | Nahradený B0 Gram testom |
| M0 | DELETED | M3 bol lepší |
| M1 | DELETED | M3 bol lepší |
| M2 | DELETED | M3 bol lepší |
| M3 | KEPT | Jediný lokalizátor |
| B1 | DELETED | Nekalibrovaný |
| B2 | DELETED | Experimentálny control |
| Redundantný 2x2 `eigvalsh` norm krok | DELETED | Priama Frobeniova norma |
| ROME detector renderer | DELETED | Žiadne graph artefakty |
| Cluster ablation runtime | DELETED | Zostal iba golden dôkaz |
| B0 clean-reference Gram test | KEPT/NEW | Jediný binárny výstup |

# Aktuálny minimálny vzorec

Celý checkpoint-only lokalizátor možno zapísať:

$$
C_l=
\frac{\operatorname{hidden\_gram}(W_l)}
{\lVert W_l\rVert_F^2},
$$

$$
N_l=\frac{C_{l-1}+C_{l+1}}2,
\qquad
R_l=C_l-N_l,
$$

$$
U_l=\operatorname{top2\_left\_svd}(R_l),
$$

$$
A_l=U_l^\top R_lU_l,
\qquad
B_l=U_l^\top N_lU_l,
$$

$$
s_l=
\left\|
B_l^{-1/2}A_lB_l^{-1/2}
\right\|_F,
$$

$$
\hat l=
\operatorname*{arg\,max}_{l\in\mathcal L}s_l.
$$

To je všetko, čo aktuálna M3 lokalizácia matematicky potrebuje.

# Záver

Simplifikácia priamo odstránila feature engineering a nekalibrované rozhodnutia
okolo weighted-spectrum skóre. Neodstránila Gram, SVD, whitening ani
Frobeniovu normu, pretože práve ich kombinácia tvorí empiricky vybrané M3.

Najdôležitejšie odstránené veci sú:

```text
DELETED rank2_energy per-layer multiplier
DELETED bilateral_coherence
DELETED bilateral_balance
DELETED morphology product
DELETED log1p + median/MAD + sqrt(2 log n)
DELETED blind-peak
DELETED blind-footprint
DELETED M0, M1, M2
DELETED B1, B2
DELETED old direct rank-one delta rule
DELETED renderer and ablation runtime
```

Aktuálny detektor je preto menší a poctivejší v rozsahu svojich tvrdení:
M3 lokalizuje kandidátnu vrstvu a B0, iba s čistou referenciou, vracia
ROME-compatible low-rank boolean.

## Auditované zdroje

- Pôvodné vysvetlenie: `explain.md`
- Pôvodná implementácia: commit `3c220c8`
- N=50 výber M3: `rome-math-n50-cluster-report.md`, commit `693a949`
- Minimálna konsolidácia: commity `420bcd8` a `c0f4222`
- Aktuálna schéma a API: `docs/detector.md`
- Aktuálny remote smoke: `rome-minimal-remote-smoke-report.md`
