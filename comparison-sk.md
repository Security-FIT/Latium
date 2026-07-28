# Súčasný minimálny ROME detektor verzus pôvodná weighted-spectrum metóda

## Krátka odpoveď

Súčasný detektor je **konsolidácia**, nie úplne nový matematický lokalizátor.

Jadro lokalizácie vrstvy z dokumentu `explain.md` zostalo zachované:

```text
editovateľná váha
  -> normalizovaná hidden Gramova matica
  -> hĺbkové reziduum voči dvom susedom
  -> vedúci dvojrozmerný SVD podpriestor
  -> 2x2 whitening podľa podpory susedov
  -> Frobeniovo skóre
  -> argmax cez povolené vrstvy
```

Odstránil sa najmä experimentálny obal okolo tohto skóre: dodatočné footprint
štatistiky, morfologické súčiny, slepé outlier pravidlá, alternatívne M0-M2
skóre, nekalibrované binárne varianty, renderovanie grafov a clusterová
ablačná infraštruktúra.

Výsledok je jednoduchší na pochopenie, testovanie a prevádzku. Robí tiež menej
práce a vytvára menšie artefakty než pôvodný plný ROME-presence alebo ablačný
pipeline. Neodstraňuje však dominantný výpočet Gramových matíc a top-2 SVD.
Asymptotická cena samotného checkpoint-only lokalizátora preto zostáva
prakticky rovnaká.

## Priame porovnanie

| Vlastnosť | Pôvodná metóda z `explain.md` a jej plný ROME-presence pipeline | Súčasný minimálny detektor |
|---|---|---|
| Verejná schéma | Viac lokalizačných a presence artefaktov | Jeden artefakt `rome-detector-minimal-v1` |
| Výstup iba z podozrivého checkpointu | Vždy vráti najsilnejšiu kandidátnu vrstvu | Rovnaké správanie |
| Binárne rozhodnutie bez čistého checkpointu | Blind-peak a blind-footprint | Odstránené, pretože neboli kalibrované na nezávislých negatívach |
| Binárne rozhodnutie s čistým checkpointom | Priamy rank-one rozdiel presne jednej váhovej matice | B0: numericky významný rank-at-most-two rozdiel hidden Gramových matíc |
| Lokalizačné skóre | Normalizovaný Gram, susedné reziduum, top-2 SVD, 2x2 whitening, Frobeniova norma | Rovnaké víťazné M3 skóre |
| Footprint hodnoty na vrstvu | Skóre, rank-two energy, bilateral coherence a bilateral balance | Iba `relative_subspace_frobenius` |
| Alternatívne skóre | Počas ablácie M0, M1, M2 a M3 | Iba M3 |
| Trimovanie | Pevný počet vrstiev, historicky `5/5` | Jedna všeobecná hodnota `10%`; zároveň sa vyradia koncové vrstvy bez oboch susedov |
| Pravidlá podľa rodiny modelu | Žiadne | Žiadne |
| Naučené prahy | Žiadne | Žiadne |
| Slepý štatistický prah | Median/MAD a univerzálny prah `sqrt(2 log n)` | Odstránený |
| Grafický výstup | Explainer grafy, CSV a JSON exporty | Žiadny detektorový grafický pipeline |
| Rozsah atribúcie | „ROME-like“ lokalizovaný rank-one edit | Opatrnejšie „ROME-compatible low-rank edit“ |
| Overená špecificita | Nie | Stále nie |

## Čo zostalo

### 1. Architektonicky neutrálna hidden Gramova matica

Pre editovateľnú projekciu $W_l$ obe metódy orientujú Gramovu maticu na menšiu
spoločnú hidden os:

$$
G_l =
\begin{cases}
W_lW_l^\top, & \text{ak počet riadkov} \leq \text{počet stĺpcov},\\
W_l^\top W_l, & \text{inak}.
\end{cases}
$$

Checkpoint-only lokalizátor ju normalizuje druhou mocninou Frobeniovej normy:

$$
C_l = \frac{G_l}{\lVert W_l\rVert_F^2}.
$$

Zostávajú tým dôležité vlastnosti pôvodnej metódy:

- žiadne routovanie podľa názvu alebo rodiny modelu;
- tolerancia voči transponovanému uloženiu váh;
- invariancia voči kladnému globálnemu preškálovaniu projekcie;
- spoločný hidden-space objekt porovnateľný medzi susednými vrstvami.

### 2. Lokálne hĺbkové reziduum

Obe metódy odhadujú normálny profil vrstvy $l$ pomocou jej susedov:

$$
N_l = \frac{C_{l-1}+C_{l+1}}{2},
\qquad
R_l = C_l-N_l.
$$

Toto je hlavný heuristický predpoklad: geometria normálneho modelu sa má cez
hĺbku meniť približne hladko, zatiaľ čo lokalizovaný edit vytvorí nezvyčajné
reziduum.

### 3. Top-2 SVD podpriestor

Obe metódy vezmú dva vedúce singulárne smery $U_l$ rezidua $R_l$. Motiváciou
je, že rank-one zmena váhy vytvorí v **nenormalizovanej** hidden Gramovej
matici zmenu s rankom najviac dva:

$$
(W+uv^\top)(W+uv^\top)^\top-WW^\top.
$$

Presný rank-two argument neprechádza bez zmeny cez samostatnú normalizáciu
oboch Gramových matíc. Toto obmedzenie z pôvodného `explain.md` stále platí.

### 4. Dvojrozmerný whitening podľa podpory susedov

Reziduum aj susedné pozadie sa premietnu do rovnakého dvojrozmerného
podpriestoru:

$$
A_l=U_l^\top R_lU_l,
\qquad
B_l=U_l^\top N_lU_l.
$$

Následne sa reziduum whitenne pomocou podpory susedov:

$$
E_l=B_l^{-1/2}A_lB_l^{-1/2}.
$$

Samotný whitening potrebuje iba eigendecomposition matice veľkosti
$2\times2$.

### 5. Frobeniovo lokalizačné skóre

Zachované M3 skóre je:

$$
s_l=\lVert E_l\rVert_F,
\qquad
\hat l=\operatorname*{arg\,max}_{l\in\mathcal L}s_l.
$$

Gramova matica, SVD, whitening a Frobeniova norma teda naďalej tvoria celé
matematické jadro lokalizácie.

## Čo sa priamo odstránilo

### Rank-two energy ako násobiteľ lokalizačného skóre

Pôvodný plný footprint ukladal podiel energie rezidua vysvetlený prvými dvoma
singulárnymi hodnotami. Experimentálne varianty ho kombinovali s jednoduchšími
skóre.

Z produkčnej lokalizácie bol odstránený, pretože vybrané M3 skóre tento
násobiteľ nepotrebovalo. Rank-two štruktúra však úplne nezmizla: B0 stále
kontroluje tail za druhou singulárnou hodnotou v nenormalizovanom Gramovom
rozdiele, kde je algebra ROME priamejšia.

### Bilateral coherence a bilateral balance

Plný presence capture meral:

- či sa stredná vrstva koherentne líši od oboch susedov;
- či majú ľavý a pravý skok podobnú energiu.

Tieto hodnoty sa snažili kódovať trojvrstvový footprint
$(-1/2,1,-1/2)$. Odstránili sa, pretože neboli súčasťou víťazného M3 skóre a
nemali preukázaný prínos pre binárnu špecificitu.

### Morfologický súčin

Pôvodné blind-footprint pravidlo násobilo:

```text
whitened spectral score
  x rank-two energy
  x bilateral coherence
  x bilateral balance
```

Takýto súčin pridával ďalšie pohyblivé časti bez nezávislých čistých a
hard-negative dát dokazujúcich, že odlišuje ROME od iných zásahov. Už sa
nepočíta.

### `log1p`, median/MAD a univerzálny extrémny prah

Pôvodné slepé pravidlá transformovali profil pomocou `log1p`, štandardizovali
peak robustným mediánom a MAD a porovnali ho s:

$$
\sqrt{2\log n}.
$$

Išlo o všeobecnú hranicu extrému pri gaussovskom šume, nie o empiricky
kalibrovaný prah pre ROME. Blind-peak aj blind-footprint boli preto
odstránené.

### M0, M1 a M2

N=50 vývojová ablácia namerala:

| Kandidát | Presná lokalizácia zo 450 prípadov |
|---|---:|
| M0 | 174/450 = 38,7 % |
| M1 | 222/450 = 49,3 % |
| M2 | 314/450 = 69,8 % |
| M3 | 386/450 = 85,8 % |

M0-M2 sa neuchovali ani ako voliteľné režimy. Produkcia ich už nepočíta,
neukladá ani nekonfiguruje.

### B1 a B2

B1 sa nedal kalibrovať bez nezávislých negatívnych checkpointov. B2 bol iba
experimentálny kontrolný variant. Minimálny detektor nevystavuje ani jeden.

### Ablačná a clusterová infraštruktúra

Z aktívnej produkčnej vetvy zmizli:

- evaluátor a porovnanie M0-M3;
- experimentálne B0-B2 cesty;
- recapture manifesty a clusterové joby určené iba na abláciu;
- veľké raw ablačné artefakty;
- bootstrap a non-inferiority výber;
- renderer a explainer grafy viazané na odstránené rozhodnutia.

Kompaktný N=50 golden fixture a výsledkové reporty zostali, aby bola zmena
auditovateľná.

### Pôvodné clean-delta pravidlo

Predchádzajúce pravidlo vyžadovalo, aby sa zmenila presne jedna kanonická MLP
output matica, a testovalo, či má priamy rozdiel váh rank jeden v rámci
numerickej chyby.

Súčasné B0 vytvorí nenormalizovaný hidden-Gram rozdiel:

$$
\Delta G_l =
G_l^{\mathrm{suspect}}-G_l^{\mathrm{clean}},
$$

a kontroluje:

1. či je jeho veľkosť nad hranicou odvodenou z dtype, rozmeru a mierky;
2. či energia za rankom dva zostáva v numerickej hranici.

Výsledok sa zámerne označuje iba ako
`generic_rank_at_most_two_gram_change`. Rovnakú stopu môže vytvoriť aj iná
rank-one editačná metóda.

## Čo sa zmenilo namiesto úplného odstránenia

### Pevný trim sa zmenil na percentuálny

Pôvodné nastavenie odrezávalo historicky päť prvých a päť posledných vrstiev.
To sa zle prenáša medzi plytkými a hlbokými modelmi.

Súčasné pravidlo odrezáva na každej strane:

$$
\left\lfloor0.10L\right\rfloor,
$$

pričom koncové vrstvy sa vždy vyradia, lebo nemajú oboch susedov. Stále je to
dizajnová konštanta, ale nie je špecifická pre konkrétnu architektúru.

### Binárny výstup má explicitný threat model

API oddeľuje dve tvrdenia:

1. **Iba podozrivý checkpoint:** nájde najsilnejšiu kandidátnu vrstvu; nejde
   o binárne áno/nie.
2. **Čistý aj podozrivý checkpoint:** B0 vráti boolean pre numericky významnú
   ROME-compatible low-rank Gramovu zmenu.

Bez čistého checkpointu je binárny výsledok `unavailable`, nie odhad založený
na nekalibrovanom slepom prahu.

## Je súčasná metóda rýchlejšia?

### Oproti pôvodnému checkpoint-only lokalizátoru

Iba mierne; bez kontrolovaného benchmarku možno rozdiel nebude merateľný.

Obe verzie vykonávajú rovnaké drahé kroky:

- hidden-space Gramove matice;
- susedné reziduá;
- randomizované top-2 SVD pre každé vyhodnotené reziduum.

Súčasná implementácia odstránila redundantný výpočet normy vlastných čísel a
berie Frobeniovu normu priamo z $2\times2$ whitened matice. V porovnaní s
veľkým Gramovým násobením a SVD je to malá úspora. Veľké zrýchlenie samotnej
M3 lokalizácie preto nie je podložené.

### Oproti pôvodnému plnému ROME-presence pipeline

Súčasná cesta nerobí:

- full-residual norm pre `rank2_energy`;
- dve bilateral jump energy;
- coherence a balance;
- morfologický súčin;
- blind-peak a blind-footprint analýzy;
- viac samostatných decision artefaktov;
- explainer grafy.

Ide prevažne o $O(h^2)$ elementwise výpočty na vrstvu a I/O. Úspora je reálna,
ale dominantný Gram/SVD zostal, takže nejde o rádové algoritmické zrýchlenie.
Ak boli predtým zapnuté grafy, úspora wall-clock času a diskového I/O môže byť
podstatne väčšia.

### Oproti M0-M3/B0 ablačnému behu

Historická experimentálna capture namerala:

- 389,3 agregovaných detektorových sekúnd pre 450 prípadov;
- približne 551 MiB maximálnej odhadovanej pracovnej pamäte.

Tento beh počítal M0-M3 aj B0. Minimálna produkčná cesta M0-M2 ani súvisiaci
evaluátor nespúšťa, takže robí menej práce. Neexistuje však párovaný
old-versus-current timing, preto nemožno poctivo tvrdiť napríklad „2x
rýchlejšie“.

### Keď je zapnuté B0

B0 potrebuje čisté aj podozrivé váhy, vytvorí ich nenormalizované Gramove
matice a odhadne rank-two tail. Je to práca navyše oproti samotnému M3.

Férové porovnania sú preto:

- M3 oproti pôvodnému weighted-spectrum lokalizátoru;
- M3+B0 oproti pôvodnému plnému ROME-presence pipeline.

## Pamäť a veľkosť artefaktov

Dominantným pracovným objektom zostáva jedna hidden-space Gramova matica.
Špičková pamäť preto zostáva $O(h^2)$. Rolling cache troch vrstiev bráni rastu
na $O(Lh^2)$, ale kvadratická závislosť od hidden rozmeru nezmizla.

Ukladaný profil je jednoznačne menší:

- starý plný footprint: štyri skalárne hodnoty na vrstvu;
- súčasný profil: jedna skalárna hodnota na vrstvu.

Ide o 75 % zníženie skalárnej časti per-layer profilu, nie nevyhnutne o 75 %
celého run adresára. Ďalšiu úsporu prináša odstránenie samostatných blind JSON,
ablačných exportov a grafov.

## Porovnanie dôkazov

Starý report uvádzal 38/40 presných lokalizácií. Pôvodné zmrazené artefakty,
ktoré by tento výsledok reprodukovali, neskôr neboli lokálne dostupné.

Väčší, ale plne odkrytý vývojový N=50 corpus pre zachované M3 ukazuje:

- 386/450 presných lokalizácií cez všetky prípady;
- 375/435 cez úspešné ROME edity;
- 434/435 pozitívnych B0 výsledkov;
- iba 9/50 lokalizácií pre Falcon.

Neskorší 13-modelový execution smoke ukázal:

- 20/25 presných lokalizácií;
- 25/25 pozitívnych B0 výsledkov;
- Falcon 0/2 a OLMo 0/2.

Protokoly sa líšia, takže 38/40 a 386/450 nie sú priame before/after porovnanie
presnosti. Zjednodušenie zachovalo M3 vzorec; nepreukázalo zvýšenie presnosti.

## Čo možno a nemožno tvrdiť

Súčasný detektor podporuje tvrdenia:

- kód nepoužíva prahy podľa rodiny modelu;
- checkpoint-only lokalizácia má jeden všeobecný vzorec;
- B0 má vysokú pozitívnu citlivosť na odkrytých ROME prípadoch;
- produkčný povrch je výrazne menší a zrozumiteľnejší.

Zatiaľ nepodporuje tvrdenia:

- spoľahlivá binárna ROME detekcia iba z podozrivého checkpointu;
- špecificita voči čistým modelom, fine-tune, kvantizácii, merge alebo iným
  editorom;
- jedinečná atribúcia programu ROME;
- spoľahlivá lokalizácia pre každú architektúru;
- konkrétny násobok zrýchlenia alebo úspory pamäte.

## Záver

Matematické zjednodušenie odstránilo najmä nepodložené vetvy okolo víťazného
skóre. Zachovalo časti s najjasnejšou ROME motiváciou: Gramovu geometriu,
top-2 SVD, 2x2 whitening a Frobeniovu lokalizáciu. Morfologické feature
engineering a nekalibrované štatistické rozhodnutia boli odstránené.

Detektor je preto menej prebudovaný a jednoduchšie obhájiteľný. Oproti plnému
presence/ablačnému workflow je menší a rýchlejší, no najdrahšie jadro sa
zámerne nezmenilo. Výraznejšie zrýchlenie by vyžadovalo implicitné
matrix-vector operácie, randomizované sketche alebo streamovaný low-rank
výpočet a následnú parity validáciu voči zmrazeným M3 dôkazom.

## Podklady porovnania

- Pôvodné matematické vysvetlenie: `explain.md`
- Pôvodná implementácia a dokumentácia: commit `3c220c8`
- N=50 ablácia: `rome-math-n50-cluster-report.md`, commit `693a949`
- Minimálna konsolidácia: commity `420bcd8` a `c0f4222`
- Súčasná dokumentácia: `docs/detector.md`
- Remote smoke: `rome-minimal-remote-smoke-report.md`
- Aktuálna vetva pred týmto dokumentom: `1044795`
