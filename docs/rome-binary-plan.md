# Binárna detekcia popri ROME reversal bez ručne ladeného prahu

Dátum: 2026-10-24. Východisko: `origin/ROME-reversal`, commit `54a829e`.
Pôvodný stav k 2026-10-24: implementovaný experimentálny prototyp
`gram-peel-mdl-v1`; reálna flotila vtedy nebola overená. Aktualizácia
2026-10-29: optimalizovaná v2 dokončila N=10 na 18 modeloch, ale má slabú
pozorovanú citlivosť. Implementácia, PBS história a konkrétne výsledky sú
v [súhrne behov](rome-binary-pbs.md). Presný kodek, spustenie a implementované rozšírenie
normalizačnej mierky sú v [gram-binary.md](gram-binary.md).

## Odporúčanie

Ponechať súčasný GRAM ako lokalizátor a existujúci peeling ako generátor
kandidátnej rank-one korekcie. Vedľa nich pridať **MDL výber medzi nulou a
jednou korekciou**: prijať korekciu iba vtedy, keď jej zavedenie skráti celkový
opis pozorovaných Gramových matíc aj po zaplatení opisu vrstvy a korekcie.

Výstup je `edited: true/false`. Rozhodnutie nepoužíva `model.layer`, causal
tracing, čistý checkpoint, faktický prompt ani ručne nastavené `collapse > t`.
Pri negatívnom výsledku `layer: null`; pôvodný GRAM argmax zostáva diagnostika.

Presný význam pozitívneho výsledku je **izolovaná rank-one anomália kompatibilná
s ROME podľa zvoleného modelu pozadia**. Zo samotných váh nemožno všeobecne
dokázať, akým algoritmom vznikli. `false` znamená, že tento model anomálie
nevyhral, nie dôkaz neprítomnosti ľubovoľnej úpravy.

MDL odstraňuje ručné ladenie rozhodovacieho prahu, nie potrebu definovať
predpoklady o prirodzených váhach. Voľba opisu pozadia je súčasť metódy a musí
byť rovnaká pre všetky rodiny. Hranica nula nižšie je hranica medzi dvoma
dĺžkami opisu, nie prah empiricky vybraný na skóre.

## Čo je na vetve dnes

| Časť | Skutočné správanie |
| --- | --- |
| `src/structural/detectors/rome_layer_localizer.py` | Stopovo normalizované hidden Gramy, priemer dvoch susedov, top-2 SVD rezídua, energeticky relatívne skóre, argmax. |
| `src/structural/analysis/detector_methods.py::analyze_gram_localization` | Vráti vrstvu a skóre. Neimplementuje binárny test zhody s configom/CT. |
| `src/graphs/gram.py` | Porovnanie predikovanej a cieľovej vrstvy je evaluačná lokalizačná metrika. Baseline argmax je výslovne kontrola, nie edit-presence verdikt. |
| `src/structural/detectors/gram_peel.py` | Hľadá smer v rovine najkladnejšieho a najzápornejšieho vlastného vektora; vykoná rank-one peeling. |
| `iterative_peel` | Skúša päť najvyššie skórovaných vrstiev a prijíma peel pri `DEFAULT_MIN_COLLAPSE = 0.7`. |
| `docs/rome-reversal.md` | Malé GPT-2/GPT-2 XL experimenty; dokumentuje aj zníženie pôvodného prahu 0.85 na 0.7. |

Nie je teda potrebné nahrádzať existujúci binárny CT test v tejto vetve.
Potrebujeme doplniť samostatné rozhodnutie a oddeliť ho od lokalizačnej accuracy.
Existujúce pole `normalized` z peelingu nie je nový binárny detektor.

## Čo z dodanej analýzy zachovať a čo opraviť

### Platná algebra sa týka rozdielu oproti skutočnému pôvodnému Gramu

Orientujme váhy ako `W ∈ R^(d×m)` s výstupným reziduálnym priestorom v riadkoch.
Pre `W' = W + abᵀ`, `G = WWᵀ`, `c = Wb` platí

\[
\Delta G = ca^\top + ac^\top + \|b\|^2 aa^\top,\qquad
\operatorname{rank}(\Delta G)\leq 2.
\]

Ak sú `a,c` nezávislé, dve nenulové vlastné čísla majú súčin

\[
\lambda_+\lambda_-=-\bigl(\|a\|^2\|c\|^2-(a^\top c)^2\bigr)<0.
\]

Nezávislosť nie je garantovaná definíciou ROME. Kolineárny nenulový update môže
mať Gramovu zmenu hodnosti jeden alebo aj nula. Ani
`tr(ΔG)=2aᵀc+||b||²||a||²` nemusí byť kladná.

### Stopová normalizácia pridáva plnohodnostný člen

Označme `A=G/tr(G)`, `A'=G'/tr(G')`, `t=tr(G)`, `t'=tr(G')` a `κ=t/t'`.
Potom presne

\[
A'=\kappa A+\frac{\Delta G}{t'},\qquad
A'-A=\frac{\Delta G}{t'}+(\kappa-1)A.
\]

Nízku hodnosť má `A'−κA`, nie všeobecne `A'−A`. Navyše detektor pozoruje
`R_l=A'_l−N_l`; susedský priemer `N_l` nie je známy čistý `A_l`.
Rank-two tvrdenie sa preto nedá bez ďalšieho preniesť na toto rezíduum.

Každé nenulové symetrické `R_l` má stopu nula a aspoň jedno kladné aj záporné
vlastné číslo. Samotná indefinitnosť je tu takmer tautológia. Opačné znamienka
dvoch absolútne najväčších vlastných čísel sú už dodatočná vlastnosť, ale ani
tá nie je nutnou a postačujúcou podmienkou ROME.

### Laplaciánová šablóna je presná pre zmenu, nie pre celé pozorované rezíduá

Pri zmene jediného `A_l` o `D` platí

\[
\delta R_{l-1}=-D/2,\quad \delta R_l=D,\quad \delta R_{l+1}=-D/2.
\]

Pozorujeme však `R_clean + δR`, nie samotné `δR`. Prirodzený izolovaný skok
vytvorí rovnakú šablónu. Použiť ju na spoločné modelovanie okolia a korektné
prepočítanie susedov; nepridávať ju ako nezávislý dôkaz s ručne zvolenou
toleranciou. Na výpočet piatich celých rezíduí treba až sedem Gramových matíc.

### Pravostranná ablácia nie je exaktné obnovenie pôvodného modelu

Aj pri znalosti pravého `b` dostaneme

\[
W'(I-bb^\top/\|b\|^2)=W(I-bb^\top/\|b\|^2),
\quad
G_{abl}=G-\frac{(Wb)(Wb)^\top}{\|b\|^2}.
\]

Zmizne edit, ale aj pôvodná zložka váh. Kolaps oboch špičiek voči susedom nie
je garantovaný. Odhad `W'ᵀu` navyše obsahuje čistú zložku `Wᵀu`; nie je
automaticky pravým faktorom editu. Pomer `1−s_after/s_before` nemusí ležať
v `[0,1]`. Existujúci peeling s dopočítaním náhradného riadka je preto lepší
východiskový kandidát než jednoduché nulovanie pravej projekcie.

BBP, kurtosis a logit lens ponechať ako výskumné diagnostiky. BBP vyžaduje
konkrétny náhodnomaticový model, ktorý pre medzivrstvové rezíduá zatiaľ nemáme.
Kurtosis neurčuje jednoznačne `C⁻¹k` a je závislá od bázy. Logit lens nemusí
izolovať cieľový token a závisí aj od normalizácie/teploty. Hodnoty `Γ>10`,
`Z>6`, `collapse>0.6` či intervaly stencil pomerov by požiadavku bez ručného
prahu opäť porušili. Tvrdenia o nulových FP a univerzálnej citlivosti zatiaľ
nemajú experimentálnu oporu.

### Malé overené protipríklady

Pre `W=I₃`, `b=e₁`, `a=e₂` má nenormalizovaná Gramova zmena vlastné čísla
`[-0.618034,0,1.618034]`, ale normalizovaná zmena
`[-0.237842,-0.083333,0.321175]`: hodnosť tri.
Pre `a=(-0.5,0.1,0)` je zmena stopy `−0.74`, hoci `a,c` nie sú kolineárne.
Pre `a=0.1e₂` pravostranná ablácia pravého `b` anomáliu vo Frobeniovej norme
zväčší; príslušný pomer kolapsu je približne `−7.675`.
Napokon `W'=I₃−2e₁e₁ᵀ` je nenulový rank-one zásah s úplne nezmeneným Gramom.
Tieto príklady boli skontrolované numericky; nie sú benchmarkom detektora.

## Matematika navrhovaného rozhodnutia

### 1. Použiť skutočnú rank-one korekciu súčasného peelingu

Pre kandidátny smer `u` a existujúci `mixer=z` peeling realizuje

\[
\hat k=W'^\top z,\qquad W_p=W'-u\hat k^\top=(I-uz^\top)W'.
\]

Pracujme v jednotkách aktuálneho normalizovaného Gramu `A` a definujme

\[
g=Az,\quad \gamma=z^\top Az,\quad
H=A-ug^\top-gu^\top+\gamma uu^\top,\quad c=\operatorname{tr}(H)>0.
\]

Peeled Gram je `B=H/c`. Platí presný rozklad

\[
\boxed{A=cB+K},\qquad
K=ug^\top+gu^\top-\gamma uu^\top,\quad \operatorname{rank}(K)\le2.
\]

To zachováva stopovú normalizáciu správne a nadväzuje priamo na terajší kód.
Kandidáta možno vyhodnocovať v Gramovom priestore; obdĺžnikové váhy treba
znovu čítať až pri požadovanej rekonštrukcii `k`. Hodnotíme aj zmenené
susedské rezíduá, nie iba skóre aktuálnej vrstvy.

### 2. Porovnať dva úplné opisy tých istých dát

Nech `X=(A₀,…,A_(L−1))` sú Gramove matice jedného podozrivého checkpointu
v pevne definovanej numerickej reprezentácii. Nech `C₀(X)` je dĺžka opisu
prirodzenej medzivrstvovej variability. Definujeme

\[
\mathcal L_0=C_0(X),
\]

\[
\mathcal L_1(l)=L(l)+L(c_l,K_l)+C_0(X^{p,l})+L(E_{recon,l}),
\]

kde `X^(p,l)` má iba `A_l` nahradené peeled Gramom `B_l`. `E_recon` kóduje
zvyšnú chybu pri rekonštrukcii pôvodných dát z konečne presných parametrov.
Hlavička voľby `H₀/H₁` má v oboch prípadoch rovnakú dĺžku. Pre známu množinu
`E` má vrstva cenu napr. `ceil(log₂ |E|)` bitov; kandidátna korekcia musí
zaplatiť aj svoje vektory, skaláre a ich presnosť.

\[
\Delta_{MDL}=\mathcal L_0-\min_{l\in E}\mathcal L_1(l),
\qquad
\boxed{\texttt{edited}=\mathbf1[\Delta_{MDL}>0]}.
\]

Pri zhode vyhrá `H₀`. Lokalizácia pozitívneho nálezu je vrstva s najkratším
opisom `H₁`; nie nevyhnutne pôvodný GRAM argmax. GRAM slúži aj na poradie
výpočtu, ale v prvej korektnej verzii sa musia vyhodnotiť všetky prípustné
vrstvy. Top-5 obmedzenie môže znovu stratiť Granite/Olmo edit.

Zlepšenie skóre po peelingu samo osebe nestačí: peeling ho priamo optimalizuje.
Práve cena dodatočnej štruktúry bráni tomu, aby každá optimalizovaná korekcia
automaticky vyhrala. Parametre sa nesmú odvodzovať z dát „zadarmo“.

### 3. Konkrétna prvá verzia opisu pozadia

Pre prvý prototyp navrhujem dvojdielny kód sekvencie horných trojuholníkov
Gramových matíc; neukladať symetrické prvky dvakrát:

1. Prvé dve matice zakódovať priamo.
2. Ďalšiu predikovať lineárnym pokračovaním `2A_(l−1)−A_(l−2)`.
3. Predikčné rezíduá kódovať symetrickým rozdelením s vrstvovým meradlom,
   napr. Student-t s pevnými tromi stupňami voľnosti. Meradlo sa odhadne
   z daného rezídua a jeho konečný zápis sa tiež zaplatí.
4. Každá vrstva má aj možnosť resetu/priamo uloženého bloku; zaplatiť bit
   voľby režimu. Tým H₀ vie opísať aj prirodzený širokopásmový prechod.
5. Pravdepodobnosti previesť na hmotnosti buniek rovnakej reprezentácie dát,
   aby obe alternatívy mali porovnateľné bitové náklady. Ak sa použije
   kvantizácia, jej chybu tiež zakódovať; nevyberať jej presnosť podľa labelov.

Je to návrh konkrétneho kodeku, nie teoréma o rozdelení váh transformerov.
Kodek a všetky hlavičky treba špecifikovať a otestovať pred meraním accuracy.
Súčin skalárnych kódov je platný opis aj pre korelované prvky, no môže ich
korelácie využívať slabo; nevytvára tým kalibrovanú pravdepodobnosť nálezu.

`K` kódovať faktorizovane, nie ako plnú maticu: napríklad cez dve bázy a malú
symetrickú maticu alebo cez `(u,g,γ)`. Pri druhom zápise je `c` určené stopou
a nekóduje sa druhýkrát. Presnosť parametrov môže vyberať samotné MDL cez
dyadické presnosti, ak sa zaplatí aj ich index a rekonštrukčné rezíduum.
Dekóder musí reprodukovať rovnaké pôvodné dáta pre `H₀` aj `H₁`.

To je zámerne explicitnejšie než nasadiť `BIC=n log(RSS/n)+k log n` nad
`d(d+1)/2` prvkami. Efektívny počet nezávislých pozorovaní tu nepoznáme a
rank-one model je pri nulovom update neidentifikovateľný. Štandardná BIC
penalizácia preto bez ďalšieho odvodenia nie je záruka korektného testu.

### 4. Whitening až ako oddelené rozšírenie kandidátov

Z analýzy má zmysel skúsiť `Q_l=N_l^(-1/2) A_l N_l^(-1/2)` na numericky
podporenom podpriestore. Keby `N_l` bolo presne čisté normalizované `A_l`,
potom `Q_l=κI+rank≤2`: treba modelovať aj posun `κI`, nie vynucovať nuly
zvyšku spektra. Pri skutočných susedoch pribúda neznáme pozadie.

V prvej verzii ponechať existujúce kandidátne roviny. Whitening pridať až
v samostatnom experimente pre maskované edity a s rovnakým MDL rozhodnutím.
Numerická tolerancia podľa dtype/rozmeru nie je štatistický dôkaz čistoty;
ak bude potrebná silnejšia regularizácia, jej voľbu a náklad zahrnúť do metódy.

## Implementačné kroky

### A. Zachytiť potrebné dáta bez leakage

- Nový modul `src/structural/detectors/gram_binary.py`; pôvodný localizer
  a `gram_peel.py` zostávajú porovnávacie metódy.
- Nový capture/analysis pár `gram-binary` v príslušných registroch a
  `detector_methods.py`. Vstup musí obsahovať Gramy alebo ich reprodukovateľný
  obmedzený cache; dnešné jediné skóre na vrstvu nestačí na nový test.
- Využiť streaming Gram okno a prípadne zdieľať výpočet s `profile_weights`;
  nenútiť ukladať všetky Gramy všetkých editov na disk.
- API `detect(weights)` alebo `detect_from_grams(grams)` neprijíma cieľovú
  vrstvu, čisté Gramy, edit request ani úspešnosť ROME. Config je prípustný na
  rozpoznanie architektúry a názvov projekcií, nie ako dôkaz editu.
- Posúdiť `GramProfileCache`: reuse susedov z čistého checkpointu je
  benchmarková optimalizácia; každý verdikt musí byť reprodukovateľný
  samostatným scanom podozrivého checkpointu bez znalosti zmenenej vrstvy.

### B. Najskôr prototyp jedného rozhodnutia

- Implementovať `C₀`, kód korekcie a round-trip rekonštrukciu.
- Vypočítať jednu korekciu na každej prípustnej vrstve cez terajší peeling.
- Pre každú vypočítať úplné `L₁`; rozhodovať vyššie uvedeným porovnaním.
- Zatiaľ jedna korekcia verzus žiadna; neinterpretovať počet peels ako počet
  editov. Multi-edit model musí platiť cenu ďalších korekcií aj ich poradia.
- Ukladať `edited`, `layer`, `mdl_delta_bits`, `null_bits`, `edit_bits`,
  rozpis ceny korekcie/pozadia/rekonštrukcie, pôvodný argmax, kolaps len ako
  diagnostiku, pokrytie vrstiev a verziu metódy/kodeku.
- Neplatné vstupy alebo neúspešná numerika majú `status: error`, nie tiché
  `edited: false`. Na platných podporovaných vstupoch je výsledok binárny.

### C. Výkon až po overení kritéria

- Terajšia uhlová mriežka opakuje veľké lineárne solve; nie je prijateľné
  odhadovať runtime iba cenou jedného dodatočného outer productu.
- Zdieľať faktorizácie a vyhodnocovať Gramovu aktualizáciu cez nízku hodnosť.
  Pri skórovaní opisu prepočítať len bloky dotknuté zmenou a prediktorom.
- Premerať čas a pamäť na reálnych rozmeroch; až potom pridať orezávanie
  kandidátov s preukázateľnou hranicou bitového zlepšenia. Bez takej hranice
  top-k screening uvádzať ako aproximáciu a zmerať jeho straty.

### D. Až potom voliteľné spojenie s reversal

- Nový verdikt najprv beží vedľa existujúceho `iterative_peel`.
- Ak sa osvedčí, prijať konkrétny peel pri skrátení opisu; tým môže nahradiť
  `min_collapse` aj pravidlo zastavenia. Každý ďalší peel platí nový opis.
- Binárna statická detekcia nespúšťa dekódovanie subjektu ani behaviorálne
  sondy. Súčasný reversal ich využíva ako ďalšiu funkciu a evaluáciu.

## Experiment, ktorý návrh môže aj vyvrátiť

1. Algebra: normalizačný člen, kolineárny update, záporná zmena stopy,
   Gramovo neviditeľný update, presný Gram/weight peeling a rekonštrukcia
   kodeku. Zmyslom je zabrániť tvrdeniam silnejším než samotná algebra.
2. Syntetika: hladký drift, prirodzený širokopásmový skok, izolovaný prirodzený
   rank-one jav, slabý/silný rank-one update, dve upravené vrstvy,
   korelované pozadie, fp32/bf16 a obe orientácie váh.
3. Malý reálny pilot: GPT-2 a GPT-2 XL na nadviazanie na reversal; Llama-2
   a oba Mistraly na FP; Granite/Olmo na masking; aspoň jeden Qwen ako
   ďalšia rodina. Bez príslušných váh sa nedá pilot nahradiť uloženým argmaxom.
4. ROME v štandardnej vrstve aj v iných prípustných vrstvách. Ground truth
   používa iba evaluator; zmena `model.layer` po vytvorení rovnakého checkpointu
   nesmie zmeniť verdikt. To isté platí pre pridanie/odstránenie CT artefaktov.
5. Kontroly: neupravené base modely a komunitné fine-tuny; normou a umiestnením
   porovnateľné náhodné rank-one/rank-two zásahy, LoRA a viacvrstvové zásahy.
   ROME, ktoré zmenilo váhy, ale nezmenilo odpoveď, oddeliť od aplikácie,
   ktorá váhy vôbec nezmenila. Úspešnosť odpovede nie je label zmeny váh.
6. Porovnať pôvodnú GRAM lokalizáciu, dnešný peel s 0.7 a nový MDL verdikt
   na rovnakých checkpointoch. Lokalizáciu a binárnu detekciu vyhodnocovať
   osobitne: FP/TN na čistých, TP/FN na upravených, pokrytie/chyby, presná
   vrstva aj odchýlka ±1, čas a pamäť. Žiadne očakávanie nulových FP vopred.
7. Pred rozšírením na celú flotilu zmraziť kodek, kandidátny algoritmus,
   numerickú reprezentáciu a rozhodovacie pravidlo. Finálnu validáciu rozdeliť
   podľa pôvodu/base rodiny; príbuzné fine-tuny nesmú vytvárať falošný dojem
   generalizácie. Úprava kodeku po výsledkoch znamená novú verziu a nový test.

Ak kodek považuje prirodzené Llama/Mistral prechody za kratšie vysvetliteľné
rank-one korekciou, zlyhal model pozadia; neopravovať to rodinným thresholdom.
Ak nerozlíši ROME od LoRA/random rank-one, výsledok je detektor rank-one
anomálií, nie detektor algoritmického pôvodu. Ak nenájde slabé Granite/Olmo
edity, ďalšou hypotézou je lepší generátor smerov alebo nedostatok informácie,
nie automaticky zníženie požadovaného bitového zisku.

## Limity a zdroje

Z jediného checkpointu existuje vždy viac možných histórií jeho vzniku.
Navyše `WWᵀ` stráca pravostrannú ortogonálnu informáciu: niektoré nenulové
rank-one zmeny nechajú Gram nezmenený. Žiadny nový rozhodovací krok nad tým
istým Gramom tieto informácie neobnoví. Výsledok musí byť empiricky validovaný
modelový úsudok, nie bezpodmienečný algebrický certifikát ROME.

MDL je štandardný princíp výberu modelu; jeho aplikácia na tento peeling,
navrhnutý kodek a jeho očakávaná užitočnosť sú tu nový, zatiaľ neoverený návrh.

- Barron, Rissanen, Yu: [The Minimum Description Length Principle in Coding and Modeling](https://research.ibm.com/publications/the-minimum-description-length-principle-in-coding-and-modeling).
- Baik, Ben Arous, Péché: [Phase transition of the largest eigenvalue for non-null complex sample covariance matrices](https://arxiv.org/abs/math/0403022). Výsledok pre konkrétny model kovariančných matíc; nedokazuje BBP model týchto rezíduí.
- Hase et al.: [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213). Podporuje oddelenie CT lokalizácie od výberu vrstvy na editovanie.
- Youssef et al.: [Tracing and Reversing Rank-One Model Edits](https://arxiv.org/abs/2505.20819). Podporuje výskum forenzných stôp a reverzie; nie univerzálne garancie navrhovaných invariantov.
