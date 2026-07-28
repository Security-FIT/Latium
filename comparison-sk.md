# Minimálny ROME lokalizátor verzus pôvodný detektor

Aktuálna implementácia zachováva najjednoduchší kandidát, ktorý sa zmestil do
vopred určenej tolerancie voči M3:

```text
jeden checkpoint
  -> normalizovaná hidden Gramova matica
  -> reziduum voči dvom susedom
  -> top-2 SVD podpriestor
  -> dve skalárne podpory smerov
  -> relatívne vektorové/Frobeniovo skóre
  -> argmax vrstvy
```

## Čo zostalo

- všeobecné nájdenie a orientácia editovateľných váh;
- normalizovaný hidden Gram;
- susedné hĺbkové reziduum;
- deterministický top-2 SVD;
- samostatná skalárna normalizácia dvoch SVD smerov;
- Frobeniovo skóre a deterministický argmax;
- všeobecné percentuálne trimovanie a numerické tolerancie.

Všetky kandidátske profily sa vypočítali v jednom zbere a potom sa
vyhodnocovali offline. Na 13 modeloch a 240 úspešných ROME editoch
diagonal-relative skóre lokalizovalo presne 196/240 prípadov (81,7 %; modelové
macro 82,7 %). M3 lokalizovalo 198/240 (82,5 %; macro 83,5 %). Rozdiel
0,81 percentuálneho bodu je v rámci vopred určenej tolerancie 2,5 bodu.
Oba rozdielne správne prípady patrili Falconu; na ostatných modeloch sa počty
správnych lokalizácií zhodovali.

Ide hlavne o matematické a implementačné zjednodušenie, nie o veľké zrýchlenie.
Čas naďalej dominuje vytvorenie Gramových matíc a top-2 SVD. Odstránenie malej
2×2 eigendecomposition a niekoľkých 2×2 násobení šetrí prácu v každej vrstve,
ale celkové zrýchlenie bude malé.

## Čo bolo odstránené

- M0, M1 a M2, pretože lokalizovali výrazne horšie;
- eigendecomposition 2×2 podpornej matice;
- rotácia podľa jej vlastných vektorov;
- inverzná odmocnina matice a obojstranný 2×2 whitening;
- rank-two multiplier, bilateral features, morfológia a `log1p`;
- B0, pretože potrebuje čistý referenčný checkpoint;
- B1 a B2, pretože nemali validovanú špecificitu;
- prahy podľa modelu alebo rodiny;
- experimentálny clusterový a grafový runtime.

## Prečo neexistuje binárny ROME verdikt

Nový vývojový korpus obsahuje 94 úspešných ROME editov, päť samostatných
čistých checkpointov a 200 hard negatívov. Najlepšie transparentné pravidlo
dosiahlo:

```text
senzitivita                      70,2 %
celková špecificita              64,4 %
špecificita na random rank-one   18,0 %
macro balanced accuracy          67,6 %
najhoršia rodina                 50,0 %
```

Malý logistický model s dvoma príznakmi bol ešte horší.

Nový diagonal-relative profil sa otestoval aj priamo na 94 úspešných ROME
editoch, piatich čistých checkpointoch a 200 nových matched negatívoch. Jeho
najlepšie macro spike pravidlo dosiahlo:

```text
senzitivita                      45,7 %
celková špecificita              75,6 %
špecificita na random rank-one   48,0 %
macro balanced accuracy          61,8 %
```

Variant s vyššou senzitivitou dosiahol 78,7 %, ale špecificita klesla na
34,6 % a rank-one špecificita na 18,0 %. Aj tento pokus zlyhal.

Je to očakávaná hranica: rank-one stopa ROME nie je jedinečná a iný program
môže vytvoriť rovnaký výsledný tensor. Produkcia preto vracia iba
diagonal-relative vrstvu a jej skóre. Netvrdí, že checkpoint definitívne alebo
pravdepodobne vytvoril ROME.

Úplné výsledky, intervaly neistoty a hashe sú v
`rome-single-checkpoint-impossibility-report.md`. Priamy experiment
matematického zjednodušenia je v
`rome-simple-gram-simplification-report.md`.
