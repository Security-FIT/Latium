# Minimálny ROME lokalizátor verzus pôvodný detektor

Aktuálna implementácia zachováva iba lokalizačné jadro M3:

```text
jeden checkpoint
  -> normalizovaná hidden Gramova matica
  -> reziduum voči dvom susedom
  -> top-2 SVD podpriestor
  -> 2x2 whitening podľa susednej podpory
  -> Frobeniovo skóre
  -> argmax vrstvy
```

## Čo zostalo

- všeobecné nájdenie a orientácia editovateľných váh;
- normalizovaný hidden Gram;
- susedné hĺbkové reziduum;
- deterministický top-2 SVD;
- 2×2 whitening;
- Frobeniovo skóre a deterministický argmax;
- všeobecné percentuálne trimovanie a numerické tolerancie.

M3 lokalizovalo 386/450 požadovaných editov v pôvodnom deväťmodelovom N=50
vývojovom experimente. Falcon zostáva známym slabým miestom.

## Čo bolo odstránené

- M0, M1 a M2, pretože lokalizovali výrazne horšie;
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

Je to očakávaná hranica: rank-one stopa ROME nie je jedinečná a iný program
môže vytvoriť rovnaký výsledný tensor. Produkcia preto vracia iba M3 vrstvu a
jej skóre. Netvrdí, že checkpoint definitívne alebo pravdepodobne vytvoril
ROME.

Úplné výsledky, intervaly neistoty a hashe sú v
`rome-single-checkpoint-impossibility-report.md`.
