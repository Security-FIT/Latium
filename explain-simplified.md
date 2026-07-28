# Zjednodušené vysvetlenie aktuálneho ROME lokalizátora

Aktívny komponent nie je binárny forenzný detektor. Z jedného podozrivého
checkpointu iba vyberie vrstvu s najsilnejšou diagonal-relative anomáliou.

## Päť krokov

Pre editovateľnú váhu `W_l` sa vytvorí normalizovaná hidden Gramova matica:

```text
C_l = hidden_gram(W_l) / ||W_l||_F^2
```

Odpočíta sa priemer susedných vrstiev:

```text
N_l = (C_{l-1} + C_{l+1}) / 2
R_l = C_l - N_l
```

Z rezidua sa vezmú dva hlavné ľavé SVD smery `u_1`, `u_2` a ich singulárne
hodnoty `sigma_1`, `sigma_2`. Susedná podpora sa v každom smere zmeria jedným
skalárom:

```text
b_i = u_i^T N_l u_i
```

Každá singulárna hodnota sa vydelí vlastnou podporou:

```text
s_l = sqrt((sigma_1 / b_1)^2 + (sigma_2 / b_2)^2)
```

Výsledkom je vrstva s najvyšším skóre:

```text
selected_layer = argmax_l s_l
```

Pôvodný M3 postup navyše diagonalizoval celú 2×2 podpornú maticu, rotoval
podpriestor, počítal inverznú odmocninu a robil obojstranný whitening. Tieto
kroky sú **DELETED**. Priamy experiment na 13 modeloch ukázal 196/240 správnych
lokalizácií pre zjednodušenie a 198/240 pre M3; modelové macro skóre sa líšilo
iba o 0,81 percentuálneho bodu.

## Čo lokalizátor nepoužíva

- čistý referenčný checkpoint;
- prompt, subject, target ani case ID;
- causal tracing alebo covariance;
- názov modelu ako feature;
- prah podľa rodiny modelu;
- morphology, bilateral features, B0, B1 alebo B2.

## Hranica binárnej detekcie

ROME zapisuje rank-one zmenu, ale rovnakú algebraickú stopu môže zapísať iný
program. Ak dva programy vytvoria identický výsledný checkpoint, detektor,
ktorý vidí iba tento checkpoint, ich nemôže rozlíšiť.

Experiment s 94 ROME pozitívami, piatimi čistými checkpointmi a 200 hard
negatívmi túto hranicu potvrdil. Najlepšie dvojštatistické pravidlo malo iba
18 % špecificitu na magnitude-matched random rank-one edit. OLMo bolo v
zvolenom M3 peaku úplne nerozlíšiteľné.

Po zjednodušení sa test zopakoval priamo s diagonal-relative profilom.
Najlepšie macro pravidlo malo iba 45,7 % senzitivitu, 75,6 % špecificitu a
48,0 % špecificitu na matched random rank-one editoch. Variant so 78,7 %
senzitivitou mal len 34,6 % celkovú špecificitu. Zjednodušenie teda nemení
záver o nemožnosti spoľahlivo určiť pôvodcu z jedného výsledného checkpointu.

Preto je poctivý verejný výstup iba:

```text
lokalizovaná vrstva + profil diagonal-relative skóre
```

Nie:

```text
ROME áno/nie
```

Pôvodný dôkaz je v `rome-single-checkpoint-impossibility-report.md`; priamy
experiment zjednodušenia je v `rome-simple-gram-simplification-report.md`.
