# Zjednodušené vysvetlenie aktuálneho ROME lokalizátora

Aktívny komponent nie je binárny forenzný detektor. Z jedného podozrivého
checkpointu iba vyberie vrstvu s najsilnejšou M3 anomáliou.

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

Reziduum a susedná podpora sa premietnu do top-2 SVD bázy:

```text
A_l = U_l^T R_l U_l
B_l = U_l^T N_l U_l
```

V malom 2×2 priestore sa odstráni mierka susednej podpory:

```text
E_l = B_l^(-1/2) A_l B_l^(-1/2)
s_l = ||E_l||_F
```

Výsledkom je vrstva s najvyšším skóre:

```text
selected_layer = argmax_l s_l
```

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

Preto je poctivý verejný výstup iba:

```text
lokalizovaná vrstva + profil M3 skóre
```

Nie:

```text
ROME áno/nie
```

Detailný dôkaz je v `rome-single-checkpoint-impossibility-report.md`.
