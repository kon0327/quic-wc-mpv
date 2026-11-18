# Compound řády proti prostým blokům: rozhodovací experiment (2026-11-16)

## Zadání a rozsah

- Uživatel 2026-11-16 se ptal, zda má QuIC smysl, když compound řády přidávají jen 0.5–3.6 % chyby. Přepočet na
  bajty (doplnění v [auditu 2026-11-12](2026-11-12-bf16-gate-optim-multilayer.md)) ukázal, že K=[1] potřebuje na
  stejnou chybu o 5–30 % víc bajtů. Uživatel schválil rozhodovací experiment **před M4-lite**, aby M4 běželo
  s ověřenou konfigurací.
- Současně se ptal, zda komprese šetří jen úložiště (odpověď v závěrečné zprávě; plán oddíl 1: MVP cílí na velikost
  checkpointu a přenos, ne na ustálenou GPU paměť).

## Plán (stanoveno před spuštěním)

- Vrstvy: SigLIP b13 fc1 (plný rank 1152) a Llama L15 q_proj (rank 1024); kalibrace
  `artifacts/pilot/2026-11-12-libero10-multilayer/calibration/`. Recept v2 (konjugované stupně, n = 8, BF16,
  3000 kroků, LR škálovaný hloubkou, výběr kroku podle trénovacího cíle), politika biasu `fixed`.
- Fit je deterministický (inicializace identitou, přesný cíl přes druhý moment, `stride` permutace bez náhody).
  Opakování se semenem by dalo stejný výsledek, proto replikace = tři struktury permutací: `stride`, `random`
  semeno 1, `random` semeno 2.
- Varianty pro každou strukturu (parametry na stupeň: K=[1] 4.5× víc, K=[1,2,3] 2.6× méně než K=[1,2]):
  - K=[1,2] (reference): q_proj L16/32/64/128, SigLIP L8/16/32/64.
  - K=[1], stejné bajty, 3000 kroků: q_proj L7/14/28, SigLIP L4/7/14.
  - K=[1], **stejný výpočet**: kroky = 3000 × čas párového K=[1,2] fitu (nejbližší bajty) / čas K=[1] (fáze B).
  - K=[1,2,3], jeden bod: q_proj L82 (≈ bajty K=[1,2] L32), SigLIP L42 (≈ K=[1,2] L16).
- Metrika: bajty, které křivka K=[1,2] téže struktury potřebuje na stejnou validační relL2 (log-log interpolace po
  Pareto frontě, bez extrapolace; mimo rozsah jen mez). Úspora K12 = 1 − bajty_K12 / bajty_varianty.
- **Kritérium (předem):** compound řády zůstanou hlavní variantou, pokud při nejmenším rozpočtu K=[1,2] ušetří
  ≥ 10 % bajtů proti K=[1] se stejným výpočtem pro každou strukturu na obou vrstvách. Jinak hlavní variantou bude
  K=[1] a compound řády zůstanou zkoumanou variantou (přeformulování přínosu — rozhodne uživatel).
- Odhad: 2.5–3 h GPU (RTX 4090).

## Rozhodnutí uživatele 2026-11-16: úspora paměti za běhu

- Uživatel na vysvětlení, že komprese zatím šetří jen úložiště, rozhodl: **po M4** rozjet obě větve — rozbalení
  vrstvy těsně před použitím a výpočet ve faktorizované podobě — a experimentálně zjistit, která dává smysl.
- Zapsáno do plánu: [instructions/implementation-plan.md](../instructions/implementation-plan.md), oddíl 12
  (návrh měření; neprovedeno).

## Soubory

- Nové: `tests/test_compare_orders.py`, `configs/orders_siglip_a.json`, `configs/orders_q_proj_a.json` (fáze A), `src/quic_wc/pilot/compare_orders.py`
  (`grid` = generování fáze B ze změřených časů, `report` = tabulka a vyhodnocení kritéria), tento audit.
- Výstupy: `artifacts/pilot/2026-11-16-orders/` (fáze B gridy `orders_*_b.json` vzniknou tam).

## Ověření před spuštěním

- Ortogonalita K=[1,2,3] v konjugovaném tvaru s náhodnou permutací (dim 1152, 4 stupně, θ ~ N(0, 0.3), float64):
  max |QᵀQ − I| = 2.9e-15.
- Smoke test celého řetězce (SigLIP, 20/40 kroků, scratchpad): fáze A → `grid` → fáze B → `report` proběhl.
  Při něm nalezeno a opraveno: (1) `entry["spec"]` je jen jméno, rodina fáze B se proto odvozuje ze jména
  (`_L<k>` se odstraní); (2) bod K=[1] horší než nejlevnější bod K=[1,2] se původně vykazoval jako „nelze
  interpolovat“, nyní jako dolní mez úspory („>“), která kritérium splní, je-li ≥ 10 %.
- Pozn.: stupeň 0 má u všech druhů permutací identitu, takže K=[1] L1 je pro všechny struktury stejný (týká se jen
  smoke testu, ne skutečného gridu).

## Průběh a výsledky

### SigLIP b13 fc1 (fáze A 10:47–11:17, fáze B 11:17–11:32; obě exit 0)

- Reprodukovatelnost: K=[1,2] `stride` L16/32/64 = 0.3891 / 0.3355 / 0.2785, shodně s během v2 z 2026-11-12.
- Fáze B: K=[1] dostal 7 700–16 100 kroků (časy K=[1,2] / K=[1] při 3000 krocích, např. 168 s / 31 s).
- Naměřeno (úspora K12 = kolik bajtů ušetří K=[1,2] proti variantě při stejné chybě; tři struktury):

| rozpočet (poměr bajtů K=[1]) | K=[1], 3000 kroků | K=[1], stejný výpočet | K=[1,2,3] (L42, 0.0148) proti K=[1,2] |
|---|---|---|---|
| 0.0165 (L4) | 40.1 / 42.2 / 41.5 % | 37.4 / 39.3 / 38.9 % | K=[1,2,3] lepší o 2.0 / 4.9 / 3.7 % bajtů, 3.6× delší trénink |
| 0.0280 (L7) | 23.6 / 24.5 / 25.2 % | 18.1 / 18.9 / 19.5 % | |
| 0.0550 (L14) | 10.9 / 12.4 / 13.1 % | 1.0 / 2.4 / 2.9 % | |

  (pořadí struktur stride / rand1 / rand2; finální čísla v `artifacts/pilot/2026-11-16-orders/report.md`)
- Interpretace: na SigLIP kritérium splněno se širokou rezervou (37–39 % ≥ 10 %). Výhoda compound řádů klesá
  s rozpočtem a při ~5.5 % bajtů a stejném výpočtu téměř mizí. Struktury permutací mění výsledek o < 2 p. b.
  K=[1] L4 je horší i než ASVD ≥B (0.4162–0.4208 vs 0.4084), K=[1,2] při menších bajtech ASVD poráží.

### Llama L15 q_proj (fáze A 11:32–13:04, fáze B 13:04–13:50; obě exit 0)

- Reprodukovatelnost: K=[1,2] `stride` L16/32/64/128 = 0.4580 / 0.4168 / 0.3705 / 0.3195 a K=[1] L7/14/28 =
  0.4324 / 0.3783 / 0.3256, shodně s během v2.
- Fáze B: K=[1] dostal 7 200–22 000 kroků (např. 590 s / 81 s při 3000 krocích).
- Naměřeno (úspora K12; stride / rand1 / rand2):

| rozpočet (poměr bajtů K=[1]) | K=[1], 3000 kroků | K=[1], stejný výpočet | K=[1,2,3] (L82, 0.0121) proti K=[1,2] |
|---|---|---|---|
| 0.0120 (L7) | 22.8 / 24.6 / 25.3 % | 18.6 / 19.5 / 20.3 % | K=[1,2,3] lepší o 2.0 / 2.5 / 1.4 % bajtů, 4.9× delší trénink |
| 0.0240 (L14) | 10.8 / 11.1 / 13.4 % | 11.4 / 2.6 / 5.0 % | |
| 0.0479 (L28) | 7.6 / −0.5 / 4.6 % | −0.2 / < −0.9 / < −0.9 % | |

  „< −0.9 %“ = K=[1] je lepší než nejlepší bod K=[1,2] (L128), skutečná hodnota není interpolovatelná.

### Vyhodnocení kritéria (`report.md`, 13:50)

- Při nejmenším rozpočtu K=[1,2] ušetří proti K=[1] se stejným výpočtem: SigLIP 37.4 / 39.3 / 38.9 %, q_proj
  18.6 / 19.5 / 20.3 % → **všech 6 případů ≥ 10 %, kritérium splněno: compound řády zůstávají hlavní variantou.**

### Interpretace

- Výhoda compound řádů je robustní při silné kompresi (≈ 1–1.6 % bajtů vrstvy): 19–39 % bajtů i při stejném
  výpočtu, rozptyl mezi strukturami permutací ≤ 2 p. b.
- S rostoucím rozpočtem klesá a kolem 5 % bajtů při stejném výpočtu mizí (q_proj −1…0 %, SigLIP 1–3 %). Velikost
  rozdílu tedy závisí na zvoleném rozpočtu; předchozí výrok „rovnocenné“ platil jen pro největší bod.
- K=[1,2,3] přidává 1.4–4.9 % bajtů za 3.6–4.9× delší trénink → jako hlavní variantu nedoporučuji; zůstává jako
  ablace.
- Pro M4-lite: K=[1,2] (recept v2). Pro budoucí větev výpočtu ve faktorizované podobě (plán oddíl 12) je
  relevantní, že při větším rozpočtu dává K=[1] stejnou kvalitu se 4.5× méně stupni (méně sekvenčních operací).

### Neúspěchy a odchylky

- Odhad doby 2.5–3 h překročen (3.05 h): bod K=[1,2,3] L82 na q_proj trvá 590 s (jako K=[1,2] L128).
- Monitor průběhu dvakrát vypršel (limit 30 min) a byl znovu nastaven; běh tím nebyl ovlivněn.
- Na q_proj `stride` L14 dal delší trénink K=[1] stejnou chybu (0.3788 vs 0.3783 při 3000 krocích); delší trénink
  pomáhá nerovnoměrně, jde o jeden bod.

### Ověření

- `run.log`: všechny 4 fáze a report exit 0; fit statusy `completed`.
- `tests/test_compare_orders.py` (nový, 3 testy: Pareto, interpolace, meze mimo křivku, rodiny fáze B);
  `pytest -q` → **76 passed**.
- Výstupy: `artifacts/pilot/2026-11-16-orders/{siglip,q_proj}_{a,b}/`, `orders_*_b.json`, `report.md`, `run.sh`,
  logy (adresář `artifacts/` je ignorovaný gitem).

### Otevřené body

- Jen dvě vrstvy; jeden checkpoint a dataset. Down_proj a DINOv2 nezahrnuty.
- Stejný výpočet = stejný čas na RTX 4090 s touto implementací; poměr časů závisí na implementaci stupňů.
- Volba K podle rozpočtu na vrstvu (K=[1,2] při silné kompresi, K=[1] při větším rozpočtu) je možná, ale
  nerozhodnutá — rozhodne uživatel.
