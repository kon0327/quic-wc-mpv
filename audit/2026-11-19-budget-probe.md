# Sonda rozpočtu skupiny MLP DINOv2 (2026-11-19)

## Zadání

- Sekvenční kalibrace při L16 skupinu nezachránila: KL 10.2 proti 12.3, viz
  [audit sekvenční kalibrace](2026-11-19-sequential-calibration.md).
- Uživatel 2026-11-19 souhlasil s přechodem na vyšší rozpočet nebo doladění. Zadal také plán tandemového
  experimentu a, b, c „po vyřešení sekvenční kalibrace“; plán je v
  [instructions/tandem-recovery-plan.md](../instructions/tandem-recovery-plan.md).
- Cíl sondy: levně zjistit, při jakém rozpočtu na vrstvu je celá skupina (48 vrstev) použitelná.
  - Použito nezávislé ASVD, které se počítá okamžitě.
  - QuIC-WC je při stejné relL2 po vrstvách o 36–72 % úspornější, takže ASVD dává horní mez potřebných bajtů.

## Provedeno

- Ruční plán `artifacts/pilot/2026-11-19-budget-probe/m4_plan.json`: ASVD rank 32 / 64 / 128 / 256 / 512 na všech
  48 vrstvách, korigovaný bias, kalibrace z 18. 11.
- Vyhodnocení `m4_lite run` → `m4/results.{json,md}`, 268 validačních snímků.
- **Chyba:** běh spadl až při psaní `results.md`, protože ruční ASVD replacementy nemají `pilot_rel_l2`; všechna
  měření byla v `results.json` hotová.
  - Oprava: `m4_lite.pilot_error()` použije `check_rel_l2`.
  - `results.md` jsem vygeneroval z `results.json`; `tests/test_m4_lite.py` → 3 passed.

## Naměřeno

| ASVD rank | poměr bajtů | ⌀ relL2 pilot | ⌀ in-situ | tf KL | tf top-1 | free shoda | free všech 7 | free = dataset |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.040 | 0.464 | 0.457 | 11.88 | 0.276 | 0.231 | 0.000 | 0.165 |
| 64 | 0.079 | 0.381 | 0.360 | 7.36 | 0.401 | 0.299 | 0.000 | 0.181 |
| 128 | 0.157 | 0.288 | 0.265 | 2.69 | 0.655 | 0.374 | 0.030 | 0.239 |
| 256 | 0.313 | 0.194 | 0.181 | 0.369 | 0.907 | 0.701 | 0.455 | 0.557 |
| 512 | 0.625 | 0.103 | 0.099 | 0.023 | 0.989 | 0.960 | 0.925 | 0.814 |
| reference | 1 | — | — | 0 | 1 | 1 | 1 | 0.836 |

Reference na konci má KL 0.

## Interpretace (není měření)

- Skupina 46 vrstev snese průměrnou chybu vrstvy kolem ~0.1, kdy je KL 0.02 a shoda 96 %. Při ~0.19 je KL 0.37
  a shoda 70 %.
- Nezávislé ASVD k tomu potřebuje 31–63 % bajtů.
- QuIC-WC by podle křivek po vrstvách potřeboval méně. Hlubší vrstvy ale mají při L32 relL2 0.45–0.5 a křivka
  klesá jen ~0.83× na zdvojnásobení L. Úroveň 0.1–0.2 by tak vyžadovala L v řádu stovek a fity v řádu hodin na
  vrstvu.
- **Důsledek:** MLP DINOv2 (2.67 % parametrů modelu) je pro skupinovou kompresi bez doladění špatný cíl. I úspěch by
  ušetřil ≤ ~2 % modelu.
- Pro uzavření otázky sekvenční kalibrace je relevantní režim blízko použitelnosti (rank 128–256). Tam se proto
  měří sekvenční ASVD, viz níže.

## Navazující běh: sekvenční ASVD r256 a r128 (provedeno)

- `run_seq.sh`: sekvenční ASVD (cíl `output`) při ranku 256 a 128, bloky 0–22, plus vyhodnocení proti nezávislému
  ASVD stejného ranku. Log `run_seq.log`.
- Řetězce `seq_asvd_r256_out` (exit 0) a `seq_asvd_r128_out` (exit 0, 13:45) i vyhodnocení `m4_seq/` (13:45–14:42,
  exit 0) doběhly.

### Chyba vyhodnocení: sdílená cache ASVD vah (zjištěno 2026-11-19 odpoledne)

- **Symptom:** v `m4_seq/results.md` mají řádky `dinov2_mlp:asvd:r256` a `seq:seq_asvd_r256_out` (stejně tak r128)
  bitově stejné metriky. Nezávislé r256 navíc nesouhlasí se sondou (KL 0.2165 proti 0.36875; in-situ 0.1900 proti
  0.1805).
- **Příčina:** `m4_lite.asvd_weights` ukládal váhy pod klíčem `(vrstva, rank, policy)` bez kalibrace. Sekvenční
  kalibrace byly v plánu později, a tak přepsaly nezávislé. Obě dvojice řádků jsou proto **sekvenční** ASVD. Dokládá to
  pilotní chyba 0.1761, která je měřená proti učiteli `W_t`, a in-situ chyba 0.1900; obě se liší od nezávislého běhu
  sondy.
- **Rozsah:** prošel jsem všechny plány a výsledky M4-lite v `artifacts/`, kolize klíče (stejná vrstva, rank
  a policy, různá kalibrace) jsou jen v `m4_seq_plan*.json` a `m4_seq/results.json`. Dřívější výsledky (M4-lite
  18. 11., skupina 18. 11., průběžné vyhodnocení sekvenční kalibrace, sonda rozpočtu) chyba nezasáhla.
- **Oprava:**
  - `m4_lite.asvd_key(r) = (kalibrace, rank, policy)`, používá se v `asvd_weights`, `Swap` i u `check_rel_l2`;
  - kontrola, že kalibrace patří nahrazované vrstvě;
  - regresní test `test_asvd_cache_separates_calibrations_of_the_same_layer`;
  - `pytest -q` → 84 passed.
- **Opakované vyhodnocení:** `rerun_m4_seq.sh` → `m4_seq_fixed/`, log `m4_seq_fixed.log`, spuštěno 2026-11-19
  odpoledne. Chybný `m4_seq/` ponechávám jako doklad, nepoužívat.

### Výsledek opraveného vyhodnocení (provedeno 14:48–15:31, exit 0)

- Zdroj: `m4_seq_fixed/results.{json,md}`. Bootstrap: `bootstrap_seq.py` → `bootstrap_seq.txt`.
- **Kontroly:**
  - nezávislé r256 i r128 reprodukují sondu (`m4/`) na všech místech;
  - reference na konci má KL 0;
  - sekvenční řádky jsou totožné s chybným během, což potvrzuje, že ten měřil sekvenční váhy.

| varianta | bajty ⌀ | relL2 pilot | in-situ relL2 (proti W) | tf KL | tf KL p95 | tf top-1 | free shoda | free všech 7 | free = dataset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASVD r256 nezávislé | 0.313 | 0.194 | 0.181 | 0.369 | 2.38 | 0.907 | 0.701 | 0.455 | 0.557 |
| **ASVD r256 sekvenční** | 0.313 | 0.176 | 0.190 | **0.217** | 1.27 | 0.936 | 0.779 | 0.605 | 0.634 |
| ASVD r128 nezávislé | 0.157 | 0.288 | 0.265 | 2.686 | 12.95 | 0.655 | 0.374 | 0.030 | 0.239 |
| **ASVD r128 sekvenční** | 0.157 | 0.256 | 0.275 | **1.325** | 8.86 | 0.780 | 0.488 | 0.172 | 0.351 |
| reference | 1 | — | — | 0 | 0 | 1 | 1 | 1 | 0.836 |

- **Poměr KL nezávislé / sekvenční (párový bootstrap, 95% CI):**
  - r256: 1.70×; po snímcích [1.49, 1.97], po epizodách [1.58, 1.90]; sekvenční je lepší na 67.5 % snímků
    a 10/10 epizodách;
  - r128: 2.03×; po snímcích [1.85, 2.23], po epizodách [1.73, 2.36]; lepší na 84.3 % snímků a 10/10 epizodách.
- **Interpretace (není měření):**
  - Sekvenční kalibrace s cílem `output` spolehlivě snižuje KL skupiny ~1.7–2×, při stejných bajtech a bez tréninku.
    Je to ale zlepšení v násobcích, ne o řád.
  - In-situ chyba proti původní W je u sekvenční varianty vyšší, a přesto má nižší KL. Vrstvy se tedy záměrně
    odchylují od W, aby kompenzovaly předchozí chyby. In-situ relL2 proti W proto pro sekvenční varianty není
    vhodná metrika kvality.
  - Ani sekvenční ASVD r256 (31 % bajtů, KL 0.22, shoda s datasetem 63 % proti 84 %) nesplňuje použitelnost.
    Skupina 46 vrstev DINOv2 MLP potřebuje víc než ~1/3 bajtů nebo doladění na výstup modelu.
