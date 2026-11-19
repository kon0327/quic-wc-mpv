# Skupina vrstev: všech 48 MLP vrstev DINOv2 (2026-11-18)

## Zadání a rozsah

- Uživatel 2026-11-18 rozhodl: absolutní kritérium M3 v KL pro skupiny souvisejících vrstev i pro celý model;
  další krok = komprese celé skupiny (MLP DINOv2) a měření skládání chyb a úspory na úrovni modelu
  (viz [audit M4-lite](2026-11-18-m4-lite.md), plán oddíl 13). Commit M4-lite: větev `m4-lite`, `a19fd4f`
  (práci z 16. 11. commitnul uživatel jako `f950abc`).

## Plán (stanoveno před spuštěním)

- Skupina: `vision_backbone.featurizer.blocks.{0..23}.mlp.{fc1,fc2}` (48 vrstev, 1024↔4096, s biasem).
- Kalibrace: `collect_vla` ve třech bězích (fc1 všech 24; fc2 bloků 0–11; fc2 bloků 12–23), protože FP64 akumulátory
  momentů fc2 (24 × 268 MB) by se s modelem (15 GB) nevešly do 24 GB; stejné epizody a snímky jako dosud.
- Fity: `configs/group_dinov2_mlp.json` — recept v2 (K=[1,2], konjugovaný stride, n = 8, BF16, 3000 kroků, LR
  škálovaný hloubkou), korigovaný bias (v M4-lite nejlepší), plný rank, sweep L = 8/16/32 od nuly pro každou vrstvu.
- Vyhodnocení (`m4_lite group-plan` + `run`): celá skupina při jednotném L = 8/16/32 (QuIC-WC a ASVD se stejnými
  bajty), podskupiny fc1 a fc2 při L16, 48 jednovrstvých variant L16 jen s teacher forcingem (test aditivity KL),
  reference na začátku a na konci.
- Otázky: (1) jak se skládá KL přes 48 vrstev (součet jednovrstvých vs skupina); (2) QuIC-WC vs ASVD na úrovni
  skupiny; (3) jaký rozpočet skupiny splní pracovní práh (KL ≤ 0.01 → kalibrace prahu pro skupiny); (4) úspora bajtů
  skupiny / vizuální větve.
- Odhad: kalibrace 3 × ~10 min, fity ~48 × 200 s ≈ 2.7 h, vyhodnocení ~40 min.

## Úpravy kódu před během

- `m4_lite.py`: varianty `tf_only` (bez greedy dekódování; KL/top-1 a in-situ chyba), ukládání KL po snímcích
  (`tf_kl`), podpříkaz `group-plan`, souhrnné sloupce pro skupiny v `results.md`.
- Ověření: `pytest -q` → 79 passed; smoke test `group-plan` + `run` (DINOv2 b0 fc1 + SigLIP b13 fc1, 8 snímků):
  proběhl, tf-only varianta ~0.07 s/snímek.

## Průběh a výsledky

### Průběh (provedeno)

- `artifacts/pilot/2026-11-18-dinov2-mlp/run.sh`, log `run.log`.
  - Kalibrace 3× bez chyby. Ověření: rozdíl hooků 0; snímky a tokeny jsou totožné s kalibrací z 11-12.
  - Fity 48 vrstev, každá ~170–180 s, všechny `exit 0`, 12:58–15:23.
  - Vyhodnocení 60 variant, 15:23–15:57 (2042 s); log `m4.log`.
  - V `run.log` nejsou žádné `Traceback` ani nenulové `exit`.
- Kontroly:
  - Fit bloku 0 fc1 reprodukuje v2 (relL2 0.1799 / 0.1295 / 0.0826 pro L8 / 16 / 32).
  - Jednovrstvá KL bloku 0 fc1 je 0.0049 (M4-lite 0.005).
  - Reference na konci běhu má KL 0.
  - QuIC varianty mají `decompress_count = 1`.
  - In-situ relL2 jednovrstvých variant odpovídá pilotu.
  - Blok 23 má KL přesně 0. Je to ověřeno v kódu: OpenVLA bere výstup předposledního bloku,
    `get_intermediate_layers(n={len(blocks)-2})`, viz `src/quic_wc/pilot/collect.py:243` a
    `external/openvla/prismatic/models/backbones/vision/dinosiglip_vit.py`.
- Analýza: `artifacts/pilot/2026-11-18-dinov2-mlp/analyze.py` → `analysis.json`. Obsahuje křivky po vrstvách, úsporu
  proti ASVD (interpolace `compare_orders.bytes_for_error`), aditivitu KL po snímcích, souhrny skupin a počty
  parametrů z checkpointu.
- Report: [docs/group-dinov2-mlp-2026-11-18.md](../docs/group-dinov2-mlp-2026-11-18.md).

### Naměřené výsledky

- **Skupina (48 vrstev):**

  | rozpočet | KL QuIC | KL ASVD | free shoda QuIC | free shoda ASVD |
  |---|---:|---:|---:|---:|
  | L8 (0.84 / 0.91 % bajtů) | 15.46 | 12.99 | 11.1 % | 23.9 % |
  | L16 (1.59 / 1.65 %) | 12.34 | 12.79 | 23.0 % | 22.3 % |
  | L32 (3.10 / 3.11 %) | 9.94 | 12.49 | 27.8 % | 22.7 % |

  Shoda s datasetem je 9–19 % (reference 83.6 %).
- **Podskupiny při L16:** fc1 KL 12.33 (QuIC) / 12.36 (ASVD), fc2 KL 8.79 / 11.93.
- **Aditivita při L16:** součet jednovrstvých KL proti KL skupiny:
  - celek 0.88 → 12.34 (14.0×);
  - fc1 0.61 → 12.33 (20.4×);
  - fc2 0.28 → 8.79 (31.7×);
  - korelace po snímcích Pearson −0.05 a Spearman −0.04.
- **In-situ relL2 ve skupině proti pilotu:**
  - fc1 (QuIC L16) ⌀ 0.745 proti 0.544;
  - poměr pro jednotlivou vrstvu až 1.9× (QuIC) a 2.9× (ASVD fc2).
- **Po vrstvách (QuIC L16 proti ASVD při stejné relL2):**
  - úspora bajtů (1 − bajty QuIC / bajty ASVD) fc1 36–47 %, fc2 63–72 %;
  - relL2 fc1 0.13–0.66, fc2 0.17–0.58;
  - KL ≤ 0.01 splní 13 ze 46 použitých vrstev, maximum je 0.082 (blok 17 fc1).
- **Parametry checkpointu:**
  - celkem 7 541 237 184;
  - váhy MLP DINOv2 201 326 592 (2.67 %);
  - DINOv2 303 M, SigLIP 428 M, projektor 71 M, LLM 6 739 M.

### Interpretace (není měření)

- Při jednotném rozpočtu 1–3 % je komprese celé skupiny nepoužitelná pro obě metody. Srovnání QuIC proti ASVD
  v režimu kolapsu nevypovídá.
- Lineární aditivita KL zde neplatí. Skupinové prahy se musí měřit přímo, ne odvozovat z jednovrstvých KL.
- Pravděpodobné příčiny (hypotézy):
  - posun vstupů mimo kalibrační rozdělení (podporuje to nárůst in-situ chyby);
  - lokální cíl fitu.
- Otázka 3 plánu (rozpočet splňující práh) zůstala bez odpovědi, protože žádná varianta není blízko prahu.
- Úspora na modelu je i při ideální kompresi DINOv2 MLP ≤ 2.7 %. Velké úspory vyžadují LLM.

### Chyby a neúspěšné pokusy

- Monitor průběhu (`br8avrv03`) vypršel a nebyl obnoven. `run.sh` byl spuštěn s `&` bez automatické notifikace.
  Náprava: samostatný čekací proces na `=== m4 exit`; ten skončil s kódem 0.
- `analyze.py`, první spuštění:
  - adresář `fits/` obsahuje i soubory `*.log`, na kterých skript spadl (`NotADirectoryError`); opraveno filtrem
    `is_dir()`;
  - původní předpoklad o pojmenování souborů tokenů byl chybný; opraveno podle `m4_lite.py`
    (`name.replace(':', '__')`).
- V reportu jsem opravil dvě nepřesná tvrzení o citlivosti bloků. Nejdřív stálo „bloky 0–4 KL ≤ 0.008“, správně je
  ≤ 0.0082. Bloky 20–22 jsem původně uváděl mezi méně citlivými, ale jejich fc1 má KL 0.024–0.027.
- Report v první verzi chybně uváděl, že „ASVD potřebuje o 36–47 % víc bajtů“. Hodnota je ale úspora
  1 − QuIC/ASVD, tedy QuIC potřebuje o 36–47 % méně bajtů a ASVD 1.6–1.9× víc. Opraveno v reportu, auditu, plánu
  i indexu.

### Nevyřešeno / další kroky (navrženo, neprovedeno, čeká na uživatele)

- Sekvenční kalibrace (fit na vstupech z modelu s již komprimovanými předchozími vrstvami).
- Větší rozpočty (L64–L256) nebo menší skupiny (bloky 0–4, jen fc2), aby se našel skupinový rozpočet pod prahem.
- Doladění na KL výstupu (destilace po blocích nebo end-to-end).
- Skupinový test SigLIP.
- Necommitnuto: rozšíření `m4_lite.py`, `configs/group_dinov2_mlp.json`, audit a report. Commit jen na pokyn
  uživatele.
