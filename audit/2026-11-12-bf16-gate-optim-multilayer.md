# Export BF16, brána M3, optimalizace a pilot na dalších vrstvách

Datum: 2026-11-12 (navazuje na [2026-11-12-m1-core-pilot.md](2026-11-12-m1-core-pilot.md)).

## Zadání a rozhodnutí uživatele

Uživatel po prvním pilotu rozhodl (zpráva z 2026-11-12):

- odchylka od původního plánu (jeden přímý součet) je u experimentálního výzkumu přijatelná;
- pokračuje se **vícestupňovou variantou s permutacemi**, počet stupňů `L` se volí **adaptivně pro každou vrstvu**,
  srovnávací baseline je **ASVD**;
- doplnit **export s θ v BF16** a ověřit, že výhoda proti ASVD platí i při stejné velikosti uložených tensorů;
- **brána M3** se mění na srovnání s ASVD při stejné velikosti, s metrikou bez biasu (odpověď „Ano“);
- před M4/M5 zopakovat pilot na **2–3 dalších vrstvách** a zkusit **lepší optimalizaci** (restarty, postupné
  přidávání stupňů) (odpověď „Ano“).

## Průběh

### 1. Export s volitelným úložným dtype (hotovo)

- `src/quic_wc/checkpoint.py`: `save_compact(..., storage_dtype=...)`, `storage_tensors`, `storage_report(...,
  storage_dtype=...)`; manifest má `storage_dtype`, `FORMAT_VERSION = 2` (`src/quic_wc/config.py`). Načtení θ/σ
  přesně převede na konstrukční dtype (FP32); ortogonální faktory se vždy staví v FP32 ze zaokrouhlených parametrů.
  Artefakty verze 1 se dál načtou; nesouhlas dtype s manifestem je chyba; úložný dtype širší než konstrukční je chyba.
- Testy (`tests/test_checkpoint_metrics.py`): BF16 artefakt rekonstruuje přesně (`torch.equal`) totéž co ruční
  zaokrouhlení θ/σ, faktory zůstávají ortonormální (1e-5), `tensor_bytes == 2 · parametry`, soubor je menší;
  načtení v1 artefaktu; odmítnutí lživého manifestu.

### 2. Jádro: konjugovaný tvar stupně a růst stupňů (hotovo)

- Problém: ve stávajícím tvaru `S = B P` je stupeň při θ = 0 permutace, ne identita, takže přidání nulového stupně
  mění funkci. Řešení: volba `SideConfig.stage_form = "conjugated"`, stupeň `R_lᵀ B_l R_l` s kumulativními
  permutacemi `R_l = P_l ⋯ P_1` (`src/quic_wc/math/orthogonal.py`, `stage_permutations`). Platí
  `B_L P_L ⋯ B_1 = R_L · Π (R_lᵀ B_l R_l)`, tedy stejná rodina až na pevnou permutaci řádků `R_L`; při θ = 0 je
  `Q = I` pro libovolné `L`. Výchozí `"chain"` zůstává kvůli zpětné kompatibilitě.
- `src/quic_wc/growth.py`: `grow_stages` (přidá nulové stupně na konec, funkce se nemění; pro chain tvar odmítne),
  `perturb_` (náhodná inicializace pro restarty).
- `src/quic_wc/distillation.py`: `FitConfig.select_on = "val" | "train"` (výběr nejlepšího kroku; nové běhy
  používají `train`, aby validační data nevstupovala do žádné volby).
- Testy (`tests/test_growth.py`): rovnost chain = `R_L`·conjugated pro stride/cyclic/random (FP64, 1e-12),
  identita při θ = 0 pro L = 1, 3, 7, zachování váhy při růstu (1e-6) a nenulové gradienty nových stupňů, odmítnutí
  chain tvaru a zmenšení.

### 3. Runner: nasaditelná metrika, stejné bajty, brána M3, vrstvy bez biasu (hotovo)

- `src/quic_wc/pilot/run.py` (schéma výsledků v3): primární `metrics` se počítají z **načteného artefaktu**
  (θ/σ v `storage_dtype`), `metrics_unrounded` z FP32 parametrů. Korigovaný bias se počítá až ze zaokrouhlené váhy.
  Reference ASVD/SVD: `r_bytes` (nejvýše stejné bajty), nově `r_bytes_ceil` (alespoň stejné bajty), `r_params`.
- Brána M3 (`gate_m3`, `layer_gate`): varianta projde, když její validační relL2 bez biasu < ASVD při
  `r_bytes_ceil` (ASVD dostane alespoň tolik bajtů), při poměru bajtů ≤ 0.25; vrstva projde, když projde aspoň
  jedna varianta dané politiky biasu. Práh 0.25 je moje volba (komprese nad čtvrtinu velikosti nemá smysl).
- Vrstvy bez biasu (Llama): jen politika `fixed`, bias se do bajtů nepočítá; `corrected` je chyba.
- Rozšíření specifikace: `storage_dtype`, `restarts`, `growth` + `growth_steps`, `rank: "full"`.
- Testy (`tests/test_pilot.py`): BF16 + růst + restarty end-to-end (roundtrip přesně 0, růst nezhorší trénovací
  cíl), vrstva bez biasu.

### 4. Ověření výhody v BF16 na hotových artefaktech (hotovo, naměřeno)

Příkaz: `PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.requantize --calibration <P>/calibration
--results <P>/grid-v1 <P>/grid-v1-controls-fixed --out <P>/requant-bf16`, kde
`P = artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1` (výstup `requant-bf16/results.{json,md}`, log
`requant-bf16.log`). Nový skript `src/quic_wc/pilot/requantize.py` nic nefituje: načte FP32 artefakt, zaokrouhlí,
přepočítá korigovaný bias, exportuje, načte a vyhodnotí.

- Konzistence: FP32 metriky z artefaktů reprodukují původní grid přesně (`all_consistent = True`, tolerance 1e-6).
- Velikost: `stride_n8K12_L16` má `layer.safetensors` 136 872 B (BF16) proti 265 256 B (FP32).
- Validační relL2 bez biasu, BF16 artefakt proti ASVD s alespoň stejnými bajty (BF16 faktory):

| varianta | poměr bajtů | QuIC-WC (BF16) | ASVD ≥B (rank: relL2) | rezerva |
|---|---:|---:|---|---:|
| stride L4 | 0.0050 | 0.2448 | 4: 0.2501 | 2.1 % |
| stride L16 | 0.0163 | 0.1345 | 13: 0.1688 | 20.3 % |
| stride L64 | 0.0614 | 0.0640 | 50: 0.0699 | 8.4 % |
| stride L16, korigovaný bias | 0.0163 | 0.1295 | 13: 0.1664 | 22.1 % |
| stride L64, korigovaný bias | 0.0614 | 0.0513 | 50: 0.0690 | 25.6 % |

- Brána M3 (nová definice) na této vrstvě: `fixed` prošla (10/14 variant, nejlepší rezerva 20.3 %), `corrected`
  prošla (2/2). Neprošly varianty přesně podle zadání (L = 1) a `ident_L16`.
- Interpretace: výhoda při stejných bajtech v BF16 se potvrdila. Zaokrouhlení θ do BF16 ale u L64 zvýší chybu
  0.0625 → 0.0640; s klesající chybou roste relativní podíl zaokrouhlení (otevřený bod: dotrénování po zaokrouhlení).

### 5. Studie optimalizace na DINOv2 b0 fc1 (hotovo, naměřeno)

Grid `configs/optim_libero10_dinov2_b0_fc1.json`, výstup `artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/optim-v1/`
(`results.{json,md}`, log `optim-v1.log`, 1783 s). BF16 úložiště, výběr kroku podle trénovacího cíle, LR 0.03,
cosine plán. Validační relL2 bez biasu z BF16 artefaktu:

| varianta | relL2 | srovnání |
|---|---:|---|
| konjugovaný tvar L16, 3000 kroků | 0.1342 | chain L16 (grid-v1, BF16): 0.1345 → tvary ekvivalentní |
| L16, 10 000 kroků | 0.1309 | −2.5 % proti 3000 krokům |
| L64, 10 000 kroků | 0.0556 (FP32 θ 0.0537) | 3000 kroků: 0.0640 → −13 %; rezerva proti ASVD r50 roste na 20.4 % |
| růst 4→8→16→32→64 (3000+4×2000 kroků) | L4 0.2452, L8 0.1921, L16 0.1426, L32 0.0999, L64 0.0797 | **horší než fit od nuly** (L16 0.1342, L64 0.0640); L64 neprojde M3 |
| 8 startů (identita + 7× θ~N(0, 0.05²)), 300 kroků screening | 0.1337 | screening vybral identitu (0.0290 vs 0.0307–0.0316); bez zlepšení |
| poz. kontrola θ·0.3, L4 (konjugovaný) | 0.4358 | chain (grid-v1): 0.4415 |
| — 10 000 kroků | 0.4284 | bariéra trvá |
| — 8 startů, měřítko 0.3 | 0.4362 | screening opět vybral identitu |
| — růst 1→2→4 | 0.4548 | horší |
| — přeparametrizovaný student L8 pro učitele L4 | 0.3599 | jediné zlepšení (−17 %) |

Interpretace:

- Postupný růst (navržený jako mechanismus adaptivního `L`) se neosvědčil: menší vyfitovaná vrstva je špatný
  start pro větší (uvízne v horším minimu). Adaptivní `L` proto volím jako **sweep fitů od nuly** přes
  geometrickou řadu `L` a výběr podle křivky chyba/bajty (`stages_sweep` v runneru). Růst zůstává v kódu jako
  levná varianta pro hrubý odhad křivky.
- Náhodné restarty nepomáhají; inicializace identitou je nejlepší ze zkoušených.
- Delší trénink pomáhá, víc při větším `L` (L64: −13 %). Výsledky s 3000 kroky jsou tedy konzervativní.
- Bariéra u pozitivní kontroly je skutečná nekonvexita; zmírní ji jen nadbytečná kapacita (L8 pro učitele L4).
  Delší trénink, restarty ani růst ji neodstraní.
- Nové zjištění: u L64 už zaokrouhlení θ do BF16 zvyšuje chybu o 3.5 % relativně (0.0537 → 0.0556). Otevřený
  bod: krátké dotrénování po zaokrouhlení (např. σ nebo straight-through).

### 6. Zrychlení stupně a gradient checkpointing (hotovo)

- `stage_blocks` skládá `C_k1 ⊕ C_k2 ⊕ I` do jedné matice → stupeň = gather + jeden `bmm` (+ kopie zbytku);
  konjugovaný tvar se počítá jako chain + jedna závěrečná permutace `R_Lᵀ`. Testy proti nezávislé dense referenci
  (chain i konjugovaný tvar, FP64 1e-12) prošly.
- Benchmark (skript ve scratchpadu relace, RTX 4090, forward+backward cíle `tr(ΔCΔᵀ)`): 4096×1024, r 1024:
  L16 13.4 ms, L64 53.4 ms (dříve ~23 ms a ~78 ms v gridu, tj. 1.5–1.7× rychleji); 4096×4096, r 1024, L64
  78.5 ms / 2.9 GB; 4096×11008, r 1024, L64 197 ms / 6.0 GB; s checkpointingem (8 segmentů) L128 466 ms / 3.3 GB.
- `QuICWCLinear.set_checkpointing(segments)`; test shody gradientů s/bez checkpointingu.

### 7. Ověření předzpracování a importů pro celý model

- Moje reimplementace větve 0 (DINOv2) z prvního pilotu dává **bitově shodné** pixely jako oficiální
  `PrismaticImageProcessor` (8 snímků LIBERO-10, max. rozdíl 0.0). Kalibrace prvního pilotu tedy odpovídá evaluaci.
- Neúspěšný pokus: kontrolní skript spuštěný s `CUDA_VISIBLE_DEVICES=""` padal na `free(): double free` při importu
  `prismatic` (import `flash_attn` bez viditelné GPU). S viditelnou GPU import funguje v obou pořadích (TF první i
  druhý). Při ukončení interpretu se po TF datasetu vypisuje neškodný `TypeError` v destruktoru.

### 8. Sběr aktivací z celého OpenVLA (hotovo, ověřeno)

- Nový `src/quic_wc/pilot/collect_vla.py`: `OpenVLAForActionPrediction` z `external/openvla` (BF16, flash-attention 2
  jako `get_vla`), dotaz po snímcích přes `predict_action(do_sample=False)` s promptem LIBERO evaluace, oficiální
  `PrismaticProcessor`; hooky vidí vizuální tokeny, prefill i 6 dekódovacích kroků. Stejné epizody a snímky jako
  první pilot (seed 0, 5 epizod/úlohu, 1 validační, každý 10. snímek).
- Běh: 1386 snímků, 532 s; výstup `artifacts/pilot/2026-11-12-libero10-multilayer/calibration/` (adresář na vrstvu +
  `summary.json`, `teacher_actions.safetensors`), log `collect.log`.
- Kontroly: hook = výstup modulu přesně (max. rozdíl 0.0) u všech 4 vrstev; BF16 vs FP32 reference relL2 0.0017–0.0029.
  Llama: 7 volání na snímek (prefill + 6 dekódovacích), ~292 tokenů/snímek (326 893 trénovacích, 78 383 validačních).
- **Konzistence s prvním pilotem:** DINOv2 b0 fc1 z celého modelu má stejné počty tokenů a matice `C`/průměry se liší
  relativně o 3–4e-6 (dávka 1 vs 64 v BF16). Kalibrace prvního pilotu je tedy potvrzená i nezávislou cestou.
- Reference pro M4-lite: greedy akční tokeny původního modelu se shodují s tokeny z datasetu v 83.6 % (validace),
  83.3 % (trénink).

### 9. Pilot na dalších vrstvách (SigLIP a q_proj hotové, naměřeno)

Gridy `configs/multilayer_vision.json` a `configs/multilayer_llm.json` (sweep `L` od nuly, konjugovaný stride,
`n = 8`, BF16, výběr podle trénovacího cíle, 3000 kroků, LR 0.03); spouštěcí skript ve scratchpadu relace, výstupy
`artifacts/pilot/2026-11-12-libero10-multilayer/<vrstva>/results.{json,md}`.

`vision_backbone.fused_featurizer.blocks.13.mlp.fc1` (SigLIP, 4304×1152, s biasem; 804 s):

| L | poměr bajtů | relL2 fixed (ASVD ≥B) | rezerva | relL2 corrected (ASVD ≥B) | rezerva |
|---:|---:|---|---:|---|---:|
| 4 | 0.0045 | 0.4867 (0.4840) | −0.6 % | 0.4733 (0.4765) | 0.7 % |
| 8 | 0.0079 | 0.4378 (0.4557) | 3.9 % | 0.4302 (0.4509) | 4.6 % |
| 16 | 0.0147 | 0.3891 (0.4178) | 6.9 % | 0.3835 (0.4155) | 7.7 % |
| 32 | 0.0284 | 0.3353 (0.3680) | 8.9 % | 0.3294 (0.3666) | 10.2 % |
| 64 | 0.0556 | 0.2863 (0.3136) | 8.7 % | 0.2734 (0.3129) | 12.6 % |

Pozitivní kontrola L4 θ·0.1: 0.0039 (obnovena); negativní kontrola L16: 0.6279 (ASVD 0.6658). M3: fixed i corrected
prošly. Vstup je méně anizotropní než u DINOv2 b0 (top-64 směrů 73 % stopy vs 99 %), ASVD potřebuje rank 512
(~45 % bajtů) pro relL2 0.095.

`language_model.model.layers.15.self_attn.q_proj` (4096×4096, bez biasu; 1859 s):

| varianta | poměr bajtů | relL2 | ASVD ≥B | rezerva | čas |
|---|---:|---:|---|---:|---:|
| K=[1,2] L16 | 0.0061 | 0.4580 | r13 0.4791 | 4.4 % | 78 s |
| K=[1,2] L32 | 0.0121 | 0.4153 | r25 0.4437 | 6.4 % | 120 s |
| K=[1,2] L64 | 0.0242 | 0.3809 | r50 0.4010 | 5.0 % | 235 s |
| K=[1,2] L128 | 0.0483 | 0.3631 | r99 0.3527 | **−3.0 %** | 587 s |
| K=[1,2] L32, rank 4096 | 0.0123 | 0.4164 | r26 0.4414 | 5.7 % | 405 s |
| K=[1] L7 | 0.0120 | 0.4324 | r25 0.4437 | 2.5 % | 47 s |
| K=[1] L14 | 0.0240 | 0.3783 | r50 0.4010 | 5.7 % | 57 s |
| K=[1] L28 | 0.0479 | 0.3256 | r99 0.3527 | 7.7 % | 78 s |

Kontroly (L16, r1024): pozitivní θ·0.1 **0.5964 — neobnovena**, negativní 0.6101. M3 (fixed) prošla (7/8). ASVD
dosáhne relL2 0.211 až při 25 % bajtů; jeden vstupní směr nese 50 % stopy.

Interpretace (SigLIP + q_proj):

- Výhoda proti ASVD při stejných bajtech přetrvává, ale je menší než u DINOv2 b0 (4–13 % místo 20 %) a u q_proj
  s rostoucím `L` mizí (L128 prohrává).
- Rank 1024 u q_proj neomezuje (4096: 0.4164 vs 0.4153).
- Compound řády: při malém rozpočtu je K=[1,2] mírně lepší (L32 0.4153 vs K=[1] L7 0.4324), při větším je lepší
  K=[1] (L28 0.3256 vs K=[1,2] L128 0.3631) a trénuje 2.5–7.5× rychleji.
- Hluboké řetězce mají dlouhé počáteční plato (q_proj L128: trénovací relMSE 1.02 po 500 krocích, L16 už 0.22) a
  pozitivní kontrola s L16 selže i při θ·0.1. Hypotéza: změna součinu roste ~L·lr, takže LR 0.03 je pro velké `L`
  příliš velký → test LR škálovaného hloubkou (bod 10, běží souběžně s down_proj; časy obou běhů jsou kvůli sdílené
  GPU nadhodnocené).

### 10. LR škálovaný hloubkou (hotovo, naměřeno)

Gridy `configs/lrdepth_dinov2_b0_fc1.json` (výstup `artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/lrdepth-v1/`) a
`configs/lrdepth_llm_q_proj.json` (`artifacts/pilot/2026-11-12-libero10-multilayer/llm_l15_q_proj_lrdepth/`), 3000
kroků, BF16. Běžely souběžně s jinými GPU úlohami, uvedené časy nejsou srovnatelné.

| vrstva, L | LR 0.03 | LR 0.03·(16/L)^0.5 | LR 0.03·16/L | ASVD ≥B |
|---|---:|---:|---:|---:|
| DINOv2 b0 fc1, L64 | 0.0660 | **0.0511** (0.015) | 0.0516 (0.0075) | r50 0.0699 |
| q_proj, L64 | 0.3809 | **0.3705** (0.015) | — | r50 0.4010 |
| q_proj, L128 | 0.3631 | **0.3194** (0.0106) | 0.3275 (0.00375) | r99 0.3527 |

- Hypotéza z bodu 9 potvrzena: pevný LR 0.03 hluboké řetězce nedotrénuje. Se škálováním 1/√L má DINOv2 L64 za 3000
  kroků lepší výsledek (0.0511) než s LR 0.03 za 10 000 kroků (0.0556); q_proj L128 se z prohry (−3 %) dostává na
  rezervu 9.4 % a je mírně lepší než K=[1] L28 (0.3256).
- Zvolený recept v2: `lr = 0.03 · min(1, (16/L)^0.5)` (`lr_depth_scaling` v runneru, test v `tests/test_pilot.py`).
- Důsledek: body s `L > 16` z bodů 5 a 9 jsou nedotrénované. Proto jsem zastavil běžící grid `down_proj` s LR 0.03
  po prvním bodu (L16 = 0.8611 proti ASVD r13 0.8786; L16 má stejný LR v obou receptech; částečný výstup
  `llm_l15_down_proj_lr0.03_stopped/results.partial.json`) a spustil kompletní sweepy v2 pro všechny čtyři vrstvy
  (`configs/multilayer_vision_v2.json`, `configs/multilayer_llm_v2.json`, výstupy `v2_*`).
- Neúspěšný pokus při zastavování: `pkill -f` se vzorem z příkazové řádky ukončil i vlastní shell nástroje (exit
  144) a obalový skript, Python proces běžel dál; ukončen podle PID. Ostatní běhy to nezasáhlo (ověřeno `pgrep`).

### 11. Sweepy receptu v2 na čtyřech vrstvách (hotovo, naměřeno)

Skript ve scratchpadu relace (`run_v2.sh`), log `artifacts/pilot/2026-11-12-libero10-multilayer/run_v2.log`
(21:30–23:25), výstupy `v2_dinov2_b0_fc1/`, `v2_siglip_b13_fc1/`, `v2_llm_l15_q_proj/`, `v2_llm_l15_down_proj/`
(`results.{json,md}`, artefakty), souhrn `summary_v2.{md,json}` (`quic_wc.pilot.summarize`).

| vrstva | M3 | nejlepší rezerva | největší L: relL2 (bajty) vs ASVD ≥B |
|---|---|---:|---|
| DINOv2 b0 fc1 | prošla (5/5, corrected 5/5) | 28.2 % (corrected 36.1 %) | L64: 0.0510 (0.061) vs 0.0699 |
| SigLIP b13 fc1 | prošla (4/5, corrected 5/5) | 11.2 % (12.9 %) | L64: 0.2785 (0.056) vs 0.3136 |
| q_proj | prošla (7/7) | 9.4 % | L128: 0.3195 (0.048) vs 0.3527 |
| down_proj | prošla (7/7) | 6.1 % | L128: 0.7425 (0.033) vs 0.7909 |

- Determinismus: body s L ≤ 16 (stejný LR jako v1) se shodují s v1 na 4 desetinná místa (např. DINOv2 L16 0.1342,
  down_proj L16 0.8611).
- Recept v2 proti v1: DINOv2 L32 0.0999 (růst)/— → 0.0879, L64 0.0640 → 0.0510; SigLIP L64 0.2863 → 0.2785 (L32
  beze změny); q_proj L128 0.3631 → 0.3195; K=[1] L28 beze změny (0.3256).
- Úspora bajtů při stejné kvalitě (log-log interpolace, interpretace): DINOv2 33–42 % (τ 0.2–0.05), SigLIP 35–38 %
  (τ 0.4–0.3), q_proj 38 % (τ 0.4), down_proj 47 % (τ 0.8).
- K=[1] proti K=[1,2] při stejných bajtech (Llama): K=[1,2] lepší o 0.5–3.6 %, K=[1] trénuje 1.8–7.2× rychleji.
- Kontroly: pozitivní θ·0.1 obnovena na DINOv2 (0.0070) a SigLIP (0.0039), na Llamě ne (q_proj 0.5964, down_proj
  0.8765 — blízko negativních kontrol 0.6101 / 0.8802). Negativní kontrola poráží ASVD na všech vrstvách.
- Spektra vstupu (podíl stopy v 64 hlavních směrech): DINOv2 99 %, SigLIP 73 %, q_proj 70 %, down_proj 27 %. ASVD
  potřebuje pro relL2 0.2 na SigLIP/q_proj 19–27 % bajtů; down_proj ani při 69 % (r2048) nedosáhne 0.33.

Interpretace: brána M3 v novém znění prošla na všech vrstvách, ale prochází i tam, kde jsou obě metody při
rozumném rozpočtu nepoužitelné (down_proj). Výhoda QuIC-WC je konzistentní (6–36 %), ale její absolutní dopad závisí
na vrstvě. Výhoda nevzniká z compound matic ani ze struktury reálných vah (negativní kontrola), ale z parametrizace
(plný rank se spektrem σ a vícestupňové ortogonální faktory).

### 12. Chyba v souhrnném skriptu a její oprava

- `summarize.py` v první verzi pro cílovou chybu, které dosáhl už nejmenší změřený bod, vracel tento bod jako
  „interpolovanou“ velikost; úspora pak vycházela zavádějící (např. −2.121 u q_proj pro τ 0.8, protože ASVD má
  menší rank 4). Oprava: takové řádky ukazují jen horní mez („≤“) a úspora se nepočítá; interpolace jen mezi dvěma
  změřenými body. Předtím jsem také zjistil, že „nejmenší změřený bod“ sám závisí na hustotě gridu (např. 49 % u τ
  0.15 jen kvůli mezeře ASVD r12/r25), proto report uvádí interpolované hodnoty a označuje je jako interpretaci.

### 13. Závěrečné ověření

- `timeout 900 .conda/openvla-gpu/bin/python -m pytest -q` → **73 passed** (CPU + CUDA).
- Report [docs/pilot-multilayer-2026-11-12.md](../docs/pilot-multilayer-2026-11-12.md): čísla zkontrolována proti
  `results.json`/`summary_v2.md`; opravena formulace poměru časů (2–7× → 1.8–7.2×).
- `git status`: uživatel mezitím commitnul práci prvního pilotu (`40cf85a M1 and M2 implementation`, 19:11); změny
  z tohoto záznamu **nejsou commitnuté**. Artefakty (4.4 GB `2026-11-12-libero10-multilayer`, 482 MB prvního pilotu)
  jsou v ignorovaném `artifacts/`.

## Vytvořené a změněné soubory

- Nové: `src/quic_wc/growth.py`, `src/quic_wc/pilot/{collect_vla,requantize,summarize}.py`, `tests/test_growth.py`,
  `configs/{optim_libero10_dinov2_b0_fc1,lrdepth_dinov2_b0_fc1,lrdepth_llm_q_proj,multilayer_vision,multilayer_llm,
  multilayer_vision_v2,multilayer_llm_v2}.json`, `docs/pilot-multilayer-2026-11-12.md`, tento audit.
- Změněné: `src/quic_wc/{checkpoint,config,distillation}.py`, `src/quic_wc/layers/linear.py`,
  `src/quic_wc/math/orthogonal.py`, `src/quic_wc/pilot/{collect,run}.py`, `tests/{test_checkpoint_metrics,
  test_orthogonal,test_pilot}.py`, `docs/m1-math-core.md` (oddíl 10), `instructions/implementation-plan.md`
  (oddíl 11), `audit/README.md`.
- Grid `configs/multilayer_vision.json`/`multilayer_llm.json` (v1, LR 0.03) zůstávají jako záznam běhů v1.

## Otevřené body a omezení

- Brána M3 nemá absolutní kritérium kvality; navrženo odvodit ho z M4-lite (vliv na výstup modelu a akční tokeny,
  reference 83.6 %). Práh poměru bajtů 0.25 zvolil agent.
- Role compound řádů (K=[1,2] vs K=[1]) — rozhodnutí uživatele.
- Optimalizace na vrstvách Llamy nevyužije strukturu učitele (pozitivní kontrola selhává) → výsledky jsou dolní mez.
- Zaokrouhlení θ do BF16 je při malé chybě znatelné (DINOv2 L64, 10 000 kroků: 0.0537 → 0.0556); dotrénování po
  zaokrouhlení neimplementováno.
- Jedna vrstva z každého typu, jedno semeno, jedna sada dat; delší trénink neproveden v receptu v2.
- `down_proj` grid v1 zastaven po L16 (záměrně, viz bod 10).
- M4 a M5 nezahájeny.
