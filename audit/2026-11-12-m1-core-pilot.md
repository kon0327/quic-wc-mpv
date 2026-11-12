# Matematické jádro QuIC-WC (M1) a první pilot jedné vrstvy

Datum: 12. 11. 2026. Průběžný záznam; doplňován během práce.

## Zadání

Uživatel požádal o návrh matematického jádra a implementaci prvního pilotu. Jako pomocný zdroj uvedl
`kon0327/openvla-quic` a jádro QuIC PEFT `kon0327/quic-peft-vla`.

## Výchozí stav a zdroje (zjištěno před implementací)

- `src/` ani testy v projektu neexistovaly; M1 nebylo implementované.
- `quic-peft-vla` (GitHub HEAD `e727b31`, 21. 10. 2026) a `openvla-quic` (HEAD `7d06820`, 12. 11. 2026) byly
  mělce naklonovány pouze ke čtení do scratchpadu relace; jejich kód se nespouštěl. Lokální
  `/home/kony/dissertation/openvla-quic-old` je starší (`deab81f`) a QuIC jádro neobsahuje;
  `/home/kony/dissertation/quic-petf-mpv` (`f1e71f6`) je zdroj, ze kterého bylo `quic-peft-vla` extrahováno.
- Převzaté poznatky z `quic-peft-vla`: explicitní polynomy C2/C3 (správný gradient i při identitě), Cayley
  přes `solve` s minimálními skew parametry, výpočet v FP32/FP64 s vypnutým autocastem, ochrana dtype
  parametrů při `.to(bfloat16)`. Kód nebyl kopírován; jádro WC je nová implementace s jinými kontrakty
  (tenké faktory, více stupňů s permutacemi, rekonstrukce váhy místo výstupní rotace adaptéru).
- Lokálně jsou checkpointy `openvla/openvla-7b` a `openvla/openvla-7b-finetuned-libero-10`
  (revize `80970322773f81baa2e22fe495d0487b93a05cfa`) v HF cache a RLDS data LIBERO v
  `/home/kony/dissertation/ovlab-data/datasets/libero/*/1.0.0/prepared`. Checkpoint
  `openvla-7b-finetuned-libero-spatial` z plánu lokálně není → pilot použije LIBERO-10 checkpoint a data.
- V této relaci (Claude Code) je GPU přímo dostupná (`nvidia-smi`: RTX 4090, CUDA v `.conda/openvla-gpu`
  funguje); sandboxové omezení z Codex relace se zde neprojevilo.

## Průběh

### 1. Matematické jádro M1 (dokončeno)

Nové soubory: [pyproject.toml](../pyproject.toml) (balíček `quic-wc`, src layout; pytest používá
`pythonpath = src`, sdílené conda prostředí se neinstalací nemění), `src/quic_wc/math/{cayley,compound,orthogonal}.py`,
`src/quic_wc/{config,decomposition,metrics,distillation,checkpoint,diagnostics}.py`, `src/quic_wc/layers/linear.py`,
testy `tests/test_{math,orthogonal,linear,checkpoint_metrics,distillation}.py`. Návrh a kontrakty:
[docs/m1-math-core.md](../docs/m1-math-core.md).

Ověření (příkaz z kořene projektu, prostředí `.conda/openvla-gpu`, GPU dostupná):

- `.conda/openvla-gpu/bin/python -m pytest -q` → **56 passed** (první běh), po rozšíření metrik o korekci biasu
  **57 passed** (11–17 s). CUDA testy běžely (nebyly přeskočeny).
- `PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.diagnostics --out artifacts/m1/diagnostics-cuda.json --device cuda`
  → Cayley ortogonalita FP32 max `1.9e-6`, FP64 `1.6e-15`; compound vs FP64 reference `≤ 6e-8` (FP32),
  `≤ 2.2e-16` (FP64); gradient při identitě (CPU, FP64): `torch.linalg.det` **0**, explicitní polynom = centrální
  diference (n=8: C2 32.965, C3 53.681). Tím je potvrzeno omezení z [GPU auditu](2026-10-01-gpu-environment.md)
  i na CPU a pro C3. Rekonstrukce 4096×1024 (r=1024) forward+backward: 3.9 ms (1 stupeň) až 72.5 ms (64 stupňů).

Zjištění při návrhu (podloženo testy):

- Stupně bez permutace se podle Cauchy–Binetovy věty slučují do jednoho stupně (test kolapsu); více stupňů má
  smysl jen s permutací mezi stupni.
- Na stupeň připadá ~`(n−1)/(n+1)` parametru na souřadnici (K=[1,2]); stupeň obou stran 4096×1024 stojí jako
  SVD faktory ranku ~0.8.
- Destilační MSE lineární vrstvy s pevným biasem je přesně kvadratická forma druhého momentu vstupu; pilot proto
  trénuje na matici `C` (1024×1024) místo minibatchů aktivací (odchylka od `phase3.md`, ekvivalentní ve střední
  hodnotě, bez šumu).

### 2. Kalibrační data pilotu (dokončeno)

`src/quic_wc/pilot/collect.py`: featurizer DINOv2 postavený přes timm s váhami načtenými jen ze safetensors shardu
(343 tensorů, bez načtení 7B modelu), BF16, `get_intermediate_layers(n={22})` jako v OpenVLA; předzpracování podle
LIBERO evaluace (lanczos3 224, center crop 0.9, normalizace procesoru). Hook na
`vision_backbone.featurizer.blocks.0.mlp.fc1` (4096×1024, bias).

Příkaz: `PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.collect --checkpoint ~/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/80970322773f81baa2e22fe495d0487b93a05cfa --dataset /home/kony/dissertation/ovlab-data/datasets/libero/libero_10/1.0.0/prepared --out artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/calibration`
→ exit 0, 43 s. 379 epizod, výběr 5 na úlohu se seedem 0 (4 train / 1 val na úlohu → 40/10), každý 10. snímek:
1118 train snímků (291 798 tokenů), 268 val snímků (69 948 tokenů). Kontroly: výstup hooku = `F.linear` v BF16
(max. rozdíl 0.0); BF16 modul vs FP32 reference na val: relL2 `0.00172`, průměrný kosinus `0.9999985`.
Manifest s vybranými epizodami, hashi vah a datasetu: `artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/calibration/manifest.json`.

Odchylka od plánu: checkpoint `openvla-7b-finetuned-libero-spatial` lokálně není → použit
`openvla/openvla-7b-finetuned-libero-10` (revize `80970322…`) a data `libero_10_no_noops` (digest `be5070fb…`).

### 3. Průzkumné běhy pilotu (dokončeno, mimo finální grid)

- `explore-lr` (500 kroků, gridy jen ve scratchpadu relace, kopie gridu je uložená v `results.json`): LR 3e-3/1e-2/3e-2.
  Jednostupňová varianta zadání zůstala na relMSE 0.82–0.90 (relL2 ≈ 0.91–0.95); stride L16 dosáhl relL2 0.155 (LR 3e-2).
  Zvolen LR 3e-2 a 3000 kroků.
- **Chybná interpretace a její náprava:** první tabulka srovnávala QuIC-WC jen se SVD vah se stejným počtem
  parametrů (stride L16 0.155 vs SVD r=12 0.469), což vypadalo jako výrazná výhoda QuIC-WC. Kontrola spektra
  ukázala extrémně anizotropní druhý moment (největší vlastní číslo `C` = 52.5 % stopy, průměr 38 % stopy,
  nejlepší směr `W C^½` nese 85.6 % výstupní energie). Aktivačně vážená SVD (ASVD) při ranku 12/16 dává 0.175/0.155,
  tj. totéž co stride L16. Runner byl proto rozšířen o ASVD a srovnání při stejném počtu parametrů **i bajtů**
  (θ ve FP32 = 4 B vs BF16 faktory 2 B), o variantu θ uložených v BF16 a o korekci biasu.
- Kosinus s biasem je pro tuto vrstvu slabé kritérium: varianta, která se téměř nic nenaučila (relL2 ≈ 0.95),
  měla kosinus s biasem 0.957; samotný bias dává relL2 s biasem 0.493. Rozhodující je kosinus/relL2 bez biasu.
- `explore-posctl` (3000 kroků): učitel ze stejné rodiny (stride L4) se z identity obnoví při θ·0.1
  (relL2 0.0033, relativní chyba vah 0.0099), ale ne při θ·0.3 (plató val relMSE ≈ 0.195, relL2 0.44, chyba vah 1.26;
  LR 1e-2 i 3e-2 stejně; izotropní cíl `C = I` ještě hůř, relMSE 0.889). Optimalizace z identity tedy uvízne v
  lokálním minimu – výsledky na reálné vrstvě jsou zčásti omezené optimalizací, nejen kapacitou.

### 4. Finální grid pilotu (dokončeno)

Grid [configs/pilot_libero10_dinov2_b0_fc1.json](../configs/pilot_libero10_dinov2_b0_fc1.json) (20 variant QuIC-WC,
SVD/ASVD baseline pro 12 ranků a obě politiky biasu), runner `src/quic_wc/pilot/run.py`.
Příkaz: `PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.run --calibration artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/calibration --grid configs/pilot_libero10_dinov2_b0_fc1.json --out artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/grid-v1`
→ exit 0, 1396 s; všech 20 fitů `completed`, roundtrip kompaktního exportu s maximální odchylkou 0.
Výsledky: `artifacts/.../grid-v1/results.{json,md}`; souhrn a interpretace: [docs/pilot-2026-11-12.md](../docs/pilot-2026-11-12.md).

Naměřené hlavní hodnoty (validace, relL2 bez biasu): varianta zadání 0.865–0.978; `ident` L16 0.867;
`stride`/`random` L16 0.134; `cyclic` L16 0.168; L64 0.0625 (s korekcí biasu 0.0507); ASVD při stejném počtu
parametrů r=12: 0.175, r=49: 0.071; ASVD při stejných bajtech (θ FP32) r=25: 0.1225, r=99: 0.0276; θ v BF16
mění relL2 nejvýše o 0.0015. Interpretace (oddělená od měření) je v reportu, oddíl 1.

### 5. Chyba nalezená v runneru a její náprava

Při revizi kódu během běhu gridu: cache ASVD rozkladů v `Context.asvd` měla klíč `(data_ptr, politika)`. Tensor
kontrolního učitele se po běhu uvolní a CUDA alokátor může jeho adresu přidělit dalšímu učiteli → stale rozklad.
Projev v `grid-v1`: ASVD reference `negctl_L16` při stejných bajtech vyšla 1.3175 (nemožné, rank 0 dává 1.0).
Reálná vrstva dotčena nebyla (tensor učitele žije po celý běh). Oprava: položka cache drží referenci na svého
učitele a při neshodě se přepočítá; regresní test `tests/test_pilot.py::test_asvd_cache_never_returns_a_stale_teacher`.
Čtyři kontroly byly zopakovány (`--only ... --skip-baselines`, výstup `grid-v1-controls-fixed/`, exit 0):
ASVD reference `negctl_L16` je 0.1842; metriky QuIC-WC všech čtyř kontrol jsou shodné s `grid-v1` (deterministické
seedy). V `grid-v1/results.json` zůstávají původní (u kontrol chybné) ASVD reference; report používá opravené.

### 6. Test posunu domény (dokončeno)

Kalibrace stejné vrstvy na LIBERO-Spatial (432 epizod, 170 433 tokenů) a LIBERO-Object (454 epizod, 201 492 tokenů)
stejným kolektorem (exit 0, 26 s / 31 s; shodný SHA-256 vah). Nový modul `src/quic_wc/pilot/shift.py` načte
kompaktní artefakty z `grid-v1` a SVD/ASVD fitované na LIBERO-10 a vyhodnotí je na všech tokenech nových sad
(exit 0, `artifacts/.../shift-v1/shift_results.{json,md}`). Naměřeno: chyby zůstávají na úrovni validace
LIBERO-10 (např. stride L16 0.134 → 0.119 / 0.136; ASVD r=12 0.175 → 0.154 / 0.169) a pořadí metod se nemění.
Relativní chyba vah QuIC-WC > 1 se v rámci LIBERO neprojevila; silnější posun netestován.

### 7. Závěrečné ověření

- `.conda/openvla-gpu/bin/python -m pytest -q` → **60 passed** (po přidání `tests/test_pilot.py`), CUDA testy běžely.
- Hodnoty v reportu byly ručně porovnány s `grid-v1/results.json`, `grid-v1-controls-fixed/results.json` a
  `shift-v1/shift_results.json`; při kontrole opraveny dvě formulace v reportu (rozsah zlepšení proti ASVD
  11–28 %, rozsah chyby vah ASVD).
- Prostředí `.conda/openvla-gpu` nebylo měněno (žádná instalace); kód se používá přes `PYTHONPATH=src`.

## Vytvořené a změněné soubory

- Nové: `pyproject.toml`, `src/quic_wc/**` (jádro, vrstva, metriky, destilace, export, diagnostika,
  `pilot/{collect,run,shift}.py`), `tests/test_{math,orthogonal,linear,checkpoint_metrics,distillation,pilot}.py`,
  `configs/pilot_libero10_dinov2_b0_fc1.json`, `docs/m1-math-core.md`, `docs/pilot-2026-11-12.md`, tento záznam.
- Změněné: [instructions/implementation-plan.md](../instructions/implementation-plan.md) (doplněn oddíl 10 s datem;
  předchozí text zachován), [audit/README.md](README.md) (odkaz).
- Lokální ignorované artefakty: `artifacts/m1/diagnostics-cuda.json`, `artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/`
  (celkem 474M, kalibrace LIBERO-10 ~229 MB, průzkumy `explore-lr`, `explore-posctl`, `grid-v1`, `grid-v1-controls-fixed`, `shift-*`, logy).

## Otevřené body a omezení

- Jedna vrstva, jeden checkpoint, jeden seed konfigurace; vliv na celý model (výstup featurizeru, akční tokeny,
  LIBERO SR) není změřen. M4/M5 nezahájeny.
- Optimalizační bariéra (pozitivní kontrola θ·0.3) neřešena; restarty ani jiná inicializace se nezkoušely.
- Uložení θ v BF16 jen vyhodnoceno, export ho nepodporuje.
- Navržená úprava brány M3 (proti ASVD při stejných bajtech, bez biasu) čeká na rozhodnutí uživatele.
- Změny nejsou commitnuté.
