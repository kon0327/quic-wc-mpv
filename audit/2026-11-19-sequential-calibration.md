# Sekvenční kalibrace MLP DINOv2 (2026-11-19)

## Zadání a rozsah

- Uživatel 2026-11-19 rozhodl: začít sekvenční kalibrací (navrženo
  v [auditu skupiny](2026-11-18-group-dinov2-mlp.md)) a commitnout dosavadní vývoj.
  - Commit: `30b4910` na větvi `m4-lite`, před ním `pytest -q` → 79 passed.
- Uživatel navrhl tandemovou architekturu: komprimovaná vrstva QuIC-WC plus kalibrační adaptér QuIC-PEFT. Moje
  stanovisko je v odpovědi uživateli a shrnuté níže v oddílu „Tandem“. Nic z toho zatím není implementováno.

## Plán (stanoveno před spuštěním)

- **Motivace:** ve skupině se chyby skládají 14–32× nadaditivně a in-situ chyba vrstev roste. Hypotéza je, že
  příčinou je posun vstupů proti kalibraci.
- **Postup:** vrstvy se kalibrují a komprimují postupně v dopředném pořadí, b0.fc1, b0.fc2, b1.fc1, … b22.fc2
  (46 vrstev).
  - Blok 23 OpenVLA nepoužívá. V hodnocení se doplní z nezávislých fitů QuIC L16; na výsledek nemá vliv.
  - Vrstva k se kalibruje na vstupech kopie featurizeru, ve které jsou vrstvy 0..k−1 už nahrazené nasazovací
    BF16 vahou.
- **Dva cíle:**
  - `input` (klasický, styl GPTQ): učitel zůstává `W, b`, jen statistiky vstupů jsou z komprimovaného řetězce.
  - `output` (asymetrický): cílem je výstup nezměněného modelu `W x_ref + b` pro spárovaný vstup `x_c`.
    - Přesně ekvivalentní fit dostane učitele `W_t = W S_rc S_cc⁻¹` s biasem `b_t = b + W μ_ref − W_t μ_c`, kde S
      jsou centrované momenty z trénovacích snímků. Ekvivalence je ověřena testem `tests/test_sequential.py`.
    - Ridge 1e-8 je zvolen předem a slouží jen k numerické stabilizaci. Mřížka ridge se jen reportuje jako
      diagnostika na validačních tokenech, nevybírá se podle ní.
- **4 řetězce:** QuIC-WC L16 (recept v2, `configs/seq_dinov2_mlp.json`) a ASVD se stejnými bajty (ranky
  z `dinov2_mlp:asvd:L16`), každý s cílem `output` a `input`.
- **Vyhodnocení M4-lite:**
  - reference;
  - nezávislé QuIC L16 a ASVD L16 (bez sekvenční kalibrace; z 18. 11. KL 12.3 a 12.8);
  - 4 sekvenční skupiny;
  - reference na konci.
- **Kontroly:**
  - referenční cesta každé vrstvy = statistiky původních kalibrací (`reference_path_vs_original_calibration`);
  - u první vrstvy posun vstupu 0;
  - diagnostiky po vrstvách: posun vstupu, chyba původní vrstvy proti referenčnímu výstupu, chyba nainstalované
    vrstvy proti referenčnímu výstupu.
- **Otázky:**
  1. Sníží sekvenční kalibrace KL skupiny, a o kolik řádů?
  2. Je cíl `output` lepší než `input`?
  3. Zůstává QuIC-WC lepší než ASVD i po sekvenční kalibraci?
- **Odhad:** QuIC řetězce 2 × 46 × ~100 s, ASVD 2 × 46 × ~40 s, vyhodnocení ~12 min, celkem ~3.5–4 h.

## Implementace (provedeno)

- `src/quic_wc/pilot/sequential.py`:
  - podpříkaz `chain`: kopie featurizeru, spárovaný sběr momentů, asymetrický učitel, fit přes
    `quic_wc.pilot.run` nebo ASVD jako v `m4_lite`, instalace váhy, `chain.json` s možností navázat;
  - podpříkaz `plan`: plán pro `m4_lite run`.
  - Spouští se jen DINOv2 featurizer (`get_intermediate_layers(n={len(blocks)-2})`, batch 1, pixel values
    z oficiálního procesoru). Hook ukončí dopředný průchod za sledovanou vrstvou.
- Testy `tests/test_sequential.py`:
  - ekvivalence asymetrického cíle s kvadratickým objektivem fitu včetně korigovaného biasu;
  - identita při shodných vstupech;
  - spárovaný výběr tokenů.
  - `pytest -q` → 82 passed.
- Smoke testy (8 + 8 snímků, fit 50 kroků, `scratchpad`) prošly: řetězec QuIC `output`, řetězec ASVD `input`,
  navázání přerušeného běhu (2 vrstvy znovu nainstalovány, 3. dofitována) a `plan`.
- Běh: `artifacts/pilot/2026-11-19-sequential/run.sh`, log `run.log`, spuštěno 2026-11-19 dopoledne.

## Chyby a neúspěšné pokusy

- **První smoke test:** fit padal, protože manifest kalibrace neměl klíč `checks` a pak `checkpoint.repo`, které
  vyžaduje `run.py`. Doplněno.
- **Výchozí ridge 1e-4 relativně k průměrnému vlastnímu číslu S_cc** zkresloval cíl i bez jakéhokoli posunu
  vstupů. Na b0.fc1 vycházela chyba lineárního cíle 0.0034 proti 2e-7 při ridge 0, protože vstupy DINOv2 jsou
  silně anizotropní. Výchozí hodnotu jsem před ostrým během změnil na 1e-8, tedy jen numerickou stabilizaci.
- **První ostrý start** (10:34) okamžitě skončil: `run.sh` odvozoval z názvu řetězce `out`/`in` místo
  `output`/`input` (argparse error) a pokračoval dál až k `plan` a `m4`. Opraveno mapováním `${c#*_}put`; skript
  teď při chybě řetězce skončí. Prázdné výstupy jsem smazal.
- **Druhý start** (10:35), po 2 vrstvách zastaven:
  - Referenční cesta seděla přesně: rozdíl `c_train` i `c_val` proti kalibraci z 18. 11. je 0.0.
  - b0.fc1 ale dala relL2 0.1344 proti 0.1295 nezávislého fitu, přestože vstupy byly totožné.
  - Příčina: ridge (1e-8, stahování k nule) nechal učitele `W_t = W S_rc (S_cc+λI)⁻¹` volného ve směrech
    s téměř nulovým rozptylem. `W_t` se lišil od `W` o 17 % ve Frobeniově normě, a to změnilo inicializaci fitu
    ze SVD učitele.
  - Oprava: ridge stahuje **korekci k původní váze**, `W_t = W + W(S_rc − S_cc)(S_cc+λI)⁻¹`. Při λ = 0 jde
    o stejné optimum; při shodných vstupech `W_t = W` přesně. Přidán test
    `test_ridge_keeps_original_weight_in_unconstrained_directions` (4 passed).
  - Výchozí ridge je teď 1e-4 (relativní tlumení korekce), zvolené před třetím startem. Transparentně: diagnostika
    druhého startu (ještě se stahováním k nule) ukázala na validačních tokenech b0.fc2 nejnižší chybu právě při
    1e-4. Hodnota tedy není úplně nezávislá na validačních datech, rozdíly v mřížce jsou ale malé (0.165–0.183).
- **Třetí start** (≈ 10:45) běží.
  - Kontrola: b0.fc1 má `W_t == W` bitově, bias stejný, relL2 0.1295 = nezávislý fit.

## Tandem: stanovisko (interpretace, neověřeno)

- Princip „komprese + doladění adaptérem“ je známý a funguje, viz QLoRA / LoftQ / kompenzace chyby nízkým rankem.
- Adaptér QuIC-PEFT (`quic-petf-mpv/quic_openvla/adapter.py`) je ortogonální rotace výstupu `R·W`. Ve QuIC-WC už
  výstupní ortogonální faktor `Q_out` je, takže rotace se do něj matematicky vstřebá a nemění singulární hodnoty.
  Per vrstvu jde tedy o ekvivalent několika dalších stupňů QuIC-WC.
- Skutečný přínos tandemu by byl v **jiném cíli tréninku** (KL na výstupu modelu, end-to-end nebo po blocích), ne
  v kapacitě adaptéru. Ten cíl jde aplikovat i přímo na θ a σ QuIC-WC.
- Doporučený test (po sekvenční kalibraci), vždy se stejnými přidanými bajty:
  - (a) doladění jen σ a biasů;
  - (b) doladění všech θ;
  - (c) zamrazené WC plus adaptér QuIC-PEFT.

## Průběh a výsledky

### Řetězec QuIC-WC L16, cíl `output` (provedeno, 10:39–11:46)

- 46 vrstev, ~60 s (fc1) / ~100 s (fc2) na vrstvu, `exit 0`.
- Kontroly:
  - referenční cesta = kalibrace z 18. 11. u všech vrstev (rozdíl momentů 0.0);
  - b0.fc1: `W_t == W`, relL2 0.1295.
- **Posun vstupů na validačních snímcích, bloky 0–11** (`interim/drift_b0-11.json`, skript `drift.py`; jde
  o samostatnou kontrolu přes timm featurizer, hodnoty sekvenčního řetězce se shodují s diagnostikami
  v `chain.json`):

  | vstup | nezávislé QuIC L16 | nezávislé ASVD L16 | sekvenční QuIC L16 (`output`) |
  |---|---:|---:|---:|
  | b1 fc1 | 0.226 | 0.262 | 0.122 |
  | b2 fc2 | 0.801 | 0.887 | 0.319 |
  | b5 fc1 | 1.020 | 1.156 | 0.688 |
  | b8 fc1 | 0.870 | 1.055 | 0.602 |
  | b11 fc1 | 0.945 | 1.118 | 0.610 |
  | b11 fc2 | 0.945 | 0.998 | 0.858 |

- **Lokální kompenzace fc2:** nainstalovaná vrstva je blíž výstupu nezměněného modelu než původní vrstva na
  stejném posunutém vstupu, např. b11 fc2 0.581 proti 0.775 a b5 fc2 0.296 proti 0.413. U fc1 to neplatí, protože
  tam dominuje chyba komprese.
- **Zesílení přes GELU:** chyba b0.fc1 0.129 posune vstup b0.fc2 o 0.479.
- **Průběžné vyhodnocení M4-lite.** Běh jsem po prvním řetězci na pokyn uživatele přerušil; `asvd_out` měl hotových
  23 vrstev. Výsledky jsou v `interim/m4/results.md`, plán v `interim/m4_plan.json`.

  | varianta | tf KL | tf top-1 | free shoda | free = dataset |
  |---|---:|---:|---:|---:|
  | nezávislé QuIC L16 | 12.337 | 0.267 | 0.230 | 0.169 |
  | nezávislé ASVD L16 | 12.794 | 0.238 | 0.223 | 0.167 |
  | **sekvenční QuIC L16 (`output`)** | **10.183** | 0.316 | 0.245 | 0.149 |

  Nezávislé varianty reprodukují běh z 18. 11. na 5 desetinných míst a reference na konci má KL 0.

### Interpretace (není měření)

- Sekvenční kalibrace funguje: posun vstupů klesá zhruba o třetinu a fc2 kompenzuje předchozí chyby.
- Při 1.6 % bajtů na vrstvu ale skupinu nezachrání, KL zůstává v režimu kolapsu (10.2 proti 12.3).
- Rozpočet L16 je pro 46 vrstev řádově malý. Zbylé řetězce (`asvd_out`, `quic_in`, `asvd_in`) při L16 proto
  nespouštím, protože by skončily v témže režimu.
- Uživatel souhlasil s přechodem na vyšší rozpočet nebo na doladění na KL výstupu. Další krok je proto levná sonda
  rozpočtu: celá skupina s nezávislým ASVD při ranku 32–512, viz
  [audit/2026-11-19-budget-probe.md](2026-11-19-budget-probe.md).

## Závěr (doplněno 2026-11-19 odpoledne)

- **Odpověď na otázku 1:** sekvenční kalibrace s cílem `output` funguje, ale jen v násobcích, ne o řády. Při stejných
  bajtech snižuje KL skupiny 46 vrstev:
  - QuIC L16: 12.34 → 10.18 (1.2×, režim kolapsu);
  - ASVD r128: 2.69 → 1.32 (2.0×, 95% CI po epizodách [1.73, 2.36]);
  - ASVD r256: 0.369 → 0.217 (1.7×, CI [1.58, 1.90]).
  - Sekvenční varianta je lepší v 10/10 validačních epizodách. Detail a chyba sdílené cache ASVD vah, opravená
    a přeměřená, jsou v [auditu sondy rozpočtu](2026-11-19-budget-probe.md).
- **Otázky 2 a 3** (`output` proti `input`, QuIC proti ASVD po sekvenční kalibraci) zůstávají nezměřené, protože
  řetězce `quic_in`, `asvd_out` a `asvd_in` při L16 jsem vědomě nespustil.
  - Výhoda QuIC-WC proti ASVD po vrstvách je doložená v [auditu skupiny](2026-11-18-group-dinov2-mlp.md).
    Interpretace: sekvenční kalibrace je nezávislá na metodě fitu, takže se nečeká, že by pořadí metod obrátila.
- **Důsledek pro tandem (interpretace):**
  - Sekvenční kalibrace se stává výchozím bodem (varianta 0).
  - Zbývající mezera (KL 0.2–1.3 při 16–31 % bajtů) se musí řešit doladěním na výstup modelu podle
    [plánu tandemu](../instructions/tandem-recovery-plan.md).

## Dotaz uživatele: „ztrácí QuIC-WC za ASVD?“ (2026-11-19 večer)

- **Odpověď:** ne. Tabulka v závěru řadí vedle sebe různé rozpočty: QuIC L16 má 1.6 % bajtů, ASVD r128 16 % a r256
  31 %, tedy 10–20× víc.
- **Při stejných bajtech** (měřeno 18. 11. a znovu dnes) vede QuIC-WC, nebo je v režimu kolapsu nerozlišitelný:
  - L16: KL 12.34 proti 12.79 (dnes reprodukováno na 5 míst);
  - L32: 9.94 proti 12.49;
  - L8: 15.46 proti 12.99, tam vede ASVD.
  - Vše je ale v kolapsu, kde KL saturuje.
- **Po vrstvách** QuIC potřebuje k dosažení stejné relL2 o 36–72 % méně bajtů.
- **Chybí:** skupinové srovnání QuIC-WC a ASVD ve vyšším rozpočtu (např. QuIC L64–L128 proti ASVD stejných bajtů),
  sekvenčně i nezávisle. ASVD se při vyšším ranku spočítá okamžitě, QuIC fity při velkém L trvají hodiny. Neprovedeno,
  uživatel řekl „zatím nepokračuj“.
- **Kontrola:** prošel jsem účtování bajtů v `m4/results.md`. Poměr QuIC L16 (0.0159) a ASVD L16 (0.0165) je
  srovnatelný, ASVD r128/r256 má 0.157/0.313.
