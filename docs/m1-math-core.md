# Matematické jádro QuIC-WC (M1)

Stav: implementováno 12. 11. 2026 v `src/quic_wc/`, ověřeno testy (`tests/`) a numerickým reportem
(`python -m quic_wc.diagnostics`). Dokument popisuje skutečně implementované kontrakty; navazuje na
[plán implementace](../instructions/implementation-plan.md), oddíly 3–4. Výsledky pilotu jsou v
[pilot-2026-11-12.md](pilot-2026-11-12.md).

## 1. Kontrakt vrstvy

Pro hustou vrstvu `W ∈ R^(m×n)` (výstup × vstup), `Y = X Wᵀ + b`, ukládá QuIC-WC pouze malé parametry a
rekonstruuje

```text
Ŵ = Û diag(σ) V̂ᵀ,   Û = Q_out E_out ∈ R^(m×r),   V̂ = Q_in E_in ∈ R^(n×r)
```

- `Q_out ∈ SO(m)`, `Q_in ∈ SO(n)` jsou strukturované QuIC transformace (oddíl 4), levá strana = výstup.
- `E` vybírá `r` různých sloupců identity, takže `Û`, `V̂` mají přesně ortonormální sloupce. Nikde se
  neořezávají řádky větší matice.
- `r` je SVD rank (≤ `min(m, n)`), nikoliv počet bloků jako `r` v compound-PEFT.
- `σ` je inicializováno z `r` největších singulárních hodnot učitele. Je buď pevné, nebo s učeným zesílením
  `σ = σ₀ ⊙ g` (zesílení může změnit znaménko). Při exportu se `g` vnoří do `σ`.
- `b` je výchozí pevný bias učitele. Pilot navíc měří korekci `b̂ = b + (W − Ŵ) μ`, viz oddíl 7.

## 2. Cayleyho báze (`math/cayley.py`)

`A = (I − S)(I + S)⁻¹` podle `phase1.md`. Protože `I − S` a `(I + S)⁻¹` komutují, počítá se
`A = solve(I + S, I − S)` bez explicitní inverze. `S` je antisymetrická a je dána `n(n−1)/2` horními prvky
`θ` (ekvivalent zadání `S = P − Pᵀ` s horní trojúhelníkovou `P`). Nulové `θ` dávají přesně `I`. Obraz je
SO(n) bez matic s vlastním číslem −1. Výpočet běží jen ve FP32/FP64 (half precision je odmítnuta) s vypnutým
autocastem. `inverse_cayley` slouží testům a konstrukci učitelů.

Konvence je opačná k `quic-peft-vla` (`(I + S)(I − S)⁻¹`): jde o záměnu `θ → −θ`, artefakty obou projektů
nejsou zaměnitelné a konvence je zapsaná v manifestu artefaktu.

## 3. Compound matice (`math/compound.py`)

`[C_k(A)]_(I,J) = det(A[I,J])` přes vzestupné lexikografické `k`-podmnožiny.

- `k = 2, 3`: explicitní polynomy determinantu, dávkově. **`torch.linalg.det` se pro ně nepoužívá.**
  V PyTorch 2.2 dává nulový gradient pro singulární minory a těch je u identity většina. Naměřeno
  (`artifacts/m1/diagnostics-cuda.json`, výpočet na CPU v FP64): norma gradientu náhodného lineárního
  funkcionálu `C_2(A(θ))` při `θ = 0` pro `n = 8` je přes `det` **0**, přes explicitní polynom **32.965**,
  centrální diference **32.965** (shoda ~1e-11). Pro `k = 3` vychází 0 / 53.681 / 53.681.
- `k = 0` vrací `1×1` jedničku ve správném dtype/zařízení. `k = n` vrací `det(A)`, pro `n > 3` přes
  `torch.linalg.det`; plná matice je u ortogonální báze regulární, takže gradient je spolehlivý.
- `4 ≤ k < n` je dostupné jen s `allow_generic_det=True`, pouze pro diagnostiku.
- Dtype báze se zachovává. FP64 kontroly jsou skutečně FP64, na rozdíl od lokálního compound-PEFT, který
  minory převádí na FP32.

Testované identity: Cauchy–Binet `C_k(AB) = C_k(A) C_k(B)`, `C_k(Aᵀ) = C_k(A)ᵀ`, a tedy ortogonalita
`C_k` ortogonální matice.

## 4. Strukturovaná transformace (`math/orthogonal.py`)

Jeden stupeň je `S = B P`: pevná permutace řádků `P` a blokově diagonální `B`. Blok `t` má tvar
`C_(k1)(A_t) ⊕ C_(k2)(A_t) ⊕ … ⊕ I_pad` s vlastní bází `A_t ∈ SO(n)`. Bloky pokrývají `N·b` souřadnic, zbytek
`dim − N·b` je identita. Výchozí šířka bloku je compound dimenze (pro `n = 8`, `K = [1,2]` je to 36), takže
`4096 = 113·36 + 28` a `1024 = 28·36 + 16`.

`Q = S_L ⋯ S_1`. Permutace se neukládají, generují se deterministicky z (druh, dim, stupeň, šířka bloku,
seed):

| druh | chování |
|---|---|
| `identity` | stejné seskupení ve všech stupních |
| `cyclic` | posun o půl bloku na stupeň (cihlový vzor, lokální míchání) |
| `stride` | prokládání: nový blok bere souřadnice z různých starých bloků (globální míchání, styl Monarch) |
| `random` | permutace se seedem |

**Důsledek Cauchy–Binetovy věty:** stupně se stejným seskupením se násobí na jediný stupeň
(`C_k(A₂)C_k(A₁) = C_k(A₂A₁)` po blocích). Další stupně tedy přidávají kapacitu jen s permutací. Na
syntetických datech to ověřuje test `test_identity_permutation_stages_collapse_to_one_stage`, na reálné
vrstvě varianta pilotu `ident_n8K12_L16`.

Tenký faktor `Q E` vzniká aplikací stupňů přímo na `E` (`dim × r`) bez materializace `Q`. Pravidlo výběru
`spread` prochází pozice uvnitř bloků (nejdřív compound souřadnice) cyklicky přes bloky a nakonec identitní
zbytek. `leading` bere prvních `r` sloupců.

Parametry: `L · N · n(n−1)/2` na stranu. Pro `K = [1,2]` a šířku bloku rovnou compound dimenzi je to
přibližně `(n−1)/(n+1)` parametru na souřadnici a stupeň. Jeden stupeň obou stran má tedy zhruba cenu
SVD faktorů ranku 0.8.

## 5. Vrstva `QuICWCLinear` (`layers/linear.py`)

- V tréninku je forward vždy diferencovatelný přes rekonstrukci: `F.linear(x, Ŵ, b)`.
- `decompress()` sestaví `Ŵ` jednou bez gradientu ve FP32/FP64, převede ho na inferenční dtype, zavolá
  `contiguous()` a uloží do nepersistentního bufferu. Eval forward pak volá jen `F.linear`. Test ověřuje, že
  opakovaný forward rekonstrukci nevolá.
- Cache je invalidována při `.to()`/změně dtype, `train()`, `load_state_dict()` a při jakékoliv změně
  uloženého stavu na místě (kontrola verzí tensorů, např. po kroku optimizéru). `strict_cache=True` místo
  líné dekomprese vyhodí `StaleCacheError`.
- Host `.to(bfloat16)` nezmění dtype `θ` ani `σ`; bias následuje dtype hostitele.
- `state_dict` obsahuje jen `left.theta`, `right.theta`, `sigma0`, (`sigma_gain`) a `bias`. Výběry,
  permutace ani cache uložené nejsou.
- `prepare_for_inference(model)` přepne vrstvy do eval a dekomprimuje je na aktuálním zařízení.

## 6. Export (`checkpoint.py`)

`layer.safetensors` (`theta_out`, `theta_in`, `sigma` se vnořeným zesílením, `bias`) a `quic_wc.json`
(konfigurace, konvence, SHA-256 tensorů, metadata). Neukládá se původní `W`, `U` ani `V`. Načtení ověřuje
hash, tvary a dtype. Přepsání existujícího artefaktu je odmítnuto. `storage_report` počítá skalární
parametry a bajty tensorů proti husté BF16 vrstvě.

## 7. Destilační cíl (`metrics.py`, `distillation.py`)

U lineární vrstvy s pevným biasem je MSE výstupů přesně kvadratická forma druhého momentu vstupu
`C = E[x xᵀ]`:

```text
E‖x (Ŵ − W)ᵀ‖² = tr((Ŵ − W) C (Ŵ − W)ᵀ)
```

Celá kalibrační sada je tedy reprezentována maticí `n × n` (zde 1024 × 1024) a průměrem `μ`. Trénink
minimalizuje `tr(ΔCΔᵀ) / tr(WCWᵀ)` bez minibatch šumu. Odpovídá MSE z `phase3.md` ve střední hodnotě,
jen bez šumu a bez ukládání gigabajtů aktivací. Kosinové metriky po tokenech se stále počítají na skutečných
validačních tokenech.

Korekce biasu `b̂ = b + (W − Ŵ) μ_train` stojí stejnou paměť jako původní bias. Odstraňuje chybu průměru;
cílem je pak `tr(Δ Σ Δᵀ)` s centrovanou kovariancí `Σ = C − μμᵀ`. Obecná přesná chyba je
`tr(ΔCΔᵀ) + 2 δbᵀ Δ μ + ‖δb‖²`, testovaná proti výpočtu po tokenech.

Smyčka: AdamW (`θ` a zesílení `σ` v samostatných skupinách), warmup + cosine schedule, globální clipping
gradientu. Nefinitní loss nebo gradient zastaví běh před krokem optimizéru. Na konci se vrací nejlepší
validační stav.

Baseline se stejnou cenou: SVD vah a ASVD (aktivačně vážená SVD, optimum ranku `r` v metrice `C`:
`[W R]_r R⁻¹`, `R = C^½`).

## 8. Ověření (12. 11. 2026)

`pytest`: 57 testů prošlo (CPU i CUDA testy, RTX 4090, PyTorch 2.2.0+cu121), viz
[audit](../audit/2026-11-12-m1-core-pilot.md). Testy pokrývají:

- compound proti nezávislé smyčce s FP64 determinanty pro `n = 2..6` a `k = 0..n`;
- `gradcheck` compound při identitě i náhodné bázi a celé řetězce `θ → Cayley → compound → tenký faktor`;
- ortogonalitu: FP64 `≤ 1e-12`; FP32 tenký faktor 4096 × 512 se 4 stupni `≤ 1e-5`;
- shodu s nezávislou hustou konstrukcí (`block_diag` + permutační matice) pro všechny druhy permutací,
  zbytky a řády `[1]`, `[2]`, `[1,2]`, `[1,2,3]`;
- kolaps stupňů bez permutace, nenulový gradient všech bloků z identity, cache a invalidaci, roundtrip
  exportu a integritu hashe;
- shodu kvadratické formy s MSE po tokenech, optimalitu ASVD a syntetickou pozitivní kontrolu (učitel ze
  stejné rodiny se z identity naučí na relMSE < 1e-2).

Numerický report `artifacts/m1/diagnostics-cuda.json` (lokální, ignorovaný artefakt):

- maximální chyba ortogonality Cayley: FP32 `1.9e-6`, FP64 `1.6e-15`;
- compound vs reference pro `n = 8`: FP32 `≤ 6e-8`, FP64 `≤ 2.2e-16`;
- rekonstrukce 4096 × 1024, `r = 1024`, forward+backward: 1 stupeň 3.9 ms, 4 stupně 6.5 ms, 16 stupňů
  17.7 ms, 64 stupňů 72.5 ms; chyba ortonormality `≤ 9e-7`.

## 9. Omezení

- Řády jsou omezené na 1–3 (explicitní polynomy). Implicitní aplikace C2 přes `R A Rᵀ` (z `quic-peft-vla`)
  není implementovaná; při `n ≤ 16` není potřeba.
- Cayley pokrývá jen SO(n) bez matic s vlastním číslem −1. Pro tenké faktory s `r < dim` to rozsah
  nezmenšuje. Při `r = dim` na obou stranách může rozhodovat znaménko determinantu; učené zesílení `σ`
  smí měnit znaménko.
- `θ` je ukládáno ve FP32 (4 B), zatímco husté a SVD faktory v BF16 (2 B). Pilot proto měří i variantu
  s `θ` zaokrouhleným na BF16.
- Optimalizace z identity nedosáhne učitele ze stejné rodiny při větších rotacích (pozitivní kontrola
  pilotu, oddíl 5 reportu). Jde o omezení optimalizace, které je nutné odlišovat od kapacity struktury.
