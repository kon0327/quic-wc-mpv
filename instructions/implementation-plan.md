# Plán implementace QuIC-WC MVP pro OpenVLA

Stav: návrh po analýze `concept.md`, `phase1.md` až `phase5.md` a lokálních zdrojů v `external/`. Datum: 1. 10. 2026. Označení MPV v názvu projektu chápu jako MVP z konceptu. Tento dokument rozvrhuje implementaci; model ani trénování ještě nejsou implementované.

## 1. Cíl a výchozí rozhodnutí

První výsledek bude reprodukovatelný experiment komprese **jedné skutečné `nn.Linear` vrstvy** v OpenVLA: extrakce → SVD baseline → aktivační destilace QuIC-WC → kompaktní checkpoint → jednorázová dekomprese → A/B evaluace v LIBERO.

Komprimovaný artefakt musí nahradit původní váhu. Nesmí potřebovat uložené původní `W`, `U` nebo `V`. Po dekompresi bude inference používat hustou váhu a `F.linear`. MVP tedy cílí na velikost checkpointu a přenos modelu; **neslibuje úsporu ustálené GPU paměti ani zrychlení inference**. Velikost vrstvy a celého modelu budeme vykazovat zvlášť.

Výchozí volby pro první experiment:

- HF varianta OpenVLA z `prismatic/extern/hf`; baseline a student vzniknou ze stejného checkpointu a revize.
- LIBERO-Spatial; kandidát checkpointu je `openvla/openvla-7b-finetuned-libero-spatial`, uvedený v lokálním README. Dostupnost a konkrétní revizi ověří až příprava experimentu.
- Kandidát první vrstvy: `vision_backbone.featurizer.blocks.0.mlp.fc1`. Existenci, rozměry a skutečné spuštění potvrdí inventář modelu a hook. Alternativa: jedna ověřená `projector.fc*` vrstva.
- Pouze jedna explicitně pojmenovaná vrstva; bez plošného nahrazování všech modulů obsahujících `proj`.
- Základní compound konfigurace `n=8`, řády `[1, 2]`, bez sdílení parametrů mezi bloky. Počet bloků pro levou a pravou stranu odvodíme nezávisle z rozměrů vrstvy.
- Matematika a destilace FP32, numerická kontrola FP64; inferenční váha ve stejném dtype jako baseline. FP16/BF16 nebudeme používat pro determinanty, Cayleyho transformaci ani SVD.
- Lokální logy a artefakty, pevné seedy, konfigurace a hash vstupů. Žádné spuštění rozsáhlého trénování není součástí této plánovací změny.

## 2. Co lokální závislosti skutečně poskytují

| Zdroj | Použitelné části | Důsledek pro návrh |
|---|---|---|
| `external/compound-peft/peft/src/peft/tuners/compound/layer.py` | `_cayley_batch`, `_compute_circuit_compound_`, `_block_diagonal`, součiny adaptérů a posunutí bloků | Matematická reference; současný `Linear` ponechává `base_layer` a upravuje jeho výstup. Není hotovou kompresní vrstvou. |
| `external/compound-peft/peft/src/peft/tuners/compound/config.py` | Řády compound, počet bloků, sdílení, více adaptérů | Vlastní konfigurace WC oddělí rozměr báze, počet bloků, SVD rank a počet stupňů. `r` v compound znamená počet bloků, nikoliv SVD rank. |
| `external/openvla/prismatic/extern/hf/modeling_prismatic.py` | HF OpenVLA, dvě vizuální větve, projektor `fc1/fc2/fc3`, `predict_action` | Integrace nad HF modulem; názvy z nativní Prismatic varianty nejsou zaměnitelné. Vizuální backbone používá předposlední blok, takže kandidáta musíme ověřit hookem. |
| `external/openvla/experiments/robot/openvla_utils.py` | Loader, processor, crop, prompt, normalizace akcí | Zachovat shodné předzpracování a `norm_stats`; loader rozšířit naším wrapperem. |
| `external/openvla/experiments/robot/libero/run_libero_eval.py` a `libero_utils.py` | Rollouty, počáteční stavy, preprocessing obrazu a akce | Základ A/B evaluace. Přidat strukturované metriky, časování a evidenci chyb. |
| `external/openvla/prismatic/vla/datasets/datasets.py` | RLDS/OpenX pipeline, episodická varianta pro jednotlivý dataset | Volitelný adaptér pro OpenX; první kalibrace může používat LIBERO demonstrace nebo baseline rollouty. |
| `external/libero/` | Simulátor, úlohy, assety a inicializační stavy | Připravený benchmarkový kód. Demonstrační dataset a modelové checkpointy v kontrolovaných cestách nejsou přiložené. |

Původní [QuIC článek](https://arxiv.org/abs/2502.06916) popisuje PEFT adaptéry. Nahrazení celé váhy pomocí aproximovaných SVD faktorů je zde **experimentální rozšíření**, jehož kvalitu nelze převzít z výsledků adaptérového fine-tuningu.

Zdroje v `external/` budou výchozí referencí. Nový kód bude v samostatném balíčku; případné převzetí kódu z compound zachová Apache-2.0 oznámení a původ. Celý upravený PEFT nebude povinnou runtime závislostí matematického jádra.

## 3. Mezery zadání, které musí implementace vyřešit

### Rozměry a konvence SVD

Pro `W ∈ R^(m×n)` platí `U_r ∈ R^(m×r)`, `s ∈ R^r`, `V_r ∈ R^(n×r)` a `W_r = (U_r * s) @ V_r.T`. Výstup je `Y = X @ W_r.T + bias`. Pravý builder vrací `V`, nikoliv nejasně pojmenované `V`/`Vh` používané střídavě v ukázkách.

Při `n_base=8`, `K=2` má přímý součet rozměr `8 + C(8,2) = 36`. Nelze ho vložit do libovolné OpenVLA vrstvy jen nastavením `sigma_dim=out_features`. Levá a pravá strana potřebují vlastní konstrukci rozměrů a výběr `r` sloupců. Bias má přesně `out_features` prvků.

Truncated SVD nemá „ortogonální čtvercové matice“: tenké faktory mají ortonormální sloupce. Testujeme `U_r.T @ U_r` a `V_r.T @ V_r`, nikoliv identitu na jejich řádkovém prostoru. `keep_ratio=0.95` v ukázce znamená podíl počtu hodnot, nikoliv zachované energie. Parametry `rank`, `energy_ratio` a `relative_threshold` budou explicitně odlišené.

### Aproximační kapacita je první experimentální brána

Přímý součet je blokově diagonální. Pokud mají oba čtvercové faktory stejný jediný rozklad `[8, 28]`, pak `Q_U @ diag(s) @ Q_V.T` zůstane blokově diagonální. Nenulové mimoblokové části husté cílové váhy tedy nikdy nenaučí. Samotné zvýšení počtu trénovacích kroků to nevyřeší.

Nejdřív otestujeme přesnou jednostupňovou variantu ze zadání, včetně strukturálního omezení. Pro skutečnou vrstvu zahrneme do omezeného experimentu také součiny ortogonálních stupňů s různými pevnými permutacemi/posuny mezi bloky, inspirované `use_offset_blocks`. Ty mohou propojit původně oddělené souřadnice, ale **nezaručují dostatečnou kapacitu**. Zkontrolujeme i dosažitelnou podporu výsledné matice; součiny stupňů se stejným rozkladem bez promíchání nepomohou.

Pokud žádná konfigurace v předem stanoveném rozpočtu nedosáhne validační kvality při menším checkpointu, výsledkem bude zdokumentovaný neúspěch této architektury. Změna na tile-wise kompresi nebo dense residual by byla další varianta metody a musí být vyhodnocená samostatně, včetně všech uložených parametrů.

### Numerika, cache a evaluace

- Cayleyho transformaci počítat pomocí `torch.linalg.solve`, bez explicitní inverze. Identita musí mít správný dtype i device. Zvolený faktor v `S = P-P.T` nebo `0.5*(P-P.T)` bude součástí konfigurace.
- Lokální compound vynucuje `.float()` uvnitř determinantů; pro FP64 kontroly musíme zachovat dtype. Hraniční řády `K=0` a `K=n` nesmí mít natvrdo CPU tensor ani obecně nahrazovat `det(A)` jedničkou mimo SO(n).
- Cayley generuje SO(n) a nepokrývá všechny ortogonální matice. Případné pevné reflexe a SVD znaménkové konvence evidovat v metadatech; nečekat přesnou aproximaci libovolného SVD faktoru touto parametrizací.
- Místo automatického „vyhlazování“ Cayleyho transformace nejprve diagnostikovat numeriku. Přičtení epsilon jen do jmenovatele může porušit požadovanou ortogonalitu.
- Z ukázky fáze 4 neplyne dekomprese během `.to('cuda')`: volá ji až první `forward`. MVP použije explicitní `prepare_for_inference()` po přesunu na cílové zařízení, před prvním měřeným rolloutem.
- Cache váhy bude nepersistentní buffer; běžný tensor atribut nezaručuje korektní přesun zařízení a `torch.save(module)` by mohl uložit i celou cache. Exportovat kompaktní `state_dict`/safetensors, nikoliv pickle celého modulu.
- SIMPLER a implementace OpenVLA-OFT nejsou v dodaných závislostech. Fáze 5 proto začne v LIBERO s odpovídajícím OpenVLA baseline. OFT/SIMPLER zůstávají pozdějším rozšířením, nikoliv zaměnitelným baseline.
- Přesná rovnost FPS `1.0×` je kvůli varianci měření nevhodné kritérium. Cílem bude srovnatelná latence v předem zvolené toleranci a na stejných vstupech.

## 4. Návrh modulů a kontraktů

```text
src/quic_wc/
  config.py                    # rozměry, řády, rank, stupně, dtype, schema
  math/cayley.py               # batched solve
  math/compound.py             # indexy minorů, dávkované determinanty
  math/orthogonal.py           # přímé součty, identity tails, permutace
  layers/linear.py             # QuICLinear + cache dekomprese
  data/activations.py          # hooky, shardy, manifest a split
  data/libero.py               # demonstrace / baseline observations
  data/openx.py                # volitelná RLDS kalibrace
  decomposition.py            # SVD a výběr ranku
  distillation.py             # training/validation/resume
  checkpoint.py               # kompaktní schema, export a load
  integrations/openvla.py     # inventář a náhrada podle přesné cesty
  evaluation/libero.py         # baseline/student, rollout metriky
  evaluation/latency.py        # pevné vstupy a synchronizované měření
  cli.py                      # vstupní příkazy
configs/                      # synthetic, single-layer, libero-spatial
tests/                        # matematika, rozměry, cache, roundtrip
docker/                       # prostředí jádra a OpenVLA/LIBERO
artifacts/                    # lokální ignorované výstupy experimentů
```

Rozměrový kontrakt builderů:

1. `CompoundBlock(base_dim, orders)` vytvoří `⊕ C_k(A)`, s velikostí `b = Σ C(base_dim,k)`.
2. `OrthogonalTransform(dim, blocks, stages, permutations)` složí nezávislé bloky a případný identitní zbytek do **přesně** `dim×dim`. Nepoužívat ořezání řádků větší ortogonální matice: nezachovává ortonormalitu tenkých faktorů. Každý stupeň a jeho permutace mají rozměr `dim`.
3. `thin_factor(rank)` vrátí `Q @ E`, kde `E` vybírá `rank` různých sloupců identity. Výběr je deterministický a uložený; u jedné blokové fáze prověřit i rozložení vybraných sloupců mezi bloky.
4. Pro vrstvu sestavit `U_hat = Q_out @ E_out`, `V_hat = Q_in @ E_in` a `W_hat = (U_hat * sigma) @ V_hat.T`.
5. Pokud je potřeba změnit orientaci, použít explicitní pevnou ortogonální reflexi. `sigma` začíná ze singulárních hodnot, v prvním experimentu bude fixní; ablace může povolit trénování nebo signed scales s jasně popsanou konvencí.

Primární API: `QuICMatrixBuilder` (případně alias `QuIC_MatrixBuilder` ze zadání), `QuICLinear(in_features, out_features, rank, left_config, right_config, bias)`, `reconstruct_weight()`, `decompress()`, `invalidate_cache()` a `prepare_for_inference(model)`.

Destilace vždy prochází diferencovatelnou rekonstrukcí. Inference používá cache vytvořenou bez gradientů. Změna checkpointu, zařízení, dtype nebo přepnutí do tréninku cache invaliduje. Po přípravě musí opakovaný forward determinanty i solve úplně vynechat.

Při větších rozměrech aplikovat jednotlivé bloky a permutace rovnou na `E`/tenké faktory, bez materializace všech `dim×dim` mezimatic. Minory počítat po blocích/chuncích podle paměťového rozpočtu: plný batch má `C(n,k)^2*k^2` prvků pro jednu bázi. Rekonstrukce hustého `W` je povolená při destilaci jedné vrstvy a při dekompresi.

## 5. Pořadí implementace a výstupní brány

### M0 — Prostředí a reprodukovatelnost

- [ ] Založit samostatný balíček, konfigurace, ignorované artefakty a CPU testy jádra.
- [ ] Připravit izolované Python 3.10 prostředí pro OpenVLA/LIBERO podle lokálního README: PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10; kompatibilní CUDA image a flash-attn 2.5.5 ověřit na cílovém stroji.
- [ ] Oddělit minimální matematické závislosti, HF/inferenční stack a volitelný OpenX stack. TensorFlow potřebuje i dodaný LIBERO preprocessing, nikoliv pouze RLDS.
- [ ] Vyřešit PEFT konflikt: OpenVLA požaduje `peft==0.11.1`, lokální fork má `0.12.1.dev0`. Pro MVP používat samostatné matematické jádro a standardní závislosti OpenVLA; neinstalovat oba PEFT balíčky do stejného prostředí.
- [ ] Local fork má v `setup.py` čtení `README.md`, který v jeho `peft/` adresáři chybí; případný samostatný referenční PEFT experiment potřebuje opravu balení nebo pracovní kopii. Není to podmínka jádra WC.
- [ ] Nastavit cache modelů/datasetů a `LIBERO_CONFIG_PATH` do zapisovatelné pracovní cesty; předem vytvořit LIBERO config, aby import nevyvolal interaktivní dotazy ani zápis do home.
- [ ] Prověřit CUDA/BF16, render backend/EGL, jedno načtení modelu a jeden výstup `predict_action`. TF nesmí zabrat paměť potřebnou PyTorchu.
- [ ] Zapsat verze, hardware, seedy, konfiguraci a hashe externích zdrojů; lokální `.git` nemá použitelné metadata, takže automaticky nepředpokládat dostupný commit hash.

**Brána:** CPU jádro funguje samostatně; cílový GPU stroj zvládne model a jeden LIBERO rollout. Konkrétní VRAM limit stanovíme podle checkpointu, husté inference a workspace SVD; komprese na disku tento limit nesnižuje.

### M1 — Diferencovatelné matematické jádro (fáze 1)

- [ ] Cayley přes `solve`, cache kombinací indexů, batched/chunked compound, direct sum.
- [ ] Vektorizaci porovnat s nezávislou smyčkou přes minory na malých maticích.
- [ ] Otestovat `K=0,1,2,3,n`, neplatné řády, identity init, náhodný init, zbytky rozměrů a několik stupňů s permutacemi.
- [ ] Ortogonalita každého čtvercového transformu a ortonormalita vybraných sloupců: maximální absolutní chyba ≤ `1e-5` v FP32.
- [ ] FP64 `gradcheck` při nenulovém náhodném init; samostatně finite gradient při identity init, kde některé minory jsou singulární. Kontrolovat také nenulový gradient při vhodné náhodné vážené loss, nejen `grad is not None`.
- [ ] Přidat pozitivní syntetický případ s teacher vahou vytvořenou stejnou rodinou QuIC a negativní případ s obecnou hustou vahou.

**Výstup:** ověřené jádro a numerický report. Pozitivní syntetický případ odhaluje chyby optimalizace, ale není důkazem vhodnosti pro OpenVLA.

### M2 — Jedna vrstva, kalibrace a SVD baseline (fáze 2)

- [ ] Inventář `named_modules()` → přesné cesty, `in_features`, `out_features`, bias, dtype, počet bajtů. Zvolit skutečně spouštěnou vrstvu s přijatelnou velikostí SVD.
- [ ] Načíst stejný finetuned checkpoint jako pro benchmark; teacher v `eval()` a `inference_mode()`.
- [ ] Získat počátečních přibližně 50 kalibračních trajektorií z LIBERO demonstrací nebo zaznamenaných baseline rolloutů. Výběr uložit do manifestu. OpenX z fáze 2 implementovat jako volitelný datový adaptér po funkčním toku LIBERO.
- [ ] Rozdělit podle trajektorií, např. 40 train / 10 validation; žádné náhodné rozdělení sousedních snímků stejné trajektorie. Konečnou benchmarkovou evaluaci oddělit od výběru hyperparametrů.
- [ ] Použít benchmarkový obrazový preprocessing, včetně orientace obrazu, resize, center crop, processoru a promptu. HDF5 demonstrace ověřit podle jejich skutečného formátu; neotáčet již otočená RLDS data znovu.
- [ ] Hookem zachytit `X`, `Y` včetně původního bias; tensor detach a okamžitý bounded/sharded zápis místo neomezených seznamů v RAM. Hook vždy odstranit. U vision vrstvy zachytit obrazový průchod, ne duplikáty z generování akčních tokenů.
- [ ] Uložit bias, weight checksum, trajektorie/split, checkpoint revizi, preprocessing, dtype, shapes a počet tokenů. První formát: tensorové `.pt` shardy; kompaktní checkpoint: safetensors.
- [ ] Pro jeden `W` provést `torch.linalg.svd(..., full_matrices=False)` v FP32; FP64 je diagnostická volba. Nedělat SVD celého modelu současně.
- [ ] Vybrat rank podle explicitní politiky a validace. Změřit cosine similarity, MSE a relativní L2 pro `F.linear(X, W_r, bias)` vůči zachycenému `Y`; vykázat i metriku bez bias. Energie SVD sama nezaručuje cosine > `0.98`.

**Brána:** SVD baseline má validační mean cosine > `0.98`; vykázat i spodní kvantily a chyby nulových výstupů. Pokud potřebuje plný rank, pokračovat jako plnorankový WC experiment a nevykazovat úsporu low-rank faktorizace. Při paměťovém selhání vybrat menší vrstvu; randomized SVD je další explicitní varianta.

### M3 — Destilace a rozhodnutí o proveditelnosti (fáze 3)

- [ ] FP32 minibatch training nad shardy; základní loss `MSE(Y_hat, Y_teacher)` s korektním bias. Vyhodnocovat zvlášť training a validation.
- [ ] AdamW, cosine schedule, společné clipping všech trénovaných parametrů normou `1.0`; logovat normu před clippingem, loss, LR, gradienty, časy a peak memory.
- [ ] Fixní sigma a bias v prvním běhu; samostatná ablace learnable sigma. Případný krátký fit vůči SVD faktorům je jen inicializace, nikoliv náhrada aktivační destilace.
- [ ] Ukončit krok při nefinite loss/gradientu; neexportovat poškozené váhy. Checkpointovat optimizer, scheduler, RNG a nejlepší validační stav; resume ověřit proti stejným datům.
- [ ] Ohraničený pilotní rozpočet: dvě velikosti báze (`4`, `8`), `K=[1,2]`, jeden nebo čtyři stupně, stejný rank; nejprve nejvýše 1 000 kroků na konfiguraci s validací každých 100 kroků. Sdílení bloků odložit. Vybrané konfiguraci povolit delší běh pouze podle výsledků pilotu.
- [ ] Vedle dense teacher vykázat truncated SVD baseline a low-rank checkpoint při stejném ranku. Porovnat také variantu `K=[1]` a direct sum bez promíchání, aby byl měřitelný přínos compound i počtu stupňů.
- [ ] Exportovat validační cosine, relativní chybu, kompaktní velikost a omezení podpory váhy. Vyhlazený či „logaritmický“ tvar training loss není akceptační test.

**Brána:** alespoň jedna konfigurace má validační mean cosine > `0.98`, stabilní numeriku a skutečně menší uložený artefakt než dense vrstva při započtení dtype i metadat. Tento práh je pracovní filtr; robotický výkon musí ještě potvrdit M5. Pokud konfigurace neuspějí, nejdřív vyhodnotit kapacitu a rozpočet, nikoliv automaticky rozšířit patch na další vrstvy.

### M4 — Kompaktní modul a integrace (fáze 4)

- [ ] `QuICLinear` ukládá parametry levých/pravých bloků, sigma, původní bias a metadata rozměrů, stupňů, výběrů sloupců a permutací. Neslibovat pouze tři tensory, pokud architektura potřebuje další stav.
- [ ] `decompress()` ve FP32, převést konečný `W` do inferenčního dtype a `contiguous()`, uložit do `register_buffer(..., persistent=False)`.
- [ ] Otestovat invalidaci při `.to()`, změně dtype, `load_state_dict()` a `train()`. Hodnota cache nikdy nesmí potichu odpovídat jiným parametrům.
- [ ] Nahradit vrstvu podle přesné plné cesty a ověřit typ, rozměry i počet náhrad; zachovat device, bias a režim modulu. Chybná či nenalezená cesta je chyba, nikoliv tichý fallback.
- [ ] Roundtrip export/load po dekompresi musí mít stejnou kompaktní velikost a stejné výstupy v daném dtype. Cache nesmí být součástí `state_dict`.
- [ ] V inference ověřit, že dekomprese proběhne jednou a další forward používá pouze `F.linear`.

První integrace může načíst dense baseline a následně provést patch — to ověří chování, ale **nedokazuje samostatně nasaditelný komprimovaný model**. Finální MVP proto potřebuje i úplný export: běžné váhy ostatních vrstev + komprimovaný stav nahrazených vrstev + HF config/processor/tokenizer/normalizační statistiky + `quic_manifest.json`.

Vlastní loader sestaví architekturu, vloží `QuICLinear` před načtením stavu a načte celý export bez původního checkpointu a bez původní váhy cílové vrstvy. Sharded/streaming load a konstrukce bez souběžné kopie celého 7B modelu ověřit na cílovém stroji. HF `save_pretrained`/`from_pretrained` nezískají automaticky podporu nového modulu pouhým monkey-patchem.

**Brána:** úspěšný samostatný roundtrip celé komprimované exportní složky a numerická shoda s referenční `F.linear(X, W_hat, bias)`. Změřit cold load, dekompresi a peak memory odděleně; čas dekomprese musí být v praktickém řádu sekund až desítek sekund na deklarovaném hardwaru.

### M5 — LIBERO A/B a report (fáze 5 upravená podle závislostí)

- [ ] Baseline, SVD a QuIC-WC se stejnou checkpoint revizí, precision, attention backendem, processor/crop, akcemi a normalizačním klíčem. Nepřimíchávat současně 4bit/8bit kvantizaci.
- [ ] Smoke evaluace: dvě úlohy, pět rolloutů každá. Je to kontrola toku, nikoliv potvrzení marginu SR.
- [ ] Finální první suite: LIBERO-Spatial, 10 úloh × 50 počátečních stavů × 3 seedy pro baseline a zvolený QuIC-WC; SVD rollouty podle výpočetního rozpočtu, alespoň pokud ukazuje významnou změnu kvality. Použít stejné task/state/seed dvojice.
- [ ] Zachovat benchmarkový gripper mapping, čekací kroky a episode limits. Logovat `task_id`, `init_state_id`, seed, success/failure, délku epizody a chyby.
- [ ] Dodaný runner zachytává výjimky a pokračuje; report musí zvlášť vykázat chyby simulátoru/inference a nesmí je skrýt pod běžné neúspěchy politiky.
- [ ] Měřit SR po úlohách i agregovaně a párový rozdíl s 95% intervalem spolehlivosti; respektovat shlukování opakování podle úlohy/počátečního stavu.
- [ ] Latenci měřit po warmupu na společném uloženém korpusu pozorování, ne jen na odlišných trajektoriích dvou politik. `time.perf_counter()` + CUDA synchronizace nebo CUDA events pro GPU úseky; oddělit preprocessing, predikci akce, simulaci a cold start. Report p50/p95 a throughput, bez videa a diskového I/O v měřeném úseku.
- [ ] Porovnat kompaktní bajty cílové vrstvy a celého exportu, inference VRAM, peak load memory a čas dekomprese. FP32 kompaktní parametry porovnat s reálnou BF16 dense baseline, nikoliv jen s počtem parametrů.

**Pracovní akceptace:** bodový propad SR nejvýše 5 procentních bodů; pro tvrzení o potvrzeném zachování výkonu musí dolní mez intervalu rozdílu zůstat nad `-5 p. b.`. Pokud interval margin protíná, výsledek je statisticky neprůkazný. Cíl inference: medián latence nejvýše `1.05×` dense baseline při stejných podmínkách; vykázat také p95 a rozptyl. Jde o navržené experimentální prahy, nikoliv již dosažené výsledky.

## 6. Artefakty a plánované příkazy

Všechny příkazy níže jsou **navrhované rozhraní, zatím neexistují**:

```text
quic-wc inspect-model --config configs/libero_spatial.yaml
quic-wc collect-activations --config configs/libero_spatial.yaml
quic-wc decompose --config configs/libero_spatial.yaml
quic-wc distill --config configs/single_layer.yaml
quic-wc export --config configs/single_layer.yaml
quic-wc eval-libero --variant baseline --config configs/libero_spatial.yaml
quic-wc eval-libero --variant quic --config configs/libero_spatial.yaml
quic-wc report --run-dir artifacts/<run-id>
```

Konfigurace obsahuje checkpoint/revizi, přesnou cestu vrstvy, dataset a split, rank policy, báze/bloky/stupně/permutace, seed, dtype, training budget, device a cesty výstupů. Konfigurace nemůže současně zadat rozporné rank policy.

Každý experiment uloží `config.yaml`, `environment.json`, `model_inventory.json`, `activation_manifest.json`, SVD report, training log, nejlepší kompaktní checkpoint, rollout JSONL a výsledný JSON/Markdown report. Training resume checkpoint může obsahovat optimizer a diagnostické SVD cíle; nasaditelný export je nebude obsahovat.

Komprese vrstvy se počítá z **celého** nasaditelného stavu: parametry všech stupňů na obou stranách + sigma + bias + persistentní strukturální údaje + metadata. Například u plných `P` je jejich počet `Σ_side Σ_stage Σ_block base_dim²`, nikoliv vždy `2*base_dim²`. Dense baseline má `m*n + m` prvků při bias. Samostatně vykázat počet parametrů, tensorové bajty a skutečnou velikost souborů.

## 7. Co už bylo ověřeno při přípravě plánu

Provedena pouze read-only analýza zdrojů a malé CPU diagnostiky; žádná instalace závislostí, stahování modelu ani benchmarkový trénink.

- V aktuálním prostředí je Python `3.13.12`, PyTorch `2.12.1+cu130`; torchvision, transformers, timm, peft, libero, tensorflow, safetensors a pytest nejsou importovatelné. Nejde o prostředí deklarované lokálním OpenVLA.
- `torch.cuda.is_available()` vrací `False`; `nvidia-smi` hlásí blokovaný přístup k GPU operačním systémem. To dokládá stav této relace, nikoliv nepřítomnost GPU na hostiteli.
- Bez importu celého PEFT byly ze zdrojového AST načteny pouze matematické metody `_cayley_batch` a `_compute_circuit_compound_`. Pro `n=8`, `K=2` vznikl direct sum `36×36`; při identity init byla chyba ortogonality `0`, při malém náhodném init přibližně `5.96e-7`. Gradient náhodné vážené loss byl v obou případech finite.
- Na FP64 vstupu lokální compound kvůli převodu determinantů do FP32 vykázal odchylku přibližně `6.92e-8` vůči smyčkové FP64 referenci. Nejde o FP64 implementaci, přestože výstup vrací do původního dtype.
- Pro jednu náhodnou hustou váhu `36×36` (seed 7 v diagnostickém skriptu) tvořila mimobloková část při rozkladu `[8,28]` přibližně `0.584` relativní Frobeniovy normy. To je dolní mez chyby libovolné aproximace se stejnou blokovou podporou; není to výsledek na OpenVLA ani tvrzení o jeho aktivační cosine similarity.

## 8. Praktické pořadí práce

1. **Nejdřív M0 CPU část + M1:** rozměrové kontrakty, matematika, gradienty, pozitivní i negativní syntetický případ. Lze dělat bez checkpointu a GPU.
2. **Potom dokončit GPU část M0 a M2:** jeden checkpoint, jedna vrstva, malé kalibrační shardy a SVD baseline. Teprve zde vzniknou reálné paměťové a časové odhady.
3. **M3 je rozhodovací experiment:** kvalita proti kompresnímu poměru, omezené porovnání jedné a více compound fází. Bez úspěšné validační konfigurace nenásleduje plošná integrace.
4. **M4:** kompaktní serializace, cache, patch jedné vrstvy a samostatný loader/export.
5. **M5:** smoke a následně plná párová LIBERO evaluace. Další vrstvy, OpenX, další LIBERO suites a OFT/SIMPLER až podle výsledků tohoto prvního toku.

Časové odhady trénování a evaluace nyní nemají podklad: chybí načtený checkpoint, kalibrační data i funkční GPU. Rozvrh proto používá závislosti a měřitelné brány; termíny a rozsah dalších kompresí lze stanovit po pilotu M2/M3.

## 9. Aktualizace prostředí — 1. 10. 2026

GPU část přípravy prostředí je ověřená v samostatném prefixu `.conda/openvla-gpu`. Vychází z ovlab snapshotu `2026-07-17` a kopie prostředí `/home/kony/miniconda3/envs/openvla`; původní instalace zůstala zachovaná. GPU bylo v předchozí diagnostice skryté sandboxem, mimo něj funguje RTX 4090 s PyTorch `2.2.0+cu121`.

Prošly CUDA FP32/BF16 operace, SVD, ortogonalita a gradienty direct sum, FlashAttention forward/backward, skutečné HF OpenVLA forward a `predict_action` na malém náhodném modelu, importy lokálních zdrojů a tři kroky LIBERO s EGL renderem. `pip check` je čistý. Byla opravena kompatibilita tensorflow-metadata/protobuf a namespace cesta lokálního LIBERO.

**Důsledek pro M1:** PyTorch 2.2/CUDA zde vrací nulový gradient compound řádu 2 při přesně identitní bázi přes `torch.linalg.det`, zatímco explicitní determinant `ad-bc` vrací nenulovou derivaci. Direct sum s řádem 1 má gradient a náhodný init prošel. Jádro musí mít kontrolu singulárních minorů a pro řád 2 využít ověřený explicitní vzorec.

Návod a přesné výsledky: [GPU prostředí](../deploy/environments/openvla/README.md). M0 zatím nepovažujeme za kompletní: zbývá načíst skutečný checkpoint OpenVLA-7B, ověřit jeho paměťový rozpočet a policy rollout. Syntetické testy nejsou evaluací politiky. Matematické jádro M1 zatím není implementované.
