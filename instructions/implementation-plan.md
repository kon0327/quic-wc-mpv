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

## 10. Aktualizace — 12. 11. 2026: M1 a první pilot jedné vrstvy

**M0 (CPU část) a M1 jsou implementované:** balíček `src/quic_wc/` (src layout, `pyproject.toml`; sdílené conda prostředí se neinstalací nemění), Cayley přes `solve`, explicitní compound C1–C3 se správným gradientem při identitě, vícestupňové strukturované transformace s permutacemi, tenké faktory, `QuICWCLinear` s nepersistentní cache a invalidací, kompaktní export. Ověření: 60 testů (CPU i CUDA) a numerický report; návrh a výsledky v [docs/m1-math-core.md](../docs/m1-math-core.md). Měřením je potvrzeno, že `torch.linalg.det` dává při identitě nulový gradient (C2 i C3), zatímco explicitní polynomy odpovídají konečným diferencím.

**Pilot M2/M3** na `vision_backbone.featurizer.blocks.0.mlp.fc1` (4096×1024) checkpointu `openvla-7b-finetuned-libero-10` (odchylka: checkpoint LIBERO-Spatial lokálně není), 40/10 epizod LIBERO-10: [docs/pilot-2026-11-12.md](../docs/pilot-2026-11-12.md). Naměřeno na validaci (relL2 bez biasu):

- Jednostupňová varianta ze zadání selhává (0.865–0.978); více stupňů bez permutace je podle Cauchy–Binetovy věty totéž.
- 16 stupňů s permutací `stride`/`random`: 0.134 (kosinus bez biasu 0.987, 3.2 % bajtů); 64 stupňů: 0.0625, s korekcí biasu 0.0507.
- Při stejném počtu parametrů je QuIC-WC o 11–28 % lepší než aktivačně vážená SVD (ASVD); při stejných bajtech s θ ve FP32 je lepší ASVD. Uložení θ v BF16 je téměř bezeztrátové.
- Compound řády nemají při stejném rozpočtu měřitelný vliv; rozhoduje počet míchaných stupňů.
- Pozitivní kontrola: učitel ze stejné rodiny se z identity obnoví jen při malých rotacích → výsledky jsou omezené optimalizací.

**Brány:** M2 (SVD baseline s kosinem > 0.98) splněna od SVD ranku 128 (0.983) a ASVD ranku 16 (0.983). M3 je formálně splněna (např. stride L16: kosinus bez biasu 0.987 s 3.2 % bajtů), ale stávající brána nerozlišuje metody; kosinus s biasem dosahuje 0.963 i pro samotný bias. **Navrhovaná úprava brány M3:** relL2/kosinus bez biasu proti ASVD při stejných bajtech, s korekcí biasu u všech metod.

**Upravené pořadí další práce (návrh, zatím neprovedeno):** (1) zopakovat pilot na 2–3 dalších vrstvách; (2) řešit optimalizační bariéru (restarty, postupné přidávání stupňů, inicializace); (3) θ v BF16 v exportu; (4) M4-lite: nahrazení vrstvy v modelu a vliv na výstup featurizeru a akční tokeny; teprve potom M5 rollouty. Rozhodnutí, zda pokračovat k M4/M5, závisí na tom, zda výhoda proti ASVD přetrvá na dalších vrstvách.

## 11. Aktualizace — 12. 11. 2026 (večer): rozhodnutí, brána M3, BF16 a pilot na čtyřech vrstvách

**Rozhodnutí uživatele (12. 11. 2026):** odchylka od jednoho přímého součtu je přijatelná; pokračuje se
vícestupňovou variantou s permutacemi, `L` se volí adaptivně pro každou vrstvu, baseline je ASVD; doplnit export
s θ v BF16; upravit bránu M3; před M4/M5 pilot na 2–3 dalších vrstvách a lepší optimalizace.

**Brána M3 (nové znění, nahrazuje bránu z oddílu M3 výše):** konfigurace vrstvy projde, když její validační relL2
výstupu bez biasu (nasaditelná podoba načtená z artefaktu, θ/σ v BF16) je nižší než u ASVD s alespoň stejným
počtem bajtů uložených tensorů (BF16 faktory, stejná politika biasu), při poměru bajtů ≤ 0.25 vůči husté BF16
vrstvě. Vrstva projde, když projde aspoň jedna konfigurace. Implementace: `gate_m3`/`layer_gate` v
`src/quic_wc/pilot/run.py`. Práh 0.25 zvolil agent. Brána neobsahuje absolutní požadavek na kvalitu; ten má určit
M4-lite (viz níže).

**Hotovo a ověřeno:** formát v2 s `storage_dtype` (BF16 θ/σ), konjugovaný tvar stupně (umožňuje růst stupňů),
výběr kroku podle trénovacího cíle, LR škálovaný hloubkou, rychlejší stupeň a checkpointing, kalibrace z celého
modelu přes `predict_action` (`collect_vla.py`), sweep `L`, souhrn přes vrstvy. Testy: 73. Report:
[docs/pilot-multilayer-2026-11-12.md](../docs/pilot-multilayer-2026-11-12.md); audit:
[audit/2026-11-12-bf16-gate-optim-multilayer.md](../audit/2026-11-12-bf16-gate-optim-multilayer.md).

**Výsledek brány M3 (recept v2, 3000 kroků, LR `0.03·min(1,(16/L)^0.5)`):**

| vrstva | M3 | nejlepší rezerva proti ASVD ≥B | relL2 QuIC-WC (poměr bajtů) |
|---|---|---:|---|
| DINOv2 blok 0 `mlp.fc1` (4096×1024) | prošla (fixed 5/5, corrected 5/5) | 28.2 % / 36.1 % | 0.0510 (0.061) |
| SigLIP blok 13 `mlp.fc1` (4304×1152) | prošla (fixed 4/5, corrected 5/5) | 11.2 % / 12.9 % | 0.2785 (0.056) |
| Llama L15 `self_attn.q_proj` (4096×4096) | prošla (7/7) | 9.4 % | 0.3195 (0.048) |
| Llama L15 `mlp.down_proj` (4096×11008) | prošla (7/7) | 6.1 % | 0.7425 (0.033) |

Brána prošla všude, ale absolutní chyba se mezi vrstvami liší řádově; na `down_proj` jsou obě metody při
užitečném rozpočtu nepoužitelné (ASVD při 69 % bajtů: 0.333). Bez absolutního kritéria brána o použitelnosti
nerozhoduje.

**Upravené pořadí další práce (návrh, zatím neprovedeno):** (1) **M4-lite** — nahradit vybrané vrstvy v modelu
artefakty QuIC-WC a měřit vliv na výstup featurizeru / LLM a shodu akčních tokenů (reference: 83.6 % shoda greedy
tokenů s datasetem); z toho odvodit cílovou chybu na vrstvu a doplnit bránu o absolutní kvalitu; (2) rozhodnout
o roli compound řádů (viz report: na q_proj je K=[1] při stejných bajtech rovnocenné a 4–7× rychlejší);
(3) delší trénink a dotrénování po zaokrouhlení do BF16 pro vybrané vrstvy; (4) M5 rollouty až po M4-lite.

## 12. Aktualizace — 16. 11. 2026: compound řády a úspora paměti za běhu

**Compound řády (rozhodnutí uživatele; dokončeno 16. 11. 13:50 — kritérium splněno, K=[1,2] zůstává hlavní
variantou; K=[1,2] šetří proti K=[1] se stejným výpočtem 19–39 % bajtů při silné kompresi, kolem 5 % bajtů rozdíl
mizí; K=[1,2,3] +1.4–4.9 % za 3.6–4.9× delší trénink):** před M4-lite rozhodovací experiment K=[1,2] vs K=[1] (stejné
bajty i stejný výpočet) a K=[1,2,3] na SigLIP b13 fc1 a Llama L15 q_proj; kritérium stanovené předem a výsledky
v [audit/2026-11-16-compound-orders.md](../audit/2026-11-16-compound-orders.md). M4-lite poběží s konfigurací,
kterou experiment potvrdí. Poznámka k bodu (2) oddílu 11: formulace „rovnocenné“ byla příliš silná (přepočet na
bajty: K=[1] potřebuje o 5–30 % víc bajtů).

**Úspora GPU paměti (rozhodnutí uživatele, plánováno po M4, neprovedeno):** MVP dál cílí na velikost checkpointu
(oddíl 1). Po M4 se experimentálně porovnají dvě větve, které mají snížit i ustálenou paměť vah:

1. **Rozbalení vrstvy těsně před použitím** — na GPU zůstávají jen kompaktní parametry (θ, σ, bias, permutace),
   hustá `Ŵ` vzniká pro jednu vrstvu (nebo blok vrstev) v pre-forward hooku a po použití se uvolní. Měřit: peak
   a ustálenou VRAM, latenci akce (OpenVLA volá LLM 7× na akci, rozbalení se tedy opakuje při každém kroku
   dekódování), čas rozbalení vrstvy, vliv velikosti bloku vrstev a případného cachování v rámci jedné akce.
2. **Výpočet ve faktorizované podobě** — `y = Q_out diag(σ) (Q_inᵀ x) + b` bez husté matice: stupně (blokové
   násobení + permutace) aplikované přímo na aktivace. Měřit: VRAM, latenci a FLOPs proti `F.linear`, číselnou
   shodu s rozbalenou vahou; nejprve implementace v PyTorch (`bmm` + `index_select`), při nadějných číslech
   vlastní kernel (Triton) se sloučenými stupni. Hrubý odhad pro 4096×4096 a L=64: FLOPs srovnatelné s hustým
   násobením, ale stovky malých sekvenčních operací.

Obě větve mění kritéria M5 (latence ≤ 1.05× husté baseline platí pro rozbalení předem); u nich se vykáže
kompromis paměť × latence a kritéria se stanoví před měřením. Možná je i kombinace po vrstvách.
