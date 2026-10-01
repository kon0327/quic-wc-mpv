# První GPU prostředí QuIC-WC

Prostředí je samostatná conda kopie ověřeného `/home/kony/miniconda3/envs/openvla`, založená na ovlab snapshotu `2026-07-17`. Prefix v tomto projektu:

```bash
conda activate /home/kony/dissertation/quic-wc-mpv/.conda/openvla-gpu
```

Původní conda prostředí a zdrojový checkout ovlabu se nemění. Kopie používá vlastní soubory (`conda --copy`), zdejší editable OpenVLA a LIBERO. Prostředí, cache a pracovní logy jsou ignorované v `.gitignore`.

Ověřeno 1. 10. 2026 na NVIDIA GeForce RTX 4090 (24 GB), driver 576.52: všech sedm kontrol prošlo a `pip check` je čistý. Finální výsledky jsou v [gpu-smoke.json](snapshots/2026-10-01/gpu-smoke.json), přesný conda/pip stav v `snapshots/2026-10-01/`. Numerické omezení determinantů popsané níže zůstává součástí výsledku.

## Příprava

Z kořene projektu:

```bash
bash deploy/scripts/prepare-openvla-env.sh
```

Výchozí zdroj klonu je `/home/kony/miniconda3/envs/openvla`. Lze zadat `QUIC_WC_CLONE_SOURCE` a cílový `QUIC_WC_ENV_PREFIX`. Skript kontroluje existenci conda metadat, připojí lokální zdroje, instaluje doplňky s omezeními verzí a vyžaduje úspěšné `pip check`.

Pokud jsou připravené všechny potřebné wheels v `.cache/wheelhouse`, instalaci lze provést offline:

```bash
QUIC_WC_OFFLINE=1 bash deploy/scripts/prepare-openvla-env.sh
```

`environment-replay.yml` uchovává úplný výchozí conda/pip předpis s opravami kompatibility. `upstream/2026-07-17/` obsahuje nedotčené kopie převzatých předpisů a původního runtime reportu; `upstream/provenance.json` eviduje zdroj a SHA-256. Bootstrap je testovaný pro klonování lokálního prostředí. Úplná instalace snapshotu od nuly zatím není ověřená; vyžaduje správné pořadí instalace PyTorchu a sestavení FlashAttention s CUDA toolkitem bez build isolation.

## Podstatné verze a opravy

| Součást | Verze |
|---|---|
| Python | 3.10.20 |
| PyTorch / torchvision / torchaudio | 2.2.0+cu121 / 0.17.0 / 2.2.0 |
| transformers / tokenizers / timm | 4.40.1 / 0.19.1 / 0.9.10 |
| FlashAttention / PEFT | 2.5.5 / 0.11.1 |
| NumPy / TensorFlow | 1.26.4 / 2.15.0 |
| tensorflow-metadata / protobuf | 1.17.2 / 4.21.12 |
| robosuite / MuJoCo / gym / bddl | 1.4.1 / 2.3.7 / 0.25.2 / 1.0.1 |

Převzaté `tensorflow-metadata==1.21.0` používalo protobuf API `runtime_version`, které v původním protobuf 4.25.9 chybí. Nová dvojice metadata/protobuf splňuje požadavky TensorFlow, tensorflow-datasets i wandb a je kontrolovaná skutečným importem OpenVLA, nejen `pip check`.

Lokální LIBERO má vnější namespace adresář bez `__init__.py`, který jeho `setup.py` při moderní editable instalaci neobjeví. Bootstrap proto přidává `quic_wc_libero_source.pth` do nové conda kopie. Jeho obsah odkazuje pouze na `external/libero` tohoto projektu.

Conda snapshot obsahuje i runtime CUDA 12.4, ale skutečný PyTorch wheel je sestavený pro CUDA 12.1 a používá odpovídající pip knihovny NVIDIA. Rozhodující ověření je `torch.version.cuda`, funkční CUDA operace a FlashAttention; samotné `pytorch-cuda` v YAML nestačí.

## Ověření

```bash
bash deploy/scripts/test-openvla-env.sh
```

Skript nastaví zapisovatelné cache v projektu, vytvoří konfiguraci LIBERO bez interaktivních dotazů, vypne síťové načítání HF modelů a spustí:

1. kontrolu verzí a importních cest lokálních zdrojů;
2. FP32 a BF16 matmul na CUDA, SVD a rekonstrukci váhy;
3. Cayleyho transformaci, compound řádu 2, ortogonalitu a backward;
4. FlashAttention BF16 forward proti FP32 SDPA a backward;
5. TensorFlow resize na CPU, s vypnutou viditelností GPU pro TensorFlow;
6. skutečné HF OpenVLA forward a `predict_action` s malým náhodně inicializovaným modelem, FlashAttention a hookem ve vizuální MLP vrstvě;
7. reset, načtení počátečního stavu, tři kroky a offscreen render první úlohy LIBERO-Spatial.

Výstupem je `artifacts/environment/gpu-smoke.json`; nenulový exit code označuje chybu některého testu. Pro diagnostiku lze vybrat např. `--only libero_render` a zadat odlišný `--report`. Výsledek vybraných testů nepředstavuje ověření celé sady.

V této Codex relaci sandbox skrývá `/dev/dxg`, zatímco mimo sandbox je dostupná RTX 4090. GPU testy proto vyžadují spuštění mimo sandbox; v běžném hostitelském terminálu není potřeba měnit konfiguraci CUDA.

Jde o test prostředí bez velkého checkpointu a bez měření robotické úspěšnosti. Zdejší syntetický model neověřuje kvalitu politiky ani paměťový rozpočet skutečného OpenVLA-7B.

## Numerické omezení pro fázi M1

Smoke test porovnává gradient `torch.linalg.det` compound řádu 2 při přesné identitě s explicitním vzorcem `ad-bc`. PyTorch 2.2/CUDA zde může vrátit nulový gradient přes singulární minory, přestože explicitní vzorec má nenulovou derivaci. Report eviduje obě normy, rozdíl a upozornění; direct sum s řádem 1 stále poskytuje nenulový gradient.

Matematické jádro musí tuto hranici zahrnout do testů. Pro řád 2 je vhodný explicitní diferencovatelný vzorec; u obecného řádu je nutné ověřit backward singulárních minorů. Malá náhodná inicializace prošla smoke testem, ale nenahrazuje obecnou kontrolu gradientů.
