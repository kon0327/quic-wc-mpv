# Příprava a ověření prvního GPU prostředí

Datum práce a doplnění záznamu: 1. 10. 2026. **Retrospektivní záznam** sestavený z průběhu práce, logů a finálního reportu.

## Zadání a změny

Připravit a otestovat GPU prostředí podle funkční konfigurace ovlabu z `/home/kony/dissertation/ovlab/deploy/environments/openvla/snapshots/2026-07-17`.

- Vytvořen samostatný conda prefix `.conda/openvla-gpu` offline klonováním `/home/kony/miniconda3/envs/openvla` s `--copy`; původní prostředí se neměnilo.
- OpenVLA připojeno jako editable z `external/openvla`; LIBERO z `external/libero` doplněno namespace cestou v nové conda kopii.
- Vytvořeny [přípravný skript](../deploy/scripts/prepare-openvla-env.sh), [testovací wrapper](../deploy/scripts/test-openvla-env.sh) a [GPU kontroly](../deploy/scripts/check_openvla_env.py).
- Vytvořeny konfigurace, constraints, LIBERO requirements, kopie upstream předpisů s SHA-256 a přesný [snapshot nové instalace](../deploy/environments/openvla/snapshots/2026-10-01/).
- Přidány `.gitignore`, [návod prostředí](../deploy/environments/openvla/README.md) a aktualizace stavu M0 v implementačním plánu.
- Wheels a cache zůstávají v `.cache/`; pracovní logy v `artifacts/environment/`. Velké checkpointy nebyly stahovány.

## Chyby, pokusy a náprava

| Problém | Postup a konečný stav |
|---|---|
| Sandbox skrýval `/dev/dxg`; CUDA nebyla viditelná | Test mimo sandbox prokázal RTX 4090 a úspěšný GPU matmul. GPU a EGL kontroly byly provedeny mimo sandbox. |
| `tensorflow-metadata==1.21.0` ze snapshotu importovalo chybějící protobuf `runtime_version` | Původní `protobuf==4.25.9` nestačil pro jeho generovaný kód; samotné původní `pip check` chybu neodhalilo. |
| První oprava metadata 1.15.0 vedla k protobuf 3.20.3 | Instalace odkryla konflikt s `wandb==0.28.0`, které požaduje protobuf >4.21.0. Tato konfigurace nebyla přijata jako finální. |
| Hledání kompatibilní metadata verze | Wheel 1.16.1 byl pouze stažen a prohlédnut; pro Python 3.10 také vyžadoval protobuf <4.21. Finální metadata 1.17.2 a protobuf 4.21.12 splnily souběžné požadavky, import i `pip check`. |
| Moderní editable instalace LIBERO nevytvořila importovatelný namespace | Legacy `find_packages()` neobjevil vnější adresář bez `__init__.py`. Bootstrap přidal `quic_wc_libero_source.pth` odkazující na zdejší `external/libero`. |
| BDDL při importu vyžadovalo nedeklarované `future` | Doplněno `future==0.18.2` z původních LIBERO requirements; import a simulace následně prošly. |
| Identity init compound řádu 2: nulový gradient přes `torch.linalg.det` | Porovnání s explicitním `ad-bc` potvrdilo rozdíl gradientů. Nejde o opravenou vlastnost PyTorchu: je zaznamenána jako numerické omezení pro M1. Direct sum s řádem 1 a náhodný init prošly. |
| Shell po úspěšném cíleném renderu hlásil neuzavřené uvozovky | Testovací wrapper byl upraven během jeho běhu. Ověřena syntaxe hotového souboru a zopakována celá sada bez dalších úprav běžících skriptů; finální exit code 0. |
| `pip freeze` hlásilo nefunkční Git metadata externích checkoutů | Přiložené `.git` odkazy míří na chybějící submodule metadata. Uloženy také conda explicit export, seznam instalovaných balíčků/verzí a zdrojové cesty; nedoplňovány smyšlené commit hashe. |

## Finální ověření

Příkaz z kořene projektu: `bash deploy/scripts/test-openvla-env.sh`, spuštěný mimo sandbox. **Exit code 0, všech sedm kontrol prošlo, `pip check` čistý.** Podklad: [gpu-smoke.json](../deploy/environments/openvla/snapshots/2026-10-01/gpu-smoke.json).

| Kontrola | Naměřený výsledek |
|---|---|
| Verze a lokální zdroje | Python 3.10.20, PyTorch 2.2.0+cu121, HF stack ze snapshotu; importní cesty vedou do tohoto projektu. |
| CUDA matematika | RTX 4090, FP32/BF16 matmul; relativní chyba SVD rekonstrukce `8.56e-6`; chyba ortogonality `3.58e-7`; finite gradient při náhodném init. |
| Gradient compound při identitě | Norma `det` gradientu 0, explicitního C2 gradientu `46.47`, direct sum gradientu `24.70`. Omezení je součástí reportu, nikoliv vyřešená chyba. |
| FlashAttention | BF16 forward/backward, finite gradienty; max. odchylka proti FP32 SDPA přibližně `0.00505`. |
| TensorFlow preprocessing | CPU resize 16×16×3; TensorFlow GPU visibility vypnutá. |
| Syntetické OpenVLA | Skutečný HF forward a `predict_action`, 7 výstupních akcí, finite hodnoty, `LlamaFlashAttention2`, hook ve vizuální MLP. |
| LIBERO | První LIBERO-Spatial úloha, reset a načtení stavu, tři kroky, neprázdný EGL obraz 128×128×3. |

SHA-256 převzatých předpisů byly porovnány s originály. Dodatečná kontrola původního prostředí potvrdila zachované metadata 1.21.0, protobuf 4.25.9 a původní editable cestu OpenVLA do ovlabu.

## Omezení a další kroky

Prostředí je připravené pro další vývoj, ale test používá malý náhodně inicializovaný OpenVLA model. Načtení skutečného OpenVLA-7B, jeho VRAM rozpočet, policy rollout a kvalita komprese zatím nejsou ověřené. Úplná instalace od nuly podle replay YAML nebyla testovaná; ověřená cesta je klonování lokálního prostředí.

Pro M1 je nutné ověřit derivace singulárních minorů a pro compound řádu 2 použít kontrolovaný explicitní determinant. GPU testy v této relaci vyžadují přístup mimo sandbox.
