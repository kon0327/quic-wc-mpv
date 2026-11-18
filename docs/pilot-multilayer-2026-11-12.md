# Pilot QuIC-WC na čtyřech vrstvách OpenVLA, export BF16 a brána M3 (12. 11. 2026)

Navazuje na [první pilot](pilot-2026-11-12.md) (jedna vrstva DINOv2) po rozhodnutí uživatele z 12. 11. 2026:
vícestupňová varianta s permutacemi, `L` volené adaptivně pro každou vrstvu, srovnání s ASVD, export s θ v BF16,
brána M3 jako srovnání s ASVD při stejné velikosti s metrikou bez biasu, pilot na dalších vrstvách a lepší
optimalizace. Průběh, chyby a ověření: [audit](../audit/2026-11-12-bf16-gate-optim-multilayer.md); jádro:
[m1-math-core.md](m1-math-core.md), oddíl 10. Integrace do modelu (M4) ani rollouty (M5) zde nejsou.

## 1. Shrnutí

Naměřené výsledky (validační epizody LIBERO-10, přesná relativní L2 chyba výstupu bez biasu z BF16 artefaktu,
nižší = lepší; „ASVD ≥B“ = aktivačně vážená SVD s alespoň stejným počtem bajtů):

1. **Export s θ a σ v BF16 je implementovaný a ověřený.** Artefakt má poloviční velikost a faktory zůstávají
   ortonormální. Na artefaktech prvního pilotu se potvrdilo, že výhoda proti ASVD platí i při stejných bajtech
   (stride L16: 0.1345 vs 0.1688).
2. **Nová brána M3 prošla na všech čtyřech vrstvách** (recept v2, 3000 kroků):

   | vrstva | rozměr | nejlepší rezerva proti ASVD ≥B | relL2 při ~5–6 % bajtů (ASVD ≥B) |
   |---|---|---:|---|
   | DINOv2 blok 0 `mlp.fc1` | 4096×1024 | 28.2 % (36.1 % s korekcí biasu) | 0.0510 (0.0699) |
   | SigLIP blok 13 `mlp.fc1` | 4304×1152 | 11.2 % (12.9 %) | 0.2785 (0.3136) |
   | Llama vrstva 15 `self_attn.q_proj` | 4096×4096 | 9.4 % | 0.3195 (0.3527) |
   | Llama vrstva 15 `mlp.down_proj` | 4096×11008 | 6.1 % | 0.7425 při 3.3 % (0.7909) |

3. **Při stejné kvalitě potřebuje QuIC-WC o 33–47 % méně bajtů než ASVD** (log-log interpolace mezi změřenými
   body, interpretace): DINOv2 33–42 % (cílová relL2 0.2–0.05), SigLIP 35–38 % (0.4–0.3), q_proj 38 % (0.4),
   down_proj 47 % (0.8).
4. **Absolutní kvalita se mezi vrstvami liší řádově.** Vstup DINOv2 b0 je silně anizotropní (64 směrů nese 99 %
   stopy), u SigLIP 73 %, u q_proj 70 %, u down_proj jen 27 %. Na těžkých vrstvách nedosáhne relL2 0.2 v
   rozumném rozpočtu ani ASVD (SigLIP i q_proj potřebují 19–27 % bajtů, down_proj ani při 69 % nedosáhne 0.33).
   Brána M3 tak prochází i tam, kde jsou obě metody pro praxi nejspíš nepoužitelné. Chybí absolutní kritérium
   kvality, které má určit M4-lite.
5. **Optimalizace:** hlavním zlepšením je LR škálovaný hloubkou (`0.03·(16/L)^0.5`): DINOv2 L64 0.066 → 0.051,
   q_proj L128 0.363 → 0.320 (z prohry nad ASVD na rezervu 9 %). Postupný růst stupňů a náhodné restarty
   nepomáhají; adaptivní `L` je proto sweep fitů od nuly.
6. **Compound řády:** na vrstvách Llamy je K=[1] (bloky SO(8) bez compound matic) při stejných bajtech rovnocenné
   s K=[1,2] (q_proj 0.3256 vs 0.3195, down_proj 0.7463 vs 0.7425) a trénuje 4.6–7× rychleji; při malém rozpočtu
   je K=[1,2] mírně lepší. Negativní kontrola (náhodné báze se skutečným spektrem) poráží ASVD na všech
   vrstvách. Interpretace: výhoda pochází z parametrizace (plný rank se zachovaným spektrem σ a vícestupňové
   ortogonální faktory), ne specificky z compound matic ani ze struktury reálných vah.


## 2. Co se změnilo v metodě a nástrojích

- **Export s úložným dtype (formát v2).** θ a σ se ukládají v BF16, faktory se při načtení staví v FP32, takže
  zůstávají ortonormální. Všechny metriky QuIC-WC v tomto reportu jsou spočtené z **načteného artefaktu**.
- **Brána M3 (nová definice, schválená uživatelem).** Varianta projde, když její validační relL2 bez biasu je nižší
  než u ASVD s **alespoň stejným** počtem bajtů uložených tensorů (ASVD faktory BF16, stejná politika biasu), při
  poměru bajtů ≤ 0.25. Vrstva projde, když projde aspoň jedna varianta. Práh 0.25 je moje volba.
- **Konjugovaný tvar stupně** `R_lᵀ B_l R_l` (kumulativní permutace): stejná rodina jako dosavadní tvar až na pevnou
  permutaci řádků (ověřeno testem i měřením: L16 0.1342 vs 0.1345), ale `Q = I` při `θ = 0` pro každé `L`, což
  umožňuje přidávat stupně bez změny funkce.
- **Výběr kroku podle trénovacího cíle** (`select_on = "train"`): validace nevstupuje do žádné volby.
- **Rychlejší stupeň** (jeden `bmm` na stupeň, konjugovaný tvar jako chain + jedna permutace): 1.5–1.7× rychlejší.
- **Kalibrace z celého modelu** (`collect_vla.py`): `predict_action` jako v LIBERO evaluaci, hooky na vizuálních
  i jazykových vrstvách včetně dekódovacích kroků akčních tokenů.

## 3. Ověření exportu BF16 na hotových artefaktech prvního pilotu

Bez nového fitování: FP32 artefakty z prvního gridu převedené do BF16, načtené a vyhodnocené
(`quic_wc.pilot.requantize`, výstup `artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1/requant-bf16/`). FP32 metriky
z artefaktů reprodukují původní grid přesně.

| varianta (DINOv2 b0 fc1) | poměr bajtů | QuIC-WC, BF16 artefakt | ASVD s ≥ bajty | rezerva |
|---|---:|---:|---|---:|
| stride L4 | 0.0050 | 0.2448 | r4: 0.2501 | 2.1 % |
| stride L16 | 0.0163 | 0.1345 | r13: 0.1688 | 20.3 % |
| stride L64 | 0.0614 | 0.0640 | r50: 0.0699 | 8.4 % |
| stride L16, korigovaný bias | 0.0163 | 0.1295 | r13: 0.1664 | 22.1 % |
| stride L64, korigovaný bias | 0.0614 | 0.0513 | r50: 0.0690 | 25.6 % |

Soubor `layer.safetensors` varianty L16 má 136 872 B (BF16) místo 265 256 B (FP32). Tvrzení z prvního pilotu, že
v BF16 platí výhoda i při stejných bajtech, se potvrdilo měřením.

## 4. Studie optimalizace (DINOv2 b0 fc1)

| zásah | výsledek (validační relL2 bez biasu, BF16) |
|---|---|
| delší trénink (10 000 místo 3000 kroků) | L16 0.1342 → 0.1309; L64 0.0640 → 0.0556 |
| postupný růst 4→8→16→32→64 (nové stupně = identita) | **horší** než fit od nuly: L16 0.1426, L64 0.0797 |
| 8 restartů (identita + náhodné θ), screening 300 kroků | 0.1337 (bez zlepšení; screening vždy vybral identitu) |
| **LR škálovaný hloubkou** `0.03·(16/L)^0.5` | **L64 0.0511** (LR 0.03: 0.0660; LR 0.03 a 10 000 kroků: 0.0556) |
| pozitivní kontrola θ·0.3 (učitel L4) | 0.436; 10 000 kroků 0.428; restarty 0.436; růst 0.455; student L8 0.360 |

Na q_proj Llamy dává škálovaný LR při L128 0.3194 místo 0.3631 (ASVD ≥B: 0.3527), při L64 0.3705 místo 0.3809.

Závěry:

- Hlavní příčinou slabších výsledků hlubokých řetězců byl **příliš velký pevný LR**: změna součinu `L` stupňů roste
  zhruba s `L·lr`, takže se hluboký řetězec začal učit až s poklesem LR v cosine plánu (q_proj L128: trénovací relMSE
  1.02 po 500 krocích, L16 už 0.22). Recept v2 používá `lr = 0.03 · min(1, (16/L)^0.5)`.
- Postupný růst a náhodné restarty nepomáhají. **Adaptivní `L` je proto sweep fitů od nuly** a výběr podle křivky
  chyba/bajty, ne růst.
- Bariéra pozitivní kontroly při větších rotacích je skutečná nekonvexita; zmírní ji jen nadbytečná kapacita.
- Při malé chybě začíná být znát zaokrouhlení θ do BF16 (L64 s 10 000 kroky: 0.0537 → 0.0556).

## 5. Kalibrace z celého modelu

`collect_vla.py`, stejné epizody a snímky jako první pilot (LIBERO-10, 40 trénovacích a 10 validačních epizod,
každý 10. snímek, 1386 snímků, 532 s). Hook se shoduje s výstupem modulu přesně u všech vrstev. Llama vidí 7 volání
na snímek (prefill + 6 dekódovacích kroků), ~292 tokenů. DINOv2 b0 fc1 z celého modelu se od kalibrace prvního
pilotu liší relativně o 4e-6 (nezávislé potvrzení). Moje předzpracování větve DINOv2 je bitově shodné s oficiálním
`PrismaticImageProcessor`. Greedy akční tokeny původního modelu se shodují s tokeny z datasetu v 83.6 % validačních
snímků (reference pro M4-lite).

## 6. Výsledky receptu v2 po vrstvách

Recept v2: konjugované stupně se `stride` permutací, `n = 8`, `K = [1,2]`, plný rank u vizuálních vrstev a rank
1024 u Llamy (rank 4096 dal u q_proj L32 0.4164 vs 0.4153 — rank 1024 neomezuje), BF16 úložiště, 3000 kroků,
LR `0.03·min(1,(16/L)^0.5)`, výběr kroku podle trénovacího cíle. Konfigurace `configs/multilayer_vision_v2.json`
a `configs/multilayer_llm_v2.json`, výstupy `artifacts/pilot/2026-11-12-libero10-multilayer/v2_*` a souhrn
`summary_v2.md`.

| vrstva | L | poměr bajtů | QuIC-WC relL2 | cos_nb | ASVD ≥B (rank: relL2) | rezerva |
|---|---:|---:|---:|---:|---|---:|
| DINOv2 b0 fc1 | 4 | 0.0050 | 0.2449 | 0.957 | 4: 0.2501 | 2.1 % |
| | 8 | 0.0087 | 0.1853 | 0.976 | 7: 0.2110 | 12.2 % |
| | 16 | 0.0163 | 0.1342 | 0.987 | 13: 0.1688 | 20.5 % |
| | 32 | 0.0313 | 0.0879 | 0.995 | 25: 0.1225 | 28.2 % |
| | 64 | 0.0614 | 0.0510 | 0.998 | 50: 0.0699 | 27.0 % |
| DINOv2 b0 fc1, korigovaný bias | 64 | 0.0614 | 0.0441 | 0.999 | 50: 0.0690 | 36.1 % |
| SigLIP b13 fc1 | 4 | 0.0045 | 0.4867 | 0.874 | 4: 0.4840 | −0.6 % |
| | 16 | 0.0147 | 0.3891 | 0.922 | 13: 0.4178 | 6.9 % |
| | 32 | 0.0284 | 0.3355 | 0.943 | 26: 0.3680 | 8.8 % |
| | 64 | 0.0556 | 0.2785 | 0.961 | 50: 0.3136 | 11.2 % |
| SigLIP b13 fc1, korigovaný bias | 64 | 0.0556 | 0.2725 | 0.963 | 50: 0.3129 | 12.9 % |
| Llama L15 q_proj | 16 | 0.0061 | 0.4580 | 0.878 | 13: 0.4791 | 4.4 % |
| | 32 | 0.0121 | 0.4168 | 0.899 | 25: 0.4437 | 6.1 % |
| | 64 | 0.0242 | 0.3705 | 0.922 | 50: 0.4010 | 7.6 % |
| | 128 | 0.0483 | 0.3195 | 0.942 | 99: 0.3527 | 9.4 % |
| Llama L15 down_proj | 16 | 0.0042 | 0.8611 | 0.498 | 13: 0.8786 | 2.0 % |
| | 32 | 0.0083 | 0.8332 | 0.549 | 25: 0.8586 | 3.0 % |
| | 64 | 0.0166 | 0.7950 | 0.609 | 50: 0.8300 | 4.2 % |
| | 128 | 0.0332 | 0.7425 | 0.677 | 100: 0.7909 | 6.1 % |

Čas fitu (RTX 4090, 3000 kroků): DINOv2/SigLIP L64 ~165 s, q_proj L128 587 s, down_proj L128 1248 s; celé sweepy
774 / 798 / 1450 / 3860 s.

**Adaptivní `L`.** Křivky jsou monotónní a rezerva proti ASVD roste s `L` (kromě DINOv2 L64, kde se ustálí).
`L` pro vrstvu se z křivky vybere podle cílové chyby nebo rozpočtu; cílovou chybu zatím nemáme (viz bod 4 shrnutí).

## 7. Compound řády proti K=[1] (Llama)

K=[1] má 4.5× víc parametrů na stupeň, proto 7/14/28 stupňů odpovídá bajtům K=[1,2] s L32/64/128.

| vrstva | bajty ≈ | K=[1,2] (L, relL2, čas) | K=[1] (L, relL2, čas) | ASVD ≥B |
|---|---:|---|---|---:|
| q_proj | 0.012 | L32: 0.4168, 121 s | L7: 0.4324, 49 s | 0.4437 |
| q_proj | 0.024 | L64: 0.3705, 237 s | L14: 0.3783, 60 s | 0.4010 |
| q_proj | 0.048 | L128: 0.3195, 587 s | L28: 0.3256, 81 s | 0.3527 |
| down_proj | 0.008 | L32: 0.8332, 360 s | L7: 0.8444, 200 s | 0.8586 |
| down_proj | 0.016 | L64: 0.7950, 594 s | L14: 0.8011, 223 s | 0.8300 |
| down_proj | 0.033 | L128: 0.7425, 1248 s | L28: 0.7463, 271 s | 0.7909 |

K=[1,2] je o 0.5–3.6 % lepší, za cenu 1.8–7.2× delšího tréninku. V prvním pilotu na DINOv2 (LR 0.03) byly řády
bez rozdílu. Obě varianty porážejí ASVD.

**Doplnění 2026-11-16 (přepočet na bajty, interpretace):** na plochých křivkách Llamy odpovídá malý rozdíl chyby
velkému rozdílu velikosti. Log-log interpolací po křivce K=[1,2] potřebuje K=[1] na stejnou chybu o 30 / 12 / 9 %
(q_proj) a 27 / 8 / 5 % (down_proj) víc bajtů, nejvíc při malém rozpočtu. Formulace „rovnocenné“ v bodě 6 shrnutí
je proto příliš silná. Srovnání při stejném výpočetním rozpočtu a s více semeny zatím chybí.

## 8. Kontroly

| vrstva | pozitivní kontrola (θ·0.1) | negativní kontrola L16 (ASVD ≥B) |
|---|---|---|
| DINOv2 b0 fc1 | L4: 0.0070 (obnovena) | 0.2205 (0.2928) |
| SigLIP b13 fc1 | L4: 0.0039 (obnovena) | 0.6279 (0.6658) |
| q_proj | L16: 0.5964 (**neobnovena**) | 0.6101 (0.6336) |
| down_proj | L16: 0.8765 (**neobnovena**) | 0.8802 (0.8972) |

U Llamy (učitel i student L16) se učitel ze stejné rodiny neobnoví ani při malých rotacích a výsledek je blízko
negativní kontroly. Optimalizace na těchto vrstvách strukturu učitele nevyužije. Výsledky jsou proto dolní mezí
toho, co rodina umí. Negativní kontrola poráží ASVD na všech vrstvách (viz interpretace v bodě 6 shrnutí).

## 9. Omezení

- Jedna vrstva z každého typu (DINOv2, SigLIP, Llama attention, Llama MLP); jedna sada dat (LIBERO-10), jeden
  checkpoint (`openvla-7b-finetuned-libero-10`), jedno semeno. Rozdíly kolem 1 % nejsou statisticky ověřené.
- 3000 kroků; delší trénink pomáhá (DINOv2 L64 o 13 % při LR 0.03), takže čísla jsou konzervativní.
- Metriky jsou výstupní chyba jedné vrstvy. Vliv na výstup modelu a akce (M4) ani na úspěšnost v LIBERO (M5)
  zatím není změřen.
- Brána M3 nemá absolutní práh kvality; práh poměru bajtů 0.25 je volba agenta.
- Interpolované úspory bajtů jsou interpretace mezi změřenými body; na krajích rozsahu se nepočítají.
- Časy LR testů (bod 4) byly měřeny při sdílené GPU a nejsou srovnatelné.

## 10. Doporučení a otevřená rozhodnutí

1. **M4-lite jako další krok:** nahradit vybrané vrstvy (např. DINOv2 b0 fc1 a SigLIP b13 fc1) artefakty QuIC-WC a
   ASVD se stejnými bajty a měřit výstup featurizeru/LLM a shodu akčních tokenů (reference: 83.6 % shoda greedy
   tokenů s datasetem). Z toho odvodit cílovou chybu na vrstvu a doplnit bránu M3 o absolutní kritérium.
2. **Výběr vrstev:** vrstvy s anizotropním vstupem (rané vizuální bloky) jsou dobří kandidáti; Llama `down_proj`
   je pro tuto rodinu (i pro ASVD) v užitečné kvalitě prakticky nestlačitelná.
3. **Role compound řádů (rozhodnutí uživatele):** K=[1,2] je identita metody QuIC a je o 0.5–3.6 % lepší, K=[1]
   je 1.8–7.2× rychlejší při téměř stejné kvalitě.
4. Pro vybrané vrstvy delší trénink a dotrénování po zaokrouhlení do BF16.

## 11. Reprodukce

```bash
P=artifacts/pilot/2026-11-12-libero10-dinov2-b0-fc1; M=artifacts/pilot/2026-11-12-libero10-multilayer
SNAP=$(ls -d ~/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/*)
# ověření BF16 na artefaktech prvního pilotu
PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.requantize --calibration $P/calibration \
  --results $P/grid-v1 $P/grid-v1-controls-fixed --out $P/requant-bf16
# studie optimalizace a LR
PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.run --calibration $P/calibration \
  --grid configs/optim_libero10_dinov2_b0_fc1.json --out $P/optim-v1 --skip-baselines
PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.run --calibration $P/calibration \
  --grid configs/lrdepth_dinov2_b0_fc1.json --out $P/lrdepth-v1 --skip-baselines
# kalibrace z celého modelu (GPU musí být viditelná: flash_attn bez GPU padá při importu)
PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.collect_vla --checkpoint "$SNAP" \
  --dataset /home/kony/dissertation/ovlab-data/datasets/libero/libero_10/1.0.0/prepared --out $M/calibration \
  --layers vision_backbone.featurizer.blocks.0.mlp.fc1 vision_backbone.fused_featurizer.blocks.13.mlp.fc1 \
  language_model.model.layers.15.self_attn.q_proj language_model.model.layers.15.mlp.down_proj
# recept v2 (analogicky pro siglip_blocks_13_mlp_fc1, llm_layers_15_self_attn_q_proj, llm_layers_15_mlp_down_proj)
PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.run --calibration $M/calibration/dinov2_blocks_0_mlp_fc1 \
  --grid configs/multilayer_vision_v2.json --out $M/v2_dinov2_b0_fc1
PYTHONPATH=src .conda/openvla-gpu/bin/python -m quic_wc.pilot.summarize --results $M/v2_dinov2_b0_fc1 \
  $M/v2_siglip_b13_fc1 $M/v2_llm_l15_q_proj $M/v2_llm_l15_down_proj --out $M/summary_v2.md \
  --targets 0.8 0.6 0.4 0.3 0.2 0.1 0.05
```
