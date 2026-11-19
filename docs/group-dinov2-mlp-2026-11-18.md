# Skupina vrstev: všech 48 MLP vrstev DINOv2 v OpenVLA (2026-11-18)

Zdroj: `artifacts/pilot/2026-11-18-dinov2-mlp/` (`run.sh`, fity `fits/<vrstva>/results.json`, vyhodnocení
`m4/results.{json,md}`, tokeny a KL po snímcích `m4/tokens/`, analýza `analyze.py` → `analysis.json`). Audit
[audit/2026-11-18-group-dinov2-mlp.md](../audit/2026-11-18-group-dinov2-mlp.md). Metodika vyhodnocení je stejná jako
v [M4-lite](m4-lite-2026-11-18.md): 268 validačních snímků, greedy `predict_action` a teacher forcing.

## 1. Shrnutí

1. **Když se při 0.8–3.1 % bajtů komprimuje všech 48 vrstev, model přestane fungovat. Platí to pro QuIC-WC i ASVD.**
   - KL na akčních pozicích je 9.9–15.5 (práh jedné vrstvy byl 0.01).
   - Shoda greedy tokenů s referencí je 11–28 %.
   - Shoda s datasetem je 9–19 %, reference má 83.6 %.
   - V tomto režimu nemá srovnání metod smysl: při L8 vychází lépe ASVD, při L32 QuIC-WC, ale obě jsou rozbité.
2. **Aditivita KL selhává.**
   - Při L16 je součet 48 jednovrstvých KL 0.88, KL skupiny je 12.3 (14×). V podskupinách je poměr 20× u fc1
     a 32× u fc2.
   - Korelace po snímcích mezi součtem a KL skupiny je ≈ 0 (Pearson −0.05).
   - Interpretace: odhad „KL skupiny ≈ součet KL vrstev“ platí jen pro malé perturbace. Jediný dřívější bod
     (all4, poměr 1.3×) do tohoto režimu nespadá. Prahy pro skupiny proto nelze odvodit z jednovrstvých měření, musí
     se měřit přímo.
3. **Chyba jedné vrstvy ve skupině roste, protože vstupy se posunou mimo kalibrační rozdělení.**
   - In-situ relL2 fc1 vrstev je ve skupině v průměru 0.745 proti 0.544 při samostatném nasazení.
   - Jednotlivé vrstvy mají až 1.9× víc u QuIC-WC a až 2.9× víc u ASVD.
4. **Po vrstvách QuIC-WC poráží ASVD na všech 48 vrstvách a s velkou rezervou.**
   - Při stejné relL2 jako ASVD potřebuje QuIC L16 o 36–47 % méně bajtů na fc1 (medián 37 %) a o 63–72 % méně na fc2
     (medián 69 %), tedy úspora = 1 − bajty QuIC / bajty ASVD. ASVD tak potřebuje 1.6–1.9× (fc1) a 2.7–3.6× (fc2)
     víc bajtů. Jde o log-log interpolaci po ASVD frontě, bez extrapolace.
5. **Absolutní chyba hlubších vrstev je ale vysoká.**
   - Blok 0 fc1 z pilotu byl netypicky snadný: podíl stopy v top-64 0.99, relL2 při L16 0.13.
   - Ostatní fc1 mají relL2 0.38–0.66, fc2 0.17–0.58, top-64 0.5–0.9.
   - Pracovní práh jedné vrstvy (KL ≤ 0.01) splní při L16 jen 13 ze 46 použitých vrstev (5 fc1, 8 fc2). Maximum je
     0.082 (blok 17 fc1).
6. **Na úrovni modelu je úspora malá i při ideální kompresi.**
   - Váhy MLP DINOv2 mají 201.3 M parametrů, tedy 2.67 % ze 7.54 G.
   - L16 by ušetřila 198 M parametrů (≈ 396 MB v BF16 z ≈ 15.1 GB), tedy ≈ 2.6 % modelu.
   - Celá DINOv2 má 303 M (4.0 %), SigLIP 428 M (5.7 %), LLM 6.74 G (89.4 %).
7. **Kontroly:**
   - Reference na konci běhu má KL 0.
   - Všechny QuIC varianty mají `decompress_count = 1`.
   - Fit bloku 0 fc1 reprodukuje v2 (0.1799 / 0.1295 / 0.0826) a jeho jednovrstvá KL 0.0049 odpovídá M4-lite
     (0.005).
   - Blok 23 má KL přesně 0, protože OpenVLA bere výstup předposledního bloku DINOv2
     (`get_intermediate_layers(n={len(blocks)-2})`). Jeho komprese tedy chování nemění; jeho bajty lze ušetřit
     odstraněním bloku, ne kompresí.

## 2. Výsledky skupin

| varianta | vrstev | poměr bajtů | ⌀ relL2 pilot | ⌀ in-situ relL2 | tf KL | tf top-1 | free shoda | free = dataset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QuIC L8 | 48 | 0.0084 | 0.549 | 0.559 | 15.46 | 0.127 | 0.111 | 0.092 |
| ASVD L8 | 48 | 0.0091 | 0.601 | 0.566 | 12.99 | 0.246 | 0.239 | 0.193 |
| QuIC L16 | 48 | 0.0159 | 0.480 | 0.500 | 12.34 | 0.267 | 0.230 | 0.169 |
| ASVD L16 | 48 | 0.0165 | 0.553 | 0.523 | 12.79 | 0.238 | 0.223 | 0.167 |
| QuIC L32 | 48 | 0.0310 | 0.402 | 0.412 | 9.94 | 0.336 | 0.278 | 0.175 |
| ASVD L32 | 48 | 0.0311 | 0.491 | 0.481 | 12.49 | 0.253 | 0.227 | 0.160 |
| QuIC L16 jen fc1 | 24 | 0.0163 | 0.544 | 0.745 | 12.33 | 0.236 | 0.222 | 0.159 |
| ASVD L16 jen fc1 | 24 | 0.0168 | 0.603 | 0.772 | 12.36 | 0.233 | 0.213 | 0.156 |
| QuIC L16 jen fc2 | 24 | 0.0155 | 0.415 | 0.446 | 8.79 | 0.368 | 0.273 | 0.166 |
| ASVD L16 jen fc2 | 24 | 0.0161 | 0.504 | 0.558 | 11.93 | 0.284 | 0.239 | 0.165 |
| reference | — | 1 | — | — | 0 | 1 | 1 | 0.836 |

## 3. Výsledky po vrstvách (QuIC L16, korigovaný bias)

| | fc1 (23 použitých) | fc2 (23 použitých) |
|---|---|---|
| relL2 L16 min–max | 0.130–0.657 | 0.174–0.576 |
| úspora bajtů QuIC proti ASVD při stejné relL2 (1 − QuIC/ASVD) | 36–47 % (medián 37 %) | 63–72 % (medián 69 %) |
| jednovrstvá KL ≤ 0.01 | 5 | 8 |
| součet jednovrstvých KL | 0.605 | 0.277 |
| KL podskupiny | 12.33 | 8.79 |

Citlivost roste s hloubkou: všechny vrstvy bloků 0–4 mají KL ≤ 0.0082, fc1 bloků 13–19 kromě bloku 16 (0.024) mají
0.033–0.082. Úplná tabulka po vrstvách je
v `analysis.json` (`layers`) a v `m4/results.md`.

## 4. Interpretace a důsledky

- **Hlavní výsledek:** relativní výhoda QuIC-WC po vrstvách se potvrdila a na fc2 je dokonce větší než v pilotu.
  Absolutní kvalita při rozpočtech 1–3 % na vrstvu ale pro celou skupinu nestačí ani zdaleka. Problém je společný
  oběma metodám: chyby přes 46 vrstev se skládají silně nadaditivně.
- **Otázka 3 plánu** („jaký rozpočet skupiny splní práh“) zůstává bez odpovědi, protože žádná testovaná varianta
  není blízko prahu. Zbývá změřit větší rozpočty nebo méně vrstev.
- **Proč to nefunguje (hypotézy, neověřené):**
  - Každá vrstva je fitovaná na vstupech nezměněného modelu. Ve skupině dostává posunuté vstupy, což potvrzuje
    nárůst in-situ chyby.
  - Fit minimalizuje lokální relL2, ne dopad na výstup modelu.
- **Úspora na modelu:** komprese vizuální větve může ušetřit nejvýš ~10 % modelu (DINOv2 + SigLIP). Smysluplná
  úspora celého modelu vyžaduje kompresi LLM (89 %). Tam M4-lite ukázal vysokou citlivost (q_proj střední,
  down_proj nepoužitelná do 3.3 %).

## 5. Navržené další kroky (zatím neprovedené, k rozhodnutí)

1. **Sekvenční kalibrace:** fitovat vrstvu na vstupech modelu, ve kterém jsou předchozí vrstvy už komprimované.
   Cílí přímo na posun vstupů. Je to levné: stejný fit, jen nová kalibrace po blocích.
2. **Větší rozpočty a menší skupiny:** sweep L64–L256 a komprese jen méně citlivých bloků (0–4) nebo jen
   fc2. Najde rozpočet, při kterém skupina splní práh, a tím kalibruje skupinový práh.
3. **Cíl na úrovni výstupu:** dodatečné doladění θ všech komprimovaných vrstev na KL výstupu modelu (destilace po
   blocích nebo end-to-end). Je dražší, ale je to standardní náprava skládání chyb.
4. **SigLIP jako skupina:** jednovrstvě nevykázal žádný vliv. Skupinový test by ukázal, jestli je vizuální větev
   v praxi komprimovatelná.

## 6. Omezení

- Jeden checkpoint, 268 snímků v 10 epizodách, korigovaný bias, jednotné L pro všechny vrstvy.
- KL v režimu kolapsu saturuje, takže poměr 14× je spíš dolní odhad nadaditivity v tomto bodě. Neříká, kde přesně
  aditivita přestává platit.
- Shoda tokenů a KL nejsou úspěšnost úlohy (M5).
