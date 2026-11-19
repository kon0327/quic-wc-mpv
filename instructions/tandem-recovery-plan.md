# Plán: doladění komprimované skupiny na výstup modelu, tandem QuIC-WC + QuIC-PEFT (2026-11-19)

Stav: **plán, neprovedeno.** Zadal ho uživatel 2026-11-19 („naplánuj experiment s variantami a, b, c po vyřešení
sekvenční kalibrace“). Spustí se až po uzavření sekvenční kalibrace a sondy rozpočtu
([audit sekvenční kalibrace](../audit/2026-11-19-sequential-calibration.md)).

**Doplněno 2026-11-19 odpoledne:** sekvenční kalibrace je uzavřená. KL klesá 1.7–2× (ASVD r256: 0.22, r128:
1.32; QuIC L16: 10.2), viz [audit sondy rozpočtu](../audit/2026-11-19-budget-probe.md). Volba výchozího bodu
(oddíl 2) čeká na rozhodnutí uživatele.

## 1. Otázka

Vrstvy QuIC-WC jsou fitované na lokální chybu vrstvy. Ve skupině se chyby násobí: KL 10–12 při 1.6 % bajtů
a sekvenční kalibrace pomůže jen částečně. Experiment odpovídá na dvě otázky:

- **Q1:** Kolik ztráty skupiny vrátí doladění na cíl na úrovni modelu?
- **Q2:** Je lepší dolaďovat parametry samotného QuIC-WC, nebo přidat kalibrační adaptér QuIC-PEFT (tandem)?

## 2. Výchozí bod

- Skupina: MLP DINOv2, bloky 0–22. Blok 23 OpenVLA nepoužívá.
- Rozpočet: zvolí se ze sondy rozpočtu. Cílem je bod, kde je skupina po sekvenční kalibraci „blízko použitelnosti“
  (KL řádově 0.1–1), aby doladění mělo co zlepšovat a výsledek byl vidět.
  - Pokud sonda ukáže, že takový bod vyžaduje víc než ~25 % bajtů, použije se menší skupina, např. bloky 0–11 nebo
    jen fc2.
- Inicializace: sekvenční kalibrace QuIC-WC s cílem `output` při zvoleném L, tedy nejlepší varianta bez doladění.

## 3. Varianty

| | trénované parametry | přidané bajty | poznámka |
|---|---|---|---|
| 0 | žádné (sekvenční kalibrace) | 0 | výchozí bod |
| a | σ a korigované biasy všech vrstev | 0 | nejlevnější, mění jen spektrum a posun |
| b | všechna θ, σ a biasy | 0 | plné doladění komprimovaných vrstev |
| c | WC zamrazené + adaptér QuIC-PEFT (výstupní rotace) na každé vrstvě | +adaptér | tandem podle návrhu uživatele |
| b+ | jako b, ale WC s větším L tak, aby bajty = c | +stejně jako c | férová kontrola pro c |

Adaptér je `quic_openvla.adapter.QuICLinear` z `quic-petf-mpv` (commit `f1e71f6`): Cayleyho ortogonální bloky
v compound řádech, inicializace identitou. Obalí dekomprimovanou vrstvu WC. Při nasazení se sloučí do husté váhy
(`merge_adapters`), takže čas inference se nemění. Při počítání ve faktorizované podobě (větev komprese za běhu)
by stál jedno násobení navíc.

**Očekávání (interpretace před měřením):**
- Výstupní rotace se matematicky vstřebá do výstupního faktoru `Q_out` WC.
- Varianta c proto nemá víc výrazové síly než b se stejnými stupni navíc.
- Případná výhoda c bude optimalizační (zamrazený základ, méně parametrů, stabilnější trénink) nebo praktická
  (hotová infrastruktura, samostatný artefakt).

## 4. Cíl a data

- **Data:**
  - trénovací snímky kalibrace (40 epizod, každý 10. snímek, ~1 100 snímků);
  - ze stejné části se po epizodách oddělí 4 epizody pro early stopping;
  - validační epizody (268 snímků) se použijí jen pro závěrečné vyhodnocení M4-lite, jako dosud.
- **Cíle, dvě úrovně (rozhodne pilotní běh):**
  1. **Příznaky:** MSE výstupních patch tokenů DINOv2 (předposlední blok) proti nezměněnému featurizeru. Je
     levná, protože nepotřebuje LLM, a dá se spočítat po celé skupině najednou.
  2. **KL na akčních tokenech:** teacher forcing na referenčních tokenech, KL(reference ‖ student) na 7 akčních
     pozicích. Gradient prochází zamrazeným LLM do vizuální větve; batch 1–2 s checkpointingem aktivací
     (odhad ~20 GB).
  - Pokud cíl 1 nesníží KL ve vyhodnocení aspoň o řád, použije se cíl 2.
- **Optimalizace:**
  - AdamW, kosinový LR, malá mřížka LR (3 hodnoty) na early-stopping epizodách;
  - stejný počet kroků pro všechny varianty;
  - deterministické pořadí snímků.

## 5. Metriky a rozhodnutí

- Na validačních snímcích (M4-lite): tf KL, tf top-1, free shoda s referencí, shoda s datasetem a in-situ relL2.
  Bajty počítané na úrovni artefaktů včetně adaptérů.
- **Kritérium, stanovené předem:**
  - Doladění je přínosné, když a nebo b sníží KL skupiny proti variantě 0 aspoň 2×.
  - Tandem c je přínosný, když proti nejlepší z variant b a b+ při stejných bajtech sníží KL aspoň o 20 %
    (relativně). Jinak se tandem nestaví a pokračuje se dolaďováním WC.
- **CI:** párový bootstrap přes snímky, s upozorněním na korelaci snímků v epizodě.

## 6. Implementace (plán)

1. `quic_wc/pilot/recover.py`:
   - načte skupinu (artefakty + sekvenční kalibrace);
   - vloží `QuICWCLinear` v trénovacím režimu (rekonstrukce váhy v každém kroku) nebo zamrazené WC
     s adaptérem;
   - smyčka tréninku s cílem 1 nebo 2;
   - export artefaktů a plánu M4-lite.
2. Adaptér: import z `quic-petf-mpv` přes cestu, se zaznamenaným commitem; případně kopie s odkazem. Před použitím
   ověřit identitu při vložení a sloučení beze ztráty.
3. Testy: gradient prochází jen zvolenými parametry, identita při nulovém kroku, uložení a načtení po doladění
   (`decompress_count = 1`).
4. Odhad: implementace ~½ dne. Běh cíle 1 v řádu desítek minut na variantu, cíle 2 v řádu hodin.

## 7. Rizika

- Přeučení na 40 epizodách LIBERO-10. Hlídá se early stoppingem po epizodách a vyhodnocením na nezávislých
  epizodách; úspěšnost úloh ověří až M5.
- Paměť GPU u cíle 2, řešení: checkpointing, batch 1, LLM bez gradientů parametrů.
- Při doladění dochází k ořezání na BF16 úložiště θ. Proto se vyhodnocuje vždy znovu načtený artefakt.
