# Audit práce na QuIC-WC

Podle požadavku uživatele z 1. 10. 2026 se zde ukládají změny, podstatná zjištění, chyby a výsledky ověření jako Markdown nebo textové soubory. Pravidlo je zachycené i v kořenovém `AGENTS.md` pro další práci v projektu.

Jeden záznam může pokrývat související kroky jednoho úkolu. Názvy používat ve tvaru `YYYY-MM-DD-tema.md`; další úkol stejného dne rozlišit tématem nebo pořadím. Existující záznamy zachovávat a navazující opravy datovat. Zpětně doplněný záznam výslovně označit jako retrospektivní.

Každý záznam podle potřeby obsahuje:

- zadání a rozsah práce;
- vytvořené nebo změněné soubory a důvod změn;
- zjištění a podklad, ze kterého vycházejí;
- chyby a neúspěšné pokusy, nápravu a konečný stav;
- provedené ověření, výsledky a odkazy na reporty;
- otevřené body a omezení.

## Záznamy

- [Analýza a plán implementace](2026-10-01-implementation-plan.md) — retrospektivní audit prvního úkolu.
- [Příprava a ověření GPU prostředí](2026-10-01-gpu-environment.md) — retrospektivní audit prostředí a chyb nalezených při jeho přípravě.
- [Zavedení auditování](2026-10-01-audit-convention.md) — zavedení pravidla a doplnění předchozích záznamů.
- [Analýza projektu pro paměť agenta](2026-11-12-agent-memory.md) — read-only analýza, paměť mimo repozitář.
- [Matematické jádro M1 a první pilot](2026-11-12-m1-core-pilot.md) — implementace `src/quic_wc`, testy, pilot na DINOv2 fc1 (LIBERO-10), chyba cache ASVD a její oprava, test posunu domény.
- [Export BF16, brána M3, optimalizace a pilot na čtyřech vrstvách](2026-11-12-bf16-gate-optim-multilayer.md) — formát v2, nová brána M3, LR škálovaný hloubkou, kalibrace z celého modelu, DINOv2/SigLIP/Llama q_proj a down_proj.
- [Compound řády proti prostým blokům](2026-11-16-compound-orders.md) — rozhodovací experiment K=[1,2] vs K=[1] (stejné bajty i stejný výpočet) a K=[1,2,3] před M4-lite.
- [M4-lite](2026-11-18-m4-lite.md) — komprimované vrstvy v celém OpenVLA: akční tokeny, KL logitů, in-situ chyba vrstvy; QuIC-WC vs ASVD.
- [Skupina MLP DINOv2](2026-11-18-group-dinov2-mlp.md) — 48 MLP vrstev DINOv2: při 1–3 % bajtů kolaps obou metod (KL 10–15), KL 14–32× nadaditivní; po vrstvách QuIC-WC potřebuje při stejné relL2 o 36–72 % méně bajtů než ASVD.
