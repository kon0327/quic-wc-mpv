# Analýza a plán implementace

Datum práce a doplnění záznamu: 1. 10. 2026. **Retrospektivní záznam** sestavený z průběhu práce a uloženého plánu.

## Zadání a změny

Analyzovat `instructions/`, zejména `concept.md`, a rozvrhnout implementaci QuIC-WC MVP s lokálními závislostmi compound, OpenVLA a LIBERO.

Výstup: [implementation-plan.md](../instructions/implementation-plan.md). Plán definuje milníky M0–M5, rozměrové kontrakty, moduly, kalibraci, destilaci, export, loader a A/B evaluaci. V tomto úkolu nebyl implementován kompresní model ani spuštěn trénink.

## Zjištění

- Lokální compound je PEFT adaptér zachovávající původní `base_layer`, nikoliv hotová komprese nahrazující váhu.
- Ukázky instrukcí neřeší obecné obdélníkové rozměry vrstev, orientaci pravého SVD faktoru a správnou velikost bias. Pro bázi 8 a compound řádu 2 má direct sum velikost 36, nikoliv libovolný rozměr OpenVLA vrstvy.
- Jeden direct sum omezuje podporu faktorů blokovou strukturou. Pro shodný blokový rozklad čtvercových faktorů zůstává rekonstruovaná váha blokově diagonální. Proto je plánovanou rozhodovací branou kvalita na jedné vrstvě; více stupňů s promícháním je experimentální varianta.
- Dodané závislosti obsahují standardní OpenVLA a LIBERO, nikoliv implementaci OpenVLA-OFT a SIMPLER. První plánovaná evaluace používá LIBERO se stejným OpenVLA checkpointem pro baseline a studenta.
- Statická dekomprese může zmenšit checkpoint; ustálená hustá inference sama nesnižuje VRAM ani nezaručuje zrychlení.
- Lokální compound vynucuje FP32 determinanty i u FP64 vstupu. Cache rekonstruované váhy vyžaduje explicitní správu zařízení, dtype, invalidace a vyloučení ze serializace.

## Provedené ověření

Read-only analýza všech šesti zadávacích dokumentů a relevantních zdrojů externích projektů. Malé CPU diagnostiky matematických metod načtených ze zdrojového AST bez instalace celého PEFT:

- direct sum 36×36 při identity init: chyba ortogonality 0;
- malý náhodný init: chyba přibližně `5.96e-7`, finite gradient;
- FP64 vstup oproti smyčkové FP64 referenci: odchylka přibližně `6.92e-8` kvůli internímu FP32 převodu;
- mimobloková část jedné náhodné husté váhy 36×36: relativní Frobeniova norma přibližně `0.584`. Jde o syntetický strukturální příklad, nikoliv výsledek na OpenVLA.

Výsledky jsou zapsané v plánu; diagnostický skript tehdy nebyl uložen jako samostatný artefakt. Ověřena také struktura dokumentu a existence hlavních odkazovaných zdrojů.

## Omezení a následná oprava interpretace

Původní kontrola v sandboxu vracela `torch.cuda.is_available() == False` a `nvidia-smi` hlásilo blokovaný přístup. Následný úkol prokázal dostupnou RTX 4090 mimo sandbox; původní výsledek popisoval omezení relace, nikoliv chybějící GPU hostitele.

Úspěšnost komprese skutečné vrstvy, načtení OpenVLA-7B a robotická kvalita zůstaly neověřené. Navazující příprava prostředí je popsána v [GPU auditu](2026-10-01-gpu-environment.md).
