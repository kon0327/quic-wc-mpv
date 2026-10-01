# phase_5_evaluation.md

# Fáze 5: Evaluace na robotických benchmarcích

## Myšlenkový rámec (Thought Process)
Konečným důkazem kompresní metody není nízká chybovost MSE, ale zachování "inteligence" robota. Zkomprimovaný model nesmí trpět degradací prostorového vnímání (což je u VLA modelů nejčastější oběť hrubé komprese). Nasadíme model do rigidního simulačního prostředí. Sledujeme dvě metriky: Success Rate (úspěšnost fyzického provedení úkolu) a Inference Latency (zda model negeneruje akční tokeny příliš pomalu na to, aby dokázal řídit reálný hardware v čase).

## Implementační instrukce (Implementation Steps)
1. **Spuštění SIMPLER Env:** Konfigurujte prostředí pro standardní WidowX a Franka sady úkolů (např. Pick Coke Can, Put Carrot in Bowl).
2. **Načtení modifikovaného modelu:** Zaveďte instanci modifikovaného OpenVLA s integrovanými `QuIC_Linear` vrstvami. Ujistěte se, že proběhla jednorázová dekomprese.
3. **A/B Testování trajektorií:** Spusťte identické scénáře s originálním OpenVLA-OFT (baseline) a komprimovaným OpenVLA.
4. **Měření FPS:** Na inferenční smyčce měřte frekvenci volání `model.predict_action()`.

## Akceptační kritéria a evaluace
* Vyhodnoťte logy ze SIMPLER Env: Absolutní propad v Success Rate (SR) nesmí překročit stanovený margin (např. $-5\%$) oproti baseline řešení.
* Inference FPS v produkční (dekomprimované) formě musí být roven `1.0x` FPS nekomprimovaného modelu. Pokud dojde k degradaci propustnosti na grafické kartě, znamená to, že dekomprimovaný tenzor $W$ nenavazuje čistě na C++ implementaci PyTorch vrstvy `F.linear` (zkontrolujte alokaci paměti a vlastnosti `contiguous()`).