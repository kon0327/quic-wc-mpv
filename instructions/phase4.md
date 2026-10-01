# phase_4_integration.md

# Fáze 4: Integrace a statická dekomprese v OpenVLA

## Myšlenkový rámec (Thought Process)
Máme natrénované parametry. Nyní je musíme zabalit do produkčně nasaditelné podoby. Primárním benefitem této architektury je zmenšení požadavků na propustnost a diskovou paměť. Třída `QuIC_Linear` musí na disku (a v `.safetensors`) existovat pouze jako tři malé vektory/matice ($P_U$, $P_V$, $\Sigma$). V momentě, kdy je model načten a odeslán na GPU (`.to('cuda')`), se spustí jednorázový "dekompresní" algoritmus, který obrovskou matici $W$ zrekonstruuje z minorů. Po zbytek běhu simulátoru se model chová přesně tak, jakoby žádná komprese nenastala.

## Implementační instrukce (Implementation Steps)
1. **Vytvořte modul `QuIC_Linear`:** Modul musí obsahovat dříve vytvořené buildery a `nn.Parameter` pro vektor $\Sigma$.
2. **Implementujte `decompress()` metodu:** Zavolejte dopředný průchod builderů, rekonstruujte a uložte kompletní tenzor $W$ jako interní stav modulu. Nastavte boolean vlajku `self.is_decompressed = True`.
3. **Dynamický forward pass:** Metoda `forward()` musí zkontrolovat vlajku dekomprese a následně provést standardní operaci `F.linear(x, self.W, self.bias)`.
4. **Monkey-patching skript:** Napište rekurzivní funkci v Pythonu, která projde model OpenVLA a nahradí specifikované cílové instance `nn.Linear` instancemi `QuIC_Linear`, přičemž zkopíruje odladěné váhy ze třetí fáze.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuIC_Linear(nn.Module):
    def __init__(self, base_dim: int, sigma_dim: int, compound_order: int, 
                 bias: bool = True):
        super().__init__()
        self.builder_U = QuIC_MatrixBuilder(base_dim, compound_order)
        self.builder_V = QuIC_MatrixBuilder(base_dim, compound_order)
        self.sigma = nn.Parameter(torch.ones(sigma_dim))
        
        if bias:
            # Reconstruct bias to match target output dimension
            # Dimensionality math depends on exact block structures
            # For simplicity, assuming output_dim is known via target
            self.bias = nn.Parameter(torch.zeros(1)) 
        else:
            self.register_parameter('bias', None)
            
        self.is_decompressed = False
        self.W_reconstructed = None

    def decompress(self):
        with torch.no_grad():
            U_hat = self.builder_U()
            V_hat = self.builder_V()
            self.W_reconstructed = torch.matmul(U_hat, self.sigma.unsqueeze(1) * V_hat)
        self.is_decompressed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_decompressed:
            self.decompress()
        return F.linear(x, self.W_reconstructed, self.bias)

def replace_linear_with_quic(module, target_name_substring="proj"):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and target_name_substring in name:
            # Example instantiation - dimensions must be dynamically extracted
            quic_layer = QuIC_Linear(base_dim=8, sigma_dim=child.out_features, compound_order=2)
            setattr(module, name, quic_layer)
        else:
            replace_linear_with_quic(child, target_name_substring)
```
Akceptační kritéria a evaluace
Uložte model po aplikaci monkey-patchingu přes torch.save. Zkontrolujte velikost souboru a porovnejte ji s velikostí originálního .bin souboru pro danou vrstvu.

Změřte (pomocí time.time()), jak dlouho trvá dekomprese při volání model.to('cuda'). Nesmí blokovat instanciaci na více než jednotky až desítky sekund v závislosti na hardwaru.
