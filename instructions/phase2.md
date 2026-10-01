# phase_2_extraction.md

# Fáze 2: Extrakce vah a aktivačních dat z OpenVLA

## Myšlenkový rámec (Thought Process)
Abychom mohli metodu uplatnit na masivní váhy OpenVLA, musíme originální matice nejdříve rozložit. Běžná váha $W$ není ortogonální, a proto ji nelze přímo nahradit QuIC strukturou. Použitím SVD získáme ortogonální složky $U$ a $V^T$, které slouží jako dokonalý cíl pro aproximaci, a diagonální matici $\Sigma$, která definuje "důležitost" parametrů. Extrakce pouze samotných vah by pro destilaci nestačila; MSE loss nad maticí vah má špatnou korelaci s výkonem sítě. Musíme zachytit reálné tenzory $X$ (aktivace vstupující do vrstvy), abychom při optimalizaci modelovali reálný distribuční posun dat (Activation-based Distillation).

## Implementační instrukce (Implementation Steps)
1. **Načtěte izolovanou vrstvu:** Z checkpointu OpenVLA (např. HuggingFace weights) izolujte jednu specifickou `nn.Linear` vrstvu (doporučeno začít na MLP projekci vizuálního enkodéru).
2. **SVD Dekompozice:** Přesuňte původní váhu na GPU v `float32` nebo `float64` (SVD je nestabilní v bf16/fp16) a aplikujte `torch.linalg.svd`. Proveďte oříznutí (Truncated SVD) smazáním singulárních hodnot menších než nastavený threshold.
3. **Generování datasetu aktivací:** Načtěte zlomek OpenX mixture (např. 50 trajektorií). Spusťte originální OpenVLA v režimu inference. Pomocí PyTorch Hook (`register_forward_hook`) odchytávejte vstupy $X$ do zvolené vrstvy a její odpovídající výstupy $Y$. Uložte tyto tenzory na disk.

```python
import torch

def perform_truncated_svd(weight_matrix: torch.Tensor, keep_ratio: float = 0.95):
    # Ensure float32 for stable SVD computation
    W = weight_matrix.to(torch.float32)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Keep top k singular values
    k = int(len(S) * keep_ratio)
    U_trunc = U[:, :k]
    S_trunc = S[:k]
    Vh_trunc = Vh[:k, :]
    
    return U_trunc, S_trunc, Vh_trunc

# Activation hook example
activations_X = []
activations_Y = []

def capture_hook(module, input, output):
    # input is a tuple
    activations_X.append(input[0].detach().cpu())
    activations_Y.append(output.detach().cpu())

# target_layer.register_forward_hook(capture_hook)
# ... run model inference ...
```

Akceptační kritéria a evaluaceRekonstruovaná matice z odříznutého SVD ($U_{trunc} \Sigma_{trunc} V^T_{trunc}$) vynásobená vstupem $X$ musí vykazovat Cosine Similarity s originálním výstupem $Y$ vyšší než 0.98.Tenzory aktivací uložte ve formátu .safetensors nebo .pt po dávkách, abyste předešli přeplnění RAM.