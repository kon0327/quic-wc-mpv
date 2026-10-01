# phase_1_math_and_prototype.md

# Fáze 1: Matematický fundament a izolovaný prototyp v PyTorchu

## Myšlenkový rámec (Thought Process)
Cílem této fáze je vytvořit samotný kompresní "motor". Metoda QuIC vyžaduje striktně ortogonální matice. Pokud bychom trénovali parametry napřímo, ortogonalita by se po několika krocích rozpadla. Proto využíváme Cayleyho transformaci, která z jakékoliv matice spolehlivě vytvoří ortogonální matici. Následně z této malé matice ($A$) generujeme obrovskou matici výpočtem determinantů jejích minorů (podmatic). Tyto výpočty musí být plně diferencovatelné, aby jimi mohl protékat gradient během budoucí destilace. Pro zamezení "bottlenecku" toku gradientu se vygenerované složené matice (compound matrices) spojují s původní maticí pomocí přímého součtu (direct sum).

## Implementační instrukce (Implementation Steps)
1. **Vytvořte třídu `QuIC_MatrixBuilder`:** Zděďte z `nn.Module`. Inicializujte trénovatelný parametr `P` s rozměry definujícími základní matici (např. 8 x 8).
2. **Implementujte `cayley_transform`:** Odečtením transpozice od matice `P` získejte antisymetrickou matici `S`. Vypočítejte $A = (I - S)(I + S)^{-1}$.
3. **Implementujte generátor minorů a determinantů:** Pro daný řád $K$ iterujte přes všechny kombinace řádků a sloupců. Využijte `torch.linalg.det` pro výpočet determinantu každé podmatice. Pro zrychlení implementujte tento krok jako vektorizovanou operaci nad batchem tenzorů.
4. **Sestavte Direct Sum ($\oplus$):** Výslednou matici zkonstruujte poskládáním základní matice a vygenerovaných "compound" matic na diagonálu blokové matice. Ostatní prvky vyplňte nulami.

```python
import torch
import torch.nn as nn
import itertools

class QuIC_MatrixBuilder(nn.Module):
    def __init__(self, base_dim: int, compound_order: int):
        super().__init__()
        self.base_dim = base_dim
        self.compound_order = compound_order
        # Trainable parameters for the base matrix
        self.P = nn.Parameter(torch.randn(base_dim, base_dim) * 0.01)

    def cayley_transform(self) -> torch.Tensor:
        # Enforce anti-symmetry to guarantee orthogonality
        S = self.P - self.P.transpose(0, 1)
        I = torch.eye(self.base_dim, device=self.P.device)
        # A = (I - S) * (I + S)^-1
        A = torch.matmul(I - S, torch.linalg.inv(I + S))
        return A

    def generate_compound_matrix(self, A: torch.Tensor, k: int) -> torch.Tensor:
        # NOTE: For MVP, a naive combinatorial approach. 
        # For production, this MUST be vectorized/batched to avoid CPU-GPU sync bottlenecks.
        indices = list(itertools.combinations(range(self.base_dim), k))
        dim_c = len(indices)
        C = torch.zeros((dim_c, dim_c), device=A.device)
        
        for i, row_idx in enumerate(indices):
            for j, col_idx in enumerate(indices):
                # Extract k x k minor and compute determinant
                minor = A[list(row_idx)][:, list(col_idx)]
                C[i, j] = torch.linalg.det(minor)
        return C

    def forward(self) -> torch.Tensor:
        A = self.cayley_transform()
        C = self.generate_compound_matrix(A, self.compound_order)
        
        # Direct sum (A \oplus C)
        total_dim = A.size(0) + C.size(0)
        out = torch.zeros((total_dim, total_dim), device=A.device)
        out[:A.size(0), :A.size(1)] = A
        out[A.size(0):, A.size(1):] = C
        return out
```

Akceptační kritéria a evaluaceVygenerujte náhodný tenzor, prožeňte jej builderem a výslednou matici $\hat{U}$ vynásobte její transpozicí $\hat{U}^T$. Výsledek musí být identitní matice (s tolerancí absolutní chyby 1e-5).Vytvořte dummy_loss = torch.sum(out). Zavolejte dummy_loss.backward(). Ověřte, že self.P.grad není None a neobsahuje NaN.