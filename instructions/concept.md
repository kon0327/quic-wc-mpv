# MVP Implementační Plán: QuIC-WC pro OpenVLA

Tento dokument slouží jako technická specifikace a checklist pro implementaci kompresní metody QuIC-WC (Weight Compression) do architektury OpenVLA. Cílem je nahradit plné váhové matice komprimovanou strukturou využívající SVD dekompozici a diferencovatelné QuIC bloky generované Cayleyho transformací a determinanty minorů.

---

## Fáze 1: Matematický fundament a izolovaný prototyp

**Cíl:** Vytvořit a otestovat základní diferencovatelné stavební bloky v PyTorchu (izolovaně od modelu OpenVLA).

### Implementační úkoly
- [ ] Založit výpočetní prostředí (Docker + NVIDIA CUDA Toolkit).
- [ ] Implementovat třídu `QuIC_MatrixBuilder` pro generování ortogonálních matic.
- [ ] Napsat funkci pro Cayleyho transformaci (převod trénovatelných parametrů $P$ na antisymetrickou matici $S$ a následně na ortogonální základní matici $A$).
- [ ] Sestavit dopředný průchod (forward pass) využívající `torch.linalg.det` pro výpočet "compound" matic řádu $K$.
- [ ] Implementovat operaci přímého součtu ($\oplus$) pro složení finální blokové matice.

```python
import torch
import torch.nn as nn

class QuIC_MatrixBuilder(nn.Module):
    def __init__(self, base_dim):
        super().__init__()
        self.P = nn.Parameter(torch.randn(base_dim, base_dim))

    def cayley_transform(self):
        # Enforce anti-symmetry
        S = self.P - self.P.transpose(0, 1)
        I = torch.eye(S.size(0), device=S.device)
        # A = (I - S) * (I + S)^-1
        A = torch.matmul(I - S, torch.linalg.inv(I + S))
        return A
        
    def forward(self):
        A = self.cayley_transform()
        # TODO: Implement determinant-based compound matrix generation and direct sum
        pass