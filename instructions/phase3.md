# phase_3_distillation.md

# Fáze 3: Optimalizační smyčka (Knowledge Distillation)

## Myšlenkový rámec (Thought Process)
Zpětně dopočítat $P_U$ a $P_V$ tak, aby vygenerovaly masivní cílové $U$ a $V^T$, je kvůli komplexitě determinantu analyticky neřešitelné. Vyřešíme to iterativním trénováním. Optimalizátor (AdamW) bude měnit hodnoty malých parametrů $P_U$ a $P_V$ tak, aby minimalizoval rozdíl mezi výstupem originální vrstvy a výstupem komprimované vrstvy. Kritickým rizikem této fáze je exploze gradientů. Derivace determinantu velké podmatice produkuje masivní hodnoty, které rozbijí parametry (tzv. "NaN contagion"). Proto musíme nasadit striktní ořezávání gradientů (Gradient Clipping).

## Implementační instrukce (Implementation Steps)
1. **Sestavte Distillation Loop:** Inicializujte dva nezávislé `QuIC_MatrixBuilder` objekty (pro stranu U a V).
2. **Kalkulace dopředného průchodu:** Vygenerujte $\hat{U}$ a $\hat{V}^T$. Spočítejte predikci $\hat{Y} = X \times (\hat{U} \Sigma \hat{V}^T)^T$.
3. **Ztrátová funkce:** Použijte MSE (Mean Squared Error) mezi původním odchyceným $Y$ a novým $\hat{Y}$.
4. **Gradient Clipping:** Ihned po volání `loss.backward()` použijte `torch.nn.utils.clip_grad_norm_` s hodnotou max_norm (např. `1.0` nebo `0.5`).
5. **Optimalizace:** Spusťte optimalizátor `AdamW` s Cosine Annealing rozvrhem learning rate.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def run_distillation(X_batch: torch.Tensor, Y_target: torch.Tensor, 
                     builder_U: QuIC_MatrixBuilder, builder_V: QuIC_MatrixBuilder, 
                     sigma: torch.Tensor, optimizer: optim.Optimizer):
    
    optimizer.zero_grad()
    
    # 1. Generate orthogonal approximations
    U_hat = builder_U()
    V_hat = builder_V()
    
    # 2. Reconstruct weight and compute forward pass
    # W = U_hat @ diag(sigma) @ V_hat
    W_approx = torch.matmul(U_hat, sigma.unsqueeze(1) * V_hat)
    
    # X_batch shape: (batch, seq_len, in_features)
    # W_approx shape: (out_features, in_features) -> requires transpose for linear processing
    Y_pred = torch.matmul(X_batch, W_approx.transpose(0, 1))
    
    # 3. Calculate activation loss
    loss = nn.functional.mse_loss(Y_pred, Y_target)
    
    # 4. Backpropagation with strict gradient clipping
    loss.backward()
    torch.nn.utils.clip_grad_norm_(builder_U.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(builder_V.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()
```

Akceptační kritéria a evaluaceKřivka Loss v TensorBoardu musí vykazovat hladký, logaritmicky klesající trend.Zkontrolujte normy gradientů před jejich oříznutím; pokud průměrná norma překračuje řád $10^3$, experimentujte se zmenšením bázových matic nebo zvětšením vyhlazení uvnitř Cayleyho transformace.