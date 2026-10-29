# DE gate

## Problem

Of the genes our predictions called differentially expressed, only about 11% actually changed. The base model
(TxPert with a Pert Mean prior) shifts all 18,533 genes a little for every perturbation.

## Idea

Add a gate branch that predicts *whether* each gene changes and multiply it into the log fold change.

```
logit_g = gate_head(z)_g + a·|prior_g| + b          z: 64-d latent of TxPert
p_g     = sigmoid(logit_g)
gated   = p ⊙ log_fold                               (log_fold = decoder + exp(gain)·prior)
loss    = base loss + BCE(logit, DE label)
```

- **soft_norm:** multiplying by p shrinks predictions, so each target's output is rescaled back to its original
  RMS (separately for measured / unmeasured genes). With uniform p the model equals the base model.
- **DE labels:** from response-store statistics, z = Δ / SE rescaled to 400 cells per target, BH-FDR 0.05 per
  target; genes under 5 CPM in control or unmeasured are masked. Against real Wilcoxon calls on H1:
  precision 0.90, recall 0.80.

## Results

- **Pre-check without training** (leave one cell line out, simple gate on the table): AUROC for "does this gene
  change" was only 0.54–0.59 whatever features were used (|prior|, control expression, DE frequency in other
  lines, gene responsiveness).
- **Leaderboard:**

| Submission | Overall |
|---|---|
| Baseline (shared control, debias −1) | 0.1319 |
| DE gate, same submission settings | 0.083 (direction fidelity −0.156) |
| DE gate, per-target control, no debias | −0.176 |

## Why it failed

- "Which genes change" learned from training cell lines did not transfer to unseen ones.
- soft_norm made the genes that passed the gate larger, so wrong-direction genes cost more.

## Files

| File | Role |
|---|---|
| `model/txpert_degate.py` | Gate model (wraps the team's `model_adapters.txpert`) |
| `model/degate_objective.py` | BCE term on top of the team's `JointObjective` |
| `configs/L0-degate-h1.yaml` | Training config |
| `tests/test_txpert_degate.py` | 11 unit tests (gate off reproduces the base loss exactly) |
| `tools/build_approx_labels.py`, `tools/h1_wilcoxon_check.py` | DE labels and their validation |
| `tools/precheck_gate.py` | Training-free leave-one-line-out pre-check |
| `tools/submit_with_gate.py` | Submission wrapper |

Requires the team's TxPert framework (not included).

## 한국어 요약

z(64)에서 "이 유전자가 변하나?" 확률을 예측해 log fold에 곱하는 게이트를 TxPert에 추가했습니다.
사전 검증에서 새 세포주의 DE 구분 AUROC가 0.54~0.59에 그쳤고, 리더보드도 0.083으로 기준(0.1319)보다
낮았습니다. 학습 세포주의 "잘 변하는 유전자" 패턴은 새 세포주로 옮겨지지 않았습니다.
