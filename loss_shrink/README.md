# L_SHRINK loss

## Problem

The model moved many genes that did not change in the real data. Those spurious changes inflate the predicted
fold changes and hurt `lfc_nmae` (the DE-table fold-change error).

## Idea

Directly push the predicted |log fold change| toward zero on measured genes that are **not** significant in the
real data.

```
L = mean_p [ mean_{g ∉ S_p, measured} | lfc_pred(p, g) | ]
lfc_pred = log2( (mean_pred + ε) / (mean_control + ε) )
```

- `S_p`: genes significant for perturbation p in the DE reference (FDR 0.05, |lfc| ≥ 0.25), the same reference
  the team's DE-MMD loss uses; ε matches that reference.
- **No clipping** of |lfc|: clipping would zero the gradient on the most wrong genes.
- Conditions with fewer than 10 significant genes are skipped, matching the official NMAE gate.
- The mean |lfc| inside `S_p` is logged (no gradient) to check that the term does not also shrink real signal.
- Note: official `lfc_nmae` scores only genes inside `S_p`, so this term acts on the scored genes indirectly.

## Results

ΔOverall is against a baseline run with the same seed. Both runs matched the baseline bit-for-bit at step 0.

| Setting | Seed | Steps | ΔOverall | Δ lfc_nmae |
|---|---|---|---|---|
| My setup (GWPS zero-shot) | 20260901 | 1,000 | +0.0140 | −0.050 |
| My setup (GWPS zero-shot) | 20260921 | 1,000 | +0.0330 | −0.122 |
| Team shared A6000 model (H1) | 20260901 | 2,400 | +0.0189 | −0.049 |
| Team shared A6000 model (H1) | 20260921 | 2,400 | −0.0081 | +0.021 |

- On my setup both seeds improved, mostly through `lfc_nmae`.
- On the shared model the seeds disagreed (seed 20260921 was +0.0141 at step 2,000, then turned negative).
  Two seeds were not enough to claim an effect, and the loss was not adopted.
- The port to the team framework was checked against my original code: on a fixed batch, total loss, each
  component and the output gradient matched bit-for-bit; a 200-step run reproduced all metrics and logs.

## Files

| File | Role |
|---|---|
| `de_shrink.py` | The loss function (standalone, PyTorch) |
| `objective_integration.patch` | My changes to the team's `JointObjective` (weight schedule, DE reference built even when DE-MMD is off, zero-weight logging, control masking fix) |
| `configs/common_a6000_shrink_w01.yaml` | Shared-model experiment config (weight 0.1); the baseline is the same file without `losses.de_shrink` |
| `tests/test_de_shrink.py` | 6 unit tests: value, zero case, gradient only on measured non-significant genes, no clipping, gate, input checks |

`python -m unittest tests/test_de_shrink.py` runs without the team framework. Four integration tests from the
original PR need the framework and are not included.

## 한국어 요약

실측에서 유의하지 않은 측정 gene의 예측 |lfc|를 0 쪽으로 직접 누르는 로스입니다. 유의 집합은 팀 DE-MMD와 같은
DE 참조에서 가져오고, 가장 크게 틀린 gene에도 gradient가 가도록 clip하지 않았습니다. 개인 세팅(GWPS zero-shot)에서는
두 seed 모두 +0.014 / +0.033으로 개선됐지만, 팀 공통 모델에서는 seed마다 +0.019 / −0.008로 엇갈려 채택되지
않았습니다.
