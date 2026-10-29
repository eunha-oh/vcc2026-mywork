# Decoder output centring

## Problem

The TxPert decoder adds a large change that is the same for every perturbation. At submission this is removed
by "debias" (subtracting the leave-one-out mean over targets); turning debias off collapsed NMAE (−1.569).
Training never saw that removal, so the decoder kept learning the common part.

## Idea

During training only, subtract from each target's raw decoder output the mean of the other targets of the same
cell line in the batch, so the loss sees what debias will see.

```
d_t ← d_t − mean_{s ≠ t, same dataset, same batch} d_s        (before the bounded head and the prior term)
L   = base loss + λ · mean(m²)                                m: common part of each group, λ = 10
```

- Groups with fewer than 4 targets in a batch are left as they are (~18% of groups).
- Centring makes the loss blind to the common part, so a small penalty keeps it near zero (≈0.5% of the loss);
  without it, eval mode (which does not centre) would carry it into predictions.
- Nothing changes at evaluation or submission.

## Result

| Submission | Overall |
|---|---|
| Baseline (multi-line training, same settings) | 0.1319 |
| With decoder centring | 0.1325 |

No real change: the submission-time debias was already removing the same component.

## Files

| File | Role |
|---|---|
| `model/txpert_centered.py` | Centring inside the model (wraps the team's `model_adapters.txpert`) |
| `model/centered_objective.py` | Adds the common-part penalty |
| `model/decoder_center_data.py` | Passes dataset ids to the model |
| `configs/B2-prior-multi5-centered-eunha.yaml` | Training config |
| `tests/test_decoder_center.py` | 13 unit tests |

Requires the team's TxPert framework (not included).

## 한국어 요약

제출 단계 debias가 지우던 "모든 표적 공통 변화"를 학습 중에도 빼도록, 같은 배치·같은 세포주 표적들의
디코더 출력 평균을 빼고 공통 성분에 작은 벌점을 줬습니다. 결과는 0.1325로 기준 0.1319와 같은 수준이었습니다.
debias가 이미 같은 일을 하고 있었기 때문입니다.
