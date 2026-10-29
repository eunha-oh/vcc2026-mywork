# Gene panel filling

## Problem

The challenge scores 18,533 genes, but the main training set (Replogle K562 genome-wide Perturb-seq, "GWPS")
measured only 7,681 of them. To train on GWPS with the full gene panel, the missing ~10,600 genes (59%) had to
be filled in for every cell, with filled and measured values kept apart by a mask.

## Approach

- **Model:** PCA (256 components) on the measured genes followed by ridge regression to the missing genes, fitted
  on log-normalised expression. Measured genes are kept as they are; missing genes get the predicted expected
  count. A per-gene confidence score (`fill_quality`) is written alongside.
- **Training data:** VIPerturb-seq K562 (10x Flex), all 906,837 cells, read in a single streaming pass so memory
  stays around 10 GB.
- **Candidates compared:** mean / kNN baselines, linear models trained on X-Atlas/Orion (HCT116, HEK293T), three
  VAE versions (mask-aware, NB likelihood, shared decoder) and a linear+VAE ensemble.

## Results

| Test | Metric | Orion-trained ensemble | **K562 linear (final)** |
|---|---|---|---|
| K562 held-out targets (VIPerturb) | Δ Pearson / Δ MSE ratio | 0.045 / 1.028 | **0.257 / 0.934** |
| GWPS ground-truth test, strong targets | Δ Pearson / Δ MSE ratio | — | **0.274 / 0.934** |

Δ = perturbation − control on the filled genes; Δ MSE ratio below 1 means better than predicting no change.

- Linear beat every VAE on tests close to the real use case; the VAE trained on Flex data was sensitive to GWPS's
  3' chemistry.
- Training on the same cell line (K562) mattered far more than model complexity: Orion-trained models did not
  transfer to K562.
- Per-target residual correction and input re-calibration did not help and were dropped.
- The filled GWPS was ultimately not used in the team's final model.

## Usage

```bash
pip install -r requirements.txt
export VCC_DATA_ROOT=/path/to/data
python fill_panel.py --input gwps.h5ad --output gwps_filled.h5ad --ckpt /path/to/linear_k562_v2
python verify_fill.py --input gwps.h5ad --output gwps_filled.h5ad   # checks measured values, mask, NaN
```

Options: `--sample-counts` (integer counts), `--mask-mode var` (store the mask in `var` instead of a layer).

| File | Role |
|---|---|
| `fill_panel.py`, `backends.py` | Inference (linear, VAE and ensemble backends) |
| `train_linear_stream.py` | Final training from VIPerturb RDS in one streaming pass |
| `train_linear.py`, `make_viperturb_sample.py` | Earlier training on a 40k-cell sample |
| `gene_quality.py` | Per-gene confidence from a held-out twin model |
| `eval_common.py` | Shared splits and metrics |
| `docs/NOTES_ko.md` | Full development notes (Korean) |

Not included: checkpoints, data, and the streaming RDS reader (`rds_stream.py`, team code) that the training
scripts import.

## 한국어 요약

GWPS(K562)는 대회 유전자 18,533개 중 7,681개만 측정해서, 나머지 59%를 채우는 모델을 만들었습니다.
VIPerturb K562 90만 세포로 학습한 PCA(256) + ridge 선형 모델이 VAE·앙상블보다 나았고, K562 시험에서
Δ Pearson 0.257 / Δ MSE ratio 0.934를 냈습니다. 같은 세포주로 학습하는 것이 모델 복잡도보다 훨씬 중요했습니다.
최종 팀 모델에는 쓰이지 않았습니다.
