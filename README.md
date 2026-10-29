# Virtual Cell Challenge 2026 — My Work

Personal portfolio of the code I wrote as a member of the KUBIG team for the
[Arc Institute Virtual Cell Challenge 2026](https://arcinstitute.org/news/virtual-cell-challenge-2026)
(August–October 2026). The team repository holds the shared training framework; this repository collects
only the parts I designed and implemented, with notes on what worked, what did not, and why.

> 한국어 요약은 [아래](#한국어-요약)에 있습니다.

## The challenge

Predict how a cell line that the model has **never seen** responds to CRISPRi knockdown of 300 genes.
For each target cell line only unperturbed (control) cells are given. A submission is 400 single cells ×
18,533 genes of raw counts per perturbation, scored by six metrics: two on pseudobulk profiles (PDS, MSE)
and four on a Wilcoxon differential-expression table (NMAE, direction fidelity, reach, Jaccard).

The strongest team baseline was a *Pert Mean table*: the average response of the same knockdown in other
cell lines, applied to the target's control cells. The central difficulty turned out to be identifiability:
across cell lines the sign of a gene's response agreed only ~54% of the time, and four to seven training
cell lines were too few for a network to learn cell-line-specific corrections.

## What is in this repository

| Folder | What I built | Outcome |
|---|---|---|
| [`loss_shrink/`](loss_shrink/) | **L_SHRINK**, a loss that penalises predicted change on genes that did not change in the data; final pick after seven loss designs | +0.014 / +0.033 on my setup, within noise on the team's shared model (not adopted) |
| [`gene_panel_fill/`](gene_panel_fill/) | Model that fills the 59% of challenge genes missing from the GWPS (K562) training data; PCA + ridge trained on 0.9M K562 cells, with per-gene confidence | Beat VAE and ensemble candidates on held-out tests; not used in the final pipeline |
| [`model_attempts/de_gate/`](model_attempts/de_gate/) | DE gate: a branch that predicts *whether* each gene changes and gates the log fold change | Leaderboard 0.083 vs 0.1319 baseline: learned "which genes change" did not transfer to unseen cell lines |
| [`model_attempts/decoder_center/`](model_attempts/decoder_center/) | Training-time centring of the decoder output within each cell line, with a penalty on the common part | 0.1325 vs 0.1319: the submission-time debias already did the same job |

Each folder has its own README with the method, results and lessons.

## Main lessons

1. **Check the evaluation split first.** A loss weight chosen on a leaky split (136 of 150 validation targets
   seen in training) reversed on a clean split; a local comparison that only held out cells could not measure
   performance on an unseen cell line.
2. **Use controls that can explain the effect away.** One loss's gain turned out to be the same as simply
   scaling predictions by 0.6.
3. **Measure the ceiling before building.** Training-free "oracle" checks (true labels, true means) repeatedly
   showed that an idea's upper bound was too small, saving training runs.
4. **Most of the score lives in rank statistics.** Four of six metrics come from a Wilcoxon test on single
   cells, so how cells are generated from a predicted mean matters as much as the mean itself.

## Notes

- **Not standalone.** `model_attempts/` code plugs into the team's TxPert-based framework
  (`model_adapters.txpert`, `training.objective`), which is not included. `gene_panel_fill/` inference runs on
  its own given a checkpoint; training reads VIPerturb RDS files with a streaming reader from the team
  repository (not included).
- **No data or checkpoints.** Several datasets used (e.g. X-Atlas/Orion, Zhu CD4+ T cells) are under
  non-commercial licences. Paths point to `${VCC_DATA_ROOT}` (default `data/`).
- Code comments and the detailed notes are mostly in Korean.
- A cross-context attention model (October 14) will be added once the backup becomes accessible again.

## Acknowledgements

Thanks to the KUBIG Virtual Cell Challenge team; the shared framework and Pert Mean tables are their work.
Data: Replogle et al. 2022 (K562 GWPS), Nadig et al. 2025, X-Atlas/Orion, VIPerturb-seq, Zhu et al. (CD4+ T),
and the challenge data from Arc Institute. Scoring follows
[cell-eval2](https://github.com/ArcInstitute/cell-eval2).

---

## 한국어 요약

KUBIG 팀으로 참가한 Virtual Cell Challenge 2026(2026년 8~10월)에서 제가 직접 설계·구현한 코드만 모은
포트폴리오입니다. 팀 공용 프레임워크는 포함하지 않았습니다.

**대회:** 처음 보는 세포주에서 유전자 300개를 각각 껐을 때(CRISPRi)의 반응을 예측합니다. 대상 세포주는
control 세포만 주어지고, perturbation마다 세포 400개의 count를 제출합니다. 지표 6개 중 4개가 세포 단위
Wilcoxon DE 표에서 나옵니다.

**담당한 작업**

| 폴더 | 내용 | 결과 |
|---|---|---|
| `loss_shrink/` | 실측에서 안 변한 유전자의 예측 변화를 억제하는 L_SHRINK 로스 (로스 7종 시도 끝의 최종안) | 개인 세팅 +0.014/+0.033, 팀 공통 모델에서는 노이즈 수준 |
| `gene_panel_fill/` | GWPS(K562)에 없는 59% 유전자를 채우는 모델 (K562 90만 세포, PCA + ridge) | VAE·앙상블보다 나았지만 최종 파이프라인에는 쓰이지 않음 |
| `model_attempts/de_gate/` | 유전자가 변할지를 예측해 log fold에 곱하는 DE 게이트 | 리더보드 0.083 (기준 0.1319). 학습 세포주의 "변할 유전자" 패턴이 새 세포주로 옮겨지지 않음 |
| `model_attempts/decoder_center/` | 학습 중 세포주별 디코더 출력 중심화 + 공통 성분 벌점 | 0.1325 (기준 0.1319). 제출 단계 debias가 이미 같은 역할 |

**배운 점:** 검증 분할(누수, 세포주 단위 분리)을 먼저 확인할 것, 효과를 설명해 버릴 수 있는 대조군을 둘 것,
만들기 전에 정답을 넣은 상한부터 잴 것, 점수 대부분이 세포 순위 통계에서 나온다는 것.

**주의:** `model_attempts/`는 팀 프레임워크 위에서만 돌아갑니다. 데이터·체크포인트는 라이선스 때문에
포함하지 않았습니다. cross-context attention 모델은 백업 접근이 가능해지면 추가할 예정입니다.
