# Gene panel 채우기

## 목적

VCC2026은 perturbation마다 유전자 18,533개(`vcc2026/gene_names.csv`)를 예측해야 한다. 학습 데이터 GWPS(K562)는 그중 7,681개만 측정했다. 이 폴더의 스크립트는 일부 유전자만 측정된 h5ad를 받아 18,533 유전자 전체로 채운다.

- 측정된 유전자는 입력 값을 그대로 둔다.
- 측정되지 않은 유전자는 모델이 예측한 기대 count로 채운다.
- 채운 값과 측정값은 mask layer와 `var`로 구분한다.

납품 기본값은 K562 데이터(VIPerturb-seq) 원본 전체 906,837세포로 학습한 선형 모델 `linear_k562_v2`다 (2026-10-04). GWPS와 같은 K562 세포주 데이터로 학습해, 이전 납품 `ensemble_v1`(Orion으로 학습)보다 K562 평가와 GWPS 실전 검증에서 모두 낫고, `linear_k562_v1`(샘플 4만 세포)보다 조금 낫다 (아래 "검증 결과: K562", "선형 개선 시험"). 채운 유전자마다 신뢰도 점수(`fill_quality`)도 함께 낸다 (아래 "유전자별 신뢰도"). 다른 체크포인트도 같은 명령으로 쓸 수 있다. `fill_panel.py`는 체크포인트 `meta.json`의 `model_type`(`linear`, `vae`, `ensemble`)을 읽어 백엔드를 고른다.

| 체크포인트 | `model_type` | 내용 |
|---|---|---|
| `linear_k562_v2` (기본) | `linear` | PCA(256) + Ridge, VIPerturb 원본 bin 전체 906,837세포로 학습. `gene_quality.csv` 포함 (아래 "방법: 선형 K562 v2") |
| `linear_k562_v1` (이전 납품) | `linear` | PCA(128) + Ridge, VIPerturb K562 샘플 39,949세포로 학습 (아래 "방법: 선형 K562 v1") |
| `ensemble_v1` (이전 납품) | `ensemble` | `linear_v1`과 `vae_v2`를 0.5씩 평균. 파일은 `meta.json` 하나이고 두 체크포인트를 상대 경로로 불러온다 |
| `vae_v2` | `vae` | 공유 디코더 VAE (아래 "방법: VAE v2") |
| `linear_v1` | `linear` | PCA + Ridge (아래 "방법: 선형") |
| `vae_v1` | `vae` | 이전 VAE. 선형보다 못해 쓰지 않음 (비교용) |

체크포인트 폴더: `${VCC_DATA_ROOT}/derived/gene-panel-fill/checkpoints/`

이 브랜치에는 납품 스크립트만 있다. VAE가 들어간 체크포인트(`vae_v2`, `vae_v1`, `ensemble_v1`)는 이 브랜치에서 실행되지 않는다. 실험 기록과 상세 결과는 `model/gene-panel-fill-eunha` 브랜치 참고.

## 실행

```bash
pip install -r gene_panel_fill/requirements.txt

python gene_panel_fill/fill_panel.py \
  --input gwps.h5ad \
  --output gwps_filled.h5ad \
  --ckpt ${VCC_DATA_ROOT}/derived/gene-panel-fill/checkpoints/linear_k562_v2
```

- `linear_k562_v2`와 `linear_k562_v1`은 CPU만 쓴다. VAE가 들어간 체크포인트는 GPU가 있으면 자동으로 쓰고, 없으면 CPU로 돈다(`--device auto`).
- 정수 count가 필요하면 `--sample-counts`를 붙인다.

출력 검증(측정값 보존, mask 일치, NaN과 음수 여부, h5ad 열기):

```bash
python gene_panel_fill/verify_fill.py --input gwps.h5ad --output gwps_filled.h5ad --expect-genes 18533
```

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--input` | 필수 | 세포 x 측정 유전자 h5ad. raw count. X는 dense, sparse 모두 가능 |
| `--output` | 필수 | 출력 h5ad. 이미 있거나 입력과 같은 경로면 중단 |
| `--ckpt` | 필수 | 체크포인트 폴더. 기본 납품은 `linear_k562_v2` |
| `--genes` | 체크포인트의 18,533 목록 | 출력 유전자 목록 csv(열 `gene_name`) |
| `--gene-col` | `var['gene_name']`가 있으면 그것, 없으면 `var_names` | 입력 유전자 심볼 열 |
| `--chunk-size` | 20000 | 한 번에 처리할 세포 수 |
| `--sample-counts` | 끔 | 채운 값을 샘플링한 정수 count로 저장. `ensemble`은 평균을 기대값으로 `vae_v2`의 θ로 음이항, `vae`는 음이항, `linear`는 Poisson |
| `--seed` | 0 | `--sample-counts` 난수 시드 |
| `--device` | `auto` | VAE 계산 장치. `auto`는 GPU가 있으면 GPU, 없으면 CPU. `cpu`, `cuda` 지정 가능 |
| `--batch-size` | 2048 | VAE 미니배치 크기 |
| `--enc-dataset` | 체크포인트 기본값(`gwps`) | VAE 인코더 데이터셋 임베딩(`gwps`, `orion`). GWPS 입력이면 바꿀 필요 없음 |
| `--dec-dataset` | 체크포인트 기본값 | `vae_v1`용. `vae_v2`와 `ensemble_v1`은 입력 데이터셋으로 고정이라 다른 값을 주면 오류 |
| `--mask-mode` | `layer` | `layer`: `layers['measured_mask']` 저장. `var`: mask layer 없이 `var['is_measured']`만 저장 |

### 입력 처리

- 입력 유전자를 심볼로 출력 축에 맞춘다. 같은 심볼이 여러 열이면 합친다(GWPS는 `TBCE`, `HSPA14` 2개).
- 출력 축에 없는 입력 유전자는 버린다(GWPS는 565개).
- 위 두 개수는 실행 로그와 `uns['gene_panel_fill']['summary']`에 남는다.

## 출력 형식

| 항목 | 내용 |
|---|---|
| `X` | dense float32, (입력 세포 수, 18,533). 유전자 순서는 18,533 목록과 같음 |
| `layers['measured_mask']` | uint8. 1은 원래 측정값, 0은 채운 값. `--mask-mode layer`일 때만 |
| `var` | index는 `gene_name`. `is_measured`(bool), `fill_source`(`measured`, `model`, `fallback`, `none`). 체크포인트에 `gene_quality.csv`가 있으면 `fill_quality`(float32, 아래 "유전자별 신뢰도") |
| `obs` | 입력 obs 그대로 |
| `uns['gene_panel_fill']` | 체크포인트 경로와 메타데이터(앙상블은 구성 모델 메타와 code_version 포함), 실행 옵션, 유전자 수 요약, 실행 시각 |

- 채운 값은 기본으로 기대 count(실수)다. 정수 count가 필요하면 `--sample-counts`를 쓴다.
- GWPS 입력이면 `linear_k562_v2`와 `linear_k562_v1`은 `measured` 7,681개, `model` 10,570개, `fallback` 282개다. `fallback`은 VCC2026 control 평균(상수)이다.
- 이전 `ensemble_v1`은 `measured` 7,681개, `model` 10,561개, `fallback` 291개다. 앙상블은 `model` 10,561개를 평균하고 `fallback` 291개는 `vae_v2` 값(VCC2026 control 비율)을 쓴다.
- `fill_source=none`은 측정되지도 않았고 모델 출력에도 없는 유전자로, 0으로 둔다. GWPS 입력에서는 0개다.
- 출력은 chunk 단위로 쓰고 `anndata.read_h5ad`의 일반 모드와 `backed='r'` 모드 모두로 열린다.

### 용량과 시간 (GWPS 1,989,578 세포, `linear_k562_v2`, `--chunk-size 20000` 기준 추정)

| 항목 | 값 |
|---|---|
| 실행 시간 (기대값) | 약 27~40분 |
| 실행 시간 (`--sample-counts`) | 약 45~60분 |
| 최대 메모리 (RAM) | 약 3GB (`linear_k562_v1` 측정 3.1GB, v2는 PCA 성분 수만 다름) |
| 출력 `X` | 약 147GB |
| 출력 `measured_mask` | 약 37GB (`--mask-mode var`로 생략 가능) |

- 20,000세포 chunk 하나의 측정값으로 계산했다 (약 100 chunk, L4 기기). chunk당 계산과 쓰기는 `linear_k562_v2` 11.0초(`--sample-counts` 22.3초), `linear_k562_v1` 10.6초(21.6초)였다. 여기에 공용 storage에서 GWPS 20,000행을 읽는 5~13초를 더했다. 1 chunk 출력은 1.86GB다.
- 선형만 쓰므로 GPU가 필요 없다. CPU 기기(RAM 14GB)의 실제 시간은 재지 않았다.
- 전체 GWPS 실행은 아직 하지 않았다.

이전 `ensemble_v1` 기준 (2026-10-03 측정):

| 항목 | L4 GPU | CPU 4코어 |
|---|---|---|
| 실행 시간 (기대값) | 약 70~80분 | 약 80~90분 |
| 실행 시간 (`--sample-counts`) | 약 70~85분 | 약 130~145분 |
| 최대 메모리 (RAM) | 약 7.6GB | 약 7.4GB |
| 출력 `X` | 약 147GB | 같음 |
| 출력 `measured_mask` | 약 37GB (`--mask-mode var`로 생략 가능) | 같음 |

- 20,000세포 chunk 하나의 측정값으로 계산했다 (약 100 chunk). chunk당 계산과 쓰기는 L4 36초(`--sample-counts` 37초), CPU 42초(73초)였다. 여기에 공용 storage에서 GWPS 20,000행을 읽는 5~13초를 더했다.
- CPU 값은 L4 기기에서 4코어로 제한해 잰 것이다 (`taskset -c 0-3`, `--device cpu`). CPU 기기의 실제 시간은 다를 수 있다.
- GPU에서도 chunk 시간의 대부분은 CPU 쪽(선형 예측, 평균 계산, 파일 쓰기)이라 GPU 이득이 작다.
- 메모리가 부족하면 `--chunk-size 10000`으로 줄인다. CPU 기기(RAM 14GB)에서는 20000으로 돌아간다.
- 공용 storage는 S3 마운트라 h5ad를 직접 쓰면 느리거나 깨질 수 있다. 로컬 디스크에 쓰고 복사한다. 로컬 디스크 여유 공간이 출력 크기보다 커야 한다.

## 방법: 선형 K562 v2 (`linear_k562_v2`, 기본)

아래 "방법: 선형"과 같은 PCA + Ridge이고, 학습 데이터와 PCA 성분 수가 다르다.

- 학습 데이터: VIPerturb-seq 원본 `genome_wide_bin{A,B,C}.RDS` 전체 906,837세포 (타깃 약 18,885개, control 7,949세포. 세 bin에 같이 들어 있는 control은 한 번만 셈). 모든 세포에서 count 합이 `nCount_RNA`와 같음을 확인했다.
- 모델: PCA 256 성분, Ridge alpha 1. 유전자 집합은 v1과 같다 (`O_model` 7,541, `M_model` 10,570, `M_fallback` 282).
- 학습 방법 (`train_linear_stream.py`): RDS를 R 없이 두 번 순차로 읽고, 세포 묶음마다 XᵀX와 XᵀY를 GPU에 float64로 누적한다. 공분산 고유분해로 정확한 PCA를 구하고, PCA 점수 위 ridge는 대각이라 바로 푼다. 메모리가 세포 수와 무관하다.
- 학습: L4 기기에서 26분, 최대 RSS 10.6GB, GPU 약 5GB. 설명 분산 0.129. code_version `90250c5`. 같은 명령으로 두 번 학습한 결과가 비트 단위로 같았다.
- 체크포인트 형식은 `train_linear.py`와 같아 `fill_panel.py`가 그대로 읽는다.

```bash
python gene_panel_fill/train_linear_stream.py --out <체크포인트 폴더>
# 유전자별 신뢰도: K562 평가 홀드아웃을 뺀 같은 모델을 만들고 점수 계산
python gene_panel_fill/train_linear_stream.py --exclude-k562-holdout \
  ${VCC_DATA_ROOT}/derived/gene-panel-fill/viperturb_k562_sample40k_vccaxis.h5ad --out <홀드아웃 제외 폴더>
python gene_panel_fill/gene_quality.py --ckpt <홀드아웃 제외 폴더> --out <체크포인트 폴더>/gene_quality.csv
```

## 방법: 선형 K562 v1 (`linear_k562_v1`, 이전 납품)

아래 "방법: 선형"과 같은 PCA(128) + Ridge(alpha=1)이고 학습 데이터만 다르다.

- 학습 데이터: VIPerturb K562 샘플 전체 39,949세포 (타깃 6,724개 × 최대 5세포 + control 6,375세포, 아래 "K562 학습 데이터 샘플").
- 유전자: `O_model` 7,541개 (GWPS 유전자이면서 VIPerturb 측정), `M_model` 10,570개, `M_fallback` 282개 (VIPerturb에 없는 유전자, 대부분 lncRNA).
- 학습: CPU에서 74초, 최대 RSS 5.0GB. code_version `89bb109`.

```bash
python gene_panel_fill/train_linear.py \
  --train-data ${VCC_DATA_ROOT}/derived/gene-panel-fill/viperturb_k562_sample40k_vccaxis.h5ad \
  --split all --out <체크포인트 폴더>
```

## 방법: 선형 (L1000 방식)

L1000처럼 측정된 유전자로 측정되지 않은 유전자를 선형 회귀로 추정한다.

| 유전자 집합 | 정의 | 개수 |
|---|---|---|
| `O_model` | GWPS 유전자, VCC 18,533, Orion 측정의 교집합. 모델 입력 | 7,545 |
| `M_model` | VCC 18,533 중 GWPS에 없고 Orion에서 측정된 유전자. 모델 출력 | 10,561 |
| `M_fallback` | VCC 18,533 중 GWPS에도 Orion에도 없는 유전자 | 291 |

1. 세포마다 `O_model` count 합을 기준으로 1e4 normalize 후 log1p.
2. PCA(128) 후 Ridge(alpha=1)로 `M_model`의 log 정규화 값을 예측.
3. `expm1` 후 음수를 0으로 하고, 세포의 `O_model` 합 / 1e4를 곱해 count로 되돌린다.
4. `M_fallback`은 VCC2026 control 3개(55,200세포)를 같은 방식으로 정규화한 유전자별 평균을 상수로 쓰고, 같은 방식으로 count로 되돌린다. perturbation 사이 차이가 없으므로 마스킹과 같은 효과다.

학습 데이터는 Orion 샘플 40,000세포(HCT116, HEK293T 각 20,000)다.

### 학습

```bash
python gene_panel_fill/train_linear.py --out <체크포인트 폴더> [--n-components 128] [--alpha 1.0]
```

- 입력 기본값은 Orion 샘플, VCC2026 control 3개, 유전자 목록, 입력 유전자 기준 GWPS var다.
- 체크포인트에는 유전자 목록, PCA mean과 components, Ridge coef와 intercept, fallback 평균, 정규화 상수, 하이퍼파라미터, 학습 데이터 경로, 생성 시각, git commit hash가 들어간다.
- 출력 폴더에 파일이 있으면 덮어쓰지 않고 중단한다.
- CPU 4코어에서 약 70초, 최대 메모리 약 6.4GB.
- `--train-data`에 학습 h5ad를 여럿 줄 수 있다 (예: Orion + VIPerturb). 이때 모든 학습 데이터가 측정한 유전자만 입력과 출력으로 쓴다. `--split k562_unseen_pert`는 K562 평가 홀드아웃 타깃을 뺀다.
- Ridge는 PCA 점수의 정규방정식을 세포 chunk마다 누적해 푼다 (sklearn Ridge와 같은 식). 기본값으로 다시 학습하면 `linear_v1`과 예측이 log1p 공간 2e-5 이내로 같다.

### 체크포인트

`${VCC_DATA_ROOT}/derived/gene-panel-fill/checkpoints/linear_v1/`

## 방법: VAE v2

측정 mask를 입력으로 받는 VAE(scVI 계열)다. 어떤 유전자 패널이 들어와도 18,533개 전체의 기대 count를 낸다.

- 인코더 입력: 보이는 유전자의 log 정규화 값, mask, 데이터셋 임베딩(`orion`, `gwps`)
- 디코더: z만 받아 18,533개 유전자 logit을 내는 공유 디코더. 데이터셋별 유전자 오프셋은 그 데이터셋이 측정한 유전자에만 더한다. Orion을 기준(오프셋 0)으로 두고 GWPS는 측정 유전자 7,681개에만 오프셋을 학습한다. dispersion θ도 같다.
- 채울 때: 입력 데이터셋(GWPS)이 측정하지 않은 유전자는 오프셋 없이 공유 출력으로 채운다. 기대 count는 보이는 유전자 count 합 × (유전자 비율 / 보이는 유전자 비율 합)이다.
- 우도: 음이항(NB). 학습은 Orion 세포를 GWPS 패턴(80%)이나 무작위 패널로 더 가리고, GWPS 세포도 함께 쓴다. 숨긴 유전자 로스에 가중치 2.
- 학습 데이터에 한 번도 측정되지 않은 291개는 VCC2026 control 비율로 채운다(`fallback`).
- 이전 `vae_v1`은 디코더에 데이터셋 임베딩을 넣어, 디코더 데이터셋을 무엇으로 고르든 실제 결측 유전자에 맞지 않았다(아래 표).

### 학습

- VAE 학습 스크립트는 이 브랜치에 없다.
- `vae_v2`: Orion 샘플 40,000세포 + GWPS 학습 샘플 60,000세포, 30 epoch, L4에서 20분. code_version `da877d9`.
- 하이퍼파라미터는 `split=val` 비교(hp_a~d)에서 고른 hp_d다.

## 방법: 앙상블

- `ensemble_v1/meta.json`에 구성 모델(`../linear_v1`, `../vae_v2`)과 가중치(0.5씩, 튜닝하지 않음)가 적혀 있다.
- 두 모델이 모두 `model`로 채우는 유전자(GWPS 입력에서 10,561개)를 `log1p(count × 1e4 / lib)` 공간에서 평균하고 count로 되돌린다. `lib`은 입력 측정 유전자 count 합이다.
- 각 구성 모델에는 단독 실행 때와 같은 입력을 준다. 선형은 자기 입력 유전자(7,545개)와 그 합을 쓴다.
- `--sample-counts`는 평균을 기대값으로 `vae_v2`의 θ로 음이항 샘플링한다.

## 검증 결과

### MVP 비교

Orion 샘플에서 `O_model`만 보고 `M_model`을 채웠을 때의 결과다. Δ는 같은 세포주 control 평균과의 pseudobulk 차이다. Δ MSE ratio 1은 Δ=0(마스킹)과 같은 수준이다.

| 분할 | 방법 | Δ Pearson (중앙값) | Δ MSE ratio (중앙값) |
|---|---|---|---|
| 처음 보는 perturbation | 평균 | 0.010 | 1.00 |
| 처음 보는 perturbation | kNN(20) | 0.183 | 0.98 |
| 처음 보는 perturbation | PCA(128) + Ridge | 0.255 | 0.93 |
| 처음 보는 세포주 (HCT116 → HEK293T) | 평균 | 0.001 | 1.00 |
| 처음 보는 세포주 (HCT116 → HEK293T) | kNN(20) | 0.119 | 1.00 |
| 처음 보는 세포주 (HCT116 → HEK293T) | PCA(128) + Ridge | 0.175 | 0.96 |

### 선형, VAE, 앙상블 비교

지표는 모두 중앙값이다. VAE는 실제 채우기와 같은 설정(v1은 enc=gwps, dec=orion, v2는 enc=gwps)이다. 앙상블은 같은 표의 pca_ridge와 VAE v2를 log 정규화 공간에서 0.5씩 평균한 것이다. 굵은 글씨는 분할별 최고값이다.

Orion 분할 (GWPS 측정 유전자 7,545개만 보고 GWPS에 없는 10,561개를 채움, 각 분할의 학습 세포로만 학습한 체크포인트):

| 분할 | 방법 | Δ Pearson | Δ MSE ratio | gene Pearson | RMSE |
|---|---|---|---|---|---|
| 처음 보는 perturbation (146그룹) | pca_ridge | 0.2551 | 0.9307 | 0.1210 | **0.1818** |
| | VAE v1 | 0.2385 | 0.9436 | 0.1201 | 0.1867 |
| | VAE v2 | 0.2396 | 0.9440 | 0.1196 | 0.1848 |
| | 앙상블 | **0.2575** | **0.9299** | **0.1274** | 0.1821 |
| 처음 보는 세포주 HCT116 → HEK293T (365그룹) | pca_ridge | 0.1746 | 0.9643 | **0.0429** | **0.2480** |
| | VAE v1 | 0.1626 | 0.9761 | 0.0371 | 0.2676 |
| | VAE v2 | 0.1567 | 0.9730 | 0.0282 | 0.2617 |
| | 앙상블 | **0.1756** | **0.9622** | 0.0376 | 0.2524 |

GWPS 자체 검증 (학습 샘플과 겹치지 않는 GWPS 20,000세포, 863그룹. GWPS 측정 유전자 1,536개를 가리고 채움. 최종 체크포인트 `vae_v1`, `vae_v2`):

| 방법 | Δ Pearson | Δ MSE ratio | gene Pearson | RMSE |
|---|---|---|---|---|
| 평균 (Δ=0) | -0.0091 | 1.0000 | 0.0000 | 0.5338 |
| pca_ridge (이 검증용으로 Orion에서 다시 학습) | 0.1468 | 0.9713 | 0.1307 | 0.5742 |
| VAE v1 | 0.0974 | 1.0194 | 0.0678 | 0.6844 |
| VAE v2 | 0.1905 | 0.9631 | 0.1252 | 0.5474 |
| 앙상블 | **0.1921** | **0.9583** | **0.1485** | **0.5372** |

해석과 한계:

- Orion 두 분할에서 VAE v2 단독은 선형보다 못하다. 앙상블은 Δ Pearson과 Δ MSE ratio가 두 분할 모두 선형보다 조금 낫다. 차이는 Δ Pearson +0.0024와 +0.0011, Δ MSE ratio -0.0008과 -0.0020으로 작고, seed 하나로 학습한 결과라 seed에 따라 뒤집힐 수 있다.
- 처음 보는 세포주 분할에서 앙상블의 gene Pearson과 RMSE는 선형보다 나쁘다.
- GWPS 자체 검증에서 가린 유전자는 GWPS가 측정하는 유전자라, VAE가 다른 GWPS 세포에서 이미 학습한 유전자다. 실제로 채울 유전자(GWPS에서 측정된 적 없음)보다 VAE에 유리한 시험이다. VAE v1의 디코더 문제(Δ MSE ratio > 1)가 v2에서 사라졌는지 확인하는 용도로만 본다.
- GWPS 자체 검증의 pca_ridge는 이 시험의 입력 패널로 다시 학습한 것이고, `linear_v1`과 같지 않다.

### 납품 스크립트 테스트: 이전 모델 (GWPS 1,000,000~1,004,999행, 5,000세포)

`ensemble_v1`, `vae_v2`, `vae_v1`, `linear_v1` 각각 `verify_fill.py --full-load`의 아래 항목을 모두 통과했다 (2026-10-03).

- 출력 shape (5,000, 18,533)
- 측정 유전자 값이 입력과 완전히 같음
- `measured_mask`, `is_measured`, `fill_source`가 서로 일치
- NaN, 음수 없음, dtype float32
- 일반 모드와 `backed='r'` 모드로 열림
- `ensemble_v1`은 `--sample-counts`, `--mask-mode var`도 각각 통과. `--sample-counts` 값은 정수이고 평균이 기대값과 같음(비율 0.999)
- 앙상블 채운 값이 두 백엔드 출력으로 직접 계산한 평균과 같음
- 이전 선형 테스트에서 sparse 입력과 float64 직접 계산 일치(상대 오차 4e-7)도 확인했다

### 검증 결과: K562 (2026-10-04)

평가용 후보는 모두 K562 홀드아웃 타깃을 학습에서 뺐다. 최종 `linear_k562_v1`은 같은 방법(`linear_k562_vip`)을 홀드아웃 없이 다시 학습한 것이다.

평가 세 가지:

- K562 처음 보는 perturbation: VIPerturb 타깃 20%(1,344개)를 학습에서 빼고, 그 세포의 GWPS 측정 유전자만 보여 준 뒤 나머지 10,570개를 채워 VIPerturb 측정값과 비교.
- GWPS 실전 검증: 홀드아웃 타깃 중 GWPS에도 있는 793개 타깃의 실제 GWPS 세포(타깃당 최대 50세포)를 채운 Δ를 같은 타깃의 VIPerturb Δ와 비교. strong은 GWPS 실제 Δ가 큰 상위 10% 타깃(79개).
- GWPS 자체 검증: GWPS 측정 유전자 1,536개를 가리고 채움. 선형 행은 이 패널에 맞춰 같은 학습 데이터로 다시 학습한 pca_ridge.

지표는 타깃별 Δ(같은 데이터 control 기준)의 중앙값이고 괄호는 타깃 부트스트랩 95% 구간이다.

| 후보 | K562 Δ Pearson | K562 Δ MSE ratio | 실전 Δ Pearson | 실전 Δ MSE ratio | 실전 strong Δ Pearson | 실전 strong Δ MSE ratio | 자체 Δ Pearson | 자체 Δ MSE ratio |
|---|---|---|---|---|---|---|---|---|
| `ensemble_v1` (이전 납품) | 0.0450 [0.0434, 0.0463] | 1.0277 | 0.0017 [0.0001, 0.0038] | 1.0063 | 0.0193 [0.0149, 0.0263] | 1.0350 | 0.1921 | 0.9583 |
| `linear_k562_vip` (= `linear_k562_v1` 방법) | **0.2441** [0.2403, 0.2486] | **0.9392** | **0.0155** [0.0111, 0.0245] | **1.0022** | **0.0937** [0.0582, 0.1210] | **0.9984** | 0.1443 | 0.9705 |
| `ens_k562` (`linear_k562_vip` + `vae_v3`) | 0.2450 [0.2391, 0.2506] | 0.9407 | 0.0100 [0.0042, 0.0154] | 1.0039 | 0.0740 [0.0367, 0.0963] | 1.0151 | 0.1909 | 0.9583 |
| 상한선 (같이 측정한 유전자에서 GWPS 실제 Δ) | | | 0.0068 | 1.1399 | 0.0650 | 1.2512 | | |

- `linear_k562_vip` - `ensemble_v1` 짝지은 차이: K562 Δ Pearson +0.199 [0.196, 0.204], Δ MSE ratio −0.089 [−0.091, −0.085]. 실전 strong Δ Pearson +0.074 [0.039, 0.098], Δ MSE ratio −0.037 [−0.054, −0.022].
- 최종 모델을 선형 단독으로 고른 이유: 목적이 GWPS 빈 유전자 채우기이고, 실제 GWPS 세포를 넣은 실전 검증(strong)에서 선형 단독이 가장 좋고 Δ MSE ratio가 유일하게 1 미만이다. 앙상블(`ens_k562`)은 K562 평가에서 선형과 차이가 구간 안이고(Δ Pearson +0.001 [−0.003, +0.003]), 실전 strong에서는 선형보다 나쁘다.
- GWPS 자체 검증에서는 선형 단독이 `ensemble_v1`보다 낮다. 이 시험은 측정된 유전자를 가리는 시험이라 VAE에 유리하고, 실제 채우기와 관련이 약하다. Orion으로 학습한 선형(0.1468 / 0.9713)과는 같은 수준이다.
- VCC2026 타깃 300개만 보면: VIPerturb 샘플에 124개가 있고, K562 평가에 26개, 실전 검증에 23개가 들어갔다. K562 26개에서 `linear_k562_vip`는 26개 모두 `ensemble_v1`보다 낫다 (Δ Pearson 0.2443 대 0.0405, Δ MSE ratio 0.9392 대 1.0284). 실전 23개에서는 Δ MSE ratio 1.0002 대 1.0079 (차이 −0.0076 [−0.0120, −0.0014]), Δ Pearson 0.0142 대 −0.0050 (차이 구간 [−0.021, 0.067]이 0 포함). 타깃 수가 적어 구간이 넓다.

### 타깃별 보정 시험: 패 (2026-10-04)

GWPS 타깃 X 세포를 선형으로 채운 뒤, VIPerturb에서 X를 껐을 때 선형이 놓친 변화(잔차 r_X)를 λ_X = n_X / (n_X + k)만큼 더하면 나아지는지 GWPS 실제 측정값으로 시험했다. 결과는 패이고 납품에 넣지 않는다.

- 시험: GWPS와 VIPerturb가 같이 측정한 7,541 유전자를 입력 6,033개와 가짜 빈칸 P 1,508개로 나눔. 선형은 VIPerturb로만 학습. r_X는 VIPerturb 원본 bin 전체 세포(타깃당 중앙값 44세포)로 계산. GWPS 정답은 예측에 쓰지 않은 세포의 P 실제 Δ. 공통 타깃 9,552개를 반으로 나눠 앞 절반에서 k를 고르고 뒤 절반(4,776개)에서 평가.
- k를 줄여 보정을 키울수록 계속 나빠져 k = ∞(보정 없음)가 골라졌다 (선택용 절반 Δ MSE ratio: k=0 2.84, k=200 1.10, k=∞ 1.053).
- 이유: VIPerturb 잔차와 GWPS에서 선형이 실제로 놓친 부분의 상관이 중앙값 0.008이다. 타깃당 수십 세포의 Δ는 잡음이 커서(GWPS 안에서도 다른 세포 50개의 Δ로 맞히면 0.040 / 3.07) 잔차를 더하면 잡음을 더하는 셈이다. 타깃 유전자 자신의 knockdown은 두 데이터에서 맞지만 유전자 하나라 지표를 거의 바꾸지 않는다.

| 평가용 타깃 | (a) 선형 Δ Pearson / Δ MSE ratio | (c) VIPerturb 실제 Δ만 |
|---|---|---|
| 전체 (4,776) | 0.0500 / 1.0539 | 0.0128 / 2.8878 |
| strong (478) | 0.2623 / 0.9384 | 0.0416 / 1.4379 |
| VCC 300 중 해당 (146) | 0.0478 / 1.0577 | 0.0175 / 2.5843 |

(b) 선형 + λ·r는 k = ∞라 (a)와 같다.

### 납품 스크립트 테스트: `linear_k562_v2` (GWPS 1,000,000~1,004,999행, 5,000세포)

기대값, `--sample-counts`, `--mask-mode var` 각각 `verify_fill.py --full-load`의 모든 항목을 통과했다 (2026-10-04). `--sample-counts` 값은 정수이고 평균이 기대값과 같다(비율 0.999). `fallback` 282개. `gene_quality.csv`를 넣은 뒤 다시 채운 값은 넣기 전과 비트 단위로 같고, `fill_quality` 검증(측정 NaN, fallback 0, 모델 −1~1)도 통과했다.

### 납품 스크립트 테스트: `linear_k562_v1` (GWPS 1,000,000~1,004,999행, 5,000세포)

기대값, `--sample-counts`, `--mask-mode var` 각각 `verify_fill.py --full-load`의 모든 항목을 통과했다 (2026-10-04). `--sample-counts` 값은 정수이고 평균이 기대값과 같다(비율 0.999). 입력 모델 유전자 누락 0개, `fallback` 282개.

### 선형 개선 시험: 학습 세포 수, PCA 성분 수, ridge (2026-10-04)

세포 수(샘플 4만 / 약 20만 / bin 전체 906,837) × PCA 성분 {64, 128, 256, 512} × ridge alpha {0.1, 1, 10} 36개를 GWPS 정답 시험(위 "타깃별 보정 시험"과 같은 분할, 같은 타깃 절반)으로 비교했다. 선택용 절반에서 고르고 평가용 절반에서 성적을 냈다. 선택은 bin 전체 + 256 성분이다 (= `linear_k562_v2`).

| 평가 | 기준선 (v1 설정: 4만, 128) | bin 전체, 256 | 짝지은 차이 Δ Pearson | 짝지은 차이 Δ MSE ratio |
|---|---|---|---|---|
| GWPS 정답 전체 (4,776 타깃) | 0.0504 / 1.0542 | 0.0527 / 1.0531 | +0.0024 [0.0010, 0.0034] | −0.0011 [−0.0021, −0.0001] |
| GWPS 정답 strong (478) | 0.2648 / 0.9384 | 0.2741 / 0.9341 | +0.0093 [0.0002, 0.0208] | −0.0044 [−0.0103, 0.0002] |
| GWPS 정답 VCC 300 해당 (146) | 0.0502 / 1.0583 | 0.0504 / 1.0554 | +0.0002 [−0.0047, 0.0098] | −0.0029 [−0.0083, 0.0042] |
| K562 홀드아웃 (보조) | 0.2453 / 0.9394 | 0.2573 / 0.9336 | +0.0120 [0.0106, 0.0145] | −0.0058 [−0.0069, −0.0047] |

- 값은 Δ Pearson / Δ MSE ratio 중앙값, 괄호는 타깃 부트스트랩 95% 구간이다.
- 세포 수를 늘리면 두 시험 모두 조금 좋아지고, 4만에서 성분 수만 늘리면 나빠진다. ridge alpha 0.1~10은 PCA 점수 고유값(수천 이상)에 비해 작아 효과가 없다.
- 한계: 개선 폭이 작다 (GWPS 정답 전체 Δ Pearson +0.002). strong과 VCC 부분집합에서는 Δ MSE ratio 차이 구간이 0을 포함한다. K562 시험의 세포 단위 유전자 Pearson은 0.042에서 0.031로 낮아진다 (pseudobulk Δ는 좋아지지만 세포 하나하나의 상관은 약해짐).
- 참고: 예측 변화 크기를 s배(채운 값 = control 예측 + s × (예측 − control 예측))로 바꾸면 strong 타깃은 s=0.75에서 Δ MSE ratio가 가장 낮다 (0.926). 기본값은 s=1이고 바꾸지 않는다.
- 입력 쪽 측정 방식 보정 (GWPS 입력을 control 기준 평균, 표준편차, 분위수로 Flex 기준에 맞춤): 애매라 쓰지 않는다. Δ MSE ratio는 낮추지만 Δ Pearson이 소폭 낮아지고 (구간이 0 포함), 예측 크기를 줄이는 효과에 가깝다.

## 유전자별 신뢰도 (`fill_quality`)

채운 값은 바꾸지 않고, 모델이 채운 유전자마다 얼마나 믿을 만한지 점수를 붙인다.

- 의미: 처음 보는 perturbation에서 그 유전자의 perturbation 효과(Δ)를 얼마나 맞히는지. K562 평가 홀드아웃 타깃 1,337개(세포 5개 이상)에 걸친 예측 pseudobulk Δ와 실제 Δ의 Pearson이다 (Δ = 타깃 평균 − control 평균, log 정규화 공간).
- 계산 (`gene_quality.py`): `linear_k562_v2`와 같은 설정이면서 K562 홀드아웃 타깃과 평가 control을 학습에서 뺀 모델(841,521세포)로 VIPerturb 평가 세포 7,983개를 GWPS 패턴 입력으로 채워 계산한다.
- 파일: `checkpoints/linear_k562_v2/gene_quality.csv` (열 `gene_name`, `fill_quality`, `fill_quality_defined`, `cell_r2`, `n_expr_cells`, `n_targets`, `n_eval_cells`). `cell_r2`는 참고용 세포 단위 R²(평가 세포 중 발현 세포 10개 미만이면 NaN).
- 출력: `fill_panel.py`는 체크포인트에 이 파일이 있으면 `var['fill_quality']`를 더한다. 측정 유전자는 NaN(`var['is_measured']`로 구분), fallback 유전자는 0, 모델 유전자는 점수. 채운 값은 파일이 없을 때와 비트 단위로 같다.
- 분포 (모델 유전자 10,570개): 중앙값 0.027, 75% 0.117, 90% 0.200, 95% 0.268, 최대 0.895. 0.1 이상 3,069개, 0.2 이상 1,055개, 0.3 이상 406개.
- 평가 세포에서 거의 발현되지 않아(발현 세포 0~1개) 실제 Δ가 변하지 않는 3,617개는 Pearson을 정할 수 없어 0으로 두고 `fill_quality_defined=False`로 표시한다. 발현 세포가 많을수록 점수가 높다 (발현 세포 10~99개 중앙값 0.065, 100~999개 0.12, 1,000개 이상 0.17).
- 타당성 확인: GWPS 정답 시험의 가짜 빈칸 1,508 유전자에서 같은 방식(K562 홀드아웃)으로 매긴 점수와 GWPS 실제 정답 기준 유전자별 정확도(9,552 타깃에 걸친 예측 Δ와 GWPS 실제 Δ의 Pearson)의 상관이 Pearson 0.58, Spearman 0.51이다. 점수 상위 25% 유전자의 GWPS 정확도 중앙값은 0.206, 하위 25%는 0.081이다. 이 확인은 시간을 줄이려고 샘플 4만 세포, 256 성분 모델 쌍으로 했다 (bin 전체 모델의 GWPS 정확도와는 Pearson 0.55).
- 한계: 정답이 VIPerturb(10x Flex) 측정값이라 Flex 측정 기준 추정치다. GWPS(10x 3′)에서의 실제 정확도와 같지 않고, 위 확인에서도 상관은 0.5~0.6 수준이다. 점수 자체도 타깃당 약 5세포 정답에서 나와 잡음이 있다.
- 쓰는 법: 채운 GWPS로 TxPert 같은 모델을 학습할 때 유전자별 로스 가중치(예: max(fill_quality, 0))로 쓰거나, 점수가 낮은 채운 유전자를 로스에서 빼는 필터(예: 0.1 미만 제외)로 쓸 수 있다. 측정 유전자는 `is_measured`로 따로 다룬다.

## K562 학습 데이터 샘플 (VIPerturb-seq)

GWPS와 같은 K562이면서 18,533 중 18,111개를 측정한 VIPerturb-seq(10x Flex)에서 샘플을 만들었다.

```bash
python gene_panel_fill/make_viperturb_sample.py
```

- 원본: `viperturb-seq/genome_wide_filtered.rds` (Seurat v5, 326,247세포, 타깃 6,724개 + control `NO-TARGET` 7,949세포). R 없이 `rds_stream.py`(브랜치 `origin/data/viperturb-seq`의 `tools/data/rds_stream.py` 복사본)로 두 번 순차로 읽는다.
- 샘플링: 타깃마다 5세포(5개 미만이면 전부), control 6,375세포, seed 0. 39,949세포.
- 출력: `derived/gene-panel-fill/viperturb_k562_sample40k_vccaxis.h5ad` (1.64GB). X는 raw count CSR float32. `var['measured_in_viperturb']` 18,111개. `obs['gene_target']`의 control은 `non-targeting`, 원본 라벨은 `obs['gene']`, `obs['guide']`.
- GWPS 빈 유전자 10,852개 중 10,570개, fallback 291개 중 10개를 측정한다.
- CPU 기기에서 4.1분, 최대 RSS 7.1GB.

## 알려진 한계

`linear_k562_v2` (기본), `linear_k562_v1`:

- 학습 데이터 VIPerturb는 K562지만 10x Flex이고, 채울 GWPS는 10x 3′다. 채움값은 Flex 측정 기준이라 3′ 측정과 유전자별로 차이가 남아 있을 수 있다. GWPS 빈 유전자의 정답은 없다.
- K562 평가의 정답도 VIPerturb(Flex) 측정값이라 VIPerturb로 학습한 모델에 유리하다.
- GWPS 실전 검증은 잡음이 크다. 대부분 타깃의 효과가 작아 전체 타깃 기준 상한선이 Δ Pearson 0.007이다. 그래서 strong 부분집합(79개 타깃)을 함께 본다. VIPerturb 정답은 타깃당 약 5세포라 잡음이 크다.
- GWPS 실제 측정값을 정답으로 한 시험(가짜 빈칸 1,508 유전자)에서 효과가 뚜렷한 타깃은 Δ=0보다 낫지만(Δ Pearson 0.262, Δ MSE ratio 0.938), 전체 타깃 기준 Δ MSE ratio는 1.054로 효과가 약한 타깃에서는 Δ=0보다 약간 못하다.
- 학습은 타깃당 최대 5세포라 타깃별 효과 정보는 적다. 모델은 세포 사이 유전자 공변동을 배운다.
- `fallback` 282개는 VCC2026 control 평균이라 perturbation 효과를 담지 못한다.
- GWPS 유전자 패널에 맞춰 학습했다. 입력에 없는 `O_model` 유전자는 0으로 들어간다 (아래 `linear_v1`과 같은 문제).

이전 모델 (`linear_v1`, `vae_v2`, `ensemble_v1`):

- 학습 데이터가 K562가 아니라 Orion(HCT116, HEK293T)이다. K562 평가에서 Δ MSE ratio가 1을 넘는다 (Δ=0보다 못함).
- GWPS에서 측정된 적 없는 유전자의 채움값은 Orion에서 배운 값(공유 출력)이다. Orion과 GWPS(10x 3′)의 측정 방식 차이가 그 유전자들에서 남아 있을 수 있다.
- `fallback` 291개는 VCC2026 control 비율이라 perturbation 효과를 담지 못한다.
- 앙상블이 선형보다 나은 정도는 작다 (위 해석 참고).
- `linear_v1`(그리고 이를 포함한 `ensemble_v1`)은 GWPS 유전자 패널에 맞춰 학습했다. 다른 패널의 데이터에 쓰면 다음 문제가 있다.
  - 입력에 없는 `O_model` 유전자는 0으로 들어가 선형 예측과 정규화 기준이 틀어진다. 실행 로그에 경고가 나온다.
  - `linear_v1` 단독이면 입력에 없는 GWPS 측정 유전자는 0(`fill_source=none`)으로 남는다.
  - 다른 패널이면 `vae_v2` 단독을 쓰거나, 그 패널로 선형을 다시 학습한다(`train_linear.py --input-genes-h5ad`).
