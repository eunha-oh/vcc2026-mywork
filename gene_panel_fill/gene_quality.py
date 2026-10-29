# 유전자별 채우기 신뢰도(fill_quality) 계산 → gene_quality.csv (채운 값은 바꾸지 않는 부가 정보)
# - --ckpt: K562 평가 홀드아웃 타깃을 학습에서 뺀 선형 체크포인트 (train_linear_stream.py --exclude-k562-holdout)
# - 평가 세포: VIPerturb 샘플의 K562 평가 세포 (eval_common.k562_split: 홀드아웃 타깃 1,344개 + control 1,275세포)
# - 입력: GWPS 패턴 (체크포인트 입력 유전자 O_model만 보여 줌). 정규화: O_model count 합 기준 1e4 + log1p (k562_eval.py와 같음)
# - 모델이 채우는 유전자(M_model)마다
#   fill_quality: 세포 5개 이상인 홀드아웃 타깃에 걸친 예측 pseudobulk Δ와 실제 Δ의 Pearson (Δ = 타깃 평균 − 평가 control 평균)
#   (평가 세포에서 거의 발현되지 않아 실제 Δ가 타깃 사이에 변하지 않는 유전자는 Pearson을 정할 수 없어 0으로 두고 fill_quality_defined=False)
#   cell_r2: 평가 세포 단위 1 − Σ(예측 − 실제)² / Σ(실제 − 평균)² (발현 세포가 --min-expr-cells 미만이면 NaN)
#   n_expr_cells: 평가 세포 중 count > 0인 세포 수
# - 정답이 VIPerturb(10x Flex) 측정값이라 Flex 측정 기준 추정치다
# 실행: python gene_panel_fill/gene_quality.py --ckpt <홀드아웃 제외 체크포인트> --out <납품 체크포인트>/gene_quality.csv
import argparse, json, os, shutil, sys, tempfile
import numpy as np, pandas as pd, anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import load_backend
from eval_common import k562_split, counts_to_log, colwise_pearson

ROOT = os.environ.get('VCC_DATA_ROOT', 'data')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='K562 홀드아웃을 뺀 선형 체크포인트')
    p.add_argument('--viperturb', default=ROOT + '/derived/gene-panel-fill/viperturb_k562_sample40k_vccaxis.h5ad')
    p.add_argument('--min-cells', type=int, default=5)
    p.add_argument('--min-expr-cells', type=int, default=10, help='cell_r2를 계산할 최소 발현 세포 수')
    p.add_argument('--out', required=True, help='gene_quality.csv 경로 (이미 있으면 중단)')
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'이미 존재함, 덮어쓰지 않음: {a.out}')
    with open(os.path.join(a.ckpt, 'meta.json')) as f:
        ex = json.load(f).get('training', {}).get('exclude_k562_holdout')
    if not ex:
        raise SystemExit('체크포인트가 K562 홀드아웃을 뺀 모델이 아님 (meta training.exclude_k562_holdout 없음)')
    be, _ = load_backend(a.ckpt)
    v = ad.read_h5ad(a.viperturb)
    genes = v.var_names.to_numpy().astype(str)
    pos = {g: i for i, g in enumerate(genes)}
    _, te, holdout = k562_split(v.obs)
    te = np.where(te)[0]
    obs = v.obs.iloc[te].reset_index(drop=True)
    X = v.X[te].toarray().astype(np.float32)
    del v
    idx = np.array([pos[g] for g in be.input_genes])
    mg = be.output_genes[be.output_source == 'model']
    midx = np.array([pos[g] for g in mg])
    XO = X[:, idx]
    lib = XO.sum(1)
    pred = be.predict_missing(XO, lib, np.ones(len(idx), bool))[:, be.output_source == 'model']
    P, Y = counts_to_log(pred, lib), counts_to_log(X[:, midx], lib)
    n_expr = (X[:, midx] > 0).sum(0)
    del X, XO, pred
    ctrl = obs['is_control'].to_numpy(bool)
    tg = obs['gene_target'].astype(str).to_numpy()
    names = [t for t, n in pd.Series(tg[~ctrl]).value_counts().items() if n >= a.min_cells]
    dp = np.stack([P[tg == t].mean(0) for t in names]) - P[ctrl].mean(0)
    dt = np.stack([Y[tg == t].mean(0) for t in names]) - Y[ctrl].mean(0)
    fq = colwise_pearson(dp, dt)
    defined = np.isfinite(fq)
    r2 = 1 - ((P - Y) ** 2).sum(0) / np.maximum(((Y - Y.mean(0)) ** 2).sum(0), 1e-12)
    r2[n_expr < a.min_expr_cells] = np.nan
    df = pd.DataFrame({'gene_name': mg, 'fill_quality': np.where(defined, fq, 0).astype(np.float32), 'fill_quality_defined': defined,
                       'cell_r2': r2.astype(np.float32), 'n_expr_cells': n_expr})
    df['n_targets'] = len(names)
    df['n_eval_cells'] = len(obs)
    tmp = tempfile.mkdtemp(prefix='gpf_q_')
    try:
        f = os.path.join(tmp, os.path.basename(a.out))
        df.to_csv(f, index=False)
        shutil.copy(f, a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    fqs = df['fill_quality'].to_numpy()
    pct = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    print(f'저장: {a.out}. 유전자 {len(mg)}, 타깃 {len(names)}, 평가 세포 {len(obs)}, Pearson을 정할 수 없어 0으로 둔 유전자 {int((~defined).sum())}')
    print('fill_quality 분위수 전체 (0,5,10,25,50,75,90,95,100%): ' + ', '.join(f'{x:.3f}' for x in np.percentile(fqs, pct)))
    print('fill_quality 분위수 정해진 유전자만: ' + ', '.join(f'{x:.3f}' for x in np.percentile(fq[defined], pct)))
    print('fill_quality ≥0.1 / ≥0.2 / ≥0.3: ' + ' / '.join(str(int((fqs >= t).sum())) for t in (0.1, 0.2, 0.3)))
    print(f'cell_r2 (발현 세포 {a.min_expr_cells}개 이상 {int(np.isfinite(r2).sum())}개) 분위수 (0,5,25,50,75,95,100%): ' + ', '.join(f'{x:.3f}' for x in np.nanpercentile(r2, [0, 5, 25, 50, 75, 95, 100])))


if __name__ == '__main__':
    main()
