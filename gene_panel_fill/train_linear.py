# 선형(L1000 방식) gene panel 채우기 모델 학습
# 입력 유전자(O_model) log 정규화 값 → PCA → Ridge → 결측 유전자(M_model) log 정규화 값
# 학습 데이터에 없는 결측 유전자(M_fallback)는 VCC2026 control 평균(상수)으로 채운다.
# 학습 데이터가 여럿이면 모든 학습 데이터에서 측정된 유전자만 O_model, M_model로 쓴다.
# Ridge는 PCA 점수로 정규방정식을 세포 chunk마다 누적해 푼다 (sklearn Ridge의 cholesky 풀이와 같은 식, 메모리 절약)
# 실행: python gene_panel_fill/train_linear.py --out <체크포인트 폴더> [--train-data a.h5ad b.h5ad] [--split k562_unseen_pert]
import argparse, datetime, json, os, shutil, subprocess, sys, tempfile, time
import h5py, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.decomposition import PCA
try: from anndata.io import read_elem
except ImportError: from anndata.experimental import read_elem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import META_NAME, ARRAYS_NAME
from eval_common import k562_split, k562_train_rows

ROOT = os.environ.get('VCC_DATA_ROOT', 'data')
D = ROOT + '/derived/gene-panel-fill'
TARGET_SUM = 1e4
CHUNK = 10000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-data', '--orion', dest='train_data', nargs='+', default=[D + '/orion_sample40k_vccaxis.h5ad'],
                   help='학습 데이터 (VCC 18,533 축 raw count, var에 measured로 시작하는 측정 mask 열 하나)')
    p.add_argument('--split', default='all', choices=['all', 'k562_unseen_pert'],
                   help='k562_unseen_pert: VIPerturb 분할의 학습 세포만, 다른 데이터셋은 홀드아웃 타깃 세포를 뺌')
    p.add_argument('--viperturb', default=D + '/viperturb_k562_sample40k_vccaxis.h5ad', help='k562 분할 기준 (obs만 읽음)')
    p.add_argument('--vcc-controls', nargs='+',
                   default=[f'{ROOT}/vcc2026/context_{c}.h5ad' for c in 'ABC'],
                   help='fallback 평균 계산용 VCC2026 control h5ad (모든 유전자 측정)')
    p.add_argument('--genes', default=ROOT + '/vcc2026/gene_names.csv', help='출력 유전자 목록 csv (열 gene_name)')
    p.add_argument('--input-genes-h5ad', default=ROOT + '/replogle2022/K562_gwps_raw_singlecell.h5ad',
                   help='입력 유전자 목록의 기준이 될 h5ad (var만 읽음)')
    p.add_argument('--input-gene-col', default='gene_name', help='위 h5ad의 var 심볼 열')
    p.add_argument('--n-components', type=int, default=128)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True, help='체크포인트 폴더 (이미 파일이 있으면 중단)')
    return p.parse_args()


def git_version():
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.check_output(['git', '-C', d, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', d, 'status', '--porcelain', '--', '.'], text=True).strip()
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def log_norm(X, o_idx):
    # 세포마다 O_model 유전자 count 합 기준 1e4 normalize + log1p (sparse 유지)
    X = X.tocsr().astype(np.float32)
    lib = np.asarray(X[:, o_idx].sum(1)).ravel()
    X = (sp.diags((TARGET_SUM / np.maximum(lib, 1)).astype(np.float32)) @ X).tocsr()
    X.data = np.log1p(X.data)
    return X


def main():
    a = parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f'출력 폴더에 이미 파일이 있음, 덮어쓰지 않음: {a.out}')
    t0 = time.time()

    genes = np.asarray(pd.read_csv(a.genes)['gene_name'], dtype=str)  # npz를 pickle 없이 읽도록 고정 길이 문자열
    with h5py.File(a.input_genes_h5ad, 'r') as f:
        in_genes = set(read_elem(f['var'])[a.input_gene_col].astype(str))

    holdout = k562_split(ad.read_h5ad(a.viperturb, backed='r').obs)[2] if a.split == 'k562_unseen_pert' else None
    Xs, cov, n_by_data = [], np.ones(len(genes), bool), []
    for path in a.train_data:
        d = ad.read_h5ad(path)
        if not np.array_equal(d.var_names.to_numpy().astype(str), genes):
            raise SystemExit(f'유전자 축이 유전자 목록과 다름: {path}')
        mcols = [c for c in d.var.columns if c.startswith('measured')]
        if len(mcols) != 1:
            raise SystemExit(f'측정 mask 열이 하나가 아님: {path} {mcols}')
        cov &= d.var[mcols[0]].to_numpy().astype(bool)
        rows = np.arange(d.n_obs) if holdout is None else np.where(k562_train_rows(d.obs, holdout))[0]
        Xs.append(sp.csr_matrix(d.X[rows]).astype(np.float32))
        n_by_data.append(dict(path=path, n_cells=int(len(rows))))  # 경로는 h5ad uns 키로 쓸 수 없어 목록으로
        print(f'학습 데이터: {os.path.basename(path)} {len(rows)}/{d.n_obs} 세포, 측정 열 {mcols[0]}', flush=True)
        del d
    chunks = lambda: (X[s:s + CHUNK] for X in Xs for s in range(0, X.shape[0], CHUNK))  # 세포 순서대로 chunk
    in_inp = np.isin(genes, list(in_genes))
    O = np.where(in_inp & cov)[0]
    M = np.where(~in_inp & cov)[0]
    F = np.where(~in_inp & ~cov)[0]
    print(f'유전자: O_model {len(O)} | M_model {len(M)} | M_fallback {len(F)} '
          f'| 입력에는 있으나 학습 데이터에 없음 {int((in_inp & ~cov).sum())}', flush=True)

    # PCA: 입력 유전자 log 정규화 값 전체로 학습
    n_cells = sum(X.shape[0] for X in Xs)
    XO = np.vstack([log_norm(c, O)[:, O].toarray() for c in chunks()])
    pca = PCA(n_components=a.n_components, svd_solver='randomized', random_state=a.seed).fit(XO)
    Z = pca.transform(XO).astype(np.float32)
    del XO
    # Ridge (절편 있음): (Zcᵀ Zc + αI) W = Zcᵀ Y, Zc = Z - 평균. Y는 chunk마다 만들어 Zcᵀ Y만 누적
    zc = Z.astype(np.float64) - Z.mean(0, dtype=np.float64)
    zty, ysum = np.zeros((Z.shape[1], len(M))), np.zeros(len(M))
    s = 0
    for c in chunks():
        Y = log_norm(c, O)[:, M].toarray().astype(np.float64)
        zty += zc[s:s + len(Y)].T @ Y
        ysum += Y.sum(0)
        s += len(Y)
        del Y
    coef = np.linalg.solve(zc.T @ zc + a.alpha * np.eye(Z.shape[1]), zty)  # (k, M)
    intercept = ysum / n_cells - Z.mean(0, dtype=np.float64) @ coef
    del Xs, Z, zc, zty
    print(f'PCA+Ridge 학습 완료: {n_cells} 세포, 설명 분산 {pca.explained_variance_ratio_.sum():.3f}, '
          f'{time.time() - t0:.0f}s', flush=True)

    # fallback: VCC control을 한 파일씩 읽어 유전자별 log 정규화 값 합만 누적
    fb_sum, fb_n = np.zeros(len(F), np.float64), 0
    for path in a.vcc_controls:
        with h5py.File(path, 'r') as f:
            vg = read_elem(f['var']).index.astype(str).to_numpy()
            if not np.array_equal(vg, genes):
                raise SystemExit(f'유전자 축이 유전자 목록과 다름: {path}')
            Xc = log_norm(read_elem(f['X']), O)
        fb_sum += np.asarray(Xc[:, F].sum(0)).ravel()
        fb_n += Xc.shape[0]
        del Xc
        print(f'fallback 누적: {os.path.basename(path)} (누적 {fb_n} 세포)', flush=True)
    fb_mean = (fb_sum / fb_n).astype(np.float32)

    meta = dict(
        model_type='linear',
        created_at=datetime.datetime.now().isoformat(timespec='seconds'),
        code_version=git_version(),
        n_genes=dict(all=len(genes), O_model=len(O), M_model=len(M), M_fallback=len(F)),
        normalization=dict(target_sum=TARGET_SUM, library='sum of O_model counts per cell', transform='log1p'),
        hyperparameters=dict(n_components=a.n_components, alpha=a.alpha, seed=a.seed),
        pca_explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
        training=dict(train_data=a.train_data, n_cells_by_data=n_by_data, split=a.split,
                      n_holdout_targets=None if holdout is None else int(len(holdout)),
                      n_cells=int(n_cells), vcc_controls=a.vcc_controls, n_control_cells=int(fb_n),
                      genes=a.genes, input_genes_h5ad=a.input_genes_h5ad, input_gene_col=a.input_gene_col,
                      ridge_solver='chunked normal equations (float64)'),
    )
    arrays = dict(
        all_genes=genes, O_model=genes[O], M_model=genes[M], M_fallback=genes[F],
        pca_mean=pca.mean_.astype(np.float32), pca_components=pca.components_.astype(np.float32),
        ridge_coef=coef.T.astype(np.float32), ridge_intercept=intercept.astype(np.float32),
        fallback_mean_log=fb_mean,
    )
    # S3 마운트에 직접 쓰지 않고 로컬 임시 폴더에 쓴 뒤 복사
    tmp = tempfile.mkdtemp(prefix='gpf_ckpt_')
    try:
        np.savez(os.path.join(tmp, ARRAYS_NAME), **arrays)
        with open(os.path.join(tmp, META_NAME), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        os.makedirs(a.out, exist_ok=True)
        for name in (ARRAYS_NAME, META_NAME):
            shutil.copy(os.path.join(tmp, name), os.path.join(a.out, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'저장: {a.out} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
