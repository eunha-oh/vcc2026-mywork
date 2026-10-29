# 선형(PCA + Ridge) gene panel 채우기 모델을 VIPerturb-seq 원본 RDS 전체 세포로 학습 (linear_k562_v2)
# train_linear.py와 같은 모델, 같은 체크포인트 형식이고 학습 방법만 다르다.
# - R 없이 rds_stream.py로 Seurat RDS(bin)를 두 번 순차로 읽는다 (1차: 구조와 counts i, 2차: counts x를 세포 묶음으로)
# - 세포 묶음마다 입력 유전자 log 정규화 값 X와 출력 Y의 XᵀX, XᵀY, 합을 float64로 누적한다. 메모리는 세포 수와 무관하다
# - 누적 통계로 정확한 PCA(공분산 고유분해)를 구한다. PCA 점수는 서로 직교라 ridge는 coef = Vᵀ(XᵀY − nμȳᵀ) / (λ + alpha)
# - 유전자 집합: O_model = 입력 유전자(GWPS var) 중 학습 데이터가 측정, M_model = 입력 유전자가 아니면서 학습 데이터가 측정,
#   M_fallback = 입력 유전자도 아니고 학습 데이터도 측정하지 않음 (train_linear.py와 같음)
# - 같은 바코드의 세포는 한 번만 센다 (genome_wide_bin A/B/C의 control 세포는 세 파일에 같이 들어 있음)
# - 세포마다 count 합이 meta.data의 nCount_RNA와 같은지 확인한다
# 실행: python gene_panel_fill/train_linear_stream.py --out <체크포인트 폴더>   (GPU 권장, 약 25분, RAM 약 10GB)
import argparse, datetime, json, os, resource, shutil, subprocess, sys, tempfile, time
import h5py, numpy as np, pandas as pd, scipy.sparse as sp, torch
try: from anndata.io import read_elem
except ImportError: from anndata.experimental import read_elem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import META_NAME, ARRAYS_NAME
from rds_stream import RdsStream, data_frame, STR

ROOT = os.environ.get('VCC_DATA_ROOT', 'data')
TARGET_SUM = 1e4
COUNTS = ('attr', 'assays', 0, 'attr', 'layers', 0, 'attr')  # RNA assay의 counts 층 (1차 읽기에서 확인)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bins', nargs='+', default=[f'{ROOT}/viperturb-seq/genome_wide_bin{b}.RDS' for b in 'ABC'],
                   help='VIPerturb-seq Seurat RDS (Assay5 RNA counts)')
    p.add_argument('--genes', default=ROOT + '/vcc2026/gene_names.csv', help='출력 유전자 목록 csv (열 gene_name)')
    p.add_argument('--input-genes-h5ad', default=ROOT + '/replogle2022/K562_gwps_raw_singlecell.h5ad',
                   help='입력 유전자 목록의 기준이 될 h5ad (var만 읽음)')
    p.add_argument('--input-gene-col', default='gene_name')
    p.add_argument('--vcc-controls', nargs='+', default=[f'{ROOT}/vcc2026/context_{c}.h5ad' for c in 'ABC'],
                   help='fallback 평균 계산용 VCC2026 control h5ad')
    p.add_argument('--n-components', type=int, default=256)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--batch-cells', type=int, default=20000)
    p.add_argument('--device', default='auto', help='누적 계산 장치. auto는 GPU가 있으면 GPU')
    p.add_argument('--max-batches', type=int, default=None, help='동작 확인용: bin마다 이 묶음 수만 처리')
    p.add_argument('--exclude-k562-holdout', default=None, metavar='VIPERTURB_SAMPLE_H5AD',
                   help='K562 평가 홀드아웃 타깃 세포와 평가 control 세포를 학습에서 뺀다 (gene_quality.py용 모델). '
                        '분할은 이 샘플의 eval_common.k562_split')
    p.add_argument('--out', required=True, help='체크포인트 폴더 (이미 파일이 있으면 중단)')
    return p.parse_args()


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg} (peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}GB)', flush=True)


def git_version():
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.check_output(['git', '-C', d, 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(['git', '-C', d, 'status', '--porcelain', '--', '.'], text=True).strip()
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


class Stop(Exception):
    pass


def read_bin(path, batch_cells, handle, max_batches=None):
    # RDS 하나를 두 번 순차로 읽고, 세포 묶음마다 handle(유전자 이름, obs, c0, c1, 행 번호, 유전자 번호, count)를 부른다
    buf = {}

    def pol1(p, rtype, n):
        if p == COUNTS + ('i',):
            buf['i'] = np.empty(n, np.uint16)

            def sink(chunk, off):
                buf['i'][off:off + len(chunk)] = chunk
            return sink
        return 'keep' if n <= 2_000_000 else 'skip'

    root = RdsStream(path, pol1).parse()
    rna = root['assays']['RNA']
    if root['assays'].names[0] != 'RNA' or rna['layers'].names[0] != 'counts' or rna['layers']['counts'].rclass != ['dgCMatrix']:
        raise SystemExit(f'{path}: RNA counts 위치가 예상과 다름')
    cnt = rna['layers']['counts']
    indptr = cnt['p'].value.astype(np.int64)
    names = []
    for axis, slot in ((0, 'features'), (1, 'cells')):
        node = cnt['Dimnames'][axis]
        names.append([str(x) for x in (node.value if node.rtype == STR else rna[slot]['dimnames'][0].value)])
    vgenes, cells = names
    obs = data_frame(root['meta.data'])
    del root, rna, cnt
    if list(obs.index) != cells or len(buf['i']) != indptr[-1]:
        raise SystemExit(f'{path}: meta.data와 counts가 맞지 않음')
    nnz, n_cells = int(indptr[-1]), len(cells)
    st = dict(start=0, buf=np.zeros(0, np.float32), off=0, nb=0)

    def sink_x(chunk, off):
        st['buf'] = np.concatenate([st['buf'], chunk.astype(np.float32)])
        end = off + len(chunk)
        c_end = np.searchsorted(indptr, end, side='right') - 1
        if c_end - st['start'] >= batch_cells or end == nnz:
            c0, lo, hi = st['start'], indptr[st['start']], indptr[c_end]
            rows = np.repeat(np.arange(c_end - c0), np.diff(indptr[c0:c_end + 1]))
            handle(vgenes, obs, c0, c_end, rows, buf['i'][lo:hi].astype(np.int64), st['buf'][lo - st['off']:hi - st['off']])
            st['buf'] = st['buf'][hi - st['off']:]; st['off'] = hi; st['start'] = c_end; st['nb'] += 1
            if max_batches and st['nb'] >= max_batches:
                raise Stop()

    try:
        RdsStream(path, lambda p, rtype, n: sink_x if p == COUNTS + ('x',) else 'skip').parse()
    except Stop:
        return False
    if st['start'] != n_cells:
        raise SystemExit(f'{path}: 처리한 세포 수가 다름 {st["start"]} / {n_cells}')
    return True


def main():
    a = parse_args()
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f'출력 폴더에 이미 파일이 있음, 덮어쓰지 않음: {a.out}')
    t0 = time.time()
    code_version = git_version()  # 시작 시점 (학습 중 다른 파일을 고쳐도 실행한 코드 기준)
    dev = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    genes = np.asarray(pd.read_csv(a.genes)['gene_name'], dtype=str)
    pos = {g: i for i, g in enumerate(genes)}
    with h5py.File(a.input_genes_h5ad, 'r') as f:
        in_inp = np.isin(genes, list(set(read_elem(f['var'])[a.input_gene_col].astype(str))))

    holdout, test_ctrl = set(), set()
    if a.exclude_k562_holdout:
        import anndata as ad
        from eval_common import k562_split
        vo = ad.read_h5ad(a.exclude_k562_holdout, backed='r').obs
        _, te, ho = k562_split(vo)
        holdout = set(ho)
        test_ctrl = set(vo['cell_barcode'][te & vo['is_control'].to_numpy(bool)].astype(str))
        log(f'K562 홀드아웃 제외: 타깃 {len(holdout)}개, 평가 control {len(test_ctrl)}세포')
    state = {}
    seen = set()
    counts = []

    def setup(vgenes):
        # 첫 bin의 유전자로 집합을 정한다. 다른 bin도 같은 유전자 목록이어야 한다
        g2v = np.array([pos.get(g, -1) for g in vgenes])
        cov = np.zeros(len(genes), bool); cov[g2v[g2v >= 0]] = True
        O, M, F = np.where(in_inp & cov)[0], np.where(~in_inp & cov)[0], np.where(~in_inp & ~cov)[0]  # train_linear.py와 같은 정의
        iO, iM = np.full(len(genes), -1), np.full(len(genes), -1)
        iO[O], iM[M] = np.arange(len(O)), np.arange(len(M))
        colO = np.where(g2v >= 0, iO[np.maximum(g2v, 0)], -1); colM = np.where(g2v >= 0, iM[np.maximum(g2v, 0)], -1)
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        state.update(vgenes=list(vgenes), O=O, M=M, F=F, colO=colO, colM=colM, n=0,
                     xs=z(len(O)), ys=z(len(M)), xtx=z(len(O), len(O)), xty=z(len(O), len(M)))
        log(f'유전자: O_model {len(O)} | M_model {len(M)} | M_fallback {len(F)} | 입력에는 있으나 학습 데이터에 없음 {int((in_inp & ~cov).sum())}')

    def handle(vgenes, obs, c0, c1, rows, gi, x):
        if not state:
            setup(vgenes)
        elif vgenes != state['vgenes']:
            raise SystemExit('bin 사이 유전자 목록이 다름')
        tot = np.bincount(rows, weights=x, minlength=c1 - c0)
        if (tot != obs['nCount_RNA'].to_numpy()[c0:c1]).any():
            raise SystemExit('count 합이 nCount_RNA와 다름')
        bc = obs.index[c0:c1].astype(str)
        keep = ~bc.isin(seen)
        seen.update(bc[keep])
        counts[-1]['duplicates_skipped'] += int((~keep).sum())
        if a.exclude_k562_holdout:
            ex = keep & (obs['gene'].astype(str).iloc[c0:c1].isin(holdout).to_numpy() | bc.isin(test_ctrl))
            counts[-1]['k562_holdout_excluded'] = counts[-1].get('k562_holdout_excluded', 0) + int(ex.sum())
            keep &= ~ex
        counts[-1]['cells'] += int(keep.sum())
        mO, mM = state['colO'][gi] >= 0, state['colM'][gi] >= 0
        XO = torch.from_numpy(sp.csr_matrix((x[mO], (rows[mO], state['colO'][gi[mO]])), shape=(c1 - c0, len(state['O']))).toarray()[keep]).to(dev)
        XM = torch.from_numpy(sp.csr_matrix((x[mM], (rows[mM], state['colM'][gi[mM]])), shape=(c1 - c0, len(state['M']))).toarray()[keep]).to(dev)
        scale = (TARGET_SUM / XO.sum(1).clamp(min=1))[:, None]
        X, Y = torch.log1p(XO * scale), torch.log1p(XM * scale)
        state['n'] += len(X)
        state['xs'] += X.sum(0, dtype=torch.float64); state['ys'] += Y.sum(0, dtype=torch.float64)
        state['xtx'] += (X.T @ X).double(); state['xty'] += (X.T @ Y).double()

    for path in a.bins:
        tb = time.time()
        counts.append(dict(file=path, cells=0, duplicates_skipped=0))
        done = read_bin(path, a.batch_cells, handle, a.max_batches)
        log(f'{os.path.basename(path)}: {counts[-1]} {"" if done else "(--max-batches로 일부만)"} ({time.time() - tb:.0f}s)')

    # 정확한 PCA + ridge
    n = state['n']
    mu, yb = state['xs'] / n, state['ys'] / n
    C = state['xtx'] - n * torch.outer(mu, mu)
    lam, V = torch.linalg.eigh(C)
    k = a.n_components
    lam, V = lam.flip(0)[:k], V.flip(1)[:, :k]
    coef = (V.T @ (state['xty'] - n * torch.outer(mu, yb))) / (lam + a.alpha)[:, None]  # (k, M)
    explained = float(lam.sum() / torch.trace(C))
    O, M, F = state['O'], state['M'], state['F']
    log(f'PCA+Ridge 학습 완료: {n} 세포, 성분 {k}, 설명 분산 {explained:.3f}')

    # fallback: train_linear.py와 같게 VCC control의 O_model 합 기준 log 정규화 평균
    fb_sum, fb_n = np.zeros(len(F), np.float64), 0
    for path in a.vcc_controls:
        with h5py.File(path, 'r') as f:
            if not np.array_equal(read_elem(f['var']).index.astype(str).to_numpy(), genes):
                raise SystemExit(f'유전자 축이 유전자 목록과 다름: {path}')
            Xc = sp.csr_matrix(read_elem(f['X'])).astype(np.float32)
        lib = np.asarray(Xc[:, O].sum(1)).ravel()
        Xc = (sp.diags((TARGET_SUM / np.maximum(lib, 1)).astype(np.float32)) @ Xc).tocsr()
        Xc.data = np.log1p(Xc.data)
        fb_sum += np.asarray(Xc[:, F].sum(0)).ravel(); fb_n += Xc.shape[0]
        del Xc
    fb_mean = (fb_sum / fb_n).astype(np.float32)

    meta = dict(
        model_type='linear',
        created_at=datetime.datetime.now().isoformat(timespec='seconds'),
        code_version=code_version,
        n_genes=dict(all=len(genes), O_model=len(O), M_model=len(M), M_fallback=len(F)),
        normalization=dict(target_sum=TARGET_SUM, library='sum of O_model counts per cell', transform='log1p'),
        hyperparameters=dict(n_components=k, alpha=a.alpha),
        pca_explained_variance_ratio_sum=explained,
        training=dict(script='gene_panel_fill/train_linear_stream.py', bins=a.bins, cells_by_file=counts, n_cells=int(n),
                      max_batches=a.max_batches, exclude_k562_holdout=a.exclude_k562_holdout, vcc_controls=a.vcc_controls, n_control_cells=int(fb_n), genes=a.genes,
                      input_genes_h5ad=a.input_genes_h5ad, input_gene_col=a.input_gene_col,
                      solver='exact covariance PCA (eigh, float64) + ridge on PCA scores'),
    )
    arrays = dict(all_genes=genes, O_model=genes[O], M_model=genes[M], M_fallback=genes[F],
                  pca_mean=mu.float().cpu().numpy(), pca_components=V.T.float().cpu().numpy(),
                  ridge_coef=coef.T.float().cpu().numpy(), ridge_intercept=yb.float().cpu().numpy(), fallback_mean_log=fb_mean)
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
    log(f'저장: {a.out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
