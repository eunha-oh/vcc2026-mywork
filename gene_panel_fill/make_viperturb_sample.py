# VIPerturb-seq K562 Seurat RDS에서 세포를 샘플링해 VCC2026 18,533 유전자 축의 raw count h5ad로 저장
# R 없이 rds_stream.py로 RDS를 두 번 순차로 읽는다. 원본은 읽기만 한다.
# - 1차: 작은 벡터만 읽어 세포 포인터(p), 유전자 이름, meta.data를 얻고 샘플 세포를 고른다
# - 2차: counts의 i와 x를 청크로 읽어 고른 세포의 값만 남긴다
# 실행: python gene_panel_fill/make_viperturb_sample.py [--source ...] [--out-dir ...]
import argparse, json, os, resource, shutil, subprocess, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from rds_stream import RdsStream, keep_small, data_frame, STR

p = argparse.ArgumentParser()
p.add_argument('--root', default=os.environ.get('VCC_DATA_ROOT', 'data'))
p.add_argument('--source', default=None, help='기본: <root>/viperturb-seq/genome_wide_filtered.rds')
p.add_argument('--out-dir', default=None, help='기본: <root>/derived/gene-panel-fill')
p.add_argument('--out-name', default='viperturb_k562_sample40k_vccaxis.h5ad')
p.add_argument('--cells-per-target', type=int, default=5)
p.add_argument('--n-control', type=int, default=6375)
p.add_argument('--control-label', default='NO-TARGET', help='meta.data gene 열의 control 라벨')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

ROOT = args.root
SRC = args.source or ROOT + '/viperturb-seq/genome_wide_filtered.rds'
OUT_DIR = args.out_dir or ROOT + '/derived/gene-panel-fill'
OUT = OUT_DIR + '/' + args.out_name
if os.path.exists(OUT):
    raise SystemExit(f'이미 존재함, 덮어쓰지 않음: {OUT}')
if os.path.realpath(OUT_DIR).startswith(os.path.realpath(os.path.dirname(SRC))):
    raise SystemExit('출력 폴더가 원본 폴더 안에 있음')
t0 = time.time()
rng = np.random.default_rng(args.seed)
log = lambda m: print(f'{time.strftime("%H:%M:%S")} {m} (peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}GB)', flush=True)

# 1차: 구조
root = RdsStream(SRC, keep_small()).parse()
rna = root['assays']['RNA']
if rna.rclass != ['Assay5']:
    raise SystemExit(f'RNA assay가 Assay5가 아님: {rna.rclass}')
cnt = rna['layers']['counts']
COUNTS = ('attr', 'assays', root['assays'].names.index('RNA'), 'attr', 'layers', rna['layers'].names.index('counts'), 'attr')
n_genes, n_cells = (int(v) for v in cnt['Dim'].value)
indptr = cnt['p'].value.astype(np.int64)
nnz = int(indptr[-1])
assert len(indptr) == n_cells + 1 and cnt['i'].length == nnz and cnt['x'].length == nnz


def dimnames(axis, slot):
    # Dimnames가 NULL이면 Assay5의 LogMap에서 읽는다
    node = cnt['Dimnames'][axis]
    if node.rtype != STR:
        node = rna[slot]['dimnames'][0]
    return [str(v) for v in node.value]


genes, cells = dimnames(0, 'features'), dimnames(1, 'cells')
obs_all = data_frame(root['meta.data'])
del root, rna, cnt
assert len(genes) == n_genes and len(set(genes)) == n_genes and list(obs_all.index) == cells
log(f'1차 끝: cells={n_cells:,} genes={n_genes:,} nnz={nnz:,}')

# 샘플 세포: 타깃마다 최대 cells-per-target개, control은 n-control개
lab = obs_all['gene'].astype(str).to_numpy()
is_ctrl = lab == args.control_label
picked = []
for t in sorted(set(lab[~is_ctrl])):
    idx = np.flatnonzero(lab == t)
    picked.append(rng.choice(idx, min(args.cells_per_target, len(idx)), replace=False))
ctrl_idx = np.flatnonzero(is_ctrl)
if len(ctrl_idx) < args.n_control:
    raise SystemExit(f'control 세포가 {len(ctrl_idx)}개뿐')
picked.append(rng.choice(ctrl_idx, args.n_control, replace=False))
sel_idx = np.sort(np.concatenate(picked))
sel = np.zeros(n_cells, bool); sel[sel_idx] = True
n_targets = len(picked) - 1
short = int(sum(len(x) < args.cells_per_target for x in picked[:-1]))
log(f'샘플: {len(sel_idx):,}세포, 타깃 {n_targets:,}개 (세포 {args.cells_per_target}개 미만 타깃 {short}개), control {args.n_control:,}')


def chunk_mask(off, n):
    # nnz 위치 [off, off+n)이 고른 세포에 속하는지
    c0 = np.searchsorted(indptr, off, side='right') - 1
    c1 = np.searchsorted(indptr, off + n - 1, side='right') - 1
    starts = np.maximum(indptr[c0:c1 + 1], off)
    ends = np.minimum(indptr[c0 + 1:c1 + 2], off + n)
    return np.repeat(sel[c0:c1 + 1], ends - starts)


# 2차: 고른 세포의 i, x만 남긴다
kept = {'i': [], 'x': []}
last = [time.time()]


def sink(slot):
    def f(chunk, off):
        m = chunk_mask(off, len(chunk))
        kept[slot].append(chunk[m].astype(np.int32 if slot == 'i' else np.float32))
        if time.time() - last[0] > 60 or off + len(chunk) == nnz:
            last[0] = time.time(); log(f'2차 {slot}: {(off + len(chunk)) / nnz:.0%}')
    return f


sinks = {COUNTS + ('i',): sink('i'), COUNTS + ('x',): sink('x')}
RdsStream(SRC, lambda path, rtype, n: sinks.get(path, 'skip')).parse()
gi = np.concatenate(kept['i']); x = np.concatenate(kept['x']); del kept
lens = (indptr[sel_idx + 1] - indptr[sel_idx]).astype(np.int64)
assert len(gi) == len(x) == lens.sum()
if (x < 0).any() or (x != np.floor(x)).any():
    raise SystemExit('count가 음수가 아닌 정수가 아님')
log(f'2차 끝: 남긴 값 {len(x):,}개')

# 검증: 고른 세포의 전체 유전자 count 합이 meta.data nCount_RNA와 같아야 한다
rows = np.repeat(np.arange(len(sel_idx), dtype=np.int32), lens)
tot = np.bincount(rows, weights=x, minlength=len(sel_idx))
obs = obs_all.iloc[sel_idx].copy()
if not np.array_equal(tot, obs['nCount_RNA'].to_numpy()):
    raise SystemExit('nCount_RNA와 count 합이 다름')

# VCC 18,533 축으로 심볼 매핑. 축에 없는 유전자는 버린다
vcc = pd.read_csv(ROOT + '/vcc2026/gene_names.csv')['gene_name'].tolist()
pos = {g: i for i, g in enumerate(vcc)}
g2col = np.array([pos.get(g, -1) for g in genes], dtype=np.int32)
col = g2col[gi]; del gi
keep = col >= 0
X_indptr = np.zeros(len(sel_idx) + 1, dtype=np.int64)
np.cumsum(np.bincount(rows[keep], minlength=len(sel_idx)), out=X_indptr[1:])
del rows
X = sp.csr_matrix((x[keep], col[keep], X_indptr), shape=(len(sel_idx), len(vcc)))
del x, col, keep
X.sort_indices()
measured = np.zeros(len(vcc), bool); measured[g2col[g2col >= 0]] = True

obs.insert(0, 'cell_barcode', obs.index.astype(str))
obs['gene_target'] = np.where(obs['gene'].astype(str) == args.control_label, 'non-targeting', obs['gene'].astype(str))
obs['is_control'] = obs['gene'].astype(str).to_numpy() == args.control_label
obs['axis_total_counts'] = np.asarray(X.sum(1)).ravel()
obs['cell_line'] = 'K562'
obs['dataset'] = 'viperturb_k562'
obs['source_file'] = os.path.basename(SRC)
for c in ['orig.ident', 'guide', 'gene', 'sample', 'gene_target', 'cell_line', 'dataset', 'source_file']:
    obs[c] = obs[c].astype('category')
obs.index = pd.Index(np.arange(len(obs)).astype(str))
var = pd.DataFrame(index=pd.Index(vcc, name='gene_name'))
var['measured_in_viperturb'] = measured

try:
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                            capture_output=True, text=True).stdout.strip()
except Exception:
    commit = ''
prov = dict(script='gene_panel_fill/make_viperturb_sample.py', code_version=commit, source=SRC,
            source_bytes=os.path.getsize(SRC), source_cells=n_cells, source_genes=n_genes, source_nnz=nnz,
            source_genes_on_axis=int((g2col >= 0).sum()), cells_per_target=args.cells_per_target,
            n_control=args.n_control, control_label=args.control_label, n_targets=n_targets,
            targets_below_cells_per_target=short, seed=args.seed, created=time.strftime('%Y-%m-%dT%H:%M:%S'))
a = ad.AnnData(X=X, obs=obs, var=var)
a.uns['note'] = ('VIPerturb-seq K562 (10x Flex) raw counts on VCC2026 18,533 axis; genes not measured are 0 and '
                 'measured_in_viperturb=False. gene_target control is non-targeting (source label NO-TARGET).')
a.uns['provenance'] = json.dumps(prov)
tmp = '/tmp/' + args.out_name
a.write_h5ad(tmp)
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copy(tmp, OUT); os.remove(tmp)
log(f'saved {OUT} {a.shape} {os.path.getsize(OUT) / 1e9:.2f}GB measured={measured.sum()} '
    f'targets={n_targets} nnz={X.nnz:,} total {(time.time() - t0) / 60:.1f}min')
