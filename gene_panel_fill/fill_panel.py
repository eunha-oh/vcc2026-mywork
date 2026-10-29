# 일부 유전자만 측정된 h5ad를 받아 전체 유전자 축(기본 VCC2026 18,533)으로 채운다.
# 측정된 유전자는 입력 값을 그대로 복사하고, 나머지는 체크포인트 백엔드(linear, vae, ensemble)로 채운다.
# 실행: python gene_panel_fill/fill_panel.py --input in.h5ad --output out.h5ad --ckpt <체크포인트 폴더>
import argparse, datetime, json, os, sys, time
import h5py, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
try: from anndata.io import write_elem
except ImportError: from anndata.experimental import write_elem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import load_backend, QUALITY_NAME


def parse_args():
    p = argparse.ArgumentParser(description='gene panel 채우기')
    p.add_argument('--input', required=True, help='세포 x 측정 유전자 h5ad (raw count)')
    p.add_argument('--output', required=True, help='출력 h5ad (이미 있으면 중단)')
    p.add_argument('--ckpt', required=True, help='체크포인트 폴더 (meta.json, model.npz)')
    p.add_argument('--genes', default=None, help='출력 유전자 목록 csv (열 gene_name). 기본: 체크포인트의 목록')
    p.add_argument('--gene-col', default=None, help='입력 var의 유전자 심볼 열. 기본: gene_name 열이 있으면 그것, 없으면 var_names')
    p.add_argument('--chunk-size', type=int, default=20000)
    p.add_argument('--sample-counts', action='store_true',
                   help='채운 값을 샘플링한 정수 count로 저장 (linear: Poisson, vae: 음이항, ensemble: 평균을 기대값으로 vae θ의 음이항)')
    p.add_argument('--seed', type=int, default=0, help='--sample-counts 난수 시드')
    p.add_argument('--device', default='auto', help='vae 백엔드(ensemble의 vae 포함) 계산 장치: auto(GPU가 있으면 GPU), cpu, cuda')
    p.add_argument('--enc-dataset', default=None, help='vae 인코더 데이터셋 임베딩 (기본: 체크포인트 fill_defaults)')
    p.add_argument('--dec-dataset', default=None, help='vae 디코더 데이터셋 임베딩 (기본: 체크포인트 fill_defaults. v2 체크포인트는 입력 데이터셋으로 고정)')
    p.add_argument('--batch-size', type=int, default=2048, help='vae 백엔드 미니배치 크기')
    p.add_argument('--mask-mode', choices=['layer', 'var'], default='layer',
                   help="layer: layers['measured_mask'] 저장, var: var['is_measured']만 저장")
    return p.parse_args()


def build_mapping(in_syms, out_genes, model_inputs):
    # 입력 열 → 고유 심볼 묶음. 출력 축이나 모델 입력에 쓰이는 심볼만 남긴다.
    keep_set = set(out_genes) | set(model_inputs)
    groups = {}
    for j, s in enumerate(in_syms):
        if s in keep_set:
            groups.setdefault(s, []).append(j)
    syms = np.array(list(groups), dtype=object)
    primary = np.array([g[0] for g in groups.values()], dtype=np.int64)
    extras = [(k, g[1:]) for k, g in enumerate(groups.values()) if len(g) > 1]
    return syms, primary, extras


def write_attrs(obj, enc_type, enc_ver):
    obj.attrs['encoding-type'] = enc_type
    obj.attrs['encoding-version'] = enc_ver


def to_h5_safe(x):
    # uns에 넣을 값에서 None 제거 (h5ad로 저장 불가)
    if isinstance(x, dict):
        return {k: to_h5_safe(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        if any(isinstance(v, dict) for v in x):  # dict의 list는 h5ad에 못 씀 → {'0': ..., '1': ...}
            return {str(i): to_h5_safe(v) for i, v in enumerate(x)}
        return [to_h5_safe(v) for v in x]
    return x


def main():
    a = parse_args()
    t0 = time.time()
    inp, out = os.path.realpath(a.input), os.path.realpath(a.output)
    if inp == out:
        raise SystemExit('출력 경로가 입력과 같음')
    if os.path.exists(out):
        raise SystemExit(f'출력 파일이 이미 있음, 덮어쓰지 않음: {out}')

    backend, meta = load_backend(a.ckpt, seed=a.seed, device=a.device, enc_dataset=a.enc_dataset,
                                 dec_dataset=a.dec_dataset, batch_size=a.batch_size)
    print(f'백엔드: {meta["model_type"]} {backend.describe()}', flush=True)
    out_genes = (pd.read_csv(a.genes)['gene_name'].astype(str).to_numpy() if a.genes else backend.all_genes)
    if len(set(out_genes)) != len(out_genes):
        raise SystemExit('출력 유전자 목록에 중복이 있음')
    G = len(out_genes)
    gpos = {g: i for i, g in enumerate(out_genes)}

    adata = ad.read_h5ad(a.input, backed='r')
    n = adata.n_obs
    gene_col = a.gene_col or ('gene_name' if 'gene_name' in adata.var.columns else None)
    in_syms = (adata.var[gene_col] if gene_col else adata.var_names).astype(str).to_numpy()
    print(f'입력: {n} 세포 x {len(in_syms)} 유전자, 심볼 열: {gene_col or "var_names"}', flush=True)

    syms, primary, extras = build_mapping(in_syms, out_genes, backend.input_genes)
    n_dropped = int((~np.isin(in_syms, out_genes)).sum())
    print(f'입력 유전자 중 출력 축에 없어 버림: {n_dropped}개', flush=True)
    if extras:
        print(f'중복 심볼 {len(extras)}개 합산: {[syms[k] for k, _ in extras]}', flush=True)

    # 측정 유전자 → 출력 열
    spos = {s: k for k, s in enumerate(syms)}
    meas_k = np.array([k for k, s in enumerate(syms) if s in gpos], dtype=np.int64)
    meas_out = np.array([gpos[syms[k]] for k in meas_k], dtype=np.int64)
    # 모델 입력 (입력에 없으면 0 열)
    o_k = np.array([spos.get(g, -1) for g in backend.input_genes], dtype=np.int64)
    present = o_k >= 0
    n_o_missing = int((~present).sum())
    if n_o_missing and backend.requires_all_inputs:
        print(f'경고: 모델 입력 유전자 {n_o_missing}/{len(o_k)}개가 입력에 없어 0으로 둠', flush=True)
    # 모델 출력 → 출력 열 (측정된 유전자는 측정값 우선)
    is_measured = np.zeros(G, bool)
    is_measured[meas_out] = True
    fill_source = np.full(G, 'none', dtype=object)
    fill_source[is_measured] = 'measured'
    pred_sel, pred_out = [], []
    for j, (g, src) in enumerate(zip(backend.output_genes, backend.output_source)):
        i = gpos.get(g)
        if i is not None and not is_measured[i]:
            pred_sel.append(j); pred_out.append(i); fill_source[i] = src
    pred_sel, pred_out = np.array(pred_sel, np.int64), np.array(pred_out, np.int64)
    n_none = int((fill_source == 'none').sum())
    if n_none:
        print(f'경고: 측정되지도 모델 출력에도 없는 유전자 {n_none}개는 0으로 둠 (fill_source=none)', flush=True)

    # 출력 파일: X와 mask를 미리 만들고 chunk 단위로 채운다
    f = h5py.File(out, 'w')
    try:
        write_attrs(f, 'anndata', '0.1.0')
        dX = f.create_dataset('X', shape=(n, G), dtype='float32')
        write_attrs(dX, 'array', '0.2.0')
        lay = f.create_group('layers')
        write_attrs(lay, 'dict', '0.1.0')
        if a.mask_mode == 'layer':
            dM = lay.create_dataset('measured_mask', shape=(n, G), dtype='uint8')
            write_attrs(dM, 'array', '0.2.0')
            mask_row = is_measured.astype(np.uint8)[None, :]

        n_chunks = (n + a.chunk_size - 1) // a.chunk_size
        warned = False
        for ci, s in enumerate(range(0, n, a.chunk_size)):
            tc = time.time()
            e = min(s + a.chunk_size, n)
            chunk = adata.X[s:e]
            chunk = chunk.toarray() if sp.issparse(chunk) else np.asarray(chunk)
            chunk = chunk.astype(np.float32, copy=False)
            if not warned and (chunk.min() < 0 or not np.all(chunk == np.round(chunk))):
                print('경고: 입력에 음수 또는 정수가 아닌 값이 있음 (raw count 입력을 가정함)', flush=True)
                warned = True
            meas = chunk[:, primary]
            for k, cols in extras:
                meas[:, k] += chunk[:, cols].sum(1)
            del chunk
            X_O = np.zeros((e - s, len(o_k)), np.float32)
            X_O[:, present] = meas[:, o_k[present]]
            lib = X_O.sum(1)
            pred = backend.predict_missing(X_O, lib, present)
            del X_O
            pred = pred[:, pred_sel]
            if a.sample_counts:
                for r in range(0, e - s, 2000):
                    pred[r:r + 2000] = backend.sample(pred[r:r + 2000], pred_sel)
            block = np.zeros((e - s, G), np.float32)
            block[:, pred_out] = pred
            del pred
            block[:, meas_out] = meas[:, meas_k]  # 측정값은 그대로 (마지막에 덮어써서 우선)
            del meas
            dX[s:e] = block
            del block
            if a.mask_mode == 'layer':
                dM[s:e] = np.broadcast_to(mask_row, (e - s, G))
            el = time.time() - t0
            print(f'[chunk {ci + 1}/{n_chunks}] 세포 {s}-{e} {time.time() - tc:.1f}s, 경과 {el:.0f}s', flush=True)

        var = pd.DataFrame({'is_measured': is_measured, 'fill_source': pd.Categorical(fill_source)},
                           index=pd.Index(out_genes, name='gene_name'))
        # 체크포인트에 유전자별 신뢰도(gene_quality.csv)가 있으면 var['fill_quality']: 모델 유전자는 점수, fallback은 0, 측정 유전자는 NaN
        qpath = os.path.join(a.ckpt, QUALITY_NAME)
        n_quality = None
        if os.path.exists(qpath):
            q = pd.read_csv(qpath).set_index('gene_name')['fill_quality']
            fq = np.full(G, np.nan, np.float32)
            fq[fill_source == 'fallback'] = 0
            mdl = fill_source == 'model'
            fq[mdl] = q.reindex(out_genes[mdl]).to_numpy(np.float32)
            n_quality = int(np.isfinite(fq[mdl]).sum())
            if n_quality < mdl.sum():
                print(f'경고: 점수가 없는 모델 유전자 {int(mdl.sum()) - n_quality}개는 fill_quality NaN', flush=True)
            var['fill_quality'] = fq
        obs = adata.obs.copy()
        summary = dict(n_measured=int(is_measured.sum()), n_model=int((fill_source == 'model').sum()),
                       n_fallback=int((fill_source == 'fallback').sum()), n_none=n_none,
                       n_input_genes_dropped=n_dropped, n_duplicate_symbols_summed=len(extras),
                       n_model_inputs_missing=n_o_missing,
                       **({} if n_quality is None else dict(n_model_with_fill_quality=n_quality)))
        uns = dict(gene_panel_fill=to_h5_safe(dict(
            checkpoint=os.path.realpath(a.ckpt), checkpoint_meta=meta,
            options=dict(input=inp, genes=a.genes, gene_col=gene_col or 'var_names', chunk_size=a.chunk_size,
                         sample_counts=a.sample_counts, seed=a.seed, mask_mode=a.mask_mode, **backend.describe()),
            values='measured: input as-is; filled: ' + ('sampled counts' if a.sample_counts else 'expected counts'),
            summary=summary, run_at=datetime.datetime.now().isoformat(timespec='seconds'))))
        write_elem(f, 'obs', obs)
        write_elem(f, 'var', var)
        write_elem(f, 'uns', uns)
        for k in ('obsm', 'varm', 'obsp', 'varp'):
            write_elem(f, k, {})
    except BaseException:
        f.close()
        os.remove(out)  # 이 실행이 만든 미완성 파일만 지움
        raise
    f.close()
    adata.file.close()

    print(f'완료: {out} ({n} x {G}), {time.time() - t0:.0f}s', flush=True)
    print('요약: ' + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
