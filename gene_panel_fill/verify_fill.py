# fill_panel.py 출력 검증: 측정값 보존, mask와 var 일치, NaN과 음수 여부, h5ad 열기
# 실행: python gene_panel_fill/verify_fill.py --input in.h5ad --output out.h5ad [--gene-col gene_name]
import argparse, sys
import numpy as np, anndata as ad, scipy.sparse as sp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gene-col', default=None, help='fill_panel.py에 준 값과 같게')
    p.add_argument('--chunk-size', type=int, default=20000)
    p.add_argument('--expect-genes', type=int, default=None, help='기대 출력 유전자 수 (예: 18533)')
    p.add_argument('--full-load', action='store_true', help='일반 모드(메모리 전체 로드)로도 열어 봄. 작은 파일에만')
    a = p.parse_args()

    res = {}
    if a.full_load:
        full = ad.read_h5ad(a.output)
        res['일반 모드로 열림'] = full.shape == (full.n_obs, full.n_vars)
        del full
    inp = ad.read_h5ad(a.input, backed='r')
    out = ad.read_h5ad(a.output, backed='r')
    res['backed 모드로 열림'] = True
    n, G = out.shape
    res['세포 수 일치'] = n == inp.n_obs
    if a.expect_genes is not None:
        res[f'유전자 수 = {a.expect_genes}'] = G == a.expect_genes
    res['obs 일치'] = out.obs.index.equals(inp.obs.index) and list(out.obs.columns) == list(inp.obs.columns)

    gene_col = a.gene_col or ('gene_name' if 'gene_name' in inp.var.columns else None)
    in_syms = (inp.var[gene_col] if gene_col else inp.var_names).astype(str).to_numpy()
    out_genes = out.var_names.to_numpy().astype(str)
    gpos = {g: i for i, g in enumerate(out_genes)}
    is_meas = out.var['is_measured'].to_numpy().astype(bool)
    src = out.var['fill_source'].astype(str).to_numpy()
    expect_meas = np.zeros(G, bool)
    expect_meas[[gpos[s] for s in in_syms if s in gpos]] = True
    res['is_measured = 입력 유전자'] = np.array_equal(is_meas, expect_meas)
    res['fill_source와 is_measured 일치'] = np.array_equal(src == 'measured', is_meas)
    has_mask = 'measured_mask' in out.layers.keys()
    if 'fill_quality' in out.var.columns:
        fq = out.var['fill_quality'].to_numpy(np.float64)
        res['fill_quality: 측정 유전자 NaN, fallback 0, 모델 유전자 -1~1'] = bool(
            np.isnan(fq[is_meas]).all() and (fq[src == 'fallback'] == 0).all()
            and (np.abs(fq[src == 'model'][np.isfinite(fq[src == 'model'])]) <= 1).all())

    # 측정값 비교용: 입력 열 → 출력 열 (중복 심볼은 합산)
    cols_in = [j for j, s in enumerate(in_syms) if s in gpos]
    M = sp.csr_matrix((np.ones(len(cols_in), np.float32), (cols_in, [gpos[in_syms[j]] for j in cols_in])),
                      shape=(len(in_syms), G))
    meas_idx = np.where(is_meas)[0]
    ok_meas = ok_nan = ok_neg = ok_mask = True
    for s in range(0, n, a.chunk_size):
        e = min(s + a.chunk_size, n)
        xi = inp.X[s:e]
        xi = xi.toarray() if sp.issparse(xi) else np.asarray(xi)
        xo = np.asarray(out.X[s:e])
        ok_meas &= np.array_equal((sp.csr_matrix(xi.astype(np.float32)) @ M).toarray()[:, meas_idx], xo[:, meas_idx])
        ok_nan &= not np.isnan(xo).any()
        ok_neg &= not (xo < 0).any()
        if has_mask:
            mk = np.asarray(out.layers['measured_mask'][s:e])
            ok_mask &= bool((mk == is_meas.astype(np.uint8)[None, :]).all())
    res['측정 유전자 값이 입력과 동일'] = bool(ok_meas)
    res['NaN 없음'] = bool(ok_nan)
    res['음수 없음'] = bool(ok_neg)
    if has_mask:
        res['measured_mask 일치'] = bool(ok_mask)
    res['dtype float32'] = out.X.dtype == np.float32

    print(f'shape {out.shape}, fill_source: ' + str(dict(zip(*np.unique(src, return_counts=True)))))
    for k, v in res.items():
        print(('PASS ' if v else 'FAIL ') + k)
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
