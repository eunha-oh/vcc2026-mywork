"""H1 (adata_Training.h5ad)에서 채점기 방식의 실제 DE 라벨을 만들고 근사 라벨과 비교한다.

채점기 방식: control 평균 CPM이 5 초과인 유전자만 검정, 표적 대 non-targeting Wilcoxon(Mann-Whitney U,
정규 근사, 연속성 보정, 동점 보정), 표적별 BH FDR < 0.05. 값은 세포별 log1p(CP10K)를 쓴다.
(채점기가 쓰는 정규화가 이것과 같은지는 확인하지 못한 가정이다.)

h5ad는 backed가 아니라 h5py로 행 구간을 순서대로 한 번만 읽고, control과 선택한 표적의 세포만 남긴다.
15.5GB 전체는 메모리에 올리지 않는다. 산출물은 --out 아래에만 쓴다.

--cells-per-target N: 표적마다 세포 N개를 무작위로 뽑아(N 미만이면 전부) 실제 라벨을 만든다. 대회 정답이 표적당
400세포라서 400세포 환산 근사 라벨(z400)과 비교할 때 쓴다. control은 전부 쓴다.

메모리: control을 CSR 조각 목록으로만 들고(vstack 복사 없음), 유전자를 블록으로 나눠 그 블록만 CSC로 바꿔 계산한다.

예: python tools/de_gate/h1_wilcoxon_check.py --n-targets 20 --cells-per-target 400 --out /tmp/de-gate-eunha/h1_400
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from de_common import CPM_MIN, approx_labels, labels_from_pvalues  # noqa: E402

H5AD = os.environ.get("VCC_DATA_ROOT", "data") + "/adata_Training.h5ad"
STORE = os.environ.get("VCC_DATA_ROOT", "data") + "/derived/context-pipeline/response_store"
PROC = psutil.Process()
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s rss={PROC.memory_info().rss / 2**30:5.2f}GB] {msg}", flush=True)


def stream_cells(path, picked_rows, ctrl_code, codes, n_genes, chunk_rows=10000):
    """선택한 행(표적별로 뽑은 세포와 control 전부)만 log1p(CP10K) CSR 조각으로 모은다.

    돌려주는 값: (control CSR 조각 목록, {코드: 표적 CSR 조각 목록}, control의 유전자별 평균 CP10K)
    """
    wanted = np.zeros(len(codes), dtype=bool)
    wanted[codes == ctrl_code] = True
    for rows in picked_rows.values():
        wanted[rows] = True
    ctrl_parts, target_parts = [], {c: [] for c in picked_rows}
    ctrl_cp10k_sum, ctrl_n = np.zeros(n_genes, dtype=np.float64), 0
    with h5py.File(path, "r") as f:
        indptr = f["X"]["indptr"][:]
        n_rows = len(indptr) - 1
        for a in range(0, n_rows, chunk_rows):
            b = min(a + chunk_rows, n_rows)
            if not wanted[a:b].any():
                continue
            lo, hi = int(indptr[a]), int(indptr[b])
            data = f["X"]["data"][lo:hi].astype(np.float32)
            indices = f["X"]["indices"][lo:hi]
            ptr = (indptr[a:b + 1] - lo).astype(np.int64)
            lengths = np.diff(ptr)
            totals = np.add.reduceat(data, ptr[:-1]) if len(data) else np.zeros(b - a)
            totals = np.where(lengths > 0, totals, 0.0)
            scale = np.where(totals > 0, 1e4 / np.maximum(totals, 1e-12), 0.0).astype(np.float32)
            data = np.log1p(data * np.repeat(scale, lengths)).astype(np.float32)
            block = sp.csr_matrix((data, indices, ptr), shape=(b - a, n_genes))
            local = codes[a:b]
            ctrl_rows = np.flatnonzero(local == ctrl_code)
            if len(ctrl_rows):
                sub = block[ctrl_rows]
                ctrl_parts.append(sub)
                ctrl_cp10k_sum += np.asarray(sub.expm1().sum(axis=0)).ravel()
                ctrl_n += sub.shape[0]
            for code, rows in picked_rows.items():
                inside = rows[(rows >= a) & (rows < b)] - a
                if len(inside):
                    target_parts[code].append(block[inside])
            del data, indices, block
            if (a // chunk_rows) % 4 == 0:
                log(f"행 {b}/{n_rows} 읽음")
    return ctrl_parts, {c: sp.vstack(p, format="csr") for c, p in target_parts.items()}, ctrl_cp10k_sum / ctrl_n


def gene_block_csc(parts, lo, hi, sort=False):
    """CSR 조각 목록에서 유전자 [lo, hi) 열만 모아 CSC로 만든다. sort면 열마다 값을 오름차순으로 정렬한다."""
    csc = sp.vstack([part[:, lo:hi] for part in parts], format="csr").tocsc()
    if sort:
        for j in range(csc.shape[1]):
            csc.data[csc.indptr[j]:csc.indptr[j + 1]].sort()
    return csc


def wilcoxon_all(ctrl_csc, targets_csc, gene_ids, n_ctrl):
    """gene_ids 유전자에 대해 표적마다 (U - nm/2, 동점 보정 T, n)을 계산한다. 희소 값만 비교하고 0은 개수로 센다."""
    nt = len(targets_csc)
    dev = np.zeros((nt, len(gene_ids)))
    ties = np.zeros((nt, len(gene_ids)))
    for k, j in enumerate(gene_ids):
        cs, ce = ctrl_csc.indptr[j], ctrl_csc.indptr[j + 1]
        cnz = ctrl_csc.data[cs:ce]
        cz = n_ctrl - len(cnz)
        _, cc = np.unique(cnz, return_counts=True)
        sc = float((cc.astype(np.float64) ** 3 - cc).sum()) + cz ** 3 - cz
        for t, (csc, n) in enumerate(targets_csc):
            v = csc.data[csc.indptr[j]:csc.indptr[j + 1]]
            tz = n - len(v)
            left = np.searchsorted(cnz, v, "left")
            right = np.searchsorted(cnz, v, "right")
            u = float((cz + left + 0.5 * (right - left)).sum()) + 0.5 * tz * cz
            dev[t, k] = u - 0.5 * n * n_ctrl
            uv, tt = np.unique(v, return_counts=True)
            tc = np.searchsorted(cnz, uv, "right") - np.searchsorted(cnz, uv, "left")
            tot = (tc + tt).astype(np.float64)
            extra = float((tot ** 3 - tot - (tc.astype(np.float64) ** 3 - tc)).sum())
            zt = cz + tz
            extra += float(zt ** 3 - zt - (cz ** 3 - cz))
            ties[t, k] = sc + extra
    return dev, ties


def wilcoxon_blocks(ctrl_parts, target_csrs, n_ctrl, tested_gene, block=1500):
    """검정 대상 유전자마다 표적별 (U - nm/2, 동점 보정 T)를 계산한다. 유전자 블록 단위로 CSC를 만들었다가 버린다."""
    n_genes = len(tested_gene)
    nt = len(target_csrs)
    dev, ties = np.zeros((nt, n_genes)), np.zeros((nt, n_genes))
    for lo in range(0, n_genes, block):
        hi = min(lo + block, n_genes)
        todo = np.flatnonzero(tested_gene[lo:hi])
        if not len(todo):
            continue
        ctrl_csc = gene_block_csc(ctrl_parts, lo, hi, sort=True)
        targets = [(t[:, lo:hi].tocsc(), t.shape[0]) for t in target_csrs]
        d, t = wilcoxon_all(ctrl_csc, targets, todo, n_ctrl)
        dev[:, lo + todo], ties[:, lo + todo] = d, t
        del ctrl_csc, targets
        log(f"유전자 {hi}/{n_genes} 블록 완료")
    return dev, ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--cells-per-target", type=int, default=None, help="표적마다 무작위로 뽑을 세포 수 (미만이면 전부)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5AD, "r") as f:
        cats = f["obs"]["target_gene"]["categories"][:].astype(str)
        codes = f["obs"]["target_gene"]["codes"][:]
        genes = f["var"]["_index"][:].astype(str)
    ctrl_code = int(np.flatnonzero(cats == "non-targeting")[0])
    counts = np.bincount(codes, minlength=len(cats))
    n_ctrl = int(counts[ctrl_code])

    store = Path(STORE)
    store_perts = [p.strip() for p in (store / "h1_hesc/perts.txt").read_text().split("\n") if p.strip()]
    pert_row = {p: i for i, p in enumerate(store_perts)}
    candidates = [c for c in range(len(cats)) if c != ctrl_code and cats[c] in pert_row]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(candidates, size=min(args.n_targets, len(candidates)), replace=False).tolist())
    picked = {}
    for c in chosen:
        rows = np.flatnonzero(codes == c)
        if args.cells_per_target and len(rows) > args.cells_per_target:
            rows = np.sort(np.random.default_rng([args.seed, c]).choice(rows, args.cells_per_target, replace=False))
        picked[c] = rows
    n_picked = int(sum(len(r) for r in picked.values()))
    log(f"예상 메모리: control {n_ctrl}세포 x 8734 nnz x 8B = {n_ctrl * 8734 * 8 / 2**30:.1f}GB (CSR 조각, 복사 없음), "
        f"표적 {n_picked}세포 x 8734 x 8B = {n_picked * 8734 * 8 / 2**30:.1f}GB, 읽기 청크 약 1.5GB, 유전자 블록 작업 공간 약 1GB")
    log(f"표적 {len(chosen)}개 선택: {[str(cats[c]) for c in chosen]}; 표적당 세포 {[len(picked[c]) for c in chosen]}")

    ctrl_parts, target_csrs, ctrl_cp10k = stream_cells(H5AD, picked, ctrl_code, codes, len(genes))
    tested_gene = ctrl_cp10k * 100.0 > CPM_MIN
    gene_ids = np.flatnonzero(tested_gene)
    log(f"읽기 끝. 검정 대상 유전자 {len(gene_ids)}/{len(genes)}")

    t1 = time.time()
    dev_full, ties_full = wilcoxon_blocks(ctrl_parts, [target_csrs[c] for c in chosen], n_ctrl, tested_gene)
    dev, ties = dev_full[:, gene_ids], ties_full[:, gene_ids]
    log(f"Wilcoxon 계산 {time.time() - t1:.0f}s")

    # 자기 검증: scipy와 몇 쌍을 비교한다
    checks = []
    csc_cache = {}
    for t, k in [(0, 5), (1, 100), (2, 2000), (len(chosen) - 1, len(gene_ids) // 2)]:
        j = gene_ids[k]
        xc = sp.vstack([part[:, j] for part in ctrl_parts]).toarray().ravel()
        xt = target_csrs[chosen[t]][:, j].toarray().ravel()
        ref = mannwhitneyu(xt, xc, use_continuity=True, method="asymptotic")
        n = len(xt)
        N = n + n_ctrl
        var = n * n_ctrl / 12.0 * ((N + 1) - ties[t, k] / (N * (N - 1)))
        checks.append((float(ref.pvalue), float(2 * norm.sf((abs(dev[t, k]) - 0.5) / np.sqrt(var)))))
    log(f"scipy 대조 (p_scipy, p_내것): {checks}")

    nt = len(chosen)
    n_arr = np.array([len(picked[c]) for c in chosen], dtype=np.float64)
    big_n = n_arr + n_ctrl
    var = n_arr[:, None] * n_ctrl / 12.0 * ((big_n[:, None] + 1) - ties / (big_n[:, None] * (big_n[:, None] - 1)))
    zstat = np.sign(dev) * np.maximum(np.abs(dev) - 0.5, 0) / np.sqrt(var)
    p = 2 * norm.sf(np.abs(zstat))

    # 근사 라벨: z400(세포 수 환산, n<50 마스크)과 환산 없는 라벨을 같은 표적, 같은 유전자 축으로 맞춘다
    d = store / "h1_hesc"
    store_genes = [g.strip() for g in (store / "genes.txt").read_text().split("\n") if g.strip()]
    sidx = {g: i for i, g in enumerate(store_genes)}
    col = np.array([sidx.get(g, -1) for g in genes])
    delta = np.load(d / "delta_log1p.npy", mmap_mode="r")
    se = np.load(d / "se.npy", mmap_mode="r")
    measured = np.load(d / "measured.npy")
    ctrl_lin = np.load(d / "ctrl_lin.npy")
    store_n = np.load(d / "n_cells.npy")
    rows = [pert_row[str(cats[c])] for c in chosen]
    have = col >= 0
    sub_delta = np.zeros((nt, len(genes)), dtype=np.float32)
    sub_se = np.zeros_like(sub_delta)
    sub_delta[:, have] = np.asarray(delta[rows])[:, col[have]]
    sub_se[:, have] = np.asarray(se[rows])[:, col[have]]
    meas = np.zeros(len(genes), dtype=bool)
    meas[have] = measured[col[have]]
    cl = np.zeros(len(genes), dtype=np.float32)
    cl[have] = ctrl_lin[col[have]]
    approx_variants = {"z400": approx_labels(sub_delta, sub_se, meas, cl, n_cells=store_n[rows]),
                       "noscale": approx_labels(sub_delta, sub_se, meas, cl)}

    tested = np.zeros((nt, len(genes)), dtype=bool)
    tested[:, gene_ids] = True
    p_full = np.ones((nt, len(genes)))
    p_full[:, gene_ids] = p
    sign_full = np.zeros((nt, len(genes)), dtype=np.int8)
    sign_full[:, gene_ids] = np.sign(dev).astype(np.int8)
    real, real_dir = labels_from_pvalues(p_full, tested, sign_full)

    from scipy.stats import spearmanr, pearsonr
    summary = dict(targets=nt, cells_per_target=args.cells_per_target, n_control=n_ctrl,
                   tested_genes=int(len(gene_ids)), mean_real_de=float((real == 1).sum(1).mean()))
    per_target = {}
    for name, (approx, approx_dir) in approx_variants.items():
        both = (real >= 0) & (approx >= 0)
        rows_report = []
        for t in range(nt):
            m = both[t]
            r, a = real[t][m] == 1, approx[t][m] == 1
            tp = int((r & a).sum())
            rows_report.append(dict(target=str(cats[chosen[t]]), n_cells_real=int(len(picked[chosen[t]])),
                n_cells_store=int(store_n[rows[t]]), real_de=int(r.sum()), approx_de=int(a.sum()), tp=tp,
                precision=tp / max(int(a.sum()), 1), recall=tp / max(int(r.sum()), 1),
                jaccard=tp / max(int((r | a).sum()), 1)))
        r_cnt = np.array([x["real_de"] for x in rows_report])
        a_cnt = np.array([x["approx_de"] for x in rows_report])
        tp_all = sum(x["tp"] for x in rows_report)
        agree = [(real_dir[t][(real[t] == 1) & (approx[t] == 1)] == approx_dir[t][(real[t] == 1) & (approx[t] == 1)]).mean()
                 for t in range(nt) if ((real[t] == 1) & (approx[t] == 1)).any()]
        summary[name] = dict(mean_approx_de=float(a_cnt.mean()),
            spearman_de_counts=float(spearmanr(r_cnt, a_cnt)[0]), pearson_de_counts=float(pearsonr(r_cnt, a_cnt)[0]),
            micro_precision=tp_all / max(int(a_cnt.sum()), 1), micro_recall=tp_all / max(int(r_cnt.sum()), 1),
            micro_jaccard=tp_all / max(int(r_cnt.sum() + a_cnt.sum() - tp_all), 1),
            sign_agreement_on_tp=float(np.mean(agree)))
        per_target[name] = rows_report
    import resource
    summary.update(seconds_total=time.time() - T0, scipy_check=checks,
                   peak_rss_gb_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    (out / "h1_check.json").write_text(json.dumps(dict(summary=summary, per_target=per_target), indent=1, ensure_ascii=False))
    np.savez_compressed(out / "h1_labels_check.npz", targets=np.array([str(cats[c]) for c in chosen]), genes=genes,
                        real=real, approx_z400=approx_variants["z400"][0], approx_noscale=approx_variants["noscale"][0])
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
