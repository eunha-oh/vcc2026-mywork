"""반응 저장소로 세포주별 근사 DE 라벨(de_labels_<line>.npz)을 만든다.

저장소 한 세포주 폴더에서 delta_log1p, se, measured, ctrl_lin 을 읽고 (읽기 전용, mmap),
표적 블록 단위로 라벨을 계산한다. 라벨 값: 1 = DE, 0 = 아님, -1 = 마스크.
마스크 조건: 미측정 유전자, control 평균 5 CPM 이하(ctrl_lin x 100), se <= 0, 세포 수 50 미만 표적(z400 라벨만).
z400 라벨은 z x sqrt(400 / n_cells)로 환산한 뒤 표적별 BH를 적용한다. _noscale 라벨은 환산하지 않는다.

예: python tools/de_gate/build_approx_labels.py --lines k562_gwps h1_hesc --out /tmp/de-gate-eunha/de_labels
"""
from __future__ import annotations

import os
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from de_common import approx_labels  # noqa: E402

STORE = Path(os.environ.get("VCC_DATA_ROOT", "data") + "/derived/context-pipeline/response_store")


def summarize(labels):
    de = (labels == 1).sum(axis=1)
    tested = (labels >= 0).sum(axis=1)
    kept = tested > 0
    return dict(mean_de=float(de[kept].mean()), median_de=float(np.median(de[kept])),
                p10_de=float(np.percentile(de[kept], 10)), p90_de=float(np.percentile(de[kept], 90)),
                zero_de_targets=int((de[kept] == 0).sum()), mean_tested=float(tested[kept].mean()),
                positive_rate_of_tested=float(de.sum() / max(tested.sum(), 1)))


def build_line(line: str, out: Path, targets: list[str] | None, block: int = 256,
               ref_cells: float = 400.0, min_cells: int = 50) -> dict:
    """z400 라벨(de_labels_<line>.npz)과 환산 없는 라벨(de_labels_<line>_noscale.npz)을 함께 만든다."""
    d = STORE / line
    perts = [p for p in (d / "perts.txt").read_text().split("\n") if p.strip()]
    genes = [g for g in (STORE / "genes.txt").read_text().split("\n") if g.strip()]
    delta = np.load(d / "delta_log1p.npy", mmap_mode="r")
    se = np.load(d / "se.npy", mmap_mode="r")
    measured = np.load(d / "measured.npy")
    ctrl_lin = np.load(d / "ctrl_lin.npy")
    n_cells = np.load(d / "n_cells.npy")
    rows = list(range(len(perts))) if targets is None else [perts.index(t) for t in targets]
    shape = (len(rows), len(genes))
    labels, direction, labels_ns, direction_ns = (np.empty(shape, dtype=np.int8) for _ in range(4))
    t0 = time.time()
    for a in range(0, len(rows), block):
        sel = rows[a:a + block]
        d_blk, s_blk = np.asarray(delta[sel], dtype=np.float32), np.asarray(se[sel], dtype=np.float32)
        sl = slice(a, a + len(sel))
        labels[sl], direction[sl] = approx_labels(d_blk, s_blk, measured, ctrl_lin, n_cells=n_cells[sel],
                                                  ref_cells=ref_cells, min_cells=min_cells)
        labels_ns[sl], direction_ns[sl] = approx_labels(d_blk, s_blk, measured, ctrl_lin)
    names = np.array([perts[r] for r in rows])
    common = dict(perts=names, genes=np.array(genes), n_cells=n_cells[rows])
    np.savez_compressed(out / f"de_labels_{line}.npz", labels=labels, direction=direction,
                        ref_cells=ref_cells, min_cells=min_cells, **common)
    np.savez_compressed(out / f"de_labels_{line}_noscale.npz", labels=labels_ns, direction=direction_ns, **common)
    excluded = int((n_cells[rows] < min_cells).sum())
    stats = dict(line=line, targets=len(rows), excluded_targets_n_lt_min=excluded,
                 masked_fraction_z400=float((labels < 0).mean()), z400=summarize(labels),
                 noscale=summarize(labels_ns), seconds=time.time() - t0,
                 peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-targets", type=int, default=None, help="시간과 메모리 측정용으로 앞쪽 표적 N개만")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for line in args.lines:
        targets = None
        if args.n_targets:
            targets = [p for p in (STORE / line / "perts.txt").read_text().split("\n") if p.strip()][:args.n_targets]
        stats.append(build_line(line, out, targets))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
