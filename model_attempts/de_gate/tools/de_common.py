"""DE 라벨 공통 함수: BH 보정과 반응 저장소 기반 근사 라벨.

라벨 값: 1 = DE, 0 = DE 아님, -1 = 마스크 (채점기가 검정하지 않는 유전자, 로스에서 제외).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

CPM_MIN = 5.0  # 채점기: control 평균이 5 CPM 이하인 유전자는 검정하지 않는다


def bh_reject(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """1차원 p값 배열에 Benjamini-Hochberg를 적용하고 기각 여부(bool)를 돌려준다."""
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    adjusted = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    reject = np.zeros(n, dtype=bool)
    reject[order] = adjusted < alpha
    return reject


def labels_from_pvalues(p: np.ndarray, tested: np.ndarray, sign: np.ndarray, alpha: float = 0.05):
    """(표적, 유전자) p값을 표적별로 BH 보정해 라벨 int8과 방향 int8을 만든다. 마스크는 -1."""
    labels = np.full(p.shape, -1, dtype=np.int8)
    direction = np.zeros(p.shape, dtype=np.int8)
    for row in range(p.shape[0]):
        use = np.flatnonzero(tested[row])
        reject = bh_reject(p[row, use], alpha)
        labels[row, use] = reject.astype(np.int8)
        direction[row, use[reject]] = sign[row, use[reject]]
    return labels, direction


def approx_labels(delta: np.ndarray, se: np.ndarray, measured: np.ndarray, ctrl_cp10k: np.ndarray,
                  alpha: float = 0.05, cpm_min: float = CPM_MIN, n_cells: np.ndarray | None = None,
                  ref_cells: float = 400.0, min_cells: int = 50):
    """반응 저장소 값으로 만든 근사 DE 라벨.

    z = delta_log1p / se (se는 delta_log1p의 표준오차), 양측 정규 p값, 표적별 BH.
    검정 대상은 측정된 유전자 중 control 평균 CPM(= ctrl_lin x 100)이 cpm_min 초과이고 se > 0 인 것이다.
    n_cells를 주면 대회 정답의 표적당 세포 수에 맞춰 z400 = z x sqrt(ref_cells / n_cells)로 환산하고,
    n_cells < min_cells 인 표적은 라벨 전체를 마스크(-1)로 한다. n_cells를 주지 않으면 환산하지 않는다.
    """
    tested = (measured & (ctrl_cp10k * 100.0 > cpm_min))[None, :] & (se > 0) & np.isfinite(delta) & np.isfinite(se)
    if n_cells is not None:
        tested = tested & (np.asarray(n_cells) >= min_cells)[:, None]
    z = np.zeros(delta.shape, dtype=np.float64)
    np.divide(delta, se, out=z, where=tested)
    if n_cells is not None:
        z = z * np.sqrt(ref_cells / np.maximum(np.asarray(n_cells, dtype=np.float64), 1.0))[:, None]
    p = 2.0 * norm.sf(np.abs(z))
    sign = np.sign(delta).astype(np.int8)
    return labels_from_pvalues(p, tested, sign, alpha)
