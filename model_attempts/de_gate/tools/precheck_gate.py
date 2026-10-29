"""학습 없는 게이트 사전 검증 (CPU): 단순 게이트를 table 예측에 씌우면 좋아지는지, 완벽한 게이트의 상한은 얼마인지 본다.

세포주 L을 하나씩 빼고(leave one line out), L을 뺀 줄의 평균으로 L의 반응을 예측한다 (PriorTable.get과 같은 규칙:
표적 세포 16개 이상인 줄만, 줄이 측정한 유전자만 평균). 세 예측을 같은 전역 배율 g로 비교한다.

  table  prior x g
  simple p = sigmoid(a |prior| + c log1p(control 발현) + b) 를 L이 아닌 세포주의 z400 라벨로 로지스틱 회귀해 맞추고,
         soft_norm(표적마다, L 측정 유전자 그룹과 나머지 그룹에서 RMS 복원)으로 prior에 씌운 뒤 x g
  oracle p = 0.9 (L의 실제 DE) / 0.1 (아님) 로 같은 soft_norm 적용 x g  (완벽한 게이트의 상한)

단위: 평가 정답과 prior 모두 delta_log1p (표적 세포의 평균 log1p(CP10K) - control). table의 lfc_lin(ln 배율)은 단위가 달라
쓰지 않고, 단위 확인에서만 비교한다. g는 L을 뺀 세포주들에서 table 예측의 MSE가 최소가 되는 전역 배율이고 세 예측에 같이 쓴다.
참고로 L에서 직접 최적화한 g(g_L)로 잰 MSE도 함께 낸다.

학습 데이터: M(L이 아닌 세포주)의 prior는 L과 M을 뺀 줄의 평균이다 (배포 때 L을 모르는 상황과 같게).
메모리: table의 delta_log1p, lfc_lin 은 npz 안의 오프셋으로 memmap 하고 표적 100개씩 읽는다.

--extra-features: 단순 게이트에 특징을 더한 게이트 세 개를 함께 비교한다.
  f1  같은 표적의 다른 세포주(L 제외) z400 라벨 DE 비율 (해당 표적, 유전자를 측정한 줄이 없으면 f1 = 0, 결측 표시 m1 = 1)
  f2  유전자 평소 반응성: L 제외 세포주들에서 그 유전자가 DE인 (표적, 줄) 비율 (검정한 적이 없으면 전체 비율)
  simple_f2 = simple + f2, simple_f1 = simple + f1 + m1, simple_f1f2 = simple + f1 + m1 + f2
누수 방지: 모든 특징은 사용할 줄 목록을 인자로 받아 그 줄의 라벨만 읽는다. L 평가에서는 L 제외 줄, 학습 표본(세포주 M)에서는
L과 M을 뺀 줄을 쓴다 (prior와 같은 규칙). 시작할 때 leak_check 가 제외된 줄의 라벨을 무작위로 바꿔 특징이 변하지 않는지,
포함된 줄을 바꾸면 변하는지(검사가 비어 있지 않은지) 확인한다.

예: python tools/de_gate/precheck_gate.py --max-targets 50 --out /tmp/de-gate-eunha/precheck/t50
"""
from __future__ import annotations

import os
import argparse
import json
import resource
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

V = Path(os.environ.get("VCC_DATA_ROOT", "data") + "/derived")
TABLE = V / "context-pipeline/prior_table/prior_table_4lines.npz"
STORE = V / "context-pipeline/response_store"
LABELS = V / "de-gate-eunha/de_labels/de_labels_{line}.npz"
LINE_DIR = {"replogle_k562_gwps": "k562_gwps", "orion_hct116": "hct116_orion", "orion_hek293t": "hek293t_orion",
            "cd4t_D1_Rest": "cd4t_D1_Rest"}
MIN_CELLS = 16  # PriorTable의 min_cells
T0 = time.time()


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time() - T0:6.1f}s peak={rss:4.2f}GB] {msg}", flush=True)


def npz_memmap(path, name):
    """압축 없이 저장된 npz 멤버를 읽기 전용 memmap으로 연다 (np.load의 mmap_mode는 npz에 동작하지 않는다)."""
    info = zipfile.ZipFile(path).getinfo(name + ".npy")
    if info.compress_type != 0:
        raise ValueError("압축된 npz 멤버는 memmap으로 열 수 없다")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        header = f.read(30)
        n, m = struct.unpack("<HH", header[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


class Data:
    def __init__(self, max_targets, seed):
        with np.load(TABLE, allow_pickle=False) as z:
            self.lines = [str(x) for x in z["lines"]]
            self.perts = [str(x) for x in z["perts"]]
            self.n_cells, self.measured = z["n_cells"], z["measured"].astype(bool)
        self.delta = npz_memmap(TABLE, "delta_log1p")
        self.lfc = npz_memmap(TABLE, "lfc_lin")
        self.pert_index = {p: i for i, p in enumerate(self.perts)}
        self.ctrl, self.labels, self.label_row = {}, {}, {}
        for i, line in enumerate(self.lines):
            self.ctrl[i] = np.log1p(np.load(STORE / LINE_DIR[line] / "ctrl_lin.npy").astype(np.float32))
            with np.load(str(LABELS).format(line=LINE_DIR[line]), allow_pickle=False) as z:
                self.labels[i] = z["labels"]
                self.label_row[i] = {str(p): j for j, p in enumerate(z["perts"])}
        self.rng = np.random.default_rng(seed)
        self.max_targets = max_targets
        self.refresh_counts()

    def refresh_counts(self):
        """유전자별로 줄마다 DE인 (표적) 수와 검정된 수 (f2 용). 라벨을 바꾸면 다시 부른다."""
        self.de_count = {i: (lab == 1).sum(0).astype(np.int64) for i, lab in self.labels.items()}
        self.test_count = {i: (lab >= 0).sum(0).astype(np.int64) for i, lab in self.labels.items()}

    def label_features(self, names, others):
        """others 줄의 라벨만으로 만든 f1, m1 (표적, 유전자)과 f2 (유전자)."""
        genes = self.delta.shape[2]
        num = np.zeros((len(names), genes), dtype=np.float32)
        den = np.zeros_like(num)
        for o in others:
            rows = np.array([self.label_row[o].get(n, -1) for n in names])
            have = rows >= 0
            if have.any():
                lab = self.labels[o][rows[have]]
                num[have] += lab == 1
                den[have] += lab >= 0
        f1 = np.where(den > 0, num / np.maximum(den, 1), 0.0).astype(np.float32)
        m1 = (den == 0).astype(np.float32)
        de = sum(self.de_count[o] for o in others)
        tested = sum(self.test_count[o] for o in others)
        f2 = np.where(tested > 0, de / np.maximum(tested, 1), de.sum() / max(tested.sum(), 1)).astype(np.float32)
        return f1, m1, f2

    def targets(self, line, others):
        """line의 라벨이 있고(마스크 아닌 유전자가 있음) others 중 한 줄 이상이 지지하는 표적의 (table 행, 라벨 행)."""
        rows = []
        for name, lr in self.label_row[line].items():
            t = self.pert_index.get(name)
            if t is None or self.n_cells[line, t] <= 0 or not (self.labels[line][lr] >= 0).any():
                continue
            if any(self.n_cells[o, t] >= MIN_CELLS for o in others):
                rows.append((t, lr))
        rows.sort()
        if self.max_targets and len(rows) > self.max_targets:
            pick = np.random.default_rng(0).choice(len(rows), self.max_targets, replace=False)
            rows = [rows[i] for i in sorted(pick)]
        return rows

    def prior(self, trows, others, source=None):
        """others 줄의 평균 (PriorTable.get과 같은 규칙). source가 None이면 delta_log1p, 'lfc'면 lfc_lin."""
        array = self.lfc if source == "lfc" else self.delta
        num = np.zeros((len(trows), array.shape[2]), dtype=np.float32)
        den = np.zeros_like(num)
        for o in others:
            support = (self.n_cells[o, trows] >= MIN_CELLS).astype(np.float32)[:, None]
            valid = self.measured[o].astype(np.float32)[None, :] * support
            num += np.asarray(array[o][trows]) * valid
            den += valid
        return num / np.maximum(den, 1.0)


def chunks(items, size=100):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fit_logistic(x, y, l2=1e-6, iters=30):
    """표준화한 특징으로 뉴턴 방법 로지스틱 회귀. 계수는 원래 특징 단위로 돌려준다 (마지막이 절편)."""
    mu, sd = x.mean(0), x.std(0) + 1e-12
    z = np.hstack([(x - mu) / sd, np.ones((len(x), 1), dtype=np.float32)]).astype(np.float64)
    w = np.zeros(z.shape[1])
    w[-1] = np.log(y.mean() / (1 - y.mean()))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z @ w)))
        grad = z.T @ (p - y) + l2 * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    coef = w[:-1] / sd
    return np.append(coef, w[-1] - float((coef * mu).sum()))


def soft_norm(prior, p, group):
    """표적마다, group(bool, 유전자)과 그 밖의 두 그룹에서 gated 를 prior 의 RMS 로 되돌린다."""
    gated = prior * p
    out = np.zeros_like(gated)
    for mask in (group, ~group):
        if not mask.any():
            continue
        ro = np.sqrt((prior[:, mask] ** 2).mean(1) + 1e-12)
        rg = np.sqrt((gated[:, mask] ** 2).mean(1) + 1e-12)
        out[:, mask] = gated[:, mask] * (ro / (rg + 1e-8))[:, None]
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def training_set(data, L, frac, seed):
    """L이 아닌 세포주 M에서 로지스틱 회귀 표본과, 전역 배율 g 의 정규방정식 합(sum prior*y, sum prior^2)을 모은다.

    특징 열: [|prior|, log1p(control), f1, m1, f2]. M의 prior와 f1, f2 는 L과 M을 뺀 줄에서만 만든다.
    """
    rng = np.random.default_rng(seed)
    xs, ys, gxy, gxx, per_line = [], [], 0.0, 0.0, {}
    for M in range(len(data.lines)):
        if M == L:
            continue
        others = [o for o in range(len(data.lines)) if o not in (L, M)]
        trows = data.targets(M, others)
        n_pos = n_all = 0
        for part in chunks(trows):
            t = [r[0] for r in part]
            prior = data.prior(t, others)
            truth = np.asarray(data.delta[M][t])
            lab = data.labels[M][[r[1] for r in part]]
            mask = lab >= 0
            gxy += float((prior * truth)[mask].sum())
            gxx += float((prior * prior)[mask].sum())
            n_pos += int((lab == 1).sum())
            n_all += int(mask.sum())
            f1, m1, f2 = data.label_features([data.perts[i] for i in t], others)
            pick = mask & (rng.random(mask.shape) < frac)
            cols = [np.abs(prior), np.broadcast_to(data.ctrl[M][None, :], prior.shape), f1, m1,
                    np.broadcast_to(f2[None, :], prior.shape)]
            xs.append(np.stack([c[pick] for c in cols], 1).astype(np.float32))
            ys.append((lab[pick] == 1).astype(np.float32))
        per_line[data.lines[M]] = dict(targets=len(trows), tested_entries=n_all, de_rate=n_pos / max(n_all, 1))
    return np.concatenate(xs), np.concatenate(ys), gxy / gxx, per_line


def auroc(score, positive, limit=400000, seed=0):
    rng = np.random.default_rng(seed)
    if len(score) > limit:
        pick = rng.choice(len(score), limit, replace=False)
        score, positive = score[pick], positive[pick]
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


GATES = {"simple": [0, 1], "simple_f2": [0, 1, 4], "simple_f1": [0, 1, 2, 3], "simple_f1f2": [0, 1, 2, 3, 4]}


def evaluate_line(data, L, frac, seed, extra):
    name = data.lines[L]
    others = [o for o in range(len(data.lines)) if o != L]
    t_fit = time.time()
    x, y, g, train_info = training_set(data, L, frac, seed)
    gates = {k: v for k, v in GATES.items() if extra or k == "simple"}
    coefs = {k: fit_logistic(x[:, cols], y) for k, cols in gates.items()}
    base = float(y.mean())
    base_logloss = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    train_logloss = {}
    for k, cols in gates.items():
        p_fit = sigmoid(x[:, cols] @ coefs[k][:-1].astype(np.float32) + coefs[k][-1])
        train_logloss[k] = float(-(y * np.log(p_fit + 1e-12) + (1 - y) * np.log(1 - p_fit + 1e-12)).mean())
    a, c, b = (float(v) for v in coefs["simple"])
    log(f"{name}: 학습 표본 {len(y):,}개, 양성 {base:.3f}, g={g:.4f}, simple a={a:.3f} c={c:.3f} b={b:.3f}, 회귀 {time.time() - t_fit:.1f}s")
    group = data.measured[L]
    ctrl = data.ctrl[L]
    methods = ("table", *gates, "oracle")
    trows = data.targets(L, others)
    sums = {m: [] for m in methods}  # 표적별 [n, yy, sy, pp, py, sp]
    precision, k_list, base_rate = {m: [] for m in methods}, [], []
    p_samples, p_labels = {k: [] for k in gates}, []
    sign_agree = []  # 실제 DE 유전자에서 prior의 부호가 정답과 같은 비율
    rng = np.random.default_rng(seed + 1)
    done = 0
    for part in chunks(trows):
        t = [r[0] for r in part]
        prior = data.prior(t, others)
        truth = np.asarray(data.delta[L][t])
        lab = data.labels[L][[r[1] for r in part]]
        f1, m1, f2 = data.label_features([data.perts[i] for i in t], others)  # others 에는 L 이 없다
        feats = [np.abs(prior), np.broadcast_to(ctrl[None, :], prior.shape), f1, m1, np.broadcast_to(f2[None, :], prior.shape)]
        p_gate = {k: sigmoid(sum(coefs[k][j] * feats[col] for j, col in enumerate(cols)) + coefs[k][-1])
                  for k, cols in gates.items()}
        p_oracle = np.where(lab == 1, 0.9, 0.1).astype(np.float32)
        preds = {"table": prior, "oracle": soft_norm(prior, p_oracle, group)}
        preds.update({k: soft_norm(prior, p, group) for k, p in p_gate.items()})
        for i in range(len(t)):
            mask = lab[i] >= 0
            yv = truth[i][mask]
            de = lab[i][mask] == 1
            k = int(de.sum())
            for m in methods:
                pv = preds[m][i][mask]
                sums[m].append([len(yv), float((yv ** 2).sum()), float(yv.sum()), float((pv ** 2).sum()),
                                float((pv * yv).sum()), float(pv.sum())])
                if k > 0:
                    top = np.argpartition(-np.abs(pv), k - 1)[:k]
                    precision[m].append(float(de[top].mean()))
                else:
                    precision[m].append(float("nan"))
            k_list.append(k)
            base_rate.append(k / len(yv))
            if k > 0:
                sign_agree.append(float((np.sign(preds["table"][i][mask][de]) == np.sign(yv[de])).mean()))
            pick = rng.random(mask.sum()) < 0.02
            for key in gates:
                p_samples[key].append(p_gate[key][i][mask][pick])
            p_labels.append(de[pick])
        done += len(t)
        log(f"{name}: 표적 {done}/{len(trows)}")
    out = {"line": name, "targets": len(trows), "g": g, "a": a, "c": c, "b": b, "train_info": train_info,
           "train_samples": int(len(y)), "train_pos_rate": base, "train_logloss": train_logloss["simple"],
           "train_logloss_base": base_logloss, "mean_de_per_target": float(np.mean(k_list)),
           "mean_base_rate": float(np.mean(base_rate)), "targets_with_de": int(np.sum(np.array(k_list) > 0)),
           "gate_coefs": {k: [float(v) for v in coefs[k]] for k in gates}, "gate_train_logloss": train_logloss}
    l_all = np.concatenate(p_labels)
    out["p_summary"] = {}
    for key in gates:
        p_all = np.concatenate(p_samples[key])
        out["p_summary"][key] = {"mean": float(p_all.mean()),
            "quantiles_5_25_50_75_95": [float(q) for q in np.quantile(p_all, [.05, .25, .5, .75, .95])],
            "mean_p_on_de": float(p_all[l_all].mean()), "mean_p_on_nonde": float(p_all[~l_all].mean()),
            "auroc": auroc(p_all, l_all)}
    out["p_summary_simple"] = out["p_summary"]["simple"]  # 이전 출력과 같은 키
    for m in methods:
        s_ = np.array(sums[m], dtype=np.float64)
        n, yy, sy, pp, py, sp = s_.T
        mse = lambda gain: (gain ** 2 * pp - 2 * gain * py + yy) / n  # noqa: E731
        cov = py / n - (sp / n) * (sy / n)
        pearson = cov / np.sqrt(np.maximum((pp / n - (sp / n) ** 2) * (yy / n - (sy / n) ** 2), 1e-20))
        if m == "table":
            out["zero_prediction_mse"] = float((yy / n).mean())
            out["table_sign_agreement_on_de"] = float(np.mean(sign_agree))
            out["true_rms"] = float(np.sqrt(yy.sum() / n.sum()))
            out["table_rms_at_g"] = float(g * np.sqrt(pp.sum() / n.sum()))
        g_line = float(py.sum() / pp.sum())  # L에서 직접 최적화한 전역 배율 (참고)
        out[m] = {"mse": float(mse(g).mean()), "mse_at_g_line": float(mse(g_line).mean()), "g_line": g_line,
                  "pearson": float(np.nanmean(pearson)), "precision_at_k": float(np.nanmean(precision[m]))}
    return out


class scrambled:
    """검사용: 주어진 줄들의 라벨을 무작위로 바꿔 두었다가 나올 때 되돌린다 (개수 표도 다시 만든다)."""

    def __init__(self, data, lines, seed=123):
        self.data, self.lines, self.seed = data, lines, seed

    def __enter__(self):
        rng = np.random.default_rng(self.seed)
        self.saved = {i: self.data.labels[i] for i in self.lines}
        for i in self.lines:
            lab = self.data.labels[i]
            self.data.labels[i] = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=lab.shape)
        self.data.refresh_counts()

    def __exit__(self, *exc):
        self.data.labels.update(self.saved)
        self.data.refresh_counts()


def leak_check(data, L, n=30):
    """특징에 L의 정보가 새지 않는지: 제외한 줄의 라벨을 바꿔도 특징이 같고, 포함한 줄을 바꾸면 특징이 달라지는지 본다."""
    lines = range(len(data.lines))
    others = [o for o in lines if o != L]
    names = [data.perts[t] for t, _ in data.targets(L, others)[:n]]
    ref = data.label_features(names, others)
    with scrambled(data, [L]):  # L 평가용 특징: L 라벨을 바꿔도 같아야 한다
        eval_same = all(np.array_equal(a, b) for a, b in zip(ref, data.label_features(names, others)))
    M = others[0]
    rest = [o for o in lines if o not in (L, M)]
    ref_m = data.label_features(names, rest)
    with scrambled(data, [L, M]):  # M 학습용 특징: L과 M 라벨을 바꿔도 같아야 한다
        train_same = all(np.array_equal(a, b) for a, b in zip(ref_m, data.label_features(names, rest)))
    with scrambled(data, [others[1]]):  # 포함한 줄을 바꾸면 달라져야 한다 (검사가 비어 있지 않다는 확인)
        changed = not all(np.array_equal(a, b) for a, b in zip(ref, data.label_features(names, others)))
    return {"line": data.lines[L], "eval_features_invariant_to_L_labels": bool(eval_same),
            "train_features_invariant_to_L_and_M_labels": bool(train_same), "check_detects_change_in_used_line": bool(changed)}


def unit_check(data, L=0, n=50):
    """prior 단위 확인: table의 delta_log1p, lfc_lin 이 반응 저장소와 같은지, 둘의 관계와 L 정답에 대한 최적 배율."""
    others = [o for o in range(len(data.lines)) if o != L]
    trows = data.targets(L, others)[:n]
    t = [r[0] for r in trows]
    lab = data.labels[L][[r[1] for r in trows]]
    mask = lab >= 0
    truth = np.asarray(data.delta[L][t])
    out = {"line": data.lines[L], "targets": len(t)}
    for source in (None, "lfc"):
        prior = data.prior(t, others, source)[mask]
        y = truth[mask]
        g = float((prior * y).sum() / (prior ** 2).sum())
        out["delta_log1p" if source is None else "lfc_lin"] = {
            "rms_prior": float(np.sqrt((prior ** 2).mean())), "rms_truth": float(np.sqrt((y ** 2).mean())),
            "best_global_gain": g, "mse_at_best_gain": float(((g * prior - y) ** 2).mean()),
            "mse_zero_prediction": float((y ** 2).mean()), "pearson": float(np.corrcoef(prior, y)[0, 1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-targets", type=int, default=None, help="세포주마다 쓸 표적 수 (시간 측정용으로 50)")
    ap.add_argument("--frac", type=float, default=0.03, help="로지스틱 회귀용으로 뽑는 검정 유전자 비율")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-features", action="store_true", help="f1, f2 를 더한 게이트 세 개도 비교")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = Data(args.max_targets, args.seed)
    log(f"table 줄 {data.lines}, 표적 {len(data.perts)}개, 유전자 {data.delta.shape[2]}개")
    result = {"max_targets": args.max_targets, "unit_check": unit_check(data), "lines": []}
    log("단위 확인: " + json.dumps(result["unit_check"], ensure_ascii=False))
    if args.extra_features:
        result["leak_check"] = [leak_check(data, L) for L in range(len(data.lines))]
        log("누수 검사: " + json.dumps(result["leak_check"], ensure_ascii=False))
        if not all(all(v for k, v in r.items() if k != "line") for r in result["leak_check"]):
            raise SystemExit("누수 검사 실패")
    for L in range(len(data.lines)):
        result["lines"].append(evaluate_line(data, L, args.frac, args.seed, args.extra_features))
    result["seconds"] = time.time() - T0
    result["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    (args.out / "precheck.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    log(f"완료 {result['seconds']:.0f}s, 피크 {result['peak_rss_gb']:.2f}GB")


if __name__ == "__main__":
    main()
