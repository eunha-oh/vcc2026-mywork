# gene panel 채우기 평가 공통 코드 (mvp_eval.py, vae_eval.py, gwps_holdout_eval.py가 같이 씀)
import numpy as np, scipy.sparse as sp

TARGET_SUM = 1e4
ORION_CTRL = 'Non-Targeting'
GWPS_CTRL = 'non-targeting'


def log_norm(X, lib_idx):
    # 세포마다 lib_idx 유전자 count 합 기준 1e4 normalize + log1p (sparse 유지)
    X = sp.csr_matrix(X).astype(np.float32)
    lib = np.asarray(X[:, lib_idx].sum(1)).ravel()
    X = (sp.diags(TARGET_SUM / np.maximum(lib, 1)) @ X).tocsr()
    X.data = np.log1p(X.data)
    return X


def counts_to_log(C, lib):
    # 기대 count를 평가 정규화(주어진 lib 기준 1e4 + log1p)로 바꾼다
    return np.log1p(np.asarray(C, np.float32) * (TARGET_SUM / np.maximum(lib, 1)).reshape(-1, 1)).astype(np.float32)


def colwise_pearson(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0)
    d = np.sqrt((P**2).sum(0) * (Y**2).sum(0))
    return np.where(d > 0, (P * Y).sum(0) / np.maximum(d, 1e-12), np.nan)


def delta_metrics(P, Y, o, ctrl, group_col, line_col=None):
    # (세포주, 표적)별 pseudobulk Δ = 그룹 평균 - 같은 세포주 평가 control 평균
    # o: 평가 세포 obs, ctrl: 평가 세포 control 여부
    lines = o[line_col].to_numpy() if line_col else np.zeros(len(o), int)
    tgt_all = o[group_col].to_numpy()
    r, ratio = [], []
    for ln in np.unique(lines):
        inl = lines == ln
        c = np.where(inl & ctrl)[0]
        if len(c) < 20: continue
        for tgt, g in o[inl & ~ctrl].groupby(group_col, observed=True).indices.items():
            if len(g) < 5: continue
            g = np.where(inl & (tgt_all == tgt))[0]
            dt = Y[g].mean(0) - Y[c].mean(0); dp = P[g].mean(0) - P[c].mean(0)
            ratio.append(((dp - dt)**2).mean() / max((dt**2).mean(), 1e-12))
            if dp.std() > 0: r.append(np.corrcoef(dp, dt)[0, 1])
    return dict(n_groups=len(ratio),
                delta_pearson_median=float(np.median(r)) if r else None,
                delta_mse_ratio_median=float(np.median(ratio)) if ratio else None)


def all_metrics(P, Y, o, ctrl, group_col, line_col=None):
    gp = colwise_pearson(P, Y)
    return dict(gene_pearson_median=float(np.nanmedian(gp)),
                rmse=float(np.sqrt(((P - Y)**2).mean())), **delta_metrics(P, Y, o, ctrl, group_col, line_col))


def orion_splits(obs, seed=0):
    # mvp_eval.py와 같은 분할. 반환: {이름: (train bool, test bool)}
    ctrl = (obs.gene_target == ORION_CTRL).to_numpy()
    line = obs.cell_line.to_numpy()
    rng = np.random.default_rng(seed)
    tg = obs.gene_target[~ctrl].unique()
    test_t = rng.choice(tg, int(0.2 * len(tg)), replace=False)
    te1 = obs.gene_target.isin(test_t).to_numpy().copy()
    ci = np.where(ctrl)[0]; rng.shuffle(ci); te1[ci[:len(ci) // 5]] = True
    return {'split_unseen_pert': (~te1, te1),
            'split_unseen_line': (line == 'HCT116', line == 'HEK293T')}


def k562_split(obs, seed=0):
    # VIPerturb K562 처음 보는 perturbation 분할: 타깃 20%와 control 20%를 평가로 뗀다
    # 반환: (train bool, test bool, 홀드아웃 타깃). 홀드아웃 타깃은 Orion, GWPS 학습 세포에서도 빼야 누수가 없다
    ctrl = obs['is_control'].to_numpy(bool)
    tgt = obs['gene_target'].astype(str)
    rng = np.random.default_rng(seed)
    tg = np.array(sorted(tgt[~ctrl].unique()))
    test_t = rng.choice(tg, int(0.2 * len(tg)), replace=False)
    te = tgt.isin(test_t).to_numpy().copy()
    ci = np.where(ctrl)[0]; rng.shuffle(ci); te[ci[:len(ci) // 5]] = True
    return ~te, te, np.sort(test_t)


def target_column(obs):
    # 데이터셋별 타깃 유전자 열 (Orion, VIPerturb: gene_target, GWPS: gene)
    return 'gene_target' if 'gene_target' in obs.columns else 'gene'


def k562_train_rows(obs, holdout_targets):
    # k562_unseen_pert 분할의 학습 세포. VIPerturb 샘플이면 분할의 train, 다른 데이터셋이면 홀드아웃 타깃 세포만 뺀다
    if 'dataset' in obs.columns and (obs['dataset'].astype(str) == 'viperturb_k562').all():
        return k562_split(obs)[0]
    return ~obs[target_column(obs)].astype(str).isin(holdout_targets).to_numpy()


def orion_val_split(obs, seed=1):
    # 하이퍼파라미터 선택용: 처음 보는 perturbation 분할의 학습 세포 안에서 표적 20%와 control 20%를 다시 뗀다
    tr, _ = orion_splits(obs)['split_unseen_pert']
    sub = obs[tr]
    ctrl = (sub.gene_target == ORION_CTRL).to_numpy()
    rng = np.random.default_rng(seed)
    tg = sub.gene_target[~ctrl].unique()
    val_t = rng.choice(tg, int(0.2 * len(tg)), replace=False)
    v = sub.gene_target.isin(val_t).to_numpy().copy()
    ci = np.where(ctrl)[0]; rng.shuffle(ci); v[ci[:len(ci) // 5]] = True
    idx = np.where(tr)[0]
    val = np.zeros(len(obs), bool); val[idx[v]] = True
    return tr & ~val, val
