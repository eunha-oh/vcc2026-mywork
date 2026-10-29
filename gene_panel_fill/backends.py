# Gene panel 채우기 모델 백엔드
# 체크포인트 폴더의 meta.json에 적힌 model_type(linear, vae, ensemble)으로 백엔드를 고른다.
# 모든 백엔드는 같은 인터페이스를 가진다:
#   input_genes  : 모델 입력 유전자 심볼 (순서 고정)
#   output_genes : 모델이 채우는 유전자 심볼
#   output_source: output_genes 각각의 출처 ('model' / 'fallback')
#   all_genes    : 기본 출력 유전자 축 (VCC2026 18,533)
#   requires_all_inputs: True면 input_genes가 모두 입력에 있어야 제대로 동작 (없으면 경고)
#   predict_missing(X_O_counts, lib_size, present) -> (세포 수, len(output_genes)) 기대 count, float32
#       present: input_genes 중 입력에 실제로 있는 유전자 (bool)
#   sample(P, cols) -> P(기대 count, output_genes의 cols 열)에서 뽑은 정수 count
import json, os
import numpy as np

META_NAME, ARRAYS_NAME, VAE_WEIGHTS_NAME = 'meta.json', 'model.npz', 'model.pt'
QUALITY_NAME = 'gene_quality.csv'  # 선택: 유전자별 채우기 신뢰도 (gene_quality.py)


class Backend:
    model_type = None
    requires_all_inputs = True

    def __init__(self, ckpt_dir, meta, seed=0, **opts):
        self.ckpt_dir, self.meta = ckpt_dir, meta
        self.rng = np.random.default_rng(seed)

    def predict_missing(self, X_O_counts, lib_size, present):
        raise NotImplementedError

    def sample(self, P, cols):
        return self.rng.poisson(P).astype(np.float32)

    def describe(self):
        return {}


class LinearBackend(Backend):
    # L1000 방식: log 정규화된 입력 → PCA → Ridge → 결측 유전자 (log 공간) → count 복원
    model_type = 'linear'

    def __init__(self, ckpt_dir, meta, seed=0, **opts):
        super().__init__(ckpt_dir, meta, seed)
        z = np.load(os.path.join(ckpt_dir, ARRAYS_NAME), allow_pickle=False)
        self.input_genes = z['O_model'].astype(str)
        m_model, m_fb = z['M_model'].astype(str), z['M_fallback'].astype(str)
        self.output_genes = np.concatenate([m_model, m_fb])
        self.output_source = np.array(['model'] * len(m_model) + ['fallback'] * len(m_fb))
        self.all_genes = z['all_genes'].astype(str)
        self.pca_mean = z['pca_mean'].astype(np.float32)
        self.pca_components_T = np.ascontiguousarray(z['pca_components'].astype(np.float32).T)  # (O, k)
        self.ridge_coef_T = np.ascontiguousarray(z['ridge_coef'].astype(np.float32).T)  # (k, M_model)
        self.ridge_intercept = z['ridge_intercept'].astype(np.float32)
        # fallback은 세포와 무관한 상수 → 정규화 공간 값을 미리 expm1
        self.fallback_scaled = np.expm1(z['fallback_mean_log'].astype(np.float32))
        self.target_sum = float(meta['normalization']['target_sum'])

    def predict_missing(self, X_O_counts, lib_size, present=None):
        lib = np.asarray(lib_size, np.float32).reshape(-1, 1)
        scale = np.float32(self.target_sum) / np.maximum(lib, 1)
        Xn = np.log1p(np.asarray(X_O_counts, np.float32) * scale)
        Xn -= self.pca_mean
        Y = (Xn @ self.pca_components_T) @ self.ridge_coef_T
        del Xn
        Y += self.ridge_intercept
        np.expm1(Y, out=Y)
        np.maximum(Y, 0, out=Y)
        back = lib / np.float32(self.target_sum)  # lib=0인 세포는 0
        out = np.empty((Y.shape[0], len(self.output_genes)), np.float32)
        nm = Y.shape[1]
        np.multiply(Y, back, out=out[:, :nm])
        out[:, nm:] = self.fallback_scaled[None, :] * back
        return out


class VAEBackend(Backend):
    # 측정 mask를 입력으로 받는 VAE. 어떤 패널이 들어와도 18,533 전체의 기대 count를 낸다
    model_type = 'vae'
    requires_all_inputs = False

    def __init__(self, ckpt_dir, meta, seed=0, device='auto', enc_dataset=None, dec_dataset=None,
                 batch_size=2048, **opts):
        super().__init__(ckpt_dir, meta, seed)
        import torch
        from vae_model import GenePanelVAE, GenePanelVAEv2, predict_counts, sample_nb
        self.torch, self._predict, self._sample_nb = torch, predict_counts, sample_nb
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else
                                   ('cpu' if device == 'auto' else device))
        torch.manual_seed(seed)
        z = np.load(os.path.join(ckpt_dir, ARRAYS_NAME), allow_pickle=False)
        self.all_genes = z['all_genes'].astype(str)
        self.input_genes = self.output_genes = self.all_genes
        # 학습 데이터에서 측정된 적 없는 유전자는 VCC control 비율로 채운다 (fallback)
        self.trained = z['trained'].astype(bool)
        self.output_source = np.where(self.trained, 'model', 'fallback')
        self.control_rho = z['control_rho'].astype(np.float64)
        self.fb_idx = np.where(~self.trained)[0]
        c = meta['model_config']
        self.arch = c.get('arch', 'v1')
        cls = GenePanelVAEv2 if self.arch == 'v2' else GenePanelVAE
        self.model = cls(c['n_genes'], c['n_datasets'], tuple(c['hidden']), c['latent'], c['dropout'], c['emb_dim'])
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, VAE_WEIGHTS_NAME), map_location='cpu', weights_only=True))
        self.model.to(self.device).eval()
        ds = meta['datasets']
        self.enc_name = enc_dataset or meta['fill_defaults']['enc_dataset']
        if self.arch == 'v2':
            # v2는 입력 데이터셋이 측정하지 않은 유전자를 공유 출력으로 채우므로 디코더 데이터셋 = 입력 데이터셋
            if dec_dataset and dec_dataset != self.enc_name:
                raise ValueError(f'v2 체크포인트는 디코더 데이터셋을 따로 고를 수 없음 (입력 {self.enc_name}, 지정 {dec_dataset})')
            self.dec_name = self.enc_name
        else:
            self.dec_name = dec_dataset or meta['fill_defaults']['dec_dataset']
        self.enc, self.dec = ds.index(self.enc_name), ds.index(self.dec_name)
        with torch.no_grad():
            self.log_theta = self.model.theta_log(torch.tensor([self.dec], device=self.device))[0]
        self.batch_size = batch_size

    def predict_missing(self, X_O_counts, lib_size, present):
        torch = self.torch
        n = X_O_counts.shape[0]
        out = np.empty((n, len(self.output_genes)), np.float32)
        m_row = torch.from_numpy(np.asarray(present, np.float32)).to(self.device)[None, :]
        for s in range(0, n, self.batch_size):
            x = torch.from_numpy(np.ascontiguousarray(X_O_counts[s:s + self.batch_size], np.float32)).to(self.device)
            b = x.shape[0]
            mu = self._predict(self.model, x, m_row.expand(b, -1),
                               torch.full((b,), self.enc, device=self.device), torch.full((b,), self.dec, device=self.device))
            out[s:s + b] = mu.cpu().numpy()
        den = self.control_rho[np.asarray(present, bool)].sum()
        lib = np.asarray(lib_size, np.float64).reshape(-1, 1)
        out[:, self.fb_idx] = (lib * (self.control_rho[self.fb_idx] / max(den, 1e-300))[None, :]).astype(np.float32)
        return out

    def sample(self, P, cols):
        # 디코더 데이터셋의 NB dispersion으로 샘플링 (생성모델 출력)
        torch = self.torch
        lt = self.log_theta[torch.from_numpy(np.asarray(cols)).to(self.device)][None, :]
        res = np.empty_like(P)
        for s in range(0, P.shape[0], self.batch_size):
            mu = torch.from_numpy(np.ascontiguousarray(P[s:s + self.batch_size])).to(self.device)
            res[s:s + mu.shape[0]] = self._sample_nb(mu, lt).cpu().numpy()
        return res

    def describe(self):
        return dict(device=str(self.device), enc_dataset=self.enc_name, dec_dataset=self.dec_name)


class EnsembleBackend(Backend):
    # 여러 백엔드의 채운 값을 log 정규화 공간에서 가중 평균한다 (지금은 linear_v1 + vae_v2, 0.5씩)
    # - 정규화: log1p(count * 1e4 / lib), lib = 입력으로 준 유전자 count 합 (vae 단독 모드와 같은 lib)
    # - 구성 모델에는 각자 단독 모드와 같은 입력을 준다 (linear는 자기 입력 유전자와 그 합)
    # - 모든 구성 모델이 'model'로 채우는 유전자만 평균하고, 나머지(fallback 등)는 vae 구성 모델 값을 쓴다
    # - 출력 축, 입력 축, output_source는 vae 구성 모델을 따른다. --sample-counts는 vae의 θ로 음이항 샘플링
    # meta.json 예: {"model_type": "ensemble", "members": [{"path": "../linear_v1", "weight": 0.5},
    #                                                      {"path": "../vae_v2", "weight": 0.5}]}
    model_type = 'ensemble'
    requires_all_inputs = False
    target_sum = 1e4

    def __init__(self, ckpt_dir, meta, seed=0, **opts):
        super().__init__(ckpt_dir, meta, seed)
        self.members, self.weights, self.paths, member_meta = [], [], [], []
        for m in meta['members']:
            path = os.path.normpath(os.path.join(ckpt_dir, m['path']))
            be, mm = load_backend(path, seed=seed, **opts)
            self.members.append(be); self.weights.append(float(m['weight'])); self.paths.append(path); member_meta.append(mm)
        self.meta = dict(meta, member_meta=member_meta)  # 출력 uns에 구성 모델 메타(code_version 등)도 남긴다
        if abs(sum(self.weights) - 1) > 1e-6:
            raise ValueError(f'앙상블 가중치 합이 1이 아님: {self.weights}')
        vaes = [b for b in self.members if b.model_type == 'vae']
        if not vaes:
            raise ValueError('앙상블에 vae 구성 모델이 필요함 (출력 축과 샘플링 기준)')
        self.base = vaes[0]
        self.all_genes = self.input_genes = self.output_genes = self.base.output_genes
        self.output_source = self.base.output_source
        G = len(self.output_genes)
        pos = {g: i for i, g in enumerate(self.output_genes)}
        in_idx, out_idx, is_model = [], [], []
        for b in self.members:
            if not (np.isin(b.input_genes, self.input_genes).all() and np.isin(b.output_genes, self.output_genes).all()):
                raise ValueError('구성 모델의 유전자 축이 vae 구성 모델 축에 포함되지 않음')
            in_idx.append(np.array([pos[g] for g in b.input_genes], np.int64))
            oi = np.array([pos[g] for g in b.output_genes], np.int64)
            out_idx.append(oi)
            im = np.zeros(G, bool); im[oi[b.output_source == 'model']] = True
            is_model.append(im)
        self.avg_pos = np.where(np.logical_and.reduce(is_model))[0]  # 평균하는 유전자 (출력 축 위치)
        self.in_idx = in_idx
        self.avg_sel = []  # 구성 모델 출력에서 avg_pos 순서로 뽑을 열
        for oi in out_idx:
            inv = np.full(G, -1, np.int64); inv[oi] = np.arange(len(oi))
            self.avg_sel.append(inv[self.avg_pos])
        self.warned = False

    def predict_missing(self, X_O_counts, lib_size, present):
        present = np.asarray(present, bool)
        scale = np.float32(self.target_sum) / np.maximum(np.asarray(lib_size, np.float32).reshape(-1, 1), 1)
        acc = np.zeros((X_O_counts.shape[0], len(self.avg_pos)), np.float32)
        out = None
        for b, w, ii, sel in zip(self.members, self.weights, self.in_idx, self.avg_sel):
            if b is self.base:
                out = b.predict_missing(X_O_counts, lib_size, present)
                P = out[:, sel]
            else:
                pres = present[ii]
                if b.requires_all_inputs and not pres.all() and not self.warned:
                    print(f'경고: {b.model_type} 구성 모델 입력 유전자 {int((~pres).sum())}/{len(pres)}개가 입력에 없어 0으로 둠', flush=True)
                    self.warned = True
                X = np.ascontiguousarray(X_O_counts[:, ii])
                P = b.predict_missing(X, X.sum(1), pres)[:, sel]
                del X
            acc += np.float32(w) * np.log1p(P * scale)
            del P
        np.expm1(acc, out=acc)
        np.maximum(acc, 0, out=acc)
        out[:, self.avg_pos] = acc / scale
        return out

    def sample(self, P, cols):
        return self.base.sample(P, cols)

    def describe(self):
        return dict(members=[dict(path=p, model_type=b.model_type, weight=w, **b.describe())
                             for p, b, w in zip(self.paths, self.members, self.weights)],
                    n_averaged_genes=int(len(self.avg_pos)), averaging='log1p(count * 1e4 / lib) 가중 평균')


BACKENDS = {b.model_type: b for b in (LinearBackend, VAEBackend, EnsembleBackend)}


def load_backend(ckpt_dir, **opts):
    # opts: seed, device, enc_dataset, dec_dataset 등. 백엔드가 쓰지 않는 값은 무시 (ensemble은 구성 모델에 그대로 넘김)
    with open(os.path.join(ckpt_dir, META_NAME)) as f:
        meta = json.load(f)
    mt = meta.get('model_type')
    if mt not in BACKENDS:
        raise ValueError(f'알 수 없는 model_type: {mt} (지원: {sorted(BACKENDS)})')
    be = BACKENDS[mt](ckpt_dir, meta, **opts)
    return be, be.meta
