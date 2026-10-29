"""TxPert B2에 DE 게이트(이 유전자가 변하나?)를 더한 모델. 기존 txpert.py는 고치지 않는다.

설정 ``model.de_gate``(기본 꺼짐)::

    model:
      factory: model_adapters.txpert_degate:build_model
      de_gate:
        enabled: true
        mode: soft_norm     # soft_norm(기본), soft, hard
        hard_threshold: 0.5
        bias_init: -1.9     # b의 초기값 (logit 단위)
        measured_path: .../measured_k562_gwps.npy   # soft_norm의 두 그룹(측정/미측정 유전자)을 나누는 bool (유전자,) 배열

구조::

    logit_j = gate_head(z)_j + a * |prior_j| + b      gate_head: latent -> hidden -> genes, 마지막 층 0 초기화
    p = sigmoid(logit)
    gated = p * log_fold          log_fold = 디코더 항 + prior 항 + 나머지 항 전체. softmax 직전에 곱한다

모드:
  soft_norm  gated를 표적(세포 세트)마다 두 그룹(측정 유전자, 미측정 유전자)에서 따로
             gated x RMS(log_fold) / RMS(gated) 로 되돌려 원래 크기를 맞춘다. p가 균일하면 게이트 없는 모델과 출력이 같다.
             measured_path가 없으면 전체를 한 그룹으로 본다. 학습과 추론에 모두 적용된다.
  soft       gated를 그대로 쓴다. 크기가 p만큼 줄어든다.
  hard       model.eval()에서만 p > hard_threshold 인 유전자의 log fold를 남기고 나머지는 0으로 한다. 학습은 soft.

``|prior|`` 는 gain을 곱하기 전의 ``inputs["prior_log_fold"]`` 절댓값이다.
꺼져 있으면 기존 ``txpert.build_model`` 결과를 그대로 돌려준다.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import txpert
from utils import expand_path

from .txpert import TxPertCountModel, counts_to_log1p_cp

GATE_KEYS = {"enabled", "mode", "hard_threshold", "bias_init", "measured_path"}
MODES = ("soft_norm", "soft", "hard")
RMS_FLOOR, RMS_EPS = 1e-12, 1e-8  # sqrt의 기울기 보호용 바닥값, 분모 eps


class TxPertDeGateModel(TxPertCountModel):
    """TxPertCountModel + DE 게이트. 게이트 모듈은 build_model이 기본 모델을 만든 뒤에 붙인다."""

    def attach_gate(self, *, hidden: int, dropout: float, mode: str, hard_threshold: float, bias_init: float,
                    group_mask: torch.Tensor | None = None):
        genes, latent = len(self.spec["gene_names"]), self.decoder[0].in_features
        self.gate_head = nn.Sequential(nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, genes))
        nn.init.zeros_(self.gate_head[-1].weight)  # 기존 디코더와 같은 방식: 시작점은 a|prior| + b 뿐
        nn.init.zeros_(self.gate_head[-1].bias)
        self.gate_a = nn.Parameter(torch.zeros(()))
        self.gate_b = nn.Parameter(torch.tensor(float(bias_init)))
        self.gate_mode, self.gate_hard_threshold = mode, float(hard_threshold)
        self._gate_logit = None  # 마지막 forward의 (sets, set_len, genes) logit. 로스 래퍼가 읽는다.
        if group_mask is not None and group_mask.shape != (genes,):
            raise ValueError("de_gate.measured_path must hold one bool per model gene")
        self.register_buffer("gate_group", None if group_mask is None else group_mask.bool().clone())

    def _match_scale(self, log_fold: torch.Tensor, gated: torch.Tensor) -> torch.Tensor:
        """gated를 표적(세트)마다, 유전자 그룹마다 log_fold의 RMS에 맞춘다."""
        group = self.gate_group
        masks = [torch.ones_like(log_fold[0, 0], dtype=torch.bool)] if group is None else [group, ~group]
        out = torch.zeros_like(gated)
        for mask in masks:
            if not bool(mask.any()):
                continue
            m = mask.to(log_fold.dtype)
            n = log_fold.shape[1] * m.sum()
            rms_original = ((log_fold.square() * m).sum((1, 2)) / n + RMS_FLOOR).sqrt()
            rms_gated = ((gated.square() * m).sum((1, 2)) / n + RMS_FLOOR).sqrt()
            scale = rms_original / (rms_gated + RMS_EPS)
            out = out + gated * m * scale[:, None, None]
        return out

    def forward(self, inputs):
        control, perturbation = inputs["ctrl_cell_emb"], inputs["pert_emb"]
        set_len, genes = self.spec["cell_set_len"], len(self.spec["gene_names"])
        if control.shape[-1] != genes or control.numel() % (set_len * genes):
            raise ValueError("TxPertCountModel expects complete cell sets on the model gene axis")
        if bool((control < 0).any()) or not bool(torch.isfinite(control).all()):
            raise ValueError("TxPertCountModel control counts must be finite and nonnegative")
        basal = control.reshape(-1, set_len, genes)
        names = inputs["pert_names"]
        if len(names) != basal.shape[0]:
            raise ValueError("pert_names must give one target gene per cell set")
        encoder_input = counts_to_log1p_cp(basal, self.target_sum)
        if self.basal_input == "set_mean":
            encoder_input = encoder_input.mean(1, keepdim=True).expand_as(encoder_input)
        latents, found = self.gnn(names)
        fallback = self.esm_fallback(perturbation.reshape(-1, set_len, perturbation.shape[-1])[:, 0])
        z_pert = torch.where(found[:, None], latents, fallback)
        if self.pert_recon_head is not None:
            feature = perturbation.reshape(-1, set_len, perturbation.shape[-1])[:, 0].detach()
            self._prior_recon = (self.pert_recon_head(z_pert), feature)
        z = self.basal(encoder_input) + z_pert[:, None, :]
        if self.set_encoder is not None:
            z = z + self.set_encoder(z)
        raw = self.decoder(z)
        log_fold = self.max_log_fold * torch.tanh(raw / self.max_log_fold)
        if self.correction_mask is not None:
            log_fold = log_fold * self.correction_mask.to(device=log_fold.device, dtype=log_fold.dtype)
        if self.correction_scale is not None:
            log_fold = log_fold * float(self.correction_scale)
        if self.shared_response is not None:
            log_fold = log_fold + self.shared_response
        prior = inputs["prior_log_fold"].to(device=basal.device, dtype=log_fold.dtype)
        if prior.shape != (basal.shape[0], genes) or not bool(torch.isfinite(prior).all()):
            raise ValueError("prior_log_fold must be one finite log fold per cell set and output gene")
        log_fold = log_fold + self.prior_log_gain.exp() * prior[:, None, :]
        if self.target_positions is not None:
            positions = torch.tensor([self.target_positions.get(self.target_aliases.get(name, name),
                                      self.target_positions.get(name, -1)) for name in names], device=basal.device)
            rows = torch.nonzero(positions >= 0).flatten()
            depth = -nn.functional.softplus(self.knockdown(z_pert)).squeeze(-1).clamp(max=2 * self.max_log_fold)
            channel = torch.zeros(basal.shape[0], genes, device=basal.device, dtype=log_fold.dtype)
            channel = channel.index_put((rows, positions[rows]), depth[rows])
            log_fold = log_fold + channel[:, None, :]
        # DE 게이트: 최종 log fold 전체에 p를 곱한다.
        logit = self.gate_head(z) + self.gate_a * prior.abs()[:, None, :] + self.gate_b
        self._gate_logit = logit
        p = torch.sigmoid(logit)
        if not self.training and self.gate_mode == "hard":
            p = (p > self.gate_hard_threshold).to(p.dtype)
        gated = log_fold * p
        log_fold = self._match_scale(log_fold, gated) if self.gate_mode == "soft_norm" else gated
        library = basal.sum(-1, keepdim=True)
        if self.pseudocount == 0:
            weights = basal * torch.exp(log_fold - log_fold.amax(-1, keepdim=True).detach())
            out = library * weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        else:
            out = library * torch.softmax(torch.log(basal + self.pseudocount) + log_fold, dim=-1)
        return out.reshape(control.shape)


def build_model(config, spec):
    original = dict(config)
    config = dict(config)
    gate = dict(config.pop("de_gate", None) or {})
    unknown = set(gate) - GATE_KEYS
    if unknown:
        raise ValueError(f"Unsupported de_gate settings: {sorted(unknown)}")
    model = txpert.build_model(config, spec)  # 게이트가 없으면 기존 모델 그대로
    if "de_gate" in original:
        model.model_config = dict(original)  # 체크포인트가 비교하는 model_config는 설정 원본과 같아야 한다
    if not gate.get("enabled", False):
        return model
    if not config.get("pert_prior", False):
        raise ValueError("model.de_gate needs model.pert_prior: true (the gate reads |prior|)")
    mode = gate.get("mode", "soft_norm")
    if mode not in MODES:
        raise ValueError(f"de_gate.mode must be one of {MODES}")
    group_mask = None
    if gate.get("measured_path"):
        group_mask = torch.from_numpy(np.load(expand_path(gate["measured_path"])).astype(bool))
    model.__class__ = TxPertDeGateModel  # 기본 모델의 파라미터 초기화 순서를 건드리지 않으려고 만든 뒤에 바꾼다
    model.attach_gate(hidden=int(config.get("hidden_dim", 512)), dropout=float(config.get("dropout", 0.1)),
                      mode=mode, hard_threshold=float(gate.get("hard_threshold", 0.5)),
                      bias_init=float(gate.get("bias_init", -1.9)), group_mask=group_mask)
    model.provenance = {**model.provenance, "model_source_path": "scripts/model_adapters/txpert_degate.py",
                        "de_gate": {**gate, "mode": mode}}
    return model
