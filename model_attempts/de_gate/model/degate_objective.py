"""JointObjective에 DE 게이트 BCE를 더하는 래퍼. 기존 objective.py는 고치지 않는다.

설정 ``losses.de_gate``를 이 래퍼가 떼어 내고 나머지(state, pds_contrastive, de_mmd)는 JointObjective에 넘긴다::

    losses:
      factory: training.degate_objective:build_objective
      de_gate:
        weight: 1.0                # lambda_gate
        labels_path: .../de_labels_k562_gwps.npz
        pos_weight: 1.0            # 기본 1.0 (p가 실제 확률이 되게). 양성 비율로 계산한 값은 로그에만 남긴다
      state: ...

로스 = JointObjective 로스 + weight * BCE(logit, DE 라벨). 라벨 -1(마스크)과 라벨 행이 없는 조건은 제외한다.
logit은 모델이 마지막 forward에 남긴 ``_gate_logit``이고, bind_model로 모델을 받는다.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .objective import JointObjective

KEYS = {"weight", "labels_path", "pos_weight"}


class DeGateObjective(torch.nn.Module):
    prediction_units = "raw_counts"

    def __init__(self, config, data):
        super().__init__()
        config = dict(config)
        gate = dict(config.pop("de_gate", None) or {})
        if set(gate) - KEYS or "labels_path" not in gate:
            raise ValueError(f"losses.de_gate needs labels_path and supports only {sorted(KEYS)}")
        self.weight = float(gate.get("weight", 1.0))
        self.pos_weight = float(gate.get("pos_weight", 1.0))
        if not (math.isfinite(self.weight) and self.weight >= 0 and math.isfinite(self.pos_weight) and self.pos_weight > 0):
            raise ValueError("de_gate weight must be >= 0 and pos_weight must be > 0")
        self.joint = JointObjective(config, data)
        with np.load(gate["labels_path"], allow_pickle=False) as handle:
            genes, perts, labels = handle["genes"].astype(str), handle["perts"].astype(str), handle["labels"]
        if list(genes) != list(data.spec["gene_names"]):
            raise ValueError("DE 라벨의 유전자 순서가 모델 유전자 축과 다르다")
        self.row = {str(name): i for i, name in enumerate(perts)}
        self.condition_genes = dict(data.prepared.condition_genes)
        self.register_buffer("labels", torch.from_numpy(np.asarray(labels, dtype=np.int8)), persistent=False)
        self.suggested_pos_weight = self._log_label_stats(data)
        self._model = None

    def _log_label_stats(self, data):
        """학습 조건 라벨의 양성 비율과 그 비율로 계산한 pos_weight(음성/양성)를 로그에 남긴다."""
        rows = [self.row[self.condition_genes[c]] for c in data.prepared.training_perturbations
                if self.condition_genes.get(c) in self.row]
        sub = self.labels[rows] if rows else self.labels[:0]
        pos, neg = int((sub == 1).sum()), int((sub == 0).sum())
        suggested = neg / max(pos, 1)
        print(f"[de_gate] 학습 조건 {len(data.prepared.training_perturbations)}개 중 라벨 있는 {len(rows)}개, "
              f"양성 {pos}, 음성 {neg}, 양성 비율 {pos / max(pos + neg, 1):.4f}, "
              f"양성 비율로 계산한 pos_weight {suggested:.2f} (사용값 {self.pos_weight:g})", flush=True)
        return suggested

    def bind_model(self, model):
        if not hasattr(model, "gate_head"):
            raise ValueError("losses.de_gate needs a model built with model.de_gate.enabled: true")
        object.__setattr__(self, "_model", model)  # 서브모듈로 등록하지 않는다 (파라미터 중복 방지)

    def forward(self, prediction, batch, *, split, step):
        total, components = self.joint(prediction, batch, split=split, step=step)
        if self._model is None or self._model._gate_logit is None:
            raise RuntimeError("bind_model(model)을 부르고 모델 forward를 먼저 실행해야 한다")
        logit = self._model._gate_logit  # (sets, set_len, genes)
        rows = [self.row.get(self.condition_genes.get(name), -1) for name in batch["conditions"]]
        index = torch.tensor(rows, device=logit.device)
        labels = self.labels[index.clamp_min(0)].to(logit.device)
        labels = torch.where((index >= 0)[:, None], labels, torch.full_like(labels, -1))[:, None, :]
        valid = (labels >= 0).expand_as(logit)
        target = (labels == 1).to(logit.dtype).expand_as(logit)
        weight = torch.where(target > 0, torch.full_like(logit, self.pos_weight), torch.ones_like(logit))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, weight=weight, reduction="none")
        count = valid.sum().clamp_min(1)
        bce = (loss * valid).sum() / count
        with torch.no_grad():
            p = torch.sigmoid(logit)
            pos, neg = valid & (target > 0), valid & (target == 0)
            unlabeled = ~valid
            mean = lambda m: (p * m).sum() / m.sum().clamp_min(1)  # noqa: E731
            components = {**components, "gate_bce": bce.detach(), "gate_p_pos": mean(pos), "gate_p_neg": mean(neg),
                          "gate_p_unlabeled": mean(unlabeled), "gate_labeled_fraction": valid.float().mean()}
            if batch.get("observed_mask") is not None:  # 데이터셋이 측정한 유전자와 아닌 유전자의 p 분포
                measured = batch["observed_mask"].to(logit.device).bool()[:, None, :].expand_as(logit)
                components.update(gate_p_measured=mean(measured), gate_p_unmeasured=mean(~measured),
                                  gate_gt05_measured=((p > 0.5) & measured).sum() / measured.sum().clamp_min(1),
                                  gate_gt05_unmeasured=((p > 0.5) & ~measured).sum() / (~measured).sum().clamp_min(1))
        return total + self.weight * bce, components


def build_objective(config, data):
    return DeGateObjective(config, data)
