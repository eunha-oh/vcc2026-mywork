"""L_SHRINK: 실측에서 유의하지 않은 gene의 예측 |lfc|를 직접 누르는 로스.

팀 저장소의 `training/losses.py`에 추가했던 함수를 그대로 옮겼다.
팀 `JointObjective`에 연결하는 부분은 `objective_integration.patch`에 있다.
"""
from __future__ import annotations

import torch


def de_shrink_loss(
    prediction_log: torch.Tensor,
    control_log: torch.Tensor,
    significant: torch.Tensor,
    *,
    lfc_epsilon: float,
    observed_mask: torch.Tensor | None = None,
    minimum_significant_genes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """실측 유의 집합 밖 gene의 예측 lfc 크기를 직접 누른다.

    `L = mean_p[ mean_{g not in S_p, 측정 gene} |lfc_pred(p,g)| ]`

    분류 없이 크기 자체를 0 쪽으로 민다. S_p는 DE 참조의 up/down 상태에서 오고,
    예측 lfc는 참조와 같은 식
    `log2((mean_pred + lfc_epsilon) / (mean_control + lfc_epsilon))`으로 만든다.
    `lfc_epsilon`은 참조를 만든 `de_mmd.lfc_epsilon`과 같아야 비교가 성립한다.

    `|lfc|`를 clip하지 않는다. clip하면 그 위에서 gradient가 0이 되어 가장 크게 틀린
    gene을 누르지 못한다.

    주의: 공식 `lfc_nmae`는 실측 유의 gene만 채점한다. 이 항이 누르는 gene은 대부분
    그 채점 밖이므로, 채점 대상 gene의 lfc는 간접적으로만 움직인다.

    반환값은 `(loss, significant_magnitude)`다. significant_magnitude는 S_p 안쪽 예측
    `|lfc|`의 평균이며 gradient가 없는 기록용이다. 이 항이 유의 gene까지 같이 누르는지 본다.
    """
    if prediction_log.ndim != 3 or control_log.shape != prediction_log.shape:
        raise ValueError(
            "de_shrink expects prediction/control [conditions, cells, genes] of the same shape, "
            f"got {tuple(prediction_log.shape)} and {tuple(control_log.shape)}"
        )
    conditions, _, gene_count = prediction_log.shape
    if significant.dtype != torch.bool or significant.shape != (conditions, gene_count):
        raise ValueError("de_shrink significant must be boolean [conditions, genes]")
    if lfc_epsilon <= 0:
        raise ValueError("de_shrink lfc_epsilon must be positive")
    if minimum_significant_genes < 1:
        raise ValueError("minimum_significant_genes must be at least 1")

    label = significant.to(prediction_log.dtype)
    valid = label.sum(dim=-1) >= float(minimum_significant_genes)
    zero = prediction_log.new_zeros(())
    if not bool(valid.any()):
        return zero, zero
    if observed_mask is None:
        observed = torch.ones_like(label)
    else:
        if observed_mask.shape != (conditions, gene_count):
            raise ValueError("de_shrink observed_mask must be [conditions, genes]")
        observed = observed_mask.to(prediction_log.dtype)
    observed = observed * valid.to(prediction_log.dtype)[:, None]
    outside = observed * (1.0 - label)
    inside = observed * label
    if not bool(outside.any()):
        return zero, zero

    predicted_mean = torch.expm1(prediction_log).mean(dim=1).clamp_min(0.0)
    control_mean = torch.expm1(control_log).mean(dim=1).clamp_min(0.0)
    magnitude = torch.log2((predicted_mean + lfc_epsilon) / (control_mean + lfc_epsilon)).abs()
    per_condition = (magnitude * outside).sum(dim=-1) / outside.sum(dim=-1).clamp_min(1.0)
    loss = per_condition[valid].mean()
    significant_magnitude = ((magnitude.detach() * inside).sum(dim=-1)
                             / inside.sum(dim=-1).clamp_min(1.0))[valid].mean()
    return loss, significant_magnitude
