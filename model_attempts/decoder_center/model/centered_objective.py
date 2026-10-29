"""``training.objective.JointObjective`` plus the decoder common-component penalty of ``model_adapters.txpert_centered``.

With ``model.decoder_center.enabled`` the model stores ``decoder_center_penalty`` = mean(m^2) for every training
forward (m: the group mean of the raw decoder output of each centred dataset group). This objective adds
``model.decoder_center.common_weight * penalty`` to the joint loss and logs two components:

    decoder_common        the penalty mean(m^2)
    decoder_common_share  weight * penalty / total loss (its share of the optimized total)

Without that model attribute (plain TxPert, centring off, eval mode) the result is the parent's, unchanged.

Config::

    losses:
      factory: training.centered_objective:build_objective
      state: ...            # as for training.objective
"""
from __future__ import annotations

from .objective import JointObjective


def add_common_penalty(loss, components, model):
    """(loss, components) with the model's stored decoder common-component penalty added; unchanged without it."""
    penalty = getattr(model, "decoder_center_penalty", None) if model is not None else None
    if penalty is None or not getattr(model, "decoder_center_enabled", False):
        return loss, components
    weighted = float(model.decoder_center_common_weight) * penalty
    total = loss + weighted
    components = dict(components, decoder_common=penalty.detach(),
                      decoder_common_share=(weighted / total.abs().clamp_min(1e-12)).detach())
    return total, components


class CenteredJointObjective(JointObjective):
    def __init__(self, config, data):
        super().__init__(config, data)
        self._model = None

    def bind_model(self, model):  # training.runner calls this after building both
        object.__setattr__(self, "_model", model)  # a plain reference, not a registered submodule

    def forward(self, prediction, batch, *, split, step):
        loss, components = super().forward(prediction, batch, split=split, step=step)
        return add_common_penalty(loss, components, self._model)


def build_objective(config, data):
    return CenteredJointObjective(config, data)
