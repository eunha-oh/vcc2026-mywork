"""TxPert count model whose decoder output is centred within each cell line of a training batch.

The submission removes the perturbation-independent part of the prediction (``--debias-perts -1``: the
leave-one-out mean over all targets, per control cell and gene). Training saw that common part in every target,
so the decoder learned it. Here, in training only, the decoder output d_t of each cell set t becomes

    d_t - mean_{s in the batch, same dataset_id, s != t} d_s

before the bounded head and the prior term (``log_fold = tanh-bounded d + exp(gain) * prior``). The mean is taken per
cell position (shared controls put the same control cells at the same position for one dataset) and per gene over
the other sets that measured the gene; it is not detached. A dataset with fewer than ``min_targets`` sets in the
batch is left as it is. In eval mode (validation, submission) nothing changes: the submission's debias does that job.

Centring makes the loss blind to the common part m of a dataset group (the group mean of the raw decoder output), so
nothing would hold it down, and eval mode, which does not centre, would carry it into the prediction. A penalty
``common_weight * mean(m^2)`` over the centred groups (masked, per cell position and gene, not detached) keeps it
small. The model stores it per training forward (``decoder_center_penalty``); ``training.centered_objective`` adds it
to the loss.

Config (``model`` section)::

    factory: model_adapters.txpert_centered:build_model
    decoder_center:
      enabled: true      # false: the model is exactly model_adapters.txpert
      min_targets: 4     # sets of one dataset in a batch needed to centre them
      log_interval: 400  # print centring statistics every N training forwards (0: never)
      common_weight: 1.0 # weight of the common-component penalty (training.centered_objective adds it)

The model needs ``inputs["dataset_id"]`` (one integer per set) and ``inputs["observed_mask"]`` (sets x genes), which
``data_adapters.decoder_center_data`` adds. The state dict keys are those of ``model_adapters.txpert``.
"""
from __future__ import annotations

import copy

import torch

from .txpert import TxPertCountModel, build_model as build_txpert


def center_decoder_output(output: torch.Tensor, dataset_id: torch.Tensor, observed: torch.Tensor | None,
                          min_targets: int) -> tuple[torch.Tensor, dict]:
    """Leave-one-out centring of ``output`` (sets, cells, genes) within each dataset id; returns (output, stats).

    ``stats["penalty_sum"] / stats["penalty_count"]`` is mean(m^2) over the centred groups, m the group mean."""
    sets, _, genes = output.shape
    if dataset_id.shape != (sets,):
        raise ValueError("dataset_id must give one id per cell set")
    mask = torch.ones(sets, genes, dtype=torch.bool, device=output.device) if observed is None else observed.to(
        device=output.device, dtype=torch.bool)
    if mask.shape != (sets, genes):
        raise ValueError("observed_mask must have shape (cell sets, genes)")
    weight = mask.to(output.dtype)[:, None, :]  # (sets, 1, genes)
    centred = output
    stats = {"sets": sets, "sets_centred": 0, "common_sq": 0.0, "total_sq": 0.0, "values": 0,
             "penalty_sum": output.new_zeros(()), "penalty_count": 0}
    for value in torch.unique(dataset_id):
        members = torch.nonzero(dataset_id == value).flatten()
        group = output[members]
        group_weight = weight[members]
        n = int(members.numel())
        with torch.no_grad():  # statistics only: size of the group's common component before centring
            mean = (group * group_weight).sum(0) / group_weight.sum(0).clamp_min(1.0)
            stats["common_sq"] += float((mean.square() * group_weight[0]).sum()) * n
            stats["total_sq"] += float((group.square() * group_weight).sum())
            stats["values"] += int(group_weight.expand_as(group).sum())
        if n < min_targets:
            continue
        total = (group * group_weight).sum(0, keepdim=True)
        count = group_weight.sum(0, keepdim=True)
        common = total / count.clamp_min(1.0)  # m: the group's common part, with gradient
        measured = (count > 0).to(output.dtype).expand_as(common)
        stats["penalty_sum"] = stats["penalty_sum"] + (common.square() * measured).sum()
        stats["penalty_count"] += int(measured.sum())
        others = count - group_weight                                  # sets other than t that measured the gene
        loo = (total - group * group_weight) / others.clamp_min(1.0)
        apply = (group_weight > 0) & (others > 0)
        new_group = torch.where(apply, group - loo, group)
        if centred is output:
            centred = output.clone()
        centred = centred.index_copy(0, members, new_group)
        stats["sets_centred"] += n
    return centred, stats


class CenteredTxPertCountModel(TxPertCountModel):
    """``TxPertCountModel`` with the decoder centring hook; created by ``build_model`` (class swap, same parameters)."""

    def _setup_center(self, settings: dict) -> None:
        self.decoder_center_enabled = bool(settings.get("enabled", False))
        self.decoder_center_min_targets = int(settings.get("min_targets", 4))
        self.decoder_center_log_interval = int(settings.get("log_interval", 0))
        self.decoder_center_common_weight = float(settings.get("common_weight", 1.0))
        if not self.decoder_center_common_weight >= 0:
            raise ValueError("decoder_center.common_weight must be >= 0")
        self.decoder_center_penalty = None
        if self.decoder_center_min_targets < 2:
            raise ValueError("decoder_center.min_targets must be at least 2 (leave-one-out needs another set)")
        if self.decoder_center_log_interval < 0:
            raise ValueError("decoder_center.log_interval must be >= 0")
        self._center_inputs = None
        self.decoder_center_totals = {"forwards": 0, "sets": 0, "sets_centred": 0, "common_sq": 0.0, "total_sq": 0.0,
                                      "values": 0}
        self.decoder_center_last = None
        if self.decoder_center_enabled or self.decoder_center_log_interval:
            self.decoder.register_forward_hook(self._decoder_hook)

    def forward(self, inputs):
        self._center_inputs = inputs if self.training else None
        self.decoder_center_penalty = None  # set by the hook for this forward (training with centring only)
        try:
            return super().forward(inputs)
        finally:
            self._center_inputs = None

    def _decoder_hook(self, module, args, output):
        inputs = self._center_inputs
        if inputs is None:
            return None  # eval mode or no context: output unchanged
        if "dataset_id" not in inputs:
            raise ValueError("decoder_center needs inputs['dataset_id'] (data_adapters.decoder_center_data)")
        dataset_id = inputs["dataset_id"].to(device=output.device)
        observed = inputs.get("observed_mask")
        min_targets = self.decoder_center_min_targets if self.decoder_center_enabled else output.shape[0] + 1
        centred, stats = center_decoder_output(output, dataset_id, observed, min_targets)
        if self.decoder_center_enabled:
            self.decoder_center_penalty = stats["penalty_sum"] / max(stats["penalty_count"], 1)
        self._record(stats)
        return centred if self.decoder_center_enabled else None

    def _record(self, stats: dict) -> None:
        totals = self.decoder_center_totals
        totals["forwards"] += 1
        for key in ("sets", "sets_centred", "common_sq", "total_sq", "values"):
            totals[key] += stats[key]
        self.decoder_center_last = stats
        interval = self.decoder_center_log_interval
        if interval and totals["forwards"] % interval == 0:
            print(f"[decoder_center] forwards={totals['forwards']} enabled={self.decoder_center_enabled} "
                  f"centred_fraction={totals['sets_centred'] / max(totals['sets'], 1):.3f} "
                  f"common_rms={(totals['common_sq'] / max(totals['values'], 1)) ** 0.5:.4g} "
                  f"total_rms={(totals['total_sq'] / max(totals['values'], 1)) ** 0.5:.4g}", flush=True)


def build_model(config, spec):
    settings = dict(config.get("decoder_center") or {})
    unknown = set(settings) - {"enabled", "min_targets", "log_interval", "common_weight"}
    if unknown:
        raise ValueError(f"Unsupported decoder_center settings: {sorted(unknown)}")
    base_config = copy.deepcopy(dict(config))
    base_config.pop("decoder_center", None)
    base_config["factory"] = "model_adapters.txpert:build_model"
    model = build_txpert(base_config, spec)
    model.__class__ = CenteredTxPertCountModel
    model._setup_center(settings)
    model.model_config = dict(config)
    model.provenance = {**model.provenance, "model_source_path": "scripts/model_adapters/txpert_centered.py",
                        "decoder_center": {"enabled": model.decoder_center_enabled,
                                           "min_targets": model.decoder_center_min_targets,
                                           "common_weight": model.decoder_center_common_weight}}
    return model
