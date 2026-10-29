"""Pert-prior data whose model inputs also carry the dataset of each cell set and its measured-gene mask.

``model_adapters.txpert_centered`` centres the decoder output within each dataset of a training batch, but the
engine hands the model only ``batch["inputs"]``. This wrapper copies two batch fields into the inputs:

    inputs["dataset_id"]     int64 (sets,)   index of batch["dataset_id"] in the config's dataset order
    inputs["observed_mask"]  bool (sets, genes)  batch["observed_mask"]

Everything else is ``data_adapters.pert_prior`` unchanged. Config (``data`` section)::

    factory: data_adapters.decoder_center_data:build_data
    base_factory: data_adapters.multi:build_data   # passed on to pert_prior
    pert_prior: {...}                              # as for data_adapters.pert_prior
"""
from __future__ import annotations

import copy

import torch

from .pert_prior import build_data as build_pert_prior


class DecoderCenterData:
    def __init__(self, base, dataset_names):
        self.base = base
        self.dataset_index = {name: i for i, name in enumerate(dataset_names)}

    def _inject(self, batch):
        names = batch["dataset_id"]
        unknown = sorted(set(names) - set(self.dataset_index))
        if unknown:
            raise ValueError(f"batch datasets not in the config: {unknown}")
        inputs = dict(batch["inputs"], dataset_id=torch.tensor([self.dataset_index[n] for n in names], dtype=torch.long))
        if batch.get("observed_mask") is not None:
            inputs["observed_mask"] = torch.as_tensor(batch["observed_mask"]).to(torch.bool)
        return dict(batch, inputs=inputs)

    def train_batches(self):
        for batch in self.base.train_batches():
            yield self._inject(batch)

    def validation_batches(self):
        for batch in self.base.validation_batches():
            yield self._inject(batch)

    def __getattr__(self, name):
        return getattr(self.base, name)


def build_data(config, plan):
    base_config = copy.deepcopy(config)
    base_config["factory"] = "data_adapters.pert_prior:build_data"
    base = build_pert_prior(base_config, plan)
    return DecoderCenterData(base, [entry["name"] for entry in config["datasets"]])
