"""DE 게이트 모델(model_adapters.txpert_degate): 끔 동일성, 게이트 극한값, hard 모드, 기울기."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_adapters import txpert, txpert_degate  # noqa: E402

GENES, SETS, SET_LEN, LATENT = 20, 3, 4, 16
SPEC = {"gene_names": [f"g{i}" for i in range(GENES)], "cell_set_len": SET_LEN, "perturbation_dim": 8}


class _StubGraph(torch.nn.Module):
    """PyG 없이 모델을 만들기 위한 GNN 대용: 모든 표적이 그래프에 없다고 보고 ESM2 fallback을 쓰게 한다."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim, self.provenance = out_dim, {}

    def forward(self, names):
        return torch.zeros(len(names), self.out_dim), torch.zeros(len(names), dtype=torch.bool)


def build(gate=None, seed=1, mask_path=None):
    config = dict(latent_dim=LATENT, hidden_dim=32, dropout=0.0, pert_prior=True)
    if gate is not None:
        config["de_gate"] = dict(gate, **({"measured_path": mask_path} if mask_path else {}))
    torch.manual_seed(seed)
    with mock.patch.object(txpert, "build_graph_encoder", lambda config, spec, out_dim: _StubGraph(out_dim)):
        return txpert_degate.build_model(config, SPEC).eval()


class DeGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.inputs = dict(ctrl_cell_emb=torch.rand(SETS * SET_LEN, GENES) * 10 + 1,
                           pert_emb=torch.randn(SETS * SET_LEN, 8), pert_names=["a", "b", "c"],
                           prior_log_fold=torch.randn(SETS, GENES))

    def test_flag_off_is_identical_to_base_model(self):
        base, off = build(), build({"enabled": False})
        self.assertIs(type(off), txpert.TxPertCountModel)
        self.assertTrue(torch.equal(base(self.inputs), off(self.inputs)))

    def test_enabled_keeps_base_initialization_and_adds_gate_parameters(self):
        base, on = build(), build({"enabled": True})
        shared = [(n, p) for n, p in on.named_parameters() if not n.startswith("gate")]
        self.assertTrue(all(torch.equal(p, q) for (_, p), (_, q) in zip(base.named_parameters(), shared)))
        self.assertEqual(sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in base.parameters()),
                         sum(p.numel() for p in on.gate_head.parameters()) + 2)

    def test_gate_limits_in_soft_mode(self):
        base, on = build(), build({"enabled": True, "mode": "soft"})
        with torch.no_grad():
            on.gate_b.fill_(30.0)  # p = 1: 게이트 없는 모델과 같다
        self.assertTrue(torch.allclose(on(self.inputs), base(self.inputs), atol=1e-5))
        with torch.no_grad():
            on.gate_b.fill_(-30.0)  # p = 0: log fold = 0
        ctrl = self.inputs["ctrl_cell_emb"].reshape(SETS, SET_LEN, GENES)
        expected = ctrl.sum(-1, keepdim=True) * torch.softmax(torch.log(ctrl + on.pseudocount), dim=-1)
        self.assertTrue(torch.allclose(on(self.inputs).reshape(SETS, SET_LEN, GENES), expected, rtol=1e-4, atol=1e-4))
        self.assertLess(float(torch.sigmoid(on._gate_logit).max()), 1e-6)

    def test_hard_mode_applies_only_in_eval(self):
        on = build({"enabled": True, "mode": "hard"})
        self.assertEqual(on.gate_mode, "hard")
        on.train()
        trained = on(self.inputs)
        on.eval()
        evaluated = on(self.inputs)
        self.assertFalse(torch.equal(trained, evaluated))
        self.assertTrue(set(torch.unique((torch.sigmoid(on._gate_logit) > 0.5).float()).tolist()) <= {0.0, 1.0})

    def test_gradients_reach_gate_with_zero_initialised_head(self):
        on = build({"enabled": True})
        on.train()
        on(self.inputs)
        torch.nn.functional.binary_cross_entropy_with_logits(on._gate_logit, torch.ones_like(on._gate_logit)).backward()
        self.assertTrue(on.gate_b.grad.abs() > 0 and on.gate_a.grad.abs() > 0)
        self.assertGreater(float(on.gate_head[-1].weight.grad.abs().sum()), 0.0)

    def test_default_mode_is_soft_norm(self):
        self.assertEqual(build({"enabled": True}).gate_mode, "soft_norm")

    def test_soft_norm_with_uniform_p_equals_gate_off(self):
        base = build()
        for logit in (-1.9, -6.0, 3.0):  # 초기값, 아주 작은 p, 큰 p
            on = build({"enabled": True, "bias_init": logit})
            self.assertTrue(torch.allclose(on(self.inputs), base(self.inputs), atol=1e-5), logit)

    def test_soft_norm_uniform_p_equals_gate_off_with_groups(self):
        import os
        import tempfile
        import numpy as np
        mask = np.zeros(GENES, dtype=bool)
        mask[:7] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "measured.npy")
            np.save(path, mask)
            on = build({"enabled": True, "bias_init": -3.0}, mask_path=path)
        self.assertTrue(torch.allclose(on(self.inputs), build()(self.inputs), atol=1e-5))

    def test_soft_norm_restores_rms_per_group_when_p_varies(self):
        import os
        import tempfile
        import numpy as np
        mask = np.zeros(GENES, dtype=bool)
        mask[:7] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "measured.npy")
            np.save(path, mask)
            on = build({"enabled": True}, mask_path=path)
        with torch.no_grad():
            on.gate_a.fill_(2.0)  # |prior|에 따라 p가 달라진다
        seen = {}
        original = on._match_scale
        on._match_scale = lambda lf, g: seen.setdefault("pair", (lf.detach(), g.detach(), original(lf, g).detach()))[2]
        on(self.inputs)
        log_fold, gated, normed = seen["pair"]
        self.assertFalse(torch.allclose(gated, log_fold))
        for sel in (torch.tensor(mask), torch.tensor(~mask)):
            rms = lambda x: x[:, :, sel].square().mean((1, 2)).sqrt()  # noqa: E731
            self.assertTrue(torch.allclose(rms(normed), rms(log_fold), rtol=1e-4))

    def test_soft_norm_is_finite_when_log_fold_is_zero(self):
        on = build({"enabled": True})
        inputs = dict(self.inputs, prior_log_fold=torch.zeros(SETS, GENES))
        on.train()
        out = on(inputs)
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in on.parameters() if p.grad is not None))

    def test_gate_requires_prior(self):
        with mock.patch.object(txpert, "build_graph_encoder", lambda config, spec, out_dim: _StubGraph(out_dim)):
            with self.assertRaises(ValueError):
                txpert_degate.build_model(dict(latent_dim=LATENT, de_gate={"enabled": True}), SPEC)


if __name__ == "__main__":
    unittest.main()
