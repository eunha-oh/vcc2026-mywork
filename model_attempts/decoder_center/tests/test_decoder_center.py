"""model_adapters.txpert_centered and data_adapters.decoder_center_data: training-only decoder centring per dataset."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from data_adapters.decoder_center_data import DecoderCenterData  # noqa: E402
from model_adapters import txpert, txpert_centered  # noqa: E402

GENES = ("P0", "P1", "P2", "G3", "G4", "G5", "G6", "G7")
SPEC = {"gene_names": GENES, "perturbation_dim": 4, "cell_set_len": 3}
NAMES = ("P0", "P1", "P2", "G3", "G4", "G5")


class DecoderCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Not removed afterwards: the tests never delete files (team rule); the files are a few kB.
        root = Path(tempfile.mkdtemp(prefix="decoder_center_test_"))
        cache = root / "esm.pt"
        generator = torch.Generator().manual_seed(7)
        torch.save({name: torch.randn(4, generator=generator) for name in GENES}, cache)
        graphs = root / "graphs"
        graphs.mkdir()
        symbols = list(GENES)
        pd.DataFrame({"symbol": symbols, "index": range(len(symbols))}).to_csv(graphs / "nodes.csv", index=False)
        edges = np.array([[0, 3, 1, 4, 2, 5], [3, 0, 4, 1, 5, 2]], dtype=np.int64)
        np.savez(graphs / "edges_string_top20.npz", edge_index=edges, edge_weight=np.ones(edges.shape[1], np.float32),
                 num_nodes=len(symbols))
        cls.base = {"factory": "model_adapters.txpert:build_model", "graph_dir": str(graphs), "graph": "string_top20",
                    "gnn_layers": 2, "gnn_hidden": 6, "gnn_heads": 2, "gnn_dropout": 0.0, "node_feature_dim": 5,
                    "esm2_cache": str(cache), "esm2_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                    "latent_dim": 8, "hidden_dim": 12, "dropout": 0.0, "pert_prior": True, "pert_prior_gain_init": 1.5}

    def models(self, enabled=True, min_targets=4, log_interval=0):
        """(plain TxPert, centred TxPert) with identical parameters and a non-zero decoder head."""
        torch.manual_seed(0)
        plain = txpert.build_model(dict(self.base), SPEC)
        with torch.no_grad():
            plain.decoder[-1].weight.normal_(0.0, 0.5)
            plain.decoder[-1].bias.normal_(0.0, 0.5)
        config = dict(self.base, factory="model_adapters.txpert_centered:build_model",
                      decoder_center={"enabled": enabled, "min_targets": min_targets, "log_interval": log_interval})
        centred = txpert_centered.build_model(config, SPEC)
        centred.load_state_dict(plain.state_dict(), strict=True)  # same keys as model_adapters.txpert
        return plain, centred

    def inputs(self, dataset_id, seed=0, mask=None):
        generator = torch.Generator().manual_seed(seed)
        sets = len(dataset_id)
        names = NAMES[:sets]
        # shared controls: the sets of one dataset hold the same control cells
        pools = {d: torch.poisson(torch.full((3, len(GENES)), 6.0), generator=generator) for d in set(dataset_id)}
        control = torch.stack([pools[d] for d in dataset_id])
        perturbation = torch.randn(sets, 1, 4, generator=generator).expand(sets, 3, 4).contiguous()
        prior = 0.3 * torch.randn(sets, len(GENES), generator=generator)
        observed = torch.ones(sets, len(GENES), dtype=torch.bool) if mask is None else mask
        return {"ctrl_cell_emb": control, "pert_emb": perturbation, "pert_names": tuple(names),
                "prior_log_fold": prior, "dataset_id": torch.tensor(dataset_id), "observed_mask": observed}

    def test_a_off_is_identical(self):
        for log_interval in (0, 1):  # log_interval > 0 only records statistics
            plain, centred = self.models(enabled=False, log_interval=log_interval)
            batch = self.inputs([0, 0, 0, 0, 1, 1])
            for train in (True, False):
                plain.train(train)
                centred.train(train)
                self.assertTrue(torch.equal(plain(batch), centred(batch)))

    def test_b_common_vector_added_to_one_dataset_changes_nothing(self):
        _, model = self.models()
        model.train()
        batch = self.inputs([0, 0, 0, 0, 0])
        before = model(batch)
        with torch.no_grad():
            model.decoder[-1].bias.add_(torch.linspace(-2.0, 2.0, len(GENES)))  # same vector on every set's output
        after = model(batch)
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)

    def test_b_function_level_invariance_and_specific_change(self):
        torch.manual_seed(1)
        output = torch.randn(5, 3, len(GENES))
        ids = torch.tensor([0, 0, 0, 0, 1])
        centred, stats = txpert_centered.center_decoder_output(output, ids, None, 4)
        shifted = output.clone()
        shifted[:4] += torch.randn(3, len(GENES))  # one common vector for all of dataset 0
        centred_shifted, _ = txpert_centered.center_decoder_output(shifted, ids, None, 4)
        torch.testing.assert_close(centred_shifted[:4], centred[:4])
        self.assertEqual(stats["sets_centred"], 4)
        expected = output[0] - output[1:4].mean(0)  # leave-one-out mean of the other three
        torch.testing.assert_close(centred[0], expected)
        changed = output.clone()
        changed[0] += 1.0  # target-specific change still moves the centred output
        self.assertFalse(torch.allclose(txpert_centered.center_decoder_output(changed, ids, None, 4)[0][0], centred[0]))

    def test_c_small_groups_are_left_alone(self):
        plain, centred = self.models(min_targets=4)
        plain.train()
        centred.train()
        batch = self.inputs([0, 0, 0, 0, 1, 1])
        out_plain, out_centred = plain(batch), centred(batch)
        self.assertTrue(torch.equal(out_plain[4:], out_centred[4:]))  # dataset 1 has 2 sets < 4
        self.assertFalse(torch.allclose(out_plain[:4], out_centred[:4]))
        self.assertEqual(centred.decoder_center_last["sets_centred"], 4)
        self.assertEqual(centred.decoder_center_last["sets"], 6)

    def test_d_eval_mode_does_not_centre(self):
        plain, centred = self.models()
        batch = self.inputs([0, 0, 0, 0, 0])
        plain.eval()
        centred.eval()
        self.assertTrue(torch.equal(plain(batch), centred(batch)))

    def test_e_mask(self):
        torch.manual_seed(2)
        output = torch.randn(4, 3, len(GENES))
        ids = torch.zeros(4, dtype=torch.long)
        mask = torch.ones(4, len(GENES), dtype=torch.bool)
        mask[0, 5] = False  # set 0 did not measure gene 5
        mask[1:, 6] = False  # only set 0 measured gene 6
        centred, _ = txpert_centered.center_decoder_output(output, ids, mask, 4)
        torch.testing.assert_close(centred[0, :, 5], output[0, :, 5])  # unmeasured: unchanged
        torch.testing.assert_close(centred[1, :, 5], output[1, :, 5] - output[2:4, :, 5].mean(0))  # mean skips set 0
        torch.testing.assert_close(centred[0, :, 6], output[0, :, 6])  # no other set measured it

    def test_gradient_flows_through_the_mean(self):
        _, model = self.models()
        model.train()
        batch = self.inputs([0, 0, 0, 0])
        captured = {}
        model.decoder.register_forward_hook(lambda m, a, out: captured.setdefault("raw", out))
        model(batch)[0].sum().backward()
        self.assertIsNotNone(model.decoder[-1].weight.grad)
        output = torch.randn(4, 3, len(GENES), requires_grad=True)
        txpert_centered.center_decoder_output(output, torch.zeros(4, dtype=torch.long), None, 4)[0][0].sum().backward()
        self.assertTrue(bool((output.grad[1:] != 0).all()))  # not detached: the other sets get gradient too

    def test_needs_dataset_id_in_training(self):
        _, model = self.models()
        model.train()
        batch = self.inputs([0, 0, 0, 0])
        batch.pop("dataset_id")
        with self.assertRaises(ValueError):
            model(batch)

    @staticmethod
    def penalty(output, ids, mask=None, min_targets=4):
        stats = txpert_centered.center_decoder_output(output, ids, mask, min_targets)[1]
        return stats["penalty_sum"] / max(stats["penalty_count"], 1)

    def test_f_penalty_gradient_shrinks_the_common_part(self):
        torch.manual_seed(3)
        output = (torch.randn(5, 3, len(GENES)) + torch.linspace(1.0, 2.0, len(GENES))).requires_grad_(True)
        ids = torch.tensor([0, 0, 0, 0, 0])
        before = self.penalty(output, ids)
        expected = output.detach().mean(0).square().mean()  # mean(m^2), m the group mean
        torch.testing.assert_close(before.detach(), expected)
        before.backward()
        stepped = output.detach() - 0.5 * output.grad / output.grad.abs().max()
        self.assertLess(float(self.penalty(stepped, ids)), float(before.detach()))
        common_before = output.detach().mean(0).square().mean()
        self.assertLess(float(stepped.mean(0).square().mean()), float(common_before))
        small = torch.randn(3, 3, len(GENES))  # a group below min_targets has no penalty
        self.assertEqual(float(self.penalty(small, torch.zeros(3, dtype=torch.long))), 0.0)

    def test_g_target_specific_change_keeps_the_penalty(self):
        torch.manual_seed(4)
        output = torch.randn(5, 3, len(GENES))
        ids = torch.zeros(5, dtype=torch.long)
        changed = output.clone()
        v = torch.randn(3, len(GENES))
        changed[1] += v
        changed[3] -= v  # specific differences that sum to zero over the group
        torch.testing.assert_close(self.penalty(changed, ids), self.penalty(output, ids))

    def test_h_off_has_no_penalty_and_objective_is_unchanged(self):
        from training.centered_objective import add_common_penalty
        loss, components = torch.tensor(2.5), {"pds": torch.tensor(1.0)}
        for enabled in (False, True):
            _, model = self.models(enabled=enabled)
            model.train()
            model(self.inputs([0, 0, 0, 0, 1]))
            total, comps = add_common_penalty(loss, components, model)
            if not enabled:
                self.assertIsNone(model.decoder_center_penalty)
                self.assertIs(total, loss)
                self.assertIs(comps, components)
            else:
                penalty = model.decoder_center_penalty
                self.assertGreater(float(penalty.detach()), 0.0)
                torch.testing.assert_close(total, loss + model.decoder_center_common_weight * penalty)
                self.assertIn("decoder_common", comps)
                self.assertIn("decoder_common_share", comps)
            model.eval()
            model(self.inputs([0, 0, 0, 0, 1]))
            self.assertIsNone(model.decoder_center_penalty)  # eval: no centring, no penalty
        self.assertIs(add_common_penalty(loss, components, None)[0], loss)

    def test_penalty_backpropagates_into_the_decoder(self):
        _, model = self.models()
        model.train()
        model(self.inputs([0, 0, 0, 0]))
        model.decoder_center_penalty.backward()
        self.assertGreater(float(model.decoder[-1].bias.grad.abs().sum()), 0.0)

    def test_data_wrapper_injects_ids_and_mask(self):
        class Base:
            spec = {"x": 1}

            def train_batches(self):
                yield {"inputs": {"a": 1}, "dataset_id": ("k562", "h1", "k562"),
                       "observed_mask": torch.ones(3, 2, dtype=torch.bool)}

        data = DecoderCenterData(Base(), ["k562", "hct116", "h1"])
        batch = next(data.train_batches())
        self.assertEqual(batch["inputs"]["dataset_id"].tolist(), [0, 2, 0])
        self.assertEqual(batch["inputs"]["observed_mask"].shape, (3, 2))
        self.assertEqual(batch["inputs"]["a"], 1)
        self.assertEqual(data.spec, {"x": 1})


if __name__ == "__main__":
    unittest.main()
