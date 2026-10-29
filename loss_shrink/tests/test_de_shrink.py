"""de_shrink_loss 단위 테스트 (팀 프레임워크 없이 실행 가능).

팀 `JointObjective` 연결 테스트 4개(weight 0 기록, DE 참조 생성 조건, control 마스킹)는
팀 프레임워크가 필요해서 제외했다.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from de_shrink import de_shrink_loss  # noqa: E402


class ShrinkTests(unittest.TestCase):
    """S_p 밖 gene의 예측 |lfc|를 누르는 항. S_p 안과 미측정 gene에는 gradient가 없어야 한다."""

    SETTINGS = dict(lfc_epsilon=1e-3, minimum_significant_genes=1)

    def _cells(self, control_rate, predicted_rate):
        control = torch.log1p(torch.tensor([control_rate] * 4, dtype=torch.float64)).unsqueeze(0)
        predicted = torch.log1p(torch.tensor([predicted_rate] * 4, dtype=torch.float64)).unsqueeze(0)
        return predicted, control

    def test_value_is_the_mean_abs_lfc_outside_the_significant_set(self):
        # gene 0은 유의(4배), gene 1은 비유의 2배, gene 2는 비유의 1/4배
        predicted, control = self._cells([100.0, 100.0, 100.0], [400.0, 200.0, 25.0])
        significant = torch.tensor([[True, False, False]])
        loss, inside = de_shrink_loss(predicted, control, significant, **self.SETTINGS)

        def lfc(p, c):
            return abs(np.log2((p + 1e-3) / (c + 1e-3)))
        self.assertAlmostEqual(float(loss), (lfc(200, 100) + lfc(25, 100)) / 2, places=6)
        self.assertAlmostEqual(float(inside), lfc(400, 100), places=6)

    def test_zero_when_prediction_equals_control_outside_the_set(self):
        predicted, control = self._cells([100.0, 100.0], [400.0, 100.0])
        loss, _ = de_shrink_loss(predicted, control, torch.tensor([[True, False]]), **self.SETTINGS)
        self.assertAlmostEqual(float(loss), 0.0, places=12)

    def test_gradient_only_reaches_measured_genes_outside_the_set(self):
        predicted, control = self._cells([100.0] * 4, [400.0, 200.0, 50.0, 300.0])
        predicted = predicted.clone().requires_grad_()
        significant = torch.tensor([[True, False, False, False]])
        mask = torch.tensor([[True, True, True, False]])        # gene 3은 미측정
        loss, _ = de_shrink_loss(predicted, control, significant, observed_mask=mask, **self.SETTINGS)
        loss.backward()
        grad = predicted.grad[0].abs().sum(dim=0)
        self.assertEqual(float(grad[0]), 0.0)                    # S_p 안
        self.assertEqual(float(grad[3]), 0.0)                    # 미측정
        self.assertGreater(float(grad[1]), 0.0)
        self.assertGreater(float(grad[2]), 0.0)
        # 경사를 따라가면 크기가 준다: 2배인 gene 1은 내려가고 1/2배인 gene 2는 올라간다.
        self.assertGreater(float(predicted.grad[0, :, 1].sum()), 0.0)
        self.assertLess(float(predicted.grad[0, :, 2].sum()), 0.0)

    def test_large_lfc_is_not_clipped(self):
        predicted, control = self._cells([100.0, 100.0], [100.0, 100.0 * 2 ** 8])
        predicted = predicted.clone().requires_grad_()
        loss, _ = de_shrink_loss(predicted, control, torch.tensor([[True, False]]), **self.SETTINGS)
        loss.backward()
        self.assertAlmostEqual(float(loss), 8.0, places=4)
        self.assertGreater(float(predicted.grad[0, :, 1].abs().sum()), 0.0)

    def test_conditions_below_the_gate_are_skipped(self):
        predicted, control = self._cells([100.0, 100.0], [400.0, 400.0])
        settings = {**self.SETTINGS, "minimum_significant_genes": 2}
        loss, inside = de_shrink_loss(predicted, control, torch.tensor([[True, False]]), **settings)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(inside), 0.0)

    def test_rejects_malformed_inputs(self):
        predicted, control = self._cells([100.0, 100.0], [400.0, 100.0])
        significant = torch.tensor([[True, False]])
        with self.assertRaisesRegex(ValueError, "de_shrink expects prediction/control"):
            de_shrink_loss(predicted[0], control[0], significant, **self.SETTINGS)
        with self.assertRaisesRegex(ValueError, "de_shrink significant must be boolean"):
            de_shrink_loss(predicted, control, significant.float(), **self.SETTINGS)
        with self.assertRaisesRegex(ValueError, "de_shrink lfc_epsilon must be positive"):
            de_shrink_loss(predicted, control, significant, lfc_epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "de_shrink observed_mask must be"):
            de_shrink_loss(predicted, control, significant,
                           observed_mask=torch.ones(1, 3, dtype=torch.bool), **self.SETTINGS)


if __name__ == "__main__":
    unittest.main()
