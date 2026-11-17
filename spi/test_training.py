import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import tables
import torch

from data import load_development, frozen_fold
from model import MaskedQualityGRU, Scaling, per_run_mean, training_loss


def fixture():
    torch.manual_seed(37)
    real = torch.randn(3, 8, 15)
    real[..., :2] = 0
    lengths = torch.tensor([3, 5, 7])
    for i, n in enumerate(lengths):
        real[i, n-1:, 0] = 1
        real[i, n:] = real[i, n-1]
        real[i, :, 1] = i % 2
    return dict(inputs=torch.tensor([[.001, 1.], [.004, 3.], [.010, 5.]]),
        point_start=real[:, 0, 2:9].clone(), sim=torch.randn(3, 8, 15), real=real,
        mask=torch.arange(8)[None] < lengths[:, None], length=lengths,
        error_mm=torch.tensor([.2, float('nan'), 2.]), error_valid=torch.tensor([True, False, True]))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.batch = fixture()
        self.model = MaskedQualityGRU(Scaling.fit(self.batch), hidden=8, layers=2, dropout=0).eval()

    def test_equal_run_weighting(self):
        values = torch.tensor([[1., 1., 999.], [4., 4., 4.]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        self.assertEqual(per_run_mean(values, mask).item(), 2.5)

    def test_padding_loss_and_gradient_zero(self):
        pred = self.model(self.batch['inputs'], self.batch['point_start'], self.batch['sim'])
        residual = pred['residual'].detach().requires_grad_(True)
        original, _ = training_loss(self.model, dict(pred, residual=residual), self.batch)
        original.backward()
        self.assertTrue(torch.all(residual.grad[~self.batch['mask']] == 0))
        altered = residual.detach().clone()
        altered[~self.batch['mask']] = float('nan')
        other, _ = training_loss(self.model, dict(pred, residual=altered), self.batch)
        self.assertTrue(torch.equal(original, other))

    def test_padding_extension_loss_invariant(self):
        pred = self.model(self.batch['inputs'], self.batch['point_start'], self.batch['sim'])
        before, _ = training_loss(self.model, pred, self.batch)
        enlarged = dict(self.batch)
        for name in ('real', 'sim'):
            enlarged[name] = torch.cat((self.batch[name], self.batch[name][:, -1:].expand(-1, 12, -1)), 1)
        enlarged['mask'] = torch.cat((self.batch['mask'], torch.zeros(3, 12, dtype=torch.bool)), 1)
        after_pred = dict(pred, residual=torch.cat((pred['residual'], torch.full((3, 12, 15), float('nan'))), 1))
        after, _ = training_loss(self.model, after_pred, enlarged)
        self.assertTrue(torch.equal(before, after))

    def test_terminal_eos_included(self):
        pred = self.model(self.batch['inputs'], self.batch['point_start'], self.batch['sim'])
        residual = pred['residual'].detach().requires_grad_(True)
        loss, _ = training_loss(self.model, dict(pred, residual=residual), self.batch)
        loss.backward()
        for i, n in enumerate(self.batch['length']):
            self.assertNotEqual(residual.grad[i, n-1, 0].item(), 0)

    def test_missing_native_error_not_imputed(self):
        pred = self.model(self.batch['inputs'], self.batch['point_start'], self.batch['sim'])
        quality = torch.tensor([1., 10000., 3.], requires_grad=True)
        loss, parts = training_loss(self.model, dict(pred, error_mm=quality), self.batch)
        loss.backward()
        self.assertAlmostEqual(parts['quality_mm2'].item(), (.8**2 + 1.) / 2, places=6)
        self.assertEqual(quality.grad[1].item(), 0)

    def test_no_valid_quality_labels(self):
        batch = dict(self.batch, error_valid=torch.zeros(3, dtype=torch.bool))
        pred = self.model(batch['inputs'], batch['point_start'], batch['sim'])
        loss, parts = training_loss(self.model, pred, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts['quality_mm2'].item(), 0)

    def test_normalization_ignores_padding(self):
        changed = {k: v.clone() for k, v in self.batch.items()}
        for name in ('real', 'sim'):
            changed[name][~changed['mask']] = 1e8
        other = Scaling.fit(changed)
        self.assertTrue(torch.equal(other.output_limits, self.model.scaling.output_limits))

    def test_future_and_padding_do_not_change_prefix(self):
        original = self.model(self.batch['inputs'], self.batch['point_start'], self.batch['sim'])
        sim = self.batch['sim'].clone(); sim[:, 3:] = 1e5
        changed = self.model(self.batch['inputs'], self.batch['point_start'], sim)
        self.assertTrue(torch.equal(original['residual'][:, :3], changed['residual'][:, :3]))

    def test_quality_independent_of_trajectory_and_padding(self):
        first = self.model(self.batch['inputs'], self.batch['point_start'], self.batch['sim'])
        other = self.model(self.batch['inputs'], self.batch['point_start'], torch.randn(3, 19, 15))
        self.assertTrue(torch.equal(first['error_mm'], other['error_mm']))
        self.assertTrue(torch.all(first['error_mm'] >= 0))

    def test_quality_has_parameter_gradients(self):
        inputs = self.batch['inputs'].clone().requires_grad_(True)
        out = self.model(inputs, self.batch['point_start'], self.batch['sim'])
        gradient, = torch.autograd.grad(out['error_mm'].sum(), inputs)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient != 0))

    def test_test_split_refused_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'forbidden.h5'
            with tables.open_file(path, 'w') as handle:
                handle.root._v_attrs.dataset_split = 'model_test'
            with self.assertRaisesRegex(ValueError, 'non-Development'):
                load_development(path)

    def test_constant_normalization_is_finite(self):
        same = dict(self.batch, inputs=torch.ones(3, 2),
                    real=torch.ones(3, 8, 15), sim=torch.ones(3, 8, 15))
        scaling = Scaling.fit(same)
        self.assertTrue(torch.isfinite(scaling.inputs(same['inputs'])).all())
        self.assertTrue(torch.isfinite(scaling.trajectory(same['real'])).all())

    def test_historical_duplicate_combination_cannot_cross_folds(self):
        data = dict(run_ids=['main', 'pilot', 'other'], inputs=torch.tensor([[.004, 2.], [.004, 2.], [.006, 3.]]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'membership.json'
            path.write_text(json.dumps(dict(data_role='development_only', folds=[dict(fold=1,
                train_run_ids=['main', 'other'], heldout_run_ids=['pilot'])])))
            with self.assertRaisesRegex(ValueError, 'combination leakage'):
                frozen_fold(data, path, 1)

    def test_no_duplicate_run_ids_in_membership(self):
        data = dict(run_ids=['a', 'b', 'c'], inputs=torch.tensor([[.004, 2.], [.004, 3.], [.006, 3.]]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'membership.json'
            path.write_text(json.dumps(dict(data_role='development_only', folds=[dict(fold=1,
                train_run_ids=['a', 'a', 'b'], heldout_run_ids=['c'])])))
            with self.assertRaisesRegex(ValueError, 'Invalid or leaking'):
                frozen_fold(data, path, 1)


if __name__ == '__main__':
    unittest.main()
