"""Real optional Torch check for the migration model's next-item interface."""
import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch runtime not installed')
class GRUMigrationTests(unittest.TestCase):
    def test_padding_batch_order_catalog_and_gradients(self):
        import torch
        path = Path(__file__).resolve().parents[1] / 'backends/sasrec/src/grurec.py'
        spec = importlib.util.spec_from_file_location('gru_migration_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(2026)
        model = module.GRURec(8, 5, hidden_size=4, dropout=0).eval()
        histories = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 0, 4, 5]])
        states = model.encode(histories)
        self.assertTrue(torch.allclose(states[1], model.encode(histories[1:])[0], atol=1e-6))
        self.assertTrue(torch.allclose(states[0], model.encode(torch.tensor([[1, 2, 3]]))[0], atol=1e-6))
        logits = model.catalog_logits(states)
        self.assertEqual(tuple(logits.shape), (2, 8))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        logits.square().mean().backward()
        self.assertGreater(float(model.encoder.weight_ih_l0.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, 'observed item'):
            model.encode(torch.zeros((1, 5), dtype=torch.long))
