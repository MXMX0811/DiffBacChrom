"""Run with: python -m unittest discover -s tests -v"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from models.crossdit import DiT as CrossDiT
from models.hic_cnn import CNNEncoder
from models.mmdit import DiT as JointAttDiT
from models.mmditx import MMDiTX
from train.rf import RF
from train.train_dit import parse_args
from train.train_vae import build_model
import structure_gen


torch.set_num_threads(2)


class AnalyticVelocity(torch.nn.Module):
    """Different conditions and a nonzero unused head expose batch/head mixups."""
    def forward(self, x, t, hic, force_drop_ids=None):
        condition = hic[:, 0, 0, 0]
        if force_drop_ids is not None:
            condition = torch.where(force_drop_ids.bool(), -torch.ones_like(condition), condition)
        velocity = (0.25 * x + condition[:, None, None]).transpose(1, 2)
        return torch.cat([velocity, torch.full_like(velocity, 1000)], dim=1)


class SamplingTests(unittest.TestCase):
    def test_cfg_matches_separate_predictions_for_every_sample(self):
        model = AnalyticVelocity().eval()
        for batch in (1, 3, 4):
            hic = torch.arange(1, batch + 1).float().reshape(batch, 1, 1, 1)
            for scale in (0.0, 1.0, 1.5):
                with self.subTest(batch=batch, scale=scale):
                    torch.manual_seed(7)
                    expected = torch.randn(batch, 8, 2)
                    for step in (3, 2, 1):
                        t = torch.full((batch,), step / 3)
                        cond = model(expected, t, hic)[:, :2]
                        uncond = model(expected, t, hic, torch.ones(batch))[:, :2]
                        velocity = uncond + scale * (cond - uncond) if scale != 1 else cond
                        expected -= velocity.transpose(1, 2) / 3
                    torch.manual_seed(7)
                    actual = RF(model).sample(hic, 3, (batch, 8, 2), scale)
                    torch.testing.assert_close(actual, expected)

    def test_sample_entry_saves_odd_tail_without_overwriting(self):
        # Use real VAE encode/decode and CrossDiT/RF; reduce transformer size only.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hic = root / 'hic.tsv'
            import pandas as pd
            pd.DataFrame(torch.eye(8).numpy()).rename_axis('hic_index').to_csv(hic, sep='\t')
            vae = build_model({'name': 'resnet-vae', 'num_res_blocks': 18})
            model = CrossDiT(input_size=8, in_channels=16, hidden_size=32, depth=1, num_heads=4)
            torch.save({'model': vae.state_dict()}, root / 'vae.pt')
            torch.save(model.state_dict(), root / 'dit.ckpt')
            argv = ['structure_gen.py', '--hic_path', str(hic), '--vae_ckpt', str(root / 'vae.pt'),
                    '--dit_ckpt', str(root / 'dit.ckpt'), '--output_root', str(root / 'samples'),
                    '--num_samples', '4', '--batch_size', '3', '--sample_steps', '2', '--cfg_scale', '1.5']
            def small_crossdit(**kwargs):
                return CrossDiT(hidden_size=32, depth=1, num_heads=4, **kwargs)
            with patch.dict(structure_gen.CrossDiT_models, {'DiT-S': small_crossdit}), \
                 patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                structure_gen.main()
            outputs = sorted((root / 'samples' / 'hic').glob('sample_*.tsv'))
            self.assertEqual([p.name for p in outputs], [f'sample_{i:04d}.tsv' for i in range(1, 5)])
            self.assertTrue((root / 'samples' / 'hic' / 'hic.tsv').is_file())
            for path in outputs:
                self.assertEqual(pd.read_csv(path, sep='\t').shape, (16, 9))


class ModelTests(unittest.TestCase):
    def test_hic_cnn_follows_callers_precision(self):
        model = CNNEncoder(ch=32, num_res_blocks=1, gradient_checkpointing=False).eval()
        hic = torch.randn(1, 1, 8, 8)
        with torch.no_grad():
            self.assertEqual(model(hic).dtype, torch.float32)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertEqual(model(hic).dtype, torch.bfloat16)

    def test_all_backbones_match_individual_conditioned_predictions(self):
        for cls in (CrossDiT, JointAttDiT, MMDiTX):
            with self.subTest(model=cls.__module__):
                model = cls(input_size=8, in_channels=2, hidden_size=32, depth=2, num_heads=4).eval()
                # Initialized output heads are zero; use nonzero weights to test actual predictions.
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if 'adaLN_modulation' in name or name.startswith('final_layer.linear'):
                            p.normal_(std=0.05)
                x, t, hic = torch.randn(3, 8, 2), torch.rand(3), torch.randn(3, 1, 8, 8)
                with torch.no_grad():
                    cond = model(x, t, hic)
                    uncond = model(x, t, hic, force_drop_ids=torch.ones(3))
                    combined = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([hic, hic]),
                                     force_drop_ids=torch.tensor([1, 1, 1, 0, 0, 0]))
                    torch.testing.assert_close(combined, torch.cat([uncond, cond]), atol=1e-5, rtol=1e-4)
                    self.assertEqual(cond.shape, (3, 4, 8))

    def test_trace_preserves_prediction_and_gradients_with_checkpointing(self):
        model = JointAttDiT(input_size=8, in_channels=2, hidden_size=32, depth=2, num_heads=4)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'adaLN_modulation' in name or name.startswith('final_layer.linear'):
                    p.normal_(std=0.05)
        latent, hic = torch.randn(2, 8, 2), torch.randn(2, 1, 8, 8)
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                model.gradient_checkpointing = checkpointing
                model.zero_grad(set_to_none=True)
                torch.manual_seed(12)
                loss = RF(model).forward(latent, hic)
                loss.backward()
                grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                torch.manual_seed(12)
                traced_loss, trace = RF(model).forward(latent, hic, return_trace=True)
                traced_loss.backward()
                torch.testing.assert_close(traced_loss, loss, rtol=0, atol=0)
                self.assertEqual(len(trace), 3)
                self.assertTrue(all(t.device.type == 'cpu' and not t.requires_grad for t in trace))
                for name, p in model.named_parameters():
                    if name in grads:
                        torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)

    def test_both_vae_configs_build_and_preserve_sequence_conventions(self):
        root = Path(__file__).resolve().parents[1] / 'train' / 'configs'
        for name, length in [('resnet', 16), ('sd', 4)]:
            with self.subTest(vae=name):
                config = yaml.safe_load((root / f'train_vae_{name}.yaml').read_text())
                model = build_model(config['model']).eval()
                with torch.no_grad():
                    recon, mu, logvar = model(torch.randn(1, 16, 16))
                self.assertEqual(recon.shape, (1, 16, 16))
                self.assertEqual(mu.shape, (1, length, 16))
                self.assertEqual(logvar.shape, mu.shape)


class ConfigTests(unittest.TestCase):
    def parse_yaml(self, config, *cli):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            return parse_args(['--config', str(path), *cli])

    def test_cli_overrides_yaml_and_yaml_overrides_defaults(self):
        args = self.parse_yaml({'model': 'CrossDiT', 'size': 'S', 'batch_size': 4,
                                'lr': 0.002, 'vae_ckpt': 'yaml.pt', 'use_global_cond': True,
                                'grad_cp': True, 'use_seq_compression': True},
                               '--batch_size', '3', '--vae_ckpt', 'cli.pt', '--no-grad_cp',
                               '--no-use_global_cond', '--no-use_seq_compression')
        self.assertEqual((args.batch_size, args.lr, args.epochs), (3, 0.002, 50))
        self.assertEqual(args.vae_ckpt, 'cli.pt')
        self.assertFalse(args.grad_cp or args.use_global_cond or args.use_seq_compression)
        self.assertEqual((args.precision, args.warmup_steps, args.cfg_scale), ('fp32', 0, 1.0))
        self.assertEqual(args.save_dir, 'checkpoints/dit/CrossDiT-S')

    def test_cli_model_selects_derived_defaults_after_yaml(self):
        args = self.parse_yaml({'model': 'CrossDiT', 'vae_ckpt': 'vae.pt'}, '--model', 'MMDiTX')
        self.assertFalse(args.grad_cp or args.use_global_cond)
        self.assertTrue(args.use_seq_compression)
        self.assertEqual((args.precision, args.warmup_steps, args.cfg_scale), ('fp16', 1000, 1.5))
        self.assertEqual(args.save_dir, 'checkpoints/dit/MMDiTX-L')

    def test_invalid_explicit_options_are_not_silently_overridden(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in [[], ['--vae_ckpt', 'v.pt', '--model', 'MMDiTX', '--grad_cp'],
                         ['--vae_ckpt', 'v.pt', '--model', 'JointAttDiT', '--use_global_cond']]:
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    parse_args(argv)


if __name__ == '__main__':
    unittest.main()
