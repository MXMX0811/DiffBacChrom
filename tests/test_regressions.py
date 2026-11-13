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
from models.sd_vae import SDVAE
from models.mmdit import DiT as JointAttDiT
from models.mmditx import MMDiTX
from train.rf import RF
from train.train_dit import parse_args
from train import train_dit
from scripts import visualize_cond_trace as trace_demo
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
        # Exercise real checkpoint loading, RF sampling, VAE decoding and TSV output.
        cases = [('CrossDiT', CrossDiT, False), ('CrossDiT', CrossDiT, True),
                 ('JointAttDiT', JointAttDiT, True), ('MMDiTX', MMDiTX, True)]
        for name, cls, compressed in cases:
            with self.subTest(model=name, compressed=compressed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                hic = root / 'hic.tsv'
                import pandas as pd
                pd.DataFrame(torch.eye(8).numpy()).rename_axis('hic_index').to_csv(hic, sep='\t')
                vae = SDVAE() if compressed else build_model({'name': 'resnet-vae', 'num_res_blocks': 18})
                model_args = {'seq_compression': compressed} if cls is CrossDiT else {}
                model = cls(input_size=8, latent_size=8 // vae.downsample_factor,
                            in_channels=vae.z_channels, hidden_size=32, depth=1, num_heads=4, **model_args)
                torch.save({'model': vae.state_dict()}, root / 'vae.pt')
                torch.save(model.state_dict(), root / 'dit.ckpt')
                argv = ['structure_gen.py', '--model', name, '--hic_path', str(hic),
                        '--vae_ckpt', str(root / 'vae.pt'), '--dit_ckpt', str(root / 'dit.ckpt'),
                        '--output_root', str(root / 'samples'), '--num_samples', '4', '--batch_size', '3',
                        '--sample_steps', '2', '--cfg_scale', '1.5']
                if compressed:
                    argv.append('--use_seq_compression')
                def small_model(**kwargs):
                    return cls(hidden_size=32, depth=1, num_heads=4, **kwargs)
                registry = {'CrossDiT': structure_gen.CrossDiT_models,
                            'JointAttDiT': structure_gen.MMDiT_models,
                            'MMDiTX': structure_gen.MMDiTX_models}[name]
                with patch.dict(registry, {'DiT-S': small_model}), \
                     patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    structure_gen.main()
                outputs = sorted((root / 'samples' / 'hic').glob('sample_*.tsv'))
                self.assertEqual([p.name for p in outputs], [f'sample_{i:04d}.tsv' for i in range(1, 5)])
                self.assertTrue((root / 'samples' / 'hic' / 'hic.tsv').is_file())
                for path in outputs:
                    self.assertEqual(pd.read_csv(path, sep='\t').shape, (16, 9))


class ModelTests(unittest.TestCase):
    def test_compressed_position_grid_covers_928_bins(self):
        for cls in (CrossDiT, JointAttDiT, MMDiTX):
            with self.subTest(model=cls.__module__):
                kwargs = {'seq_compression': True} if cls is CrossDiT else {}
                model = cls(input_size=928, latent_size=232, in_channels=16,
                            hidden_size=32, depth=1, num_heads=4, **kwargs)
                cond_pos = model.hic_encoder.hic_encoder.pos_embed
                self.assertEqual(model.pos_embed.shape, (1, 232, 32))
                self.assertEqual(cond_pos.shape[1], 232)
                # The first sine component uses frequency 1 in both embedding widths.
                expected = torch.arange(232).float().sin()
                torch.testing.assert_close(model.pos_embed[0, :, 0], expected)
                torch.testing.assert_close(cond_pos[0, :, 0], expected)
                self.assertEqual(model.hic_encoder.input_size, 928)

    def test_compressed_training_step_for_all_backbones(self):
        vae = SDVAE().eval()
        with torch.no_grad():
            latent = vae.encode(torch.randn(2, 16, 16)).latent_dist.sample()
        self.assertEqual(latent.shape, (2, 4, 16))
        for cls in (CrossDiT, JointAttDiT, MMDiTX):
            with self.subTest(model=cls.__module__):
                kwargs = {'seq_compression': True} if cls is CrossDiT else {}
                model = cls(input_size=16, latent_size=4, in_channels=16,
                            hidden_size=32, depth=1, num_heads=4, **kwargs).train()
                loss = RF(model).forward(latent, torch.randn(2, 1, 16, 16))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(model.final_layer.linear.weight.grad.abs().sum().item(), 0)

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
                model = cls(input_size=8, latent_size=8, in_channels=2, hidden_size=32, depth=2, num_heads=4).eval()
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
        model = JointAttDiT(input_size=8, latent_size=2, in_channels=2, hidden_size=32, depth=2, num_heads=4)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'adaLN_modulation' in name or name.startswith('final_layer.linear'):
                    p.normal_(std=0.05)
        latent, hic = torch.randn(2, 2, 2), torch.randn(2, 1, 8, 8)
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
        for name, length in [('resnet', 928), ('sd', 232)]:
            with self.subTest(vae=name):
                config = yaml.safe_load((root / f'train_vae_{name}.yaml').read_text())
                model = build_model(config['model']).eval()
                with torch.no_grad():
                    recon, mu, logvar = model(torch.randn(1, 928, 16))
                self.assertEqual(recon.shape, (1, 928, 16))
                self.assertEqual(mu.shape, (1, length, 16))
                self.assertEqual(logvar.shape, mu.shape)


class TraceDemoTests(unittest.TestCase):
    def test_demo_uses_formal_config_with_cli_overrides(self):
        config = Path(__file__).resolve().parents[1] / 'train/configs/train_mmdit.yaml'
        cli = ['--config', str(config), '--vae_ckpt', 'sd-vae.pt', '--size', 'S',
               '--precision', 'fp32', '--latent_scale', '2.0', '--no-grad_cp']
        formal = parse_args(cli)
        demo = trace_demo.parse_args(cli)
        for name, value in vars(formal).items():
            self.assertEqual(getattr(demo, name), value, name)
        self.assertTrue(demo.use_seq_compression)

    def test_training_demo_collects_compressed_embeddings_and_plots(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = root / 'Pair_1'
            pair.mkdir()
            pd.DataFrame(torch.eye(8).numpy()).rename_axis('hic_index').to_csv(
                pair / 'Pair_1_sim_hic_freq.tsv', sep='\t')
            columns = ['hic_index', 'bead_index1', 'x1', 'y1', 'z1', 'mask1',
                       'bead_index2', 'x2', 'y2', 'z2', 'mask2']
            for sample in range(2):
                rows = [[i // 2, i * 2, *torch.randn(3).tolist(), 1,
                         i * 2 + 1, *torch.randn(3).tolist(), 1] for i in range(16)]
                pd.DataFrame(rows, columns=columns).to_csv(pair / f'structure_{sample}.tsv', sep='\t', index=False)
            torch.save({'model': SDVAE().state_dict()}, root / 'vae.pt')
            cli = ['--train_dir', str(root), '--vae_ckpt', str(root / 'vae.pt'),
                   '--size', 'S', '--precision', 'fp32', '--train_steps', '2',
                   '--batch_size', '2', '--log_interval', '2', '--warmup_steps', '1']
            built = []
            def small_joint(**kwargs):
                model = JointAttDiT(hidden_size=32, depth=2, num_heads=4, **kwargs)
                built.append(model)
                return model
            def cpu_loader(*args, **kwargs):
                kwargs.update(num_workers=0, pin_memory=False)
                return torch.utils.data.DataLoader(*args, **kwargs)
            with patch.dict(train_dit.MMDiT_models, {'DiT-S': small_joint}), \
                 patch.object(trace_demo, 'DataLoader', side_effect=cpu_loader), \
                 patch.object(trace_demo, 'plot_tsne', wraps=trace_demo.plot_tsne) as plot, \
                 patch.object(trace_demo, 'wandb') as logger, \
                 patch('sys.argv', ['visualize_cond_trace.py', *cli]), \
                 contextlib.redirect_stdout(io.StringIO()):
                trace_demo.main()
                self.assertEqual(logger.log.call_count, 2)
                self.assertEqual(logger.init.call_args.kwargs['project'], 'diffbacchrom-cond-trace')
                self.assertEqual(plot.call_count, 1)
                embeddings = plot.call_args.args[0]
                self.assertEqual(len(embeddings), 3)
                self.assertTrue(all(t.shape == (2, 2, 32) and not t.requires_grad for t in embeddings))
                self.assertEqual(built[0].pos_embed.shape, (1, 2, 32))
                self.assertGreater(built[0].final_layer.linear.weight.abs().sum().item(), 0)
                figure = logger.Image.call_args.args[0]
                self.assertEqual(len(figure.axes[0].collections), 6)
                self.assertEqual(trace_demo.plt.get_fignums(), [])
                # Partial weights must fail instead of silently mixing in initialized parameters.
                state = built[0].state_dict()
                state.pop('pos_embed')
                torch.save(state, root / 'partial.ckpt')
                args = trace_demo.parse_args([*cli, '--dit_ckpt', str(root / 'partial.ckpt')])
                with self.assertRaises(RuntimeError):
                    trace_demo.load_models(args, 8, torch.device('cpu'))


class SchedulerTests(unittest.TestCase):
    def test_warmup_checkpoint_loads_with_weights_only(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = train_dit.get_scheduler(optimizer, warmup_steps=1, total_steps=3)
        for _ in range(2):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
        checkpoint = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        self.assertIsInstance(optimizer.param_groups[0]['lr'], float)


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
