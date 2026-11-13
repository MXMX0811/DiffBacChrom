# DiffBacChrom
A repository of diffusion models for bacteria chromosome structure generation.

## Training and sampling

DiT configuration precedence is **explicit command line > YAML > code defaults**.
Boolean flags use `--grad_cp` / `--no-grad_cp`, `--use_global_cond` /
`--no-use_global_cond`, and `--use_seq_compression` / `--no-use_seq_compression`.
Model-dependent defaults are resolved after the configuration is merged.
Unsupported explicit combinations raise an error instead of changing the options.

Select the VAE checkpoint explicitly; VAE training saves `epoch_XXX.pt` files.
The `vae_ckpt: null` entries in the DiT YAML files must be filled in or overridden:

```bash
python train/train_dit.py --config train/configs/train_crossdit.yaml \
  --vae_ckpt /path/to/selected/epoch_XXX.pt
python structure_gen.py --model CrossDiT --size L \
  --vae_ckpt /path/to/selected/epoch_XXX.pt \
  --dit_ckpt /path/to/matching/final.ckpt --hic_path /path/to/hic.tsv \
  --latent_scale 1.335256 --cfg_scale 1.0 --batch_size 10 --num_samples 500
```

Use the same VAE checkpoint, latent scale, model size and global-condition setting
for training and sampling. Both entry points support `CrossDiT`, `JointAttDiT`,
`MMDiTX` and `DiMamba`. CrossDiT and DiMamba use the non-compressing ResNet VAE by default; select
`--use_seq_compression` for its SD VAE comparison. JointAttDiT and MMDiTX use
SD VAE compression by default, matching their training configurations.

The original Hi-C width and the VAE latent length are separate model arguments:
`input_size` is the original width and `latent_size` is the length after encoding.
The entry points derive `latent_size` from the selected VAE's downsampling factor.

| Branch | Hi-C input | Structure latent | Condition tokens | Position indices |
| --- | --- | --- | --- | --- |
| CrossDiT / DiMamba without compression | 928 × 928 | 928 × 16 | 928 | 0…927 |
| Compressed CrossDiT / DiMamba / JointAttDiT / MMDiTX | 928 × 928 | 232 × 16 | 232 | 0…231 |

Compressed sampling starts with 232 latent tokens and decodes them back to 928
structure tokens. Each modality uses its existing embedding width on the same
compressed position grid. MMDiTX directly creates this grid without center cropping,
position offsets or rescaling. The VAE and Hi-C encoder layers are unchanged.

For compressed CrossDiT, add `--use_seq_compression` to both training and sampling,
select the SD VAE checkpoint, and pass the latent scale used for that training run.
The SD VAE scale still needs to be established for the selected checkpoint; the
existing `1.335256` value has not been validated for SD VAE. Compressed CrossDiT
uses a separate default checkpoint/output directory ending in `-compression`.

Earlier compressed DiT checkpoints with a 928-position table require retraining
under the corrected position convention. They are not automatically converted.
Uncompressed CrossDiT and VAE checkpoint parameter names and shapes are unchanged.
The standalone trace demonstration uses the same JointAttDiT construction,
SD VAE, latent preparation and configuration parser as formal training. Both
structure and condition sequences therefore have 232 tokens for a 928-bin input.

CPU training of the attention backbones requires `--precision fp32`. DiMamba uses
the official CUDA implementation described below. The Hi-C CNN follows the caller's
autocast context instead of forcing CUDA fp16 internally.

## Diffusion Mamba (DiMamba) comparison

`DiMamba` (Diffusion Mamba) reuses CrossDiT's blocks, conditioning, Rectified Flow
training and sampling. Each self-attention module is replaced by two independent
official
[Mamba-2 modules](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/mamba_ssm/modules/mamba2.py).
The reverse branch flips the token sequence, scans it, and restores the original
order. The two outputs are averaged. Cross-attention, MLP, adaLN-zero and the
global Hi-C condition follow CrossDiT. Condition tokens remain read-only across
backbone layers. Each RF model call starts a fresh scan; no recurrent state is
carried between sampling steps.

S/B/L/XL reuse exactly the CrossDiT width, depth and cross-attention head settings.
Mamba-2 uses `d_state=128`, `d_conv=4`, `expand=2` and `headdim=64`, retaining
its official initialization. Parameter counts increase when replacing attention;
the existing training log reports the actual counts.

The primary comparison uses 928 latent tokens; the compressed comparison uses
232. The two YAML files mirror their corresponding CrossDiT configurations,
including size L for the uncompressed template and S for the compressed template.
For a comparison between sequence lengths, select the same size and adjust the
save directory and run name in both configurations. Use the matching VAE checkpoint
and latent scale for each branch; the SD VAE scale still needs checkpoint-specific
validation.

Install the pinned Mamba dependencies in the Linux CUDA training environment,
after installing the project's PyTorch version:

```bash
pip install --no-build-isolation causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3
python train/train_dit.py --config train/configs/train_dimamba.yaml \
  --vae_ckpt /path/to/resnet-vae/epoch_XXX.pt
python train/train_dit.py --config train/configs/train_dimamba_compression.yaml \
  --vae_ckpt /path/to/sd-vae/epoch_XXX.pt --latent_scale <selected-scale>
python structure_gen.py --model DiMamba --size L \
  --vae_ckpt /path/to/resnet-vae/epoch_XXX.pt \
  --dit_ckpt checkpoints/dit/DiMamba-L/final.ckpt \
  --hic_path /path/to/hic.tsv --latent_scale 1.335256 --cfg_scale 1.0
```

Compressed sampling additionally uses `--use_seq_compression`, the SD VAE and
the size/latent scale from its training configuration. Its default checkpoint and
output directories end in `-compression`. Other backbones can be imported and
run without installing Mamba; the dependency is imported when a Mamba model is built.

## RF sampling and condition traces

CFG batching belongs to `RF.sample`: it temporarily builds an unconditional-first
2B batch and combines the predictions into B velocities before each Euler step,
following [Diffusers 0.33.1](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py).
Backbones return `(B, 2C, T)` tensors with the existing heads; RF uses the first C
channels as velocity. `cfg_scale` is passed to the sampler, not the backbones.

JointAttDiT returns condition traces only with `return_trace=True`. The trace
script collects them on logging steps, retaining the original recording points
(initial tokens and tokens after each joint-attention update, before its MLP).
This is a standalone training demonstration for visualizing condition-token
evolution in intermediate layers. Updating model weights during visualization
is intentional. Formal training experiments use `train/train_dit.py`.
The demo defaults to `train/configs/train_mmdit.yaml`; CLI arguments override
that file. Use `--train_dir` to select data and provide the same SD VAE checkpoint
and latent scale as the formal run. Optional `--dit_ckpt` weights are loaded
strictly into the shared backbone, without restoring optimizer state.

```bash
python scripts/visualize_cond_trace.py --config train/configs/train_mmdit.yaml \
  --vae_ckpt /path/to/sd-vae/epoch_XXX.pt \
  --train_steps 1000 --log_interval 100 --run_name JointAttDiT-cond-trace
```

The demo remains a separate training run with W&B project `diffbacchrom-cond-trace`.
It supports JointAttDiT and records the initial Hi-C embedding plus each layer's
post-attention embedding before the condition MLP, each shaped `[B, 232, D]`.
t-SNE pools the token dimension to one vector per sample per layer. Only logging
steps collect traces. The optimizer and optional warmup/cosine schedule follow
the formal implementation, with `train_steps` as the demo's schedule horizon;
`warmup_steps` must be smaller than that horizon. Formal-only options such as
`epochs`, `val_dir` and `save_dir` do not drive the demo loop, validation or saving.

The mask rule remains decoder logit > 0.5: values above the threshold become
1, all others become 0, and coordinates with mask 0 are zeroed.

Evaluation issues awaiting resolution before formal validation are recorded in
[the evaluation follow-up notes](doc/evaluation_followups.md).
