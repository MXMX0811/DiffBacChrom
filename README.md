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
for training and sampling. Sampling calls the joint-attention backbone
`JointAttDiT`, matching the training entry point. The sampling entry point still
uses the non-compressing ResNet VAE. Compressed sampling, positional alignment,
and the latent scale for SD VAE remain pending experimental decisions.
CPU training requires `--precision fp32`. The Hi-C CNN follows the caller's
autocast context instead of forcing CUDA fp16 internally.

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

The mask rule remains decoder logit > 0.5: values above the threshold become
1, all others become 0, and coordinates with mask 0 are zeroed.

Run the CPU regression checks with the project dependencies installed:

```bash
python -m unittest discover -s tests -v
```
