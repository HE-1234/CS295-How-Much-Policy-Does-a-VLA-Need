# Frozen-LAPA Adapter Experiment

This folder trains a small causal LM inside the original LAPA embedding/head space:

```text
frozen LAPA vte/wte/dte -> Linear(4096 -> small_hidden)
    -> small causal LM -> Linear(small_hidden -> 4096)
    -> frozen LAPA delta_head
```

The small-run target backbone is `EleutherAI/pythia-160m`; the multi-GPU 7B
target is `EleutherAI/pythia-6.9b`. The original LAPA JAX code is not modified.

Python dependencies for the PyTorch adapter side are listed in `requirements_adapter.txt`.

## 1. Extract Frozen LAPA Tensors

Run from the `LAPA` directory:

```bash
python adapter_distill/extract_lapa_weights.py \
  --lapa_checkpoint params::lapa_checkpoints/params \
  --output adapter_distill/artifacts/lapa_frozen.npz
```

The extractor defaults to `--jax_platform cpu` because it only copies
checkpoint tensors and does not need a GPU. This also avoids CUDA/cuDNN loader
errors from mixed cluster libraries. If you intentionally want JAX to select a
GPU, add `--jax_platform auto`.

This saves:

- `vte`: vision-token embedding `(8448, 4096)`
- `wte`: text-token embedding `(32000, 4096)`
- `dte`: latent-action embedding `(8, 4096)`
- `delta_head`: latent-action head `(4096, 8)`

## 2. Train

Single-GPU small-model run:

```bash
python adapter_distill/train_adapter.py \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --data_path data/latent_action_pretraining_openx.jsonl \
  --model_name EleutherAI/pythia-160m \
  --batch_size 8 \
  --num_epochs 3 \
  --output_dir adapter_distill/checkpoints/pythia160m
```

Multi-GPU 7B run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  adapter_distill/train_adapter.py \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --data_path data/latent_action_pretraining_openx.jsonl \
  --model_name EleutherAI/pythia-6.9b \
  --batch_size 1 \
  --num_epochs 15 \
  --output_dir adapter_distill/checkpoints/pythia6p9b
```

For `torchrun`, `--batch_size` is per GPU. The effective global batch size is
`batch_size * world_size`.

Only the projection layers and Pythia are trainable. The extracted LAPA tensors stay frozen.
By default, training reads the full JSONL file and uses every record exactly
once: the first 97% for training and the last 3% for validation. On the local
OpenX file with 277,758 records, this is about 269K train and 8K validation
examples.

For capped runs, pass explicit sample counts. `none` and `all` mean "use the
remaining records":

```bash
python adapter_distill/train_adapter.py \
  --train_samples 20000 \
  --val_samples 1000
```

Training writes checkpoints and metrics under `--output_dir`. By default:

- `last.pt`: latest adapter checkpoint
- `best.pt`: best adapter checkpoint by validation accuracy
- `metrics.jsonl`: run config, sampled train losses, and epoch validation metrics

Example metric records:

```json
{"event": "train_step", "epoch": 0, "global_step": 50, "loss": 1.923, "lr": 0.0001}
{"event": "epoch", "epoch": 0, "train_loss": 2.104, "val_loss": 1.887, "val_acc": 0.312}
```

Use `--log_every 10` to record train-step loss more frequently, or
`--metrics_path path/to/metrics.jsonl` to write metrics somewhere else.

The OpenX trainer supports single-node FSDP for 7B Pythia runs. Relevant
options:

```bash
--distributed_strategy auto   # auto|fsdp|none
--precision auto              # auto|bf16|fp16|fp32
--no_gradient_checkpointing
--seed 1234
--num_workers 2
```

`auto` uses FSDP when launched by `torchrun` with `WORLD_SIZE > 1`. Precision
defaults to BF16 on supported CUDA GPUs, otherwise FP16 on CUDA and FP32 on CPU.

The trainer skips malformed JSONL records by default and prints the skipped
line number. This is useful when a downloaded dataset has a truncated final
record. Pass `--strict_jsonl` if you prefer the run to fail on any malformed
line.

For a quick smoke test:

```bash
python adapter_distill/train_adapter.py \
  --model_name EleutherAI/pythia-160m \
  --train_samples 8 \
  --val_samples 4 \
  --batch_size 2 \
  --num_epochs 1 \
  --output_dir adapter_distill/checkpoints/pythia160m_smoke
```

## 3. Inference from JSONL Vision Tokens

```bash
python adapter_distill/infer_adapter.py \
  --checkpoint adapter_distill/checkpoints/pythia160m/best.pt \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --jsonl data/latent_action_pretraining_openx.jsonl \
  --index 0
```

This prints the predicted 4-token latent action and, for JSONL input, the target `delta`.

## 4. Inference from Raw Image

```bash
python adapter_distill/infer_adapter.py \
  --checkpoint adapter_distill/checkpoints/pythia160m/best.pt \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --vqgan_checkpoint lapa_checkpoints/vqgan \
  --image imgs/bridge_inference.jpg \
  --instruction "<s> You are a helpful assistant. USER: take the broccoli out of the pot ASSISTANT:"
```

Raw-image inference uses the original LAPA VQGAN to produce 256 vision tokens.

## 5. SIMPLER Action Fine-tuning

The PyTorch adapter checkpoint is not a JAX `params::...` LAPA checkpoint. For
SIMPLER, fine-tune this adapter directly on the 7-token `action` field in
`data/simpler.jsonl`.

Smoke test from the `LAPA` directory:

```bash
conda run -n lapaTemp python adapter_distill/train_simpler_adapter.py \
  --base_checkpoint adapter_distill/checkpoints/pythia160m/best.pt \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --data_path data/simpler.jsonl \
  --output_dir adapter_distill/checkpoints/pythia160m_simpler_smoke \
  --train_samples 8 \
  --val_samples 4 \
  --num_epochs 1 \
  --batch_size 2
```

Full fine-tune:

```bash
conda run -n lapaTemp python adapter_distill/train_simpler_adapter.py \
  --base_checkpoint adapter_distill/checkpoints/pythia160m/best.pt \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --data_path data/simpler.jsonl \
  --output_dir adapter_distill/checkpoints/pythia160m_simpler \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 2e-5
```

The output directory contains:

- `best.pt`: best checkpoint by validation action-token accuracy
- `last.pt`: latest checkpoint
- `metrics.jsonl`: run config, train loss samples, and epoch metrics

Action inference smoke test:

```bash
conda run -n lapaTemp python adapter_distill/infer_adapter.py \
  --mode action \
  --checkpoint adapter_distill/checkpoints/pythia160m_simpler/best.pt \
  --frozen_lapa adapter_distill/artifacts/lapa_frozen.npz \
  --vocab_file lapa_checkpoints/tokenizer.model \
  --jsonl data/simpler.jsonl \
  --index 0
```

SIMPLER smoke eval:

```bash
cd SimplerEnv
./scripts/lapa_adapter_bridge.sh smoke
```

Full Bridge eval:

```bash
cd SimplerEnv
./scripts/lapa_adapter_bridge.sh full
```

The eval script defaults to
`adapter_distill/checkpoints/pythia160m_simpler/best.pt`. Override paths with
`CKPT_PATH`, `FROZEN_LAPA_PATH`, `ACTION_SCALE_FILE`, `VOCAB_FILE`, and
`VQGAN_CHECKPOINT` environment variables.

## Notes

- The instruction is tokenized with the LAPA/LLaMA tokenizer, not the Pythia tokenizer.
- Adapter checkpoints intentionally do not include the large frozen LAPA tensors; keep `lapa_frozen.npz` alongside them.
- If loading `EleutherAI/pythia-160m` fails, install the packages in `requirements_adapter.txt` in the environment used for this experiment.
