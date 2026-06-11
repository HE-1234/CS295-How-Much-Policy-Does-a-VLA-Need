# How Much Policy Does a VLA Need?

This repository studies how language-model backbone size affects latent-action
learning in a Vision-Language-Action (VLA) policy. It builds on
[LAPA: Latent Action Pretraining from Videos](https://latentactionpretraining.github.io/)
and compares Pythia backbones from 160M to 6.9B parameters while keeping the
visual representation, action representation, data, and training objective as
fixed as possible.

The main experiment places a trainable Pythia model between frozen LAPA
embeddings and LAPA's frozen latent-action head:

```text
frozen LAPA vision/text/action embeddings
                    |
          Linear(4096 -> hidden)
                    |
           trainable Pythia LM
                    |
          Linear(hidden -> 4096)
                    |
       frozen LAPA latent-action head
```

## Current Finding

The completed and partial runs currently reach roughly the same validation
token accuracy:

| Backbone | Best validation accuracy | Run status |
| --- | ---: | --- |
| Pythia 160M | 47.1% | Complete |
| Pythia 410M | 47.8% | Complete enough to compare |
| Pythia 1B | 46.9% | Partial, 3 of 15 epochs |
| Pythia 6.9B | Not available | Launched, no completed epoch |

Random accuracy is 12.5%. The provisional result is that backbone size from
160M to 1B is not the dominant bottleneck in this frozen-representation
fine-tuning regime. Overfitting, regularization, and data scale have a larger
effect. These numbers are not directly comparable to the original LAPA paper
because this project fine-tunes pretrained Pythia models rather than
pretraining a LLaMA backbone from scratch.

The repository contains two complementary policy-capacity studies:

1. `vla_foundry/` varies diffusion-policy transformer depth with a frozen
   Foundry-VLM backbone.
2. `LAPA_torch/LAPA/` varies the Pythia backbone inside a frozen LAPA
   representation.

See the [LAPA progress report](LAPA_torch/LAPA/REPORT.md) and the
[VLA Foundry runbook](vla_foundry/RUNBOOK.md) for experiment-specific results,
limitations, and remaining work.

## Repository Guide

### VLA Foundry Policy-Size Study

| Path | Purpose |
| --- | --- |
| [`vla_foundry/`](vla_foundry) | VLA Foundry source, tests, documentation, and local policy-size changes |
| [`RUNBOOK.md`](vla_foundry/RUNBOOK.md) | Policy-depth sweep design, commands, results, and known issues |
| [`report_questions.md`](vla_foundry/report_questions.md) | Report-oriented interpretation and experiment details |
| [`scripts/sweep/`](vla_foundry/scripts/sweep) | Dataset, training, validation, and evaluation workflow |
| [`config_presets/models/`](vla_foundry/vla_foundry/config_presets/models) | 77M, 205M, and 410M transformer configurations |
| [`experiments/`](vla_foundry/experiments) | Tracked validation summaries only; large data and run outputs are excluded |

### LAPA Backbone-Size Study

| Path | Purpose |
| --- | --- |
| [`REPORT.md`](LAPA_torch/LAPA/REPORT.md) | Research motivation, design decisions, results, and next steps |
| [`walk through.md`](LAPA_torch/LAPA/walk%20through.md) | Reproducible JAX and PyTorch GPU environment setup |
| [`adapter_distill/`](LAPA_torch/LAPA/adapter_distill) | Main frozen-LAPA adapter implementation, training, and inference |
| [`adapter_distill/README.md`](LAPA_torch/LAPA/adapter_distill/README.md) | Detailed commands for extraction, training, inference, and SIMPLER |
| [`train_policy.py`](LAPA_torch/LAPA/train_policy.py) | Simpler PyTorch rewrite used as a secondary baseline |
| [`latent_pretraining/`](LAPA_torch/LAPA/latent_pretraining) | Original LAPA latent-pretraining code plus evaluation utilities |
| [`SimplerEnv/`](LAPA_torch/LAPA/SimplerEnv) | Downstream simulation integration and Bridge evaluation scripts |
| [`data/`](LAPA_torch/LAPA/data) | Small tracked SIMPLER data and preprocessing utilities |

The original LAPA documentation remains available at
[`LAPA_torch/LAPA/README.md`](LAPA_torch/LAPA/README.md).

## Main Project Additions

- `adapter_distill/models.py`: frozen-LAPA adapter policy.
- `adapter_distill/train_adapter.py`: Open-X latent-action training, including
  multi-GPU FSDP support.
- `adapter_distill/train_simpler_adapter.py`: downstream action fine-tuning.
- `adapter_distill/infer_adapter.py`: latent-action and robot-action inference.
- `SimplerEnv/simpler_env/policies/lapa_adapter/`: SIMPLER policy wrapper.
- `SimplerEnv/scripts/lapa_adapter_bridge.sh`: smoke and full Bridge evaluations.
- `latent_pretraining/sample_openx_loss.py`: official LAPA loss reference.

## Setup

The project combines an older JAX/Flax stack used by LAPA with a PyTorch stack
used by the adapter experiments. Follow the full
[environment walkthrough](LAPA_torch/LAPA/walk%20through.md); in particular,
use the documented JAX/cuDNN versions and do not install
`tensorflow[and-cuda]` into that environment.

For only the PyTorch adapter dependencies:

```bash
cd LAPA_torch/LAPA
python -m pip install -r adapter_distill/requirements_adapter.txt
```

## Required External Artifacts

Large datasets, model weights, extracted tensors, and experiment checkpoints
are intentionally excluded from Git. Before running the main experiment,
provide:

```text
LAPA_torch/LAPA/lapa_checkpoints/tokenizer.model
LAPA_torch/LAPA/lapa_checkpoints/vqgan
LAPA_torch/LAPA/lapa_checkpoints/params
LAPA_torch/LAPA/data/latent_action_pretraining_openx.jsonl
LAPA_torch/LAPA/adapter_distill/artifacts/lapa_frozen.npz
```

The official LAPA checkpoint files and Open-X JSONL are available from the
[LAPA-7B-openx Hugging Face repository](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx).
Generate `lapa_frozen.npz` from the downloaded LAPA parameters:

```bash
cd LAPA_torch/LAPA
python adapter_distill/extract_lapa_weights.py \
  --lapa_checkpoint params::lapa_checkpoints/params \
  --output adapter_distill/artifacts/lapa_frozen.npz
```

## Run the Main Experiment

From `LAPA_torch/LAPA`, a small single-GPU run is:

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

For a fast installation and data-path check, add:

```text
--train_samples 8 --val_samples 4 --batch_size 2 --num_epochs 1
```

Detailed multi-GPU, inference, downstream fine-tuning, and SIMPLER commands are
documented in the
[adapter experiment guide](LAPA_torch/LAPA/adapter_distill/README.md).

## Project Status

- Frozen-LAPA adapter and small-model training: implemented.
- Pythia 160M and 410M comparisons: completed.
- Pythia 1B and 6.9B comparisons: incomplete.
- SIMPLER integration: implemented end to end.
- Downstream action generalization: currently limited by the small fine-tuning
  dataset.

The original LAPA code is distributed under the MIT License; see
[`LAPA_torch/LAPA/LICENSE`](LAPA_torch/LAPA/LICENSE).
