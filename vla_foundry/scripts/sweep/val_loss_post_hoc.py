"""Compute held-out validation loss on a single trained checkpoint.

Decouples val from training so training runs at full throughput. Reuses
`validate_one_checkpoint` from `vla_foundry/validate.py` so the loss is
computed exactly the way main.py would compute it inline.

Usage:
    uv run python scripts/sweep/val_loss_post_hoc.py \\
        --config_path  <run-dir>/config.yaml \\
        --checkpoint   <run-dir>/checkpoints/checkpoint_1.pt \\
        --val_manifest <data-dir>/manifest_val.jsonl \\
        --val_stats    <data-dir>/stats.json \\
        --val_samples  2048 \\
        [--rung_name <label>]

Prints one CSV row:
    <rung_name>,<checkpoint_num>,<step>,<val_loss>,<samples>

`05_val_loss_post_hoc.sh` is the wrapper that calls this once per rung.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

from vla_foundry.data.dataloader import get_datastring_input, get_wds_dataloader
from vla_foundry.distributed import get_model_precision
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.losses import get_loss_function
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml
from vla_foundry.validate import validate_one_checkpoint


def _override(obj, attr, value):
    """Override a field on a frozen draccus dataclass."""
    object.__setattr__(obj, attr, value)


def _parse_checkpoint_num(checkpoint_path: str) -> int:
    """Extract N from .../checkpoint_N.pt; default to 0 if not parseable."""
    m = re.search(r"checkpoint_(\d+)\.pt$", checkpoint_path)
    return int(m.group(1)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config_path", required=True, help="Path to saved config.yaml from training.")
    ap.add_argument("--checkpoint", required=True, help="Path to a single checkpoint_N.pt to evaluate.")
    ap.add_argument("--val_manifest", required=True, help="Path to manifest_val.jsonl (held-out shards).")
    ap.add_argument("--val_stats", required=True, help="Path to stats.json (normally same file as train).")
    ap.add_argument("--val_samples", type=int, default=2048, help="Number of val samples to score.")
    ap.add_argument("--rung_name", default="rung", help="Label written into the CSV first column.")
    ap.add_argument("--header", action="store_true", help="Also print the CSV header line.")
    args = ap.parse_args()

    # 1. Load the saved config and override val fields. We pass the *saved* config
    #    (not the chassis YAML) because it has the resolved action_dim / proprio
    #    dim baked in, plus the exact transformer dims used at train time.
    cfg = load_experiment_params_from_yaml(args.config_path)
    _override(cfg.data, "val_dataset_manifest", [args.val_manifest])
    _override(cfg.data, "val_dataset_statistics", [args.val_stats])
    _override(cfg.data, "val_dataset_weighting", [1.0])
    _override(cfg, "total_val_samples", args.val_samples)
    # Single-GPU; no need to fight FSDP for a small forward-only pass.
    _override(cfg.distributed, "fsdp", False)

    device = torch.device(cfg.distributed.device)

    # 2. Build model (architecture only — we'll load weights from the checkpoint).
    model = create_model(cfg.model, load_pretrained=False)
    model = model.to(device, dtype=get_model_precision(cfg))

    # 3. Load the checkpoint state into the model.
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    start_checkpoint_num, global_step, _ = load_model_checkpoint(model, args.checkpoint)

    # 3a. Force num_action_head_repeats=1 for val.
    #
    # At training time, num_action_head_repeats=8 makes the action expert
    # denoise N=8 noise samples per VLM forward (a gradient noise-reduction
    # trick). The tiling [B] -> [B*N] happens inside the batch handler's
    # slice_inputs_for_accumulation, which runs during gradient accumulation
    # in the train loop. `validate_one_checkpoint` only calls
    # prepare_inputs_and_targets and never tiles, so the model's forward
    # assertion (actions.shape[0] == vlm_batch * num_repeats) fails.
    #
    # The 8x replication doesn't affect what we're measuring — expected MSE
    # per sample is identical whether you average over 1 or 8 noise samples
    # per VLM forward. Setting repeats=1 sidesteps the tiling requirement.
    if getattr(model, "num_action_head_repeats", None) and model.num_action_head_repeats > 1:
        model.num_action_head_repeats = 1

    # 4. Build val dataloader the same way main.py does.
    val_datastrings, val_num_samples_per_dataset, _, _ = get_datastring_input(
        num_samples=cfg.total_val_samples,
        curr_shard_idx_per_dataset=[0 for _ in cfg.data.val_dataset_manifest],
        shard_shuffle_seed_per_dataset=[cfg.hparams.seed for _ in cfg.data.val_dataset_manifest],
        manifest_paths=cfg.data.val_dataset_manifest,
        dataset_weighting=cfg.data.val_dataset_weighting,
        allow_multiple_epochs=True,
        num_workers_per_gpu=cfg.data.num_workers,
        world_size=cfg.distributed.world_size,
    )
    val_dataloader = get_wds_dataloader(val_datastrings, val_num_samples_per_dataset, 0, cfg)

    # 5. Compute val loss using the same code path as inline val.
    loss_fn = get_loss_function(cfg.hparams.loss_function, cfg.hparams)
    checkpoint_num = _parse_checkpoint_num(args.checkpoint)
    avg_loss = validate_one_checkpoint(
        model=model,
        val_dataloader=val_dataloader,
        loss=loss_fn,
        checkpoint_num=checkpoint_num,
        step=global_step,
        cfg=cfg,
    )

    if args.header:
        print("rung,checkpoint,step,val_loss,samples", flush=True)
    print(f"{args.rung_name},{checkpoint_num},{global_step},{avg_loss:.6f},{cfg.total_val_samples}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
