"""Print a per-submodule parameter breakdown for a resolved training config.

Used to verify each policy-size rung lands where the design says before any
training spend.

Usage:
    uv run python scripts/print_model_size_breakdown.py <path-to-config.yaml>

The config can be:
  - A resolved config produced by `vla_foundry/main.py --resolve_configs True
    --resolve_configs_path <dir>` (writes `resolved_config.yaml`)
  - Any TrainExperimentParams YAML the launcher would otherwise accept (e.g.
    `vla_foundry/config_presets/training_jobs/vla_pickandplace_box.yaml`).

By default the model is built with `load_pretrained=False` so we don't pull
multi-GB checkpoints just to count tensors. Pass `--load_pretrained` if you
specifically want to validate that the backbone load succeeds.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict

from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


def _fmt(n: int) -> str:
    return f"{n:>15,} ({n / 1e6:7.2f}M)"


def _agg_by_prefix(named_params, prefix: str):
    total = 0
    trainable = 0
    for name, p in named_params:
        if name.startswith(prefix):
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
    return total, trainable


def _top_level_submodules(model) -> list[str]:
    """First-level child module names, in registration order."""
    return [name for name, _ in model.named_children()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to resolved config.yaml or a training-job preset YAML.")
    ap.add_argument(
        "--load_pretrained",
        action="store_true",
        help="If set, also load backbone pretrained weights (default: arch-only, no weight download).",
    )
    args = ap.parse_args()

    cfg = load_experiment_params_from_yaml(args.config)

    # Build model. create_model() dispatches on cfg.model.type via the registry.
    # diffusion_policy's create function takes (params, load_pretrained=...).
    model = create_model(cfg.model, load_pretrained=args.load_pretrained)

    # Apply freeze flags exactly the way BaseModel._post_init would for any
    # submodule with model_params.freeze=True. create_model already calls
    # _post_init when constructing the diffusion policy and its children, so
    # freeze flags should already be applied; we just inspect them here.
    named = list(model.named_parameters())

    total = sum(p.numel() for _, p in named)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)

    print("=" * 80)
    print("Model size summary")
    print("=" * 80)
    print(f"Total model params:              {_fmt(total)}")
    print(f"Trainable total params:          {_fmt(trainable)}")

    # Backbone is always under `vision_language_backbone.` in the
    # DiffusionPolicy module tree (see vla_foundry/models/diffusion_policy/diffusion_policy.py).
    bb_total, bb_trainable = _agg_by_prefix(named, "vision_language_backbone.")
    non_bb_total = total - bb_total
    non_bb_trainable = trainable - bb_trainable

    print(f"VLM backbone params:             {_fmt(bb_total)}")
    print(f"Trainable VLM backbone params:   {_fmt(bb_trainable)}")
    print(f"Non-VLM params:                  {_fmt(non_bb_total)}")
    print(f"Trainable non-VLM params:        {_fmt(non_bb_trainable)}")

    print()
    print("Diffusion/action policy")
    print("-" * 80)
    print(f"Diffusion policy params:         {_fmt(non_bb_total)}")
    print(f"Trainable diffusion params:      {_fmt(non_bb_trainable)}")

    print()
    print("Diffusion/action breakdown")
    print("-" * 80)
    # Aggregate by top-level child module name, excluding the backbone (already shown above).
    seen = OrderedDict()
    for child_name in _top_level_submodules(model):
        if child_name == "vision_language_backbone":
            continue
        t, tr = _agg_by_prefix(named, f"{child_name}.")
        if t == 0:
            continue
        seen[child_name] = (t, tr)

    # Sort by size descending so the dominant block is on top.
    for name, (t, tr) in sorted(seen.items(), key=lambda kv: -kv[1][0]):
        marker = "" if tr == t else f"  (trainable {tr:,})"
        print(f"  {name:<22} {_fmt(t)}{marker}")

    # Sanity warnings
    if cfg.model.vision_language_backbone is not None and getattr(
        cfg.model.vision_language_backbone, "freeze", False
    ):
        if bb_trainable != 0:
            print()
            print(
                f"WARNING: backbone freeze=True but {bb_trainable:,} backbone params "
                "still have requires_grad=True. Freeze did not take.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
