# Policy-Size Ablation — Runbook

A coauthor-facing project log. Covers what the experiment is, what we ran, what
we changed during the run, and what's deferred.

## Experiment in one paragraph

Train a diffusion policy on a single LBM sim task with the VLM backbone held
frozen, varying only the policy transformer size. Three rungs
(`transformer_77m`, `transformer_205m`, `transformer_410m`) × one budget
(40 000 samples seen) × one task per sweep. Three tasks have been swept so
far: `PickAndPlaceBox`, `PushBox`, and `PutOrangeOnSaucer`, each under its
own experiment root. The currently-active task (per `env.sh`) is `PushBox`.
Backbone is `hf://TRI-ML/Foundry-VLM-1.3B-200M` loaded via the
`vlm_foundry_backbone` model type with `freeze: true`.

## Where the experiment lives

| Path | Contents |
|---|---|
| `scripts/sweep/` | The eight numbered shell scripts that run the sweep end-to-end. |
| `scripts/sweep/env.sh` | All tunables (paths, budget, batch size, LR, warmup, eval settings). |
| `vla_foundry/config_presets/training_jobs/vla_pickandplace_box.yaml` | Chassis YAML — composes data + model defaults. CLI overrides per rung come from `04_train_rungs.sh`. |
| `vla_foundry/config_presets/models/transformer_{77m,205m,410m}.yaml` | The three policy-size presets. |
| `experiments/policy_size_pickandplace/` | First sweep workspace (PickAndPlaceBox). Data, trained runs, post-hoc val losses. |
| `experiments/policy_size_pushbox/` | Sweep workspace (PushBox). Same layout. |
| `experiments/policy_size_putorangeonsaucer/` | Sweep workspace (PutOrangeOnSaucer). Same layout. |

## Data

All three tasks share an embodiment, which is what makes them swappable with
the same training-job preset:

- 2 Panda arms (bimanual)
- 6 cameras: `scene_{left,right}_0` + 4 wrist cameras (`wrist_{left,right}_{minus,plus}`)
- 20-dim action: bimanual 6-DoF pose (3 xyz + 6-D rotation per arm) + per-arm
  gripper. Verified via the cumulative `action_index_fields` in
  `vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml`.

Sequence counts: PickAndPlaceBox ≈ 5.2K, PutOrangeOnSaucer ≈ 5.3K,
PushBox ≈ 2.1K.

## Model configuration

| Knob | Value | Notes |
|---|---|---|
| VLM backbone | `hf://TRI-ML/Foundry-VLM-1.3B-200M`, frozen | Set by `vla_pickandplace_box.yaml` (`vision_language_backbone.freeze: true`). |
| Policy transformer | `transformer_77m`, `transformer_205m`, or `transformer_410m` | Swapped via `--model.transformer "include …"` on the CLI (per-rung). |
| `--model.transformer.is_causal` | `False` | Policy uses bidirectional attention. |
| `--model.transformer.vocab_size` | `0` | Vocab embedding is unused in diffusion-policy mode. |
| `action_dim` / `proprioception_dim` | auto-populated from data config | Do not set manually. |

Per-rung overrides go in via the CLI (not new YAML files) — the repo's
`<<: !include` directive can't partial-override an included block.

## Training command

We don't invoke `main.py` directly; everything goes through
`scripts/sweep/04_train_rungs.sh`. That script reads `env.sh` and launches one
`uv run python vla_foundry/main.py …` per rung. Single GPU, no FSDP, no
`torch.compile`. The canonical invocation is the script itself — see it for
the exact CLI flags.

To train one rung instead of all three:
```bash
bash scripts/sweep/04_train_rungs.sh transformer_77m
```

## Eval plan

Two independent signals:

- **Open-loop val MSE** via `05_val_loss_post_hoc.sh` — runs after training
  finishes, no Docker needed. Writes `$EXP_ROOT/val_losses.csv` (one row per
  rung × checkpoint).
- **Closed-loop success rate** via `06_eval_sim.sh` — pulls
  `toyotaresearch/lbm-eval-oss:vla-foundry` and rolls out 200 episodes per
  rung. Requires Docker. Results land in `$EXP_ROOT/rollouts/`. Visualize via
  `07_open_dashboard.sh` (Streamlit on `localhost:8505`).

`EVAL_MAX_SAMPLE_SIZE=200` must stay fixed across all rungs we ever compare —
it's the STEP-test decision boundary, set per
`tutorials/STATISTICAL_COMPARISON.md`.

## Session changes (what we edited during this run)

Chronological. Each row names the file and the why.

| Change | File | Why |
|---|---|---|
| `--num_checkpoints` set to `1` | `scripts/sweep/04_train_rungs.sh` | One save per rung is enough for this ablation. Deferred follow-up: the saved checkpoint still contains the frozen VLM backbone (~2.5 GB redundant). See "Known issues" below. |
| Added dataloader concurrency knobs: `--data.num_workers 4 --data.prefetch_factor 2 --data.shuffle_buffer_size 400` | `scripts/sweep/04_train_rungs.sh` | Iterated on these to balance two failure modes on a shared NFS-backed box: (a) too aggressive → OOM kills the dataloader worker, (b) too conservative → GPU starvation, 0% util. Current values are a compromise; expect to retune if the host load profile changes. |
| `WARMUP=1000 → 50` | `scripts/sweep/env.sh` | Total optimizer steps per rung ≈ 312 (`BUDGET=40000 / GLOBAL_BATCH_SIZE=128`). A 1000-step warmup meant the LR never reached its target — early loss curves were essentially "model at LR ≈ 0". |
| `TASK=PickAndPlaceBox → PushBox` and matching `EXP_ROOT` switch | `scripts/sweep/env.sh` | Moved the active sweep to PushBox. Both tasks share embodiment so no other config changes were needed. PickAndPlaceBox runs are preserved under their own experiment root. |
| `WITH_VAL_LOSS=0` (default) | `scripts/sweep/env.sh` | Inline val during training costs ~1.75× throughput because the val dataloader's workers compete with the train dataloader. We run val post-hoc via script `05` instead. |

## Known issues / deferred

These were noticed and consciously deferred — fix when convenient.

1. **VLM backbone is re-saved in every checkpoint.** `save_checkpoint`
   (`vla_foundry/file_utils.py`) dumps the full `model.state_dict()`, which
   includes the frozen backbone (~2.5 GB per checkpoint). Three changes would
   fix it: filter `vision_language_backbone.*` keys at save, switch
   `model.load_state_dict(sd)` → `model.load_state_dict(sd, strict=False)` at
   load, and flip `from_pretrained` in `vla_foundry/models/base_model.py` from
   `load_pretrained=False` → `True` so the backbone is re-fetched from HF at
   inference. Touches framework code; deferred to keep the sweep moving.
2. **Data + HF cache live on NFS.** `/home/jianch2` is mounted from
   `tardigrade.ics.uci.edu:/grad/home/jianch2`. Every shard read and every
   weight load goes over the network. Local scratch is available at
   `/scratch` (1.7 TB, ~360 GB free) and would remove the NFS bottleneck.
   Considered, not staged.
3. **Shared box, no scheduler.** `pyxis.ics.uci.edu` is shared with ~25
   users and has no SLURM/cgroup isolation. Training has hit OOM kills when
   neighbors spike. Mitigation discussed but not wired in:
   ```bash
   systemd-run --user --scope -p MemoryMax=25G -p MemorySwapMax=0 \
     scripts/sweep/04_train_rungs.sh
   ```
4. **WARMUP × total-steps coupling is implicit.** `WARMUP=50` works for
   `BUDGET=40000` + `GLOBAL_BATCH_SIZE=128` (≈ 312 steps). If you change
   either, recheck warmup.

## Status

| Task | Rungs trained | Val loss (post-hoc) | Closed-loop eval |
|---|---|---|---|
| PickAndPlaceBox | 77m, 205m, 410m | `val_losses.csv` ✓ | not yet (no Docker) |
| PushBox | 77m, 205m, 410m | `val_losses.csv` ✓ | not yet (no Docker) |
| PutOrangeOnSaucer | 77m, 205m, 410m | `val_losses.csv` ✓ | not yet (no Docker) |

Trained run dirs live under
`experiments/policy_size_{pickandplace,pushbox,putorangeonsaucer}/runs/<TASK>_<rung>_budget-40000/policy_size_<rung>/<timestamp>/checkpoints/`.

## Re-running

From the repo root, with `env.sh` pointing at the task you want:

```bash
bash scripts/sweep/00_check_gpu.sh
bash scripts/sweep/01_download_data.sh
bash scripts/sweep/02_split_val.sh      # if WITH_VAL_LOSS=1, otherwise optional
bash scripts/sweep/03_smoke_test.sh
bash scripts/sweep/04_train_rungs.sh    # the long step (~1–2h per rung)
bash scripts/sweep/05_val_loss_post_hoc.sh
bash scripts/sweep/06_eval_sim.sh       # Docker required
bash scripts/sweep/07_open_dashboard.sh
```

To switch between the two tasks, edit `env.sh` (`TASK`, `EXP_ROOT`) and
re-run from `01_` (downloads are idempotent if the data is already on disk).
