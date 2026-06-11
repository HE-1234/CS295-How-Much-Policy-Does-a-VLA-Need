# Policy-Size Sweep — Scripts

Operational playbook for the eight numbered scripts in this directory. For
narrative (what the experiment is, what we've run, what's deferred), see
`../../RUNBOOK.md`.

All scripts source `env.sh` for paths and hyperparameters. Edit `env.sh` (not
the individual scripts) to change anything global.

## Scripts

| # | Script | One-line purpose |
|---|---|---|
| 00 | `00_check_gpu.sh` | Verify PyTorch can use the GPU; fails loudly if not. |
| 01 | `01_download_data.sh` | Pull the task's preprocessed shards (~1.2 GB) into `$EXP_ROOT/data/$TASK/`. Idempotent. |
| 02 | `02_split_val.sh` | Split `manifest.jsonl` into 90/10 train/val. Only needed if `WITH_VAL_LOSS=1` *or* if you'll run `05_val_loss_post_hoc.sh`. |
| 03 | `03_smoke_test.sh` | ~5-min single-rung run that flushes integration bugs end-to-end before the long sweep. |
| 04 | `04_train_rungs.sh` | The real sweep. Trains all three rungs sequentially (or one if you pass its name). |
| 05 | `05_val_loss_post_hoc.sh` | Compute held-out val MSE on each saved checkpoint. Decoupled from training so train runs at full throughput. No Docker. Writes `$EXP_ROOT/val_losses.csv`. |
| 06 | `06_eval_sim.sh` | Closed-loop sim eval via `lbm-eval-oss`. **Requires Docker.** |
| 07 | `07_open_dashboard.sh` | Open the Streamlit results dashboard on `http://localhost:8505`. |

`val_loss_post_hoc.py` (not numbered) is a standalone Python helper called by
`05_*.sh`; it loads one checkpoint and reuses `validate_one_checkpoint`.

## Recommended order for a fresh task

```bash
cd /home/jianch2/CS295/vla_foundry

bash scripts/sweep/00_check_gpu.sh
bash scripts/sweep/01_download_data.sh
bash scripts/sweep/02_split_val.sh
bash scripts/sweep/03_smoke_test.sh
bash scripts/sweep/04_train_rungs.sh        # the long step

# After training, either or both eval signals:
bash scripts/sweep/05_val_loss_post_hoc.sh  # open-loop val MSE per rung
bash scripts/sweep/06_eval_sim.sh           # closed-loop success rate (Docker)
bash scripts/sweep/07_open_dashboard.sh
```

## Common targeted runs

Single rung only:
```bash
bash scripts/sweep/04_train_rungs.sh transformer_205m
```

Different task — edit `TASK` and `EXP_ROOT` in `env.sh`, then re-run from
`01_`. Sanity-check that the new task uses the same 6-camera bimanual /
20-dim-action embodiment (the chassis YAML
`vla_foundry/config_presets/training_jobs/vla_pickandplace_box.yaml` assumes
that layout).

## Tunables

All in `env.sh`. The values you'll most likely touch:

- `TASK`, `EXP_ROOT` — which task and where to write everything.
- `BUDGET` — total training samples per rung. Halve for faster iteration.
- `PER_GPU_BATCH_SIZE` — drop if you OOM. Don't touch `GLOBAL_BATCH_SIZE`
  unless you've thought about it (it's the effective batch).
- `WITH_VAL_LOSS` — `0` (default) skips inline val; we compute it post-hoc via
  `05_*`. `1` is ~1.75× slower because the val dataloader steals workers.
- `EVAL_NUM_EPISODES` — `"0:200"` matches the published recipe; drop to
  `"0:50"` for a quick first read. `EVAL_MAX_SAMPLE_SIZE` must stay constant
  across every rung you compare (locks STEP's stopping boundary).

For the dataloader concurrency knobs (`num_workers`, `prefetch_factor`,
`shuffle_buffer_size`), those are set inline in `04_train_rungs.sh` because we
tuned them mid-sweep to balance OOM safety against GPU starvation on a shared
NFS-backed host. See `RUNBOOK.md` § *Session changes*.
