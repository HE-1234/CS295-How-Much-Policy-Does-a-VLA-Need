# Report Q&A — Answers from the Repo

Where the answer is grounded in the repo or this session's work, I cite the
source. Where it's a stylistic / personal decision or requires data I don't
have, I flag it as **needs your input** and skip the substantive answer.

---

## 1. Authors, course, institution

**Partial.** From repo paths and host:
- Course: **CS295** (path: `~/CS295/vla_foundry`)
- Institution: **UC Irvine, ICS** (host `pyxis.ics.uci.edu`, home mounted from `tardigrade.ics.uci.edu:/grad/home/jianch2`)
- Git author: `HE-1234` (account `jianch2`)

**Needs your input:** full author name(s), coauthor name(s), instructor, term.

## 2. Homepage vs `/posts/vla-policy-size/`

**Needs your input** — stylistic choice. Mild lean: keep under
`/posts/vla-policy-size/` and add a homepage link. The result is a focused
ablation, not a portfolio capstone, so dedicating the homepage to it
over-weights this single project.

## 3. Final one-sentence conclusion

**Draft (based on the val-loss data — see Q9):** *Across three LBM bimanual
tasks (PickAndPlaceBox, PushBox, PutOrangeOnSaucer) trained at 40K samples
with a frozen Foundry-VLM-1.3B backbone, deeper policy heads consistently
beat shallower ones on held-out val MSE; the end-to-end gap (77M → 410M)
ranges from ~3% (PushBox) to ~10% (PickAndPlaceBox), and on
PutOrangeOnSaucer essentially all of the improvement comes from going 205M
→ 410M (77M ≈ 205M) — suggesting capacity is starting to bind only at the
largest rung at this data scale.*

This will need revising once closed-loop success rates are in.

## 4. Audience

**Needs your input.** Default suggestion: classmates + technical readers
familiar with transformer training, but not assumed to know VLAs. Frame the
VLM/diffusion-policy split early.

## 5. Architectural difference: 77M / 205M / 410M

**Single axis: `n_layers` only.** Every other knob held constant.

| | 77M | 205M | 410M |
|---|---:|---:|---:|
| `n_layers` | 6 | 12 | 24 |
| `hidden_dim` | 1024 | 1024 | 1024 |
| `n_heads` (head dim = 64) | 16 | 16 | 16 |
| `max_seq_len` | 2048 | 2048 | 2048 |
| `post_embed_norm` | false | false | false |
| `weight_tying` | false | false | false |

Source: `vla_foundry/config_presets/models/transformer_{77m,205m,410m}.yaml`.

Two CLI overrides applied to every rung in `scripts/sweep/04_train_rungs.sh`:
- `--model.transformer.is_causal False` — bidirectional attention (policy, not LM)
- `--model.transformer.vocab_size 0` — drops the unused vocab embedding table

So this is a clean **depth-only** scaling axis. The name "410M" is computed
with the vocab embed counted; runtime trainable params are smaller
(~325M for the 410m rung, per the training log earlier this session).

Implication: results are about **depth (compute per token)**, not width or
total parameter count. A negative result here doesn't rule out width.

## 6. Foundry-VLA repo / paper / checkpoint / dataset

**Partial.**
- **Framework repo:** [`TRI-ML/vla-foundry`](https://github.com/TRI-ML/vla-foundry) (upstream of this fork).
- **Backbone checkpoint used:** `hf://TRI-ML/Foundry-VLM-1.3B-200M`, specifically `checkpoints/checkpoint_75.pt`. Loaded frozen via the `vlm_foundry_backbone` model type.
- **Dataset:** LBM sim preprocessed shards published at `https://vla-foundry.s3.amazonaws.com/datasets/lbm_sim/preprocessed/v0.1.0`. Three tasks pulled and swept: `PickAndPlaceBox`, `PushBox`, `PutOrangeOnSaucer`.

**Needs your input:** exact paper(s) to cite for the Foundry-VLA release and
LBM. Candidates: TRI-ML's Large Behavior Model paper; the Foundry-VLA
release post / arXiv if one exists.

## 7. Training data per task / splits

| Task | Total sequences | Train shards / Val shards | Approx split |
|---|---:|:---:|:---:|
| PickAndPlaceBox | 5,175 | 46 / 6 | 88% / 12% |
| PutOrangeOnSaucer | 5,324 | 48 / 6 | 89% / 11% |
| PushBox | 2,111 | 19 / 3 | 86% / 14% |

Sources: `experiments/policy_size_*/data/<TASK>/manifest_{train,val}.jsonl`
line counts, sequence totals from the dataset registry.

Splits are produced by `scripts/sweep/02_split_val.sh` (nominally 90/10 by
shard; the realized split is constrained by the integer number of shards).

The 2K-sample val budget used in `05_val_loss_post_hoc.sh` (`VAL_SAMPLES=2048`
in `env.sh`) is sampled from the val manifest; with PushBox at 2,111 train+val
sequences, val MSE measurements have wider uncertainty than on PickAndPlaceBox.

## 8. Hardware, duration, hyperparameters, seeds, checkpoint selection

| | Value | Source |
|---|---|---|
| GPU | 1 × NVIDIA RTX A6000 (49 GB) | `nvidia-smi` on `pyxis.ics.uci.edu` |
| Distributed | Single-process, FSDP off, `torch.compile` off | `04_train_rungs.sh` |
| Optimizer | AdamW | `04_train_rungs.sh` |
| Learning rate | 5e-5 | `env.sh` (`LR`) |
| LR scheduler | cosine, warmup = 50 steps | `env.sh` (`WARMUP`) |
| Effective batch | 128 (per-GPU 16, accumulation 8) | `env.sh` (`GLOBAL_BATCH_SIZE`, `PER_GPU_BATCH_SIZE`) |
| Total samples / steps | 40,000 / ~312 optimizer steps | `BUDGET / GLOBAL_BATCH_SIZE` |
| Epochs | ~7.7 (PickAndPlaceBox) / ~19 (PushBox) | `BUDGET / sequences` |
| Seed | 42 (default) | `vla_foundry/params/hyper_params.py:12` |
| Checkpoint selection | Last checkpoint only (`--num_checkpoints 1`) | `04_train_rungs.sh` |
| Wall-clock per rung | ~1–2 h (varies with NFS load) | `04_train_rungs.sh` header |

**Caveats to report:**
- `WARMUP` was previously set to 1000 (exceeds total steps); fixed to 50
  partway through this session. Earlier runs trained at near-zero LR; re-runs
  may or may not be in the reported set — check timestamps under
  `experiments/.../runs/.../checkpoints/`.
- Dataloader concurrency (`num_workers`, `prefetch_factor`,
  `shuffle_buffer_size`) was retuned mid-session for OOM safety. Doesn't
  change the optimization, but does change the shuffle order seen by the
  model, which slightly perturbs reproducibility across re-runs.
- Single seed only — no confidence intervals on val loss or success rate.

## 9. Raw validation losses

**Available**, post-hoc, in `experiments/<exp>/val_losses.csv`:

**PickAndPlaceBox** (held-out val, 2048 samples):

| Rung | Step | Val MSE |
|---|---:|---:|
| transformer_77m | 312 | 0.300314 |
| transformer_205m | 312 | 0.280176 |
| transformer_410m | 312 | 0.270337 |

**PutOrangeOnSaucer** (held-out val, 2048 samples):

| Rung | Step | Val MSE |
|---|---:|---:|
| transformer_77m | 312 | 0.303680 |
| transformer_205m | 312 | 0.301670 |
| transformer_410m | 312 | 0.283980 |

**PushBox** (held-out val, 2048 samples):

| Rung | Step | Val MSE |
|---|---:|---:|
| transformer_77m | 312 | 0.308910 |
| transformer_205m | 312 | 0.300479 |
| transformer_410m | 312 | 0.290322 |

Monotonic ordering on all three tasks: bigger = better. End-to-end spread
77M → 410M: ~10% (PickAndPlaceBox), ~7% (PutOrangeOnSaucer), ~6% (PushBox).
Note the shape difference on PutOrangeOnSaucer — almost all of the
improvement is in the 205M → 410M step (77M and 205M are within 0.7% of each
other), unlike the other two tasks where the gains are more evenly
distributed across rungs.

**Note on training curves:** we save 1 checkpoint per rung
(`--num_checkpoints 1`), so we have one (step, val_loss) point per rung — not
a curve. If a per-step training-loss trace is needed for the report, it's
emitted in stdout/log files in the run dirs; to get a clean training curve
we'd need to re-run with W&B/CSV logging on, or grep the log files.

## 10. OpenVLA / Pythia 12.6%

**Skip — need from you.** I don't have the source of that number or what it's
measuring (success rate on what task set? language eval? param-efficiency
metric?).

## 11. Physical robot, cameras, controller, action freq, compute

**Sim only — no physical robot in this experiment.** All evaluations are
intended to run in LBM sim. For completeness:
- Simulated embodiment: 2 Franka Panda arms (bimanual)
- Cameras: 6 in sim — `scene_{left,right}_0`, `wrist_{left,right}_{minus,plus}`, fed to the model at 224×224
- Action: 20-dim — per arm: 3 xyz + 6-D rotation + 1 gripper
- Compute (training + eval): 1 × NVIDIA RTX A6000

**Don't know:** action frequency / control rate of the sim. Skip.

## 12. Success definition per task

**Skip — need from you / LBM docs.** Success is a binary `is_success` per
episode returned by `toyotaresearch/lbm-eval-oss:vla-foundry`. The per-task
criteria (object pose tolerance, etc.) live inside the eval container; I
don't have them surfaced here.

## 13. Rollouts per model × task

**Planned:** 200 episodes per (rung × task) — set by `EVAL_NUM_EPISODES="0:200"`
in `env.sh`. With 3 rungs × 3 tasks = 1,800 rollouts.

**Completed: 0.** Closed-loop eval (`06_eval_sim.sh`) hasn't been run yet —
requires Docker, which isn't set up on the box.

## 14. Complete success counts/rates including failed runs

**Skip — none collected yet.** See Q13. The val-loss table in Q9 is the
only quantitative comparison available so far.

## 15. Latency, memory, throughput, model size

**Partial.**

| Metric | Value | Source |
|---|---|---|
| Trainable params (410m rung) | ~325 M | Training log this session |
| Trainable params (205m rung) | ~150 M | extrapolated (linear in n_layers) |
| Trainable params (77m rung) | ~75 M | extrapolated |
| Total params incl. frozen VLM | ~1.85 B | Training log this session |
| Training memory (410m rung) | ~17 GB VRAM | `nvidia-smi --query-gpu=memory.used` mid-run |
| Training step time (410m rung, batch=16, 8 accum) | ~11 s / optimizer step ≈ 1.4 s / micro-batch | `tqdm` output |
| Disk per checkpoint | ~3 GB (includes redundant frozen VLM — see Known Issues in RUNBOOK) | `du -sh experiments/.../checkpoint_*.pt` |
| Hardware | NVIDIA RTX A6000 (Ampere, 49 GB) | `nvidia-smi` |

**Don't know:** inference latency / throughput. Skip — needs a benchmark
script we haven't written.

## 16. Qualitative differences (smoothness, hesitation, precision)

**Skip — no rollouts yet (Q13/Q14).**

## 17. Available rollout videos / GIFs / charts / logs

**Partial.**
- **Have:** training log files under `experiments/.../runs/.../<timestamp>/`, checkpoint `.pt` files, two `val_losses.csv` files.
- **Have, derivable:** simple val-loss bar chart or table from the CSVs above (matplotlib one-liner).
- **Don't have:** rollout videos, GIFs, success/failure clips — these come out of sim eval (`06_eval_sim.sh`), which hasn't been run.
- **Don't have:** training-loss curves — only the terminal val loss was saved (`--num_checkpoints 1`).

## 18. Video storage (local / YouTube / Drive)

**Needs your input.** No video assets exist yet, so we can defer.

## 19. Limitations / confounders / incomplete experiments

Things the report should explicitly acknowledge:

1. **Closed-loop eval not run** — Q9 results are open-loop val MSE only. The
   brief identifies this as a known proxy that can disagree with success rate.
2. **Single seed** — no confidence intervals, no error bars. Differences as
   small as the 3–10% spreads in Q9 are within plausible seed variance for
   policies trained on a few thousand sequences.
3. **Single budget (40K samples)** — the policy-size effect could look very
   different at 4K or 400K samples. The original sweep design called for
   multiple budgets; we ran one.
4. **Depth-only scaling** — width, head count, and head dim are held fixed
   (Q5). Results don't generalize to width scaling.
5. **Frozen backbone** — observed differences may be VLM-bottlenecked. A
   parallel sweep with the backbone unfrozen (or LoRA-adapted) would say
   whether the policy head is the bottleneck.
6. **Small tasks** — PushBox has 2,111 sequences, so 40K samples = ~19 epochs;
   PickAndPlaceBox at 5,175 sequences = ~7.7 epochs. PushBox is at higher
   overfitting risk; val-loss split is also noisier (3 val shards).
7. **Mid-experiment config changes** — WARMUP was wrong (1000 vs. ≈312 total
   steps) earlier in this session; dataloader knobs were retuned for
   stability. If earlier runs are in the reported set, they're not directly
   comparable.
8. **Saved-checkpoint provenance** — VLM backbone is re-saved in every
   checkpoint, doubling disk footprint. Not a correctness issue, but
   complicates checkpoint sharing.
9. **NFS-backed data** — variable training throughput depending on neighbor
   load on the shared box. Doesn't affect final model, does affect wall-clock
   numbers in Q8.

## 20. Papers / related projects to cite

**Partial — confirm exact citations with your coauthor.**
- TRI-ML VLA Foundry (the framework — see Q6)
- TRI-ML Large Behavior Models (the data + sim — original LBM paper)
- Diffusion Policy (Chi et al., 2023) — for the action-prediction backbone
- Foundry-VLA-1.7B-full / Foundry-VLM-1.3B-200M release write-ups (if any)
- OpenVLA / Pythia if the 12.6% comparison stays in the report (Q10)

## 21. Visual identity (personal-blog vs polished)

**Needs your input.** Stylistic.

## 22. Light only vs light/dark themes

**Needs your input.** Stylistic.

## 23. Code / checkpoints / datasets / slides / contact public?

**Needs your input.** Some of this is decidable from the repo state:
- Code: this fork is currently local; making it public requires a deliberate `git remote` push.
- Checkpoints: ~3 GB each × 6 → too large for a typical GitHub release; would need HF Hub or similar.
- Dataset: already public at the TRI-ML S3 bucket (Q6).
- Slides / contact: don't know — your call.

## 24. Content beyond the current article

**Skip — don't know what "the current article" is.** If there's a draft post
or repo elsewhere I should diff against, point me at it.
