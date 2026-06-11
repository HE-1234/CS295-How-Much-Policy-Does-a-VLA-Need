# Running closed-loop sim eval on Lambda Cloud

pyxis has no container runtime, so the LBM sim can't run there. These scripts run
the eval on a rented Lambda box instead. The finetuned checkpoints are pulled from
**public** HuggingFace repos under the `Eric-H` namespace, and the frozen VLM
backbone (`hf://TRI-ML/Foundry-VLM-1.3B-200M`) is auto-downloaded — so nothing
here depends on pyxis once the checkpoints are pushed.

## One-time, on pyxis: publish the checkpoints

```bash
for run in experiments/*/runs/*/policy_size_*; do
  task_size=$(basename "$(dirname "$run")")
  repo="Eric-H/$(echo "$task_size" | tr '[:upper:]_' '[:lower:]-')"
  python -m vla_foundry.hf_hub push "$run" "$repo" --collection ""   # public
done
```

Then commit + push these scripts so they're in the GitHub clone:
```bash
git add scripts/lambda && git commit -m "Add Lambda eval scripts" && git push
```

## On the Lambda box

SSH in with the `.pem`, then work inside `tmux` so a disconnect can't kill the run:

```bash
ssh -i <lambda-key.pem> ubuntu@<lambda-ip>
tmux new -s eval                 # Ctrl-b then d to detach; `tmux attach -t eval` to return

git clone https://github.com/HE-1234/vla_foundry.git
cd vla_foundry

bash scripts/lambda/00_preflight.sh      # MUST pass before spending GPU-hours
bash scripts/lambda/01_setup.sh          # uv sync + docker pull
bash scripts/lambda/02_smoke_test.sh     # 1 checkpoint, 2 episodes - proves the pipeline
bash scripts/lambda/03_run_eval.sh       # full 3x3 sweep (hours; keep in tmux)
bash scripts/lambda/04_sync_results.sh   # tar results to pull off before terminating
```

## Tuning (edit `scripts/lambda/env.sh`)

| Var | Meaning |
|-----|---------|
| `NUM_GPUS` | set to the number of GPUs you rented (one policy server + sim each) |
| `NUM_EPISODES` | `0:200` = paper default; lower for a cheaper partial run |
| `TASKS` / `SIZES` | the 3x3 grid; trim to run a subset |
| `HF_NAMESPACE` | where the checkpoints live (`Eric-H`) |

## Don't forget
- **Terminate the Lambda instance when done** — it bills by the hour.
- Pull results (`04_sync_results.sh`) **before** terminating; the disk is ephemeral.
- Analyze locally (no GPU/Docker needed):
  `uv run --group dashboard python vla_foundry/eval/results_explorer.py rollouts/`
