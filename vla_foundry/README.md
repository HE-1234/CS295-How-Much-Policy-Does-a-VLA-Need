# Policy-Size Ablation on LBM Sim Tasks

Fork of [VLA Foundry](https://github.com/TRI-ML/vla-foundry) used to run a
single-task policy-size ablation: three diffusion-policy transformer sizes
(`transformer_77m / 205m / 410m`) trained on top of a **frozen**
`Foundry-VLM-1.3B-200M` backbone, on two LBM bimanual pick-and-place tasks
(`PickAndPlaceBox` and `PushBox`).

This README is a landing page. The actual experiment lives in two places:

- **`RUNBOOK.md`** — narrative: what we set up, what decisions we made, what
  we changed during the run, what's still pending. Read this first.
- **`scripts/sweep/`** — the eight numbered shell scripts that execute the
  sweep. See `scripts/sweep/README.md` for the script-by-script playbook.

## Setup pointer

```bash
uv sync
source scripts/sweep/env.sh
```

That's enough to read the doc paths. To actually re-run anything, follow the
order in `scripts/sweep/README.md`.

Everything below the top-level (params, dataloader, model registry, training
loop) is unmodified upstream code — refer to the upstream
[VLA Foundry README](https://github.com/TRI-ML/vla-foundry) for framework
documentation, repo structure, and citations.
