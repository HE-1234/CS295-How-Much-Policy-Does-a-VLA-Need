"""Re-upload sanitized config.yaml to each published checkpoint repo.

The push tool uploaded config.yaml verbatim, so it still carries the absolute
training-machine path in data.dataset_statistics. On any other machine that
path doesn't exist and the policy server crashes at startup
(normalization_params.init_shared_attributes -> preprocessing_config.yaml).

Fix: null the stale path (data.dataset_statistics = [None]) so the loader takes
the "published checkpoint" early-return. Numeric stats are loaded separately by
RoboticsProcessor from the repo's stats.json, so this is safe.

Only config.yaml is re-uploaded (tiny) -- the .pt checkpoints are untouched.
"""

import os
import tempfile

import yaml
from huggingface_hub import HfApi

NAMESPACE = "Eric-H"
BUDGET = 40000
TASKS = ["PushBox", "PickAndPlaceBox", "PutOrangeOnSaucer"]
SIZES = ["transformer_77m", "transformer_205m", "transformer_410m"]
EXP_ROOT = {
    "PushBox": "experiments/policy_size_pushbox",
    "PickAndPlaceBox": "experiments/policy_size_pickandplace",
    "PutOrangeOnSaucer": "experiments/policy_size_putorangeonsaucer",
}

api = HfApi()

for task in TASKS:
    for size in SIZES:
        run_dir = f"{EXP_ROOT[task]}/runs/{task}_{size}_budget-{BUDGET}/policy_size_{size}"
        cfg_path = os.path.join(run_dir, "config.yaml")
        repo_id = f"{NAMESPACE}/" + f"{task}_{size}_budget-{BUDGET}".lower().replace("_", "-")

        if not os.path.exists(cfg_path):
            print(f"SKIP (no local config): {cfg_path}")
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        before = cfg.get("data", {}).get("dataset_statistics")
        cfg["data"]["dataset_statistics"] = [None]  # sanitized "published" form

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(cfg, tf, sort_keys=False)
            tmp_path = tf.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="config.yaml",
            repo_id=repo_id,
            commit_message="Sanitize dataset_statistics path for portable inference",
        )
        os.remove(tmp_path)
        print(f"OK  {repo_id}\n      was: {before}\n      now: [null]")

print("\nDone. Re-running inference now resolves the new config from HF automatically.")
