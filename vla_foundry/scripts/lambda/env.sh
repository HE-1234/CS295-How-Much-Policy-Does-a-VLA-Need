# Shared config for running closed-loop sim eval on a rented Lambda Cloud box.
# Every other script in this directory `source`s this file. Edit values here.
#
# The checkpoints are pulled from PUBLIC HuggingFace repos (namespace below),
# so nothing here references pyxis. The frozen VLM backbone
# (hf://TRI-ML/Foundry-VLM-1.3B-200M) is auto-downloaded by the policy server.

# --- HuggingFace namespace where the finetuned checkpoints were pushed ---
export HF_NAMESPACE="Eric-H"

# --- The experiment grid: 3 tasks x 3 policy sizes = 9 checkpoints ---
# Task names MUST be the PascalCase names the LBM sim harness knows.
export TASKS=(PushBox PickAndPlaceBox PutOrangeOnSaucer)
export SIZES=(transformer_77m transformer_205m transformer_410m)
export BUDGET=40000   # matches the "...-budget-40000" suffix in the repo names

# --- Eval settings ---
export NUM_EPISODES="0:200"     # full run = paper default. Smoke test overrides to 0:2.
export MAX_SAMPLE_SIZE=200      # stopping boundary for the stat test; decide BEFORE seeing results.
export NUM_GPUS=1               # bump to the number of GPUs you actually rented.
export OUTPUT_DIR="rollouts"    # all results land here.
export DOCKER_IMAGE="toyotaresearch/lbm-eval-oss:vla-foundry"

# --- Helper: map (task, size) -> the public HF repo id ---
# Mirrors the push naming: PushBox_transformer_77m_budget-40000 -> pushbox-transformer-77m-budget-40000
hf_repo_for() {
  local task="$1" size="$2"
  echo "${HF_NAMESPACE}/$(echo "${task}_${size}_budget-${BUDGET}" | tr '[:upper:]_' '[:lower:]-')"
}
