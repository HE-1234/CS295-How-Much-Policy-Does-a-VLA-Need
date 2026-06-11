# Shared environment for the policy-size sweep on PickAndPlaceBox.
# All other scripts in this directory `source` this file. Edit values here.

# Repo root (absolute) — every script cd's here so they can be run from anywhere.
export REPO_ROOT="/home/jianch2/CS295/vla_foundry"

# Workspace root for the whole experiment (data + runs + rollouts).
export EXP_ROOT="${REPO_ROOT}/experiments/policy_size_pushbox"

# Single-task target.
export TASK="PushBox"

# Three policy size rungs (order matters: full → half → quarter).
# Defined in vla_foundry/config_presets/models/.
export RUNGS=(transformer_410m transformer_205m transformer_77m)

# Training budget per rung (total samples seen).
# 40,000 samples ÷ global_batch_size 128 ≈ 312 optimizer steps
# ≈ ~7.7 epochs over PickAndPlaceBox's 5,175 sequences.
export BUDGET=40000
export GLOBAL_BATCH_SIZE=128
export PER_GPU_BATCH_SIZE=16    # drop if you OOM on the full rung
export LR="5e-5"
export WARMUP="50"

# Inline validation-loss during training is OFF by default — the val
# dataloader's construction at train startup steals ~1.75× train throughput
# even when val itself only runs once.
#
# We instead compute val loss POST-HOC on saved checkpoints via
# scripts/sweep/05_val_loss_post_hoc.sh (no contention with training).
# Set this to 1 to re-enable inline val (not recommended).
export WITH_VAL_LOSS=0

# Used by both inline-val (if WITH_VAL_LOSS=1) and post-hoc val.
export VAL_SAMPLES=2048
export VAL_EVERY_N_CHECKPOINTS=1

# Closed-loop eval settings (Step 5). Only used if Docker is available.
export EVAL_NUM_EPISODES="0:200"
export EVAL_MAX_SAMPLE_SIZE=200
export EVAL_NUM_GPUS=1
export DOCKER_IMAGE="toyotaresearch/lbm-eval-oss:vla-foundry"
