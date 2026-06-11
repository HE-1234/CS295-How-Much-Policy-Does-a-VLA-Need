#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SIMPLER_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
LAPA_DIR="$( cd -- "$( dirname -- "$SIMPLER_DIR" )" &> /dev/null && pwd )"
cd "$SIMPLER_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$LAPA_DIR:$SIMPLER_DIR"

MODE="${1:-smoke}"
GPU_ID="${GPU_ID:-0}"
CKPT_PATH="${CKPT_PATH:-../adapter_distill/checkpoints/pythia160m_simpler/best.pt}"
FROZEN_LAPA_PATH="${FROZEN_LAPA_PATH:-../adapter_distill/artifacts/lapa_frozen.npz}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-../data/simpler.csv}"
VOCAB_FILE="${VOCAB_FILE:-../lapa_checkpoints/tokenizer.model}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-../lapa_checkpoints/vqgan}"
LOGGING_DIR="${LOGGING_DIR:-./results_lapa_adapter}"

if [[ "$MODE" == "full" ]]; then
  OBJ_START=0
  OBJ_END=24
else
  OBJ_START=0
  OBJ_END=1
fi

run_task() {
  local env_name="$1"
  local scene_name="$2"
  local robot="$3"
  local rgb_overlay_path="$4"
  local robot_init_x="$5"
  local robot_init_y="$6"
  local max_episode_steps="$7"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python simpler_env/main_inference_lapa.py \
    --policy-model lapa-adapter \
    --ckpt-path "$CKPT_PATH" \
    --adapter-frozen-lapa "$FROZEN_LAPA_PATH" \
    --robot "$robot" \
    --policy-setup widowx_bridge \
    --action-scale-file "$ACTION_SCALE_FILE" \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps "$max_episode_steps" \
    --env-name "$env_name" \
    --scene-name "$scene_name" \
    --rgb-overlay-path "$rgb_overlay_path" \
    --robot-init-x-range "$robot_init_x" "$robot_init_x" 1 \
    --robot-init-y-range "$robot_init_y" "$robot_init_y" 1 \
    --obj-variation-mode episode \
    --obj-episode-range "$OBJ_START" "$OBJ_END" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --vocab-file "$VOCAB_FILE" \
    --vqgan-checkpoint "$VQGAN_CHECKPOINT" \
    --logging-dir "$LOGGING_DIR"
}

run_task \
  StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
  bridge_table_1_v1 \
  widowx \
  ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  0.147 \
  0.028 \
  60

if [[ "$MODE" == "full" ]]; then
  run_task \
    PutCarrotOnPlateInScene-v0 \
    bridge_table_1_v1 \
    widowx \
    ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    0.147 \
    0.028 \
    60

  run_task \
    PutSpoonOnTableClothInScene-v0 \
    bridge_table_1_v1 \
    widowx \
    ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    0.147 \
    0.028 \
    60

  run_task \
    PutEggplantInBasketScene-v0 \
    bridge_table_1_v2 \
    widowx_sink_camera_setup \
    ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
    0.127 \
    0.06 \
    120
fi
