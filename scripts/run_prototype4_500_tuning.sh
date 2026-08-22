#!/bin/zsh
set -euo pipefail

python_bin=/opt/anaconda3/envs/kcc2/bin/python
runner=scripts/run_graph_ca_visual_prototype.py

configs=(
  "01 0.0005 0.0015 0.0010 1.0"
  "02 0.0010 0.0030 0.0010 1.0"
  "03 0.0020 0.0030 0.0010 1.0"
  "04 0.0010 0.0030 0.0003 1.0"
  "05 0.0010 0.0030 0.0030 1.0"
  "06 0.0010 0.0030 0.0010 2.0"
)

for config in $configs; do
  read id ca_lr readout_lr ridge clip <<< "$config"
  echo "TUNING_CONFIG $id ca_lr=$ca_lr readout_lr=$readout_lr ridge=$ridge clip=$clip"
  SME_CA_RULE=inertial_reaction_diffusion \
  SME_GENERATIONS=125 \
  SME_RUN_NAME="graph_ca_inertial_500_tuning_$id" \
  SME_CA_LR="$ca_lr" \
  SME_READOUT_LR="$readout_lr" \
  SME_RIDGE="$ridge" \
  SME_GRAD_CLIP="$clip" \
  SME_MAX_EPOCHS=8 \
  SME_PATIENCE=4 \
  SME_MIN_DELTA=0.003 \
  SME_TUNING_ONLY=1 \
  "$python_bin" "$runner" train
done
