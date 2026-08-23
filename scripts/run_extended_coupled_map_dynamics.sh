#!/bin/sh
set -eu

export SME_DEVICE="${SME_DEVICE:-auto}"
export SME_CHECKPOINT="${SME_CHECKPOINT:-results/production_coupled_map_enhanced_v3/runs/final_model/model.pt}"
export SME_SCREENING_CSV="${SME_SCREENING_CSV:-results/production_coupled_map_enhanced_v3/runs/final_model/validation_dynamics.csv}"
export SME_EXTENDED_OUTPUT="${SME_EXTENDED_OUTPUT:-results/coupled_map_5000_generation_dynamics}"
export SME_EXTENDED_GENERATIONS="${SME_EXTENDED_GENERATIONS:-5000}"
export SME_EXTENDED_CANDIDATES="${SME_EXTENDED_CANDIDATES:-100}"
export SME_EXTENDED_BURN_IN="${SME_EXTENDED_BURN_IN:-1000}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

python scripts/run_graph_ca_visual_prototype.py extended-dynamics
