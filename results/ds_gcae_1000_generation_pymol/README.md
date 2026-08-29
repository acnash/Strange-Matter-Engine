# DS-GCAE 1,000-generation PyMOL trajectories

This visual set contains ten validation molecule-CYP trajectories, two from each transition-rule family in DS-GCAE v1. Every trajectory was extended to 1,000 generations from a frozen seed-1701 multiscale checkpoint. No optimisation, ridge fitting, parameter update, or blinded-test selection occurred.

## Open and play

1. Start PyMOL.
2. Choose **File > Run Script**.
3. Select `load_ds_gcae_10_trajectories.pml` from this directory.
4. The first trajectory is enabled automatically.
5. Playback starts automatically and plays all 1,001 frames once at 0.05 seconds per frame.
6. Enter `gca_play 0.05, 1` to replay it with one command. Enter `gca_stop` to stop. Use `gca_state 500`, `gca_next`, or `gca_previous` for manual inspection.

Enable one `traj_*` object at a time in the PyMOL object panel. Atom colours encode the within-trajectory magnitude of the learned dynamical state, robustly scaled between its first and ninety-ninth percentiles. Cyan indicates lower magnitude and magenta indicates higher magnitude. Coordinates are display-only RDKit conformers reconstructed from SMILES; the graph CA used molecular connectivity and chemical features rather than 3D coordinates.

The lossless atom-by-channel trajectories and display arrays are retained in `display_values`. Selection metrics and provenance are recorded in `trajectory_manifest.csv`.
