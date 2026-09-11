# Graph-CA refinement score ledger

| Date | Method | Sealed point MA-ST-RAE | RMSE, pIC50 | Status |
|---|---|---:|---:|---|
| 2 September 2026 | CIA-EA-CV-CYP-GCA | 0.748490 | 0.847738 | External reference model for the current refinement lineage |
| 2 September 2026 | PP-CIA-EA-CV-CYP-GCA | 0.760855 | 0.858781 | Experimental result |
| 3 September 2026 | TA-CIA-EA-CV-CYP-GCA | 0.755021 | 0.850082 | Experimental result |
| 3 September 2026 | FG-CIA-EA-CV-CYP-GCA | 0.751384 | 0.848686 | Experimental result |
| 4 September 2026 | PC-CIA-EA-CV-CYP-GCA | 0.759449 | 0.852318 | Experimental result |
| 5 September 2026 | ISA-CIA-EA-CV-CYP-GCA | 0.757657 | 0.853247 | Experimental result |
| 6 September 2026 | TDO-CIA-EA-CV-CYP-GCA | 0.755675 | 0.851798 | Experimental result |
| 6 September 2026 | AB-CIA-EA-CV-CYP-GCA | 0.750740 | 0.849659 | Experimental result |
| 6 September 2026 | BMR-CIA-EA-CV-CYP-GCA | 0.755994 | 0.852721 | Experimental result |
| 7 September 2026 | DNC-CIA-EA-CV-CYP-GCA | 1.223519 | 1.293131 | Rejected; degree softening degraded generalisation |
| 7 September 2026 | SBR-CIA-EA-CV-CYP-GCA | 0.755412 | 0.852409 | Experimental result; support/query rebalancing did not improve leader |
| 8 September 2026 | BTS-CIA-EA-CV-CYP-GCA | 0.755150 | 0.849111 | Experimental result; typed-bond temperature scaling did not improve leader |
| 8 September 2026 | HWR-CIA-EA-CV-CYP-GCA | 0.751954 | 0.851544 | Experimental result; wider cellular state did not improve leader |
| 8 September 2026 | GDR-CIA-EA-CV-CYP-GCA | 0.752414 | 0.848947 | Experimental result; longer recurrent evolution did not improve leader |
| 9 September 2026 | USR-CIA-EA-CV-CYP-GCA | **0.748039** | **0.845606** | Current sealed-validation leader; blind evaluation did not confirm the incremental internal improvement |
| 11 September 2026 | ISR-CIA-EA-CV-CYP-GCA | 0.752851 | 0.851661 | Experimental result; initial-state amplitude refinement did not improve the sealed leader |

All entries use the same reserved scaffold holdout. Model and hyperparameter
selection occur within the fitting pool; the reserved holdout is excluded from
selection. Blind challenge labels are unavailable to the campaigns.

## Official blind evaluation

| Model | Recorded rank | MA-ST-RAE | Macro MAE | Macro R-squared | Macro Spearman rho | Macro Kendall tau | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| EA-CV-CYP-GCA | 99 of 111 | **1.0071** | **1.0778** | **-0.0715** | 0.5345 | 0.3750 | Lowest official primary error among the recorded submissions |
| CIA-EA-CV-CYP-GCA | 102 of 112 | 1.0075 | 1.0781 | -0.0716 | **0.5379** | **0.3788** | External reference for the current credible-interval refinement lineage; strongest rank correlations |
| USR-CIA-EA-CV-CYP-GCA | 123 of 137 | 1.0092 | 1.0789 | -0.0744 | 0.5345 | 0.3762 | Latest submission; sealed improvement did not transfer to the blind set |

Ranks are time-specific snapshots because leaderboard membership changed between
submissions. Official blind metrics are retained for external reporting and are
not used retroactively to redefine the sealed model-selection result.
