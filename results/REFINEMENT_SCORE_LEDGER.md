# Graph-CA refinement score ledger

| Date | Method | Sealed point MA-ST-RAE | RMSE, pIC50 | Status |
|---|---|---:|---:|---|
| 2 September 2026 | CIA-EA-CV-CYP-GCA | **0.748490** | **0.847738** | Current internal leader |
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

All entries use the same reserved scaffold holdout. Model and hyperparameter
selection occur within the fitting pool; the reserved holdout is excluded from
selection. Blind challenge labels are unavailable to the campaigns.
