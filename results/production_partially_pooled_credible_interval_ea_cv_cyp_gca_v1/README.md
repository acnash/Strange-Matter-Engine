# PP-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign begins from the completed CIA-EA-CV-CYP-GCA specialists and
tests controlled cross-isoform supervision during recurrent Graph-CA training.
The active CYP retains full loss weight. Observations from the other three CYP
isoforms contribute a combined auxiliary weight selected from 0.00, 0.05,
0.15, and 0.30. This allows shared chemical information to stabilise the
learned local transition dynamics while endpoint-specific scaffold validation
continues to determine selection.

The molecular bonded graph, atom-as-cell representation, explicit recurrent
generations, nonlinear transition rules, complete trajectory fingerprints,
backpropagation through time, and differentiable ridge readout are retained.
The reserved scaffold holdout remains sealed until configuration selection is
complete, and blind labels remain unavailable. The current leader used for
comparison is CIA-EA-CV-CYP-GCA, with point MA-ST-RAE 0.748490 and RMSE
0.847738 pIC50.

## Completed result

The campaign completed on 2 September 2026. Reserved-holdout point MA-ST-RAE
was 0.760855 and RMSE was 0.858781 pIC50. Both diagnostics were weaker than
the CIA-EA-CV-CYP-GCA baseline, so the partially pooled model was retained as
an experimental result and was not promoted as the submission leader. The
bootstrap mean MA-ST-RAE was 0.761493 with a 95% interval from 0.723346 to
0.804152. Endpoint point ST-RAE values were 0.816270 for CYP1A2, 0.731897 for
CYP2C9, 0.946202 for CYP2D6, and 0.549052 for CYP3A4.
