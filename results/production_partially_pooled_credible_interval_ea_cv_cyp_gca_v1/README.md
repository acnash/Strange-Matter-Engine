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
