# Frozen model bundles

Each task directory contains `zero_shot.txt` (700-tree GBM), `within_membrane.txt`
(backfitted contrast GBM), `pairwise.txt` (pairwise transfer GBM), and `metadata.json`
(ordered encoding vocabulary, feature names, variance parameters and training hashes).
The frozen point prediction averages the two contrast learners with weight 0.5.
With anchors, there is no zero-shot prior shrinkage. Models were fitted solely to the
corrected 2025 snapshot; pure-solvent training has 1,934 rows/121 papers and solution
training has 3,190 rows/192 papers. The other domain supplies auxiliary contrasts.

Targets are log10 permeance (L m^-2 h^-1 bar^-1 before transformation). These models
are for accepted single organic solvent conditions within the represented domain.
They do not predict arbitrary solvent mixtures, field performance, or solute rejection.
Missing metadata can weaken predictions. Measured anchors should use the same
membrane and compatible test conditions. Raw model SD is not calibrated coverage.

Evaluation data, native reload parity and exact file hashes are supplied under reports/.
Deployment fits do not replace the grouped/future/external evaluation results.
