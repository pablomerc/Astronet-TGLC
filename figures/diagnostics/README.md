# Ad-hoc diagnostics

One-off checks kept for reference. See `../before_after_folded/` for the systematic before/after set.

## TIC 364302118 — original training baseline vs reprocessed, and a stale period

This target is a training-set **EB** whose original file was `mk_hlsp_qlp_..s0012..` — a **single-sector
S12** light curve (~27 d, 903 good points), labelled with **P = 0.672809 d**. The reprocessed curve now
spans S1→S98 (~2720 d, ~242k points).

### `s12_vs_full_364302118.png`
Original training input (**S12 only**, green) vs the full reprocessed curve (≤S103). Folded on the
catalog period 0.6728 d, S12 folds cleanly (one eclipse) but the **full curve is a smeared mess** — the
eclipses no longer line up over the long baseline.

### `period_fix_364302118.png`
Global view of the full curve folded on the catalog **0.6728 d (half period, left — chaotic)** vs the
**BLS-refined 1.34534 d (right — clean primary + secondary eclipse)**.

**Finding:** 0.6728 d is the aliased **half-period** (the 27-day S12 baseline couldn't separate primary
from secondary). Full-baseline BLS gives **P ≈ 1.34534 d ≈ 2×**, with ~10× the BLS power.

**Implication for retraining:** the reprocessed set carries the *original* catalog periods. Folding a
long-baseline reprocessed curve on a stale/aliased period yields garbage phase-folded views (as above),
which would hurt the model. Ephemerides should be **re-derived/refined** (period + 0.5×/2× harmonics) on
the reprocessed curves — or at least targets that fold poorly should be flagged — before generating
TFRecords. Generated with the same env as `../../diagnose_before_after.py`
(`daniel_env_cloned_v2`); BLS via `astropy.timeseries.BoxLeastSquares`.
