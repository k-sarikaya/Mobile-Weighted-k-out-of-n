# Archive

Earlier scripts, kept for provenance. Nothing here is used by the results
reported in the paper.

- `validate_analytical.py`, `validation_results.csv`,
  `fig_validation_convergence.pdf` — an earlier engine-verification script that
  evaluated the capacity constraint alone and reported MTTF. The verification in
  the paper is produced by `run_experiments.py --verify-only`, which drives the
  full engine (capacity, connectivity and delay) and reports RMST.
- `compute_csv_mttf.py` — MTTF summary. The paper reports restricted mean
  survival time RMST(T) instead, because the mission horizon is finite and a
  substantial fraction of replications survive it.
