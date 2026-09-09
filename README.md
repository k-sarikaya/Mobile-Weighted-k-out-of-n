# Mobile Weighted $k$-out-of-$n$:G Swarm Reliability Simulation

Simulation code, data and figures for:

> **Mobile Weighted $k$-out-of-$n$:G Systems: A Dual-Layer Spatio-Temporal
> Reliability Framework for Mobile Swarms**
> Kadir Sarıkaya, *Reliability Engineering & System Safety*

The framework evaluates mission reliability for a heterogeneous mobile swarm as
the joint satisfaction of three constraints at every mission epoch: weighted
capacity, network connectivity, and end-to-end latency. Two scripts produce
everything reported in the paper.

## Contents

| Path | Purpose |
|---|---|
| `run_experiments.py` | Unified Monte Carlo engine. Produces Tables 2-4 and Figures 2, 8, 9 and 10. |
| `experiment_outputs/` | Its outputs, committed so they can be inspected without re-running. |
| `verify_claims.py` | Re-checks every number quoted in the paper against the output file it came from. |
| `run_scenarios.py` | Per-epoch survival curves for the three degradation scenarios and the three literature baselines. |
| `plot_results.py` | Draws Figures 4-7 from those curves. |
| `results/` | Scenario curve data. |
| `amovfly_regression.py` | AMOVFLY energy-model calibration reported in Appendix A. |
| `parameters.md` | Full parameter manifest: agent weights, failure rates, ranges, copula settings, horizon, seeds. |
| `figure_mapping.md` | Which script produces which figure and table. |

## Quick start

```bash
pip install -r requirements.txt

python run_experiments.py               # full suite, 5000 replications
python run_experiments.py --smoke       # 100 replications, for a quick check
python run_experiments.py --verify-only # engine verification against the closed form

python run_scenarios.py                 # scenario survival curves
python plot_results.py                  # Figures 4-7

python verify_claims.py                 # check the paper's numbers against the outputs
```

`run_experiments.py` writes to `experiment_outputs/`. The largest run-level
files are stored gzipped (`*.csv.gz`) and are read transparently; a fresh run
writes them uncompressed. The directory contains no file that the script does
not produce.

Individual stages can be rerun in isolation: `--edge-only` for the edge-model
sensitivity, `--dependence-only` for the phase sweep and its marginal check,
`--boundary-only` for the phase-boundary sensitivity, and `--figures-only` to
redraw from stored outputs.

## The unified engine

`run_experiments.py` generates one trajectory per replication and evaluates that
same trajectory at every capacity threshold ($K = 2 \dots 23$) and under every
edge model. Comparisons across thresholds, copulas and edge models are therefore
*paired*, which is what makes the reported paired max-$t$ simultaneous confidence
bands valid: a difference between two settings is measured within a replication
rather than between independent experiments.

Random streams are derived as
`numpy.random.SeedSequence([20260822, run_id]).spawn(4)`, giving each replication
four independent substreams — mobility, copula sampling, edge generation and
latency. Replication *r* therefore sees the same mobility realisation under every
threshold, copula and edge model. Re-running reproduces the committed outputs
given the same NumPy version.

### What it computes

| Output | Content |
|---|---|
| `main_scenario_runs.csv.gz`, `main_scenario_summary.csv`, `main_binding_constraints.csv` | Per-replication RMST, horizon survival, first-passage time and first binding constraint for the three scenarios and five copulas (Table 2) |
| `threshold_sweep_runs.csv.gz`, `threshold_sweep_summary.csv`, `threshold_sweep_simultaneous_bands.csv` | Capacity-threshold sweep with paired simultaneous bands (Figure 9) |
| `threshold_crossover.csv`, `design_zones.csv` | Threshold at which the sign of the dependence effect changes, and the resulting design zones (Table 4) |
| `correlation_sweep_runs.csv.gz`, `correlation_sweep_summary.csv`, `correlation_sweep_simultaneous_bands.csv` | Dependence-strength sweep over Kendall's $\tau$ for four copula families, plus Student-$t$ across four degrees of freedom (Figure 8) |
| `phase_sweep_runs.csv.gz`, `phase_sweep_summary.csv` | Phase-switching copula over the $\tau_{\text{early}} \times \tau_{\text{late}}$ grid, with paired intervals (Figure 10a) |
| `phase_boundary_runs.csv.gz`, `phase_boundary_summary.csv`, `phase_boundary_asymmetry.csv` | The same corners re-evaluated at three phase boundaries, testing whether the early-versus-late asymmetry is a property of the mechanism or of the boundary (Figure 10b) |
| `phase_marginal_validation.csv` | Marginal exactness of the phase switch: recovered rate against nominal over 200,000 draws |
| `edge_model_sensitivity_runs.csv.gz`, `edge_model_sensitivity_summary.csv`, `edge_model_paired_differences.csv`, `edge_model_binding_constraints.csv` | i.i.d. Bernoulli vs. deterministic edge models at matched link probability, paired differences and binding-constraint shares (Table 3) |
| `analytic_benchmark.csv`, `analytic_benchmark_curves.csv` | Engine verification against the closed-form benchmark (Figure 2) |
| `copula_sampler_validation.csv`, `copula_sampler_validation_by_seed.csv` | Kolmogorov-Smirnov and Kendall's $\tau$ diagnostics for the copula samplers, across ten seeds |
| `geometry_scale_invariance.csv` | Homothety residual and SIR scale-invariance check for the prescribed trajectory |
| `experiment_metadata.json` | Replication count, horizon, threshold grid, copulas, edge models, seed scheme |

## Time-varying dependence

Two formulations relax the assumption that the dependence between component
lifetimes is constant over the mission. The **phase-switching copula** partitions
the horizon at a boundary and draws residual lifetimes from a copula with a
different Kendall's $\tau$ on each side; because the exponential marginals are
memoryless, the two exponential factors telescope and the marginal survival
function is preserved exactly, so the dependence changes without perturbing any
marginal. This is the arm evaluated in the paper.

The **environmental frailty** formulation instead derives the dependence from a
shared stress process. It is stated in the paper but not evaluated, because
comparing it against the copula arm at fixed marginals requires the baseline
hazards to be recalibrated numerically for each stress pair, and that
recalibration matches the survival function at the horizon without matching it
throughout. `frailty_lifetimes()` and `calibrate_frailty_baseline()` implement
the construction and its recalibration for anyone wishing to pursue it.

**The frailty code is not used by any reported result.** Every function in
`run_experiments.py` whose name contains `frailty` is unreachable from
`main()`, produces none of the CSVs in `experiment_outputs/`, and is covered by
no assertion in `verify_claims.py`; a block comment above `frailty_lifetimes()`
says so at the source. Deleting it would change no figure, table or number in
the paper. It is kept so the formulation of Eq. (22) is inspectable and so the
calibration failure that keeps the arm out of the results can be reproduced —
`frailty_marginal_diagnostic()` records the $\sigma$ range over which the
marginal error exceeds the tolerance.

## Requirements

Python 3.8+. `pip install -r requirements.txt`
(`run_experiments.py` needs only `numpy`, `scipy`, `pandas` and `matplotlib`;
`run_scenarios.py` and `plot_results.py` additionally use `networkx` and
`seaborn`.)

`requirements.txt` gives minimum supported versions. To reproduce the deposited
numbers exactly, use the pinned environment instead:
`pip install -r requirements-lock.txt` (CPython 3.12.5, win-amd64; numpy 2.1.1,
scipy 1.14.1, pandas 2.2.2, networkx 3.6.1, matplotlib 3.9.2, seaborn 0.13.2).

## License

MIT — see `LICENSE`.
