# Figure and table mapping

Which script produces each figure and table in the paper, from which data, and where the rendered outputs are stored.

## Figures

| Figure | Content | Output File | Produced by | Source data |
|---|---|---|---|---|
| 1 | Dual-layer framework schematic | `Figure_1.pdf` / `.png` | `framework_fig.tex` (standalone TikZ; `pdflatex framework_fig.tex`) | none (schematic, no simulation data) |
| 2 | Monte Carlo engine verification against the closed-form benchmark | `Figure_2_verification.pdf` / `.png` | `run_experiments.py::verify_engine()` | `experiment_outputs/analytic_benchmark_curves.csv`, `analytic_benchmark.csv` |
| 3 | Scenario setup: heterogeneous swarm topology and spiral formation expansion | `Figure_3.png` / `.pdf` | static illustration of the configuration in `parameters.md` | none (schematic, no simulation data) |
| 4 | Scenario S1, spatial degradation only | `Figure_4.pdf` / `.png` | `plot_results.py::plot_s1()` | `results/S1_Spatial.csv` |
| 5 | Scenario S2, temporal degradation only | `Figure_5.pdf` / `.png` | `plot_results.py::plot_s2()` | `results/S2_Temporal.csv` |
| 6 | Scenario S3, joint spatio-temporal with copula dependence | `Figure_6.pdf` / `.png` | `plot_results.py::plot_s3()` | `results/S3_Joint_*.csv` |
| 7 | Scenario S3 baseline comparison | `Figure_7.pdf` / `.png` | `plot_results.py::plot_s3_baselines()` | `results/S3_Joint_T.csv` |
| 8 | Dependence-strength sweep over Kendall's tau | `Figure_8_correlation_sweep.pdf` / `.png` | `run_experiments.py::make_correlation_figure()` | `experiment_outputs/correlation_sweep_summary.csv`, `correlation_sweep_simultaneous_bands.csv` |
| 9 | Capacity-threshold sweep with paired simultaneous bands | `Figure_9_threshold_sensitivity.pdf` / `.png` | `run_experiments.py::make_figures()` | `experiment_outputs/threshold_sweep_summary.csv`, `threshold_sweep_simultaneous_bands.csv` |
| 10 | Phase-dependent dependence: the tau_early x tau_late map, and its sensitivity to the phase boundary | `Figure_10_time_varying_dependence.pdf` / `.png` | `run_experiments.py::make_dependence_figure()` | `experiment_outputs/phase_sweep_summary.csv`, `phase_boundary_asymmetry.csv` |

## Tables

| Table | Content | Produced by | Source data |
|---|---|---|---|
| 2 | RMST, confidence intervals and horizon survival across degradation regimes | `run_experiments.py::run_main_scenarios()` | `experiment_outputs/main_scenario_summary.csv`, `main_scenario_runs.csv.gz`, `main_binding_constraints.csv` |
| 3 | Paired edge-model sensitivity and binding-constraint shares | `run_experiments.py::run_edge_sensitivity()` | `experiment_outputs/edge_model_sensitivity_summary.csv`, `edge_model_paired_differences.csv`, `edge_model_binding_constraints.csv` |
| 4 | Design zones by normalized capacity threshold | `run_experiments.py::design_zones()` | `experiment_outputs/design_zones.csv`, `threshold_crossover.csv` |
| A.1 | AMOVFLY energy-model calibration | `amovfly_regression.py` | flight-log regression, see Appendix A |

## Supporting diagnostics

These are reported in the text rather than as numbered floats.

| Quantity | Produced by | Output |
|---|---|---|
| Copula sampler validation (Kolmogorov-Smirnov, Kendall's tau), across ten seeds | `run_experiments.py::validate_copulas()` | `experiment_outputs/copula_sampler_validation.csv`, `copula_sampler_validation_by_seed.csv` |
| Homothety residual and SIR scale invariance of the prescribed trajectory | `run_experiments.py::validate_geometry()` | `experiment_outputs/geometry_scale_invariance.csv` |
| Marginal exactness of the phase-switching construction: recovered rate against nominal over 200,000 draws | `run_experiments.py::validate_phase_marginals()` | `experiment_outputs/phase_marginal_validation.csv` |
| Phase-boundary sensitivity of the early-versus-late asymmetry, at t_1 = 20, 40 and 80 | `run_experiments.py::run_phase_boundary_check()` | `experiment_outputs/phase_boundary_summary.csv`, `phase_boundary_asymmetry.csv`, `phase_boundary_runs.csv.gz` |
| Run configuration: replications, horizon, threshold grid, copulas, edge models, seed scheme | `run_experiments.py::main()` | `experiment_outputs/experiment_metadata.json` |

## Note on two unused plotting functions

`plot_results.py` also defines `plot_mttf()` and `plot_sensitivity()`. Neither
corresponds to a figure in the paper: mean time to failure is not reported
(RMST(T) is used instead, because the mission horizon is finite and a
substantial fraction of replications survive it), and the dependence sweep is
produced by `run_experiments.py` so that it carries paired simultaneous bands.

## Checking the paper against these outputs

`verify_claims.py` asserts every number quoted in the revised sections of the
manuscript against the output file it came from, and fails with the mismatched
values if any has drifted. It also checks the verbal claims that accompany them
— that a stated interval really excludes zero, that a stated ordering really
holds.

```bash
python verify_claims.py
```

## Regenerating everything

```bash
python run_experiments.py    # Figures 2, 8, 9, 10 and Tables 2-4
python run_scenarios.py      # scenario curves into results/
python plot_results.py       # Figures 4-7
```

Individual stages, when the full run is not needed:

```bash
python run_experiments.py --verify-only      # closed-form engine verification (Figure 2)
python run_experiments.py --edge-only        # edge-model sensitivity (Table 3)
python run_experiments.py --dependence-only  # phase sweep and its marginal check (Figure 10a)
python run_experiments.py --boundary-only    # phase-boundary sensitivity (Figure 10b)
python run_experiments.py --figures-only     # redraw from stored outputs
python run_experiments.py --smoke            # whole suite at 100 replications
```

There is no flag for the environmental frailty arm, because no figure or table
comes from it. The `frailty*` functions in `run_experiments.py` implement the
formulation of Eq. (22) and are unreachable from `main()`; see the block comment
above `frailty_lifetimes()` and the README for why the arm is reported as a
formulation rather than an evaluated result.

The pinned environment used to produce the outputs listed above is in
`requirements-lock.txt`.
