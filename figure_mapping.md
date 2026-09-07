# Figure and table mapping

Which script produces each figure and table in the paper, from which data, and where the rendered outputs are stored.

## Figures

| Figure | Content | Output File in `figures/` | Produced by | Source data |
|---|---|---|---|---|
| 1 | Dual-layer framework schematic | `Figure_1.pdf` / `.png` | `framework_fig.tex` (standalone TikZ; `pdflatex framework_fig.tex`) | none (schematic, no simulation data) |
| 2 | Monte Carlo engine verification against the closed-form benchmark | `Figure_2_verification.pdf` / `.png` | `run_experiments.py::verify_engine()` | `experiment_outputs/analytic_benchmark_curves.csv`, `analytic_benchmark.csv` |
| 3 | Scenario setup: heterogeneous swarm topology and spiral formation expansion | `Figure_3.png` / `.pdf` | static illustration of the configuration in `parameters.md` | none (schematic, no simulation data) |
| 4 | Scenario S1, spatial degradation only | `Figure_4_S1_spatial_results.pdf` / `.png` (alias: `fig3_...`) | `plot_results.py::plot_s1()` | `results/S1_Spatial.csv` |
| 5 | Scenario S2, temporal degradation only | `Figure_5_S2_temporal_results.pdf` / `.png` (alias: `fig4_...`) | `plot_results.py::plot_s2()` | `results/S2_Temporal.csv` |
| 6 | Scenario S3, joint spatio-temporal with copula dependence | `Figure_6_S3_joint_results.pdf` / `.png` (alias: `fig5_...`) | `plot_results.py::plot_s3()` | `results/S3_Joint_*.csv` |
| 7 | Scenario S3 baseline comparison | `Figure_7_S3_baselines_comparison.pdf` / `.png` (alias: `fig5b_...`) | `plot_results.py::plot_s3_baselines()` | `results/S3_Joint_T.csv` |
| 8 | Dependence-strength sweep over Kendall's tau | `Figure_8_correlation_sweep.pdf` / `.png` | `run_experiments.py::make_correlation_figure()` | `experiment_outputs/correlation_sweep_summary.csv`, `correlation_sweep_simultaneous_bands.csv` |
| 9 | Capacity-threshold sweep with paired simultaneous bands | `Figure_9_threshold_sensitivity.pdf` / `.png` | `run_experiments.py::make_figures()` | `experiment_outputs/threshold_sweep_summary.csv`, `threshold_sweep_simultaneous_bands.csv` |

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
| Copula sampler validation (Kolmogorov-Smirnov, Kendall's tau) | `run_experiments.py::validate_copulas()` | `experiment_outputs/copula_sampler_validation.csv` |
| Homothety residual and SIR scale invariance of the prescribed trajectory | `run_experiments.py::validate_geometry()` | `experiment_outputs/geometry_scale_invariance.csv` |
| Run configuration: replications, horizon, threshold grid, copulas, edge models, seed scheme | `run_experiments.py::main()` | `experiment_outputs/experiment_metadata.json` |

## Note on two unused plotting functions

`plot_results.py` also defines `plot_mttf()` and `plot_sensitivity()`. Neither
corresponds to a figure in the paper: mean time to failure is not reported
(RMST(T) is used instead, because the mission horizon is finite and a
substantial fraction of replications survive it), and the dependence sweep is
produced by `run_experiments.py` so that it carries paired simultaneous bands.

## Regenerating everything

```bash
python run_experiments.py    # Figures 2, 8, 9 and Tables 2-4
python run_scenarios.py      # scenario curves into results/
python plot_results.py       # Figures 4-7 into figures/
```
