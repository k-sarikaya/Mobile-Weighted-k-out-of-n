# Figure to Code Mapping Manifest

This document maps the figures presented in the revised RESS manuscript to the corresponding plotting functions in the simulation script.

| Figure Number | Filename | Description | Generating Code Function | Source Data |
|---|---|---|---|---|
| **Figure 1** | `fig1_framework_schematic.pdf` | Conceptual block diagram of the dual-layer MW-$k$/$n$:G reliability model. | TikZ source in `framework_fig.tex` | N/A |
| **Figure 3** | `fig3_S1_spatial_results.pdf` | Scenario S1 (Spatial-only) mission reliability and layer status. | `plot_results.py::plot_s1()` | `results/S1_Spatial.csv` |
| **Figure 4** | `fig4_S2_temporal_results.pdf` | Scenario S2 (Temporal-only) mission reliability and layer status. | `plot_results.py::plot_s2()` | `results/S2_Temporal.csv` |
| **Figure 5** | `fig5_S3_joint_results.pdf` | Scenario S3 (Joint Spatio-Temporal) and Copula comparison. | `plot_results.py::plot_s3()` | `results/S3_Joint_*.csv` |
| **Figure 6** | `fig6_mttf_comparison.pdf` | Mean Time to Failure (MTTF) across baseline and joint scenarios. | `plot_results.py::plot_mttf()` | `results/*.csv` |
| **Figure 7** | `fig7_sensitivity_analysis.pdf` | Continuous sensitivity sweep of MTTF against correlation $\rho$ and tail dependence $\nu$. | `plot_results.py::plot_sensitivity()` | `results/sensitivity_analysis.csv` |

## Execution
To generate all figures, run:
```bash
python plot_results.py
```
Figures will be written directly to the `RESS_v2/figures/` directory as high-resolution PDFs.
