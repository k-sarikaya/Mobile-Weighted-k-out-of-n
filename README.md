# Mobile Weighted $k$-out-of-$n$:G Swarm Reliability Simulation

This repository contains the official implementation of the simulation experiments, sensitivity analysis, and figure plotting code for the paper:

> **Mobile Weighted $k$-out-of-$n$:G Systems: A Dual-Layer Spatio-Temporal Reliability Framework for Mobile Swarms**

## Repository Structure

- `run_scenarios.py`: Core simulation script that executes the Monte Carlo replications for Scenarios S1, S2, and S3, computes system reliability metrics over the mission horizon, and performs the copula sensitivity sweeps.
- `plot_results.py`: Script that generates the high-resolution figures (PDFs) from the simulation results for inclusion in the manuscript.
- `parameters.md`: Complete manifest of all physical and environmental parameters (agent weights, ranges, exponential failure rates, copula configurations, seeds).
- `figure_mapping.md`: Mapping file linking specific manuscript figures to their corresponding plotting functions in `plot_results.py`.
- `results/`: Directory containing the pre-computed Monte Carlo simulation output in CSV format.

## Prerequisites

The simulation and plotting scripts require Python 3.8+ and the following standard scientific computing libraries:
- `numpy`
- `scipy`
- `pandas`
- `networkx`
- `matplotlib`
- `seaborn`

You can install the dependencies via pip:
```bash
pip install numpy scipy pandas networkx matplotlib seaborn
```

## Running the Simulation

To re-run the entire suite of Monte Carlo simulations (5,000 replications per scenario, plus the continuous sensitivity sweep over $\rho \in [0, 0.8]$ and $\nu \in [3, 30]$), run:
```bash
python run_scenarios.py
```
*Note: Due to the 5,000-replication horizon and multi-dimensional parameter sweep, the full simulation can take several minutes depending on your system's hardware.*

## Generating the Figures

Once the simulation completes and outputs the CSV files into the `results/` folder, you can generate all figures (saved as PDFs in a parent `figures/` folder) by running:
```bash
python plot_results.py
```
The figures will match the ones published in the manuscript.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
