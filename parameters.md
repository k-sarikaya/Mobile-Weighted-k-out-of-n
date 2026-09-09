# Simulation Parameters Manifest

This document details the complete parameters and configurations used in the simulation experiments for the Mobile Weighted $k$-out-of-$n$:G (MW-$k$/$n$:G) reliability framework.

## Swarm Parameters
- **Total Agents ($n$)**: 12
- **Agent Roles and Capability Weights ($w_i$)**:
  - **Type A (3 agents, heavy payload/sensor leaders)**: $w_i = 3.0$ each, $\lambda_i = 0.005$
  - **Type B (5 agents, medium relays)**: $w_i = 2.0$ each, $\lambda_i = 0.008$
  - **Type C (4 agents, light sensors)**: $w_i = 1.0$ each, $\lambda_i = 0.012$
- **Total Swarm Weight**: $3 \times 3.0 + 5 \times 2.0 + 4 \times 1.0 = 23.0$
- **System Operational Threshold ($K$)**: 18

## Network & Connectivity Parameters
- **Communication Range ($R_c$)**: 250 m (Geometric disk model)
- **Maximum Delay Path Length ($H_{\max}$)**: 4 hops
- **Time Step Size ($\Delta t$)**: 1.0
- **Total Mission Horizon ($T$)**: 120 steps

## Node Reliability Parameters
- **Marginal Lifetime Distributions**: Exponential degradation $\tau_i \sim \text{Exp}(\lambda_i)$
- **Failure Rates ($\lambda_i$)**:
  - **Type A**: $\lambda_i = 0.005$ (Mean Time to Failure $MTTF_i = 200.0$ steps)
  - **Type B**: $\lambda_i = 0.008$ (Mean Time to Failure $MTTF_i = 125.0$ steps)
  - **Type C**: $\lambda_i = 0.012$ (Mean Time to Failure $MTTF_i = 83.3$ steps)

## Copula Dependence Configurations
- **Scenarios**:
  - **S1 (Spatial-only)**: Perfect node reliability ($\lambda_i = 0.0$, $p_i(t) = 1.0$), active agent movement.
  - **S2 (Temporal-only)**: Hovering tight formation, exponential component degradation under independence.
  - **S3 (Joint)**: Combining spatial movement and component degradation under three dependence structures:
    - **Independent Copula**: $\rho = 0.0$
    - **Gaussian Copula**: Correlation $\rho = 0.6$
    - **Student-$t$ Copula**: Correlation $\rho = 0.6$, Degrees of Freedom $\nu = 3$
- **Sensitivity Sweep**:
  - Correlation parameter $\rho \in [0.0, 0.2, 0.4, 0.6, 0.8]$
  - Degrees of freedom $\nu \in [3, 5, 10, 30]$

## Simulation Execution
- **Monte Carlo Replications ($N_{\mathrm{rep}}$)**: 5,000
- **Random Number Generation Seed**: 42 (ensures exact reproducibility of joint copula samples and failure trajectories)
