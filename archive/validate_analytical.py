"""
validate_analytical.py
======================
Monte Carlo engine verification for the MW-k/n:G framework.

Computes the closed-form analytical system reliability R(t) for Scenario S2
(temporal-only, independence copula, fixed formation) using the exact
combinatorial formula for a weighted k-out-of-n:G system, then compares
against Monte Carlo estimates at varying N_rep to produce a convergence plot.

Outputs:
  figures/fig_validation_convergence.pdf  -- survival curve + error convergence
  validation_results.csv                  -- numerical table for paper
"""

import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm as sp_norm

# ---------------------------------------------------------------------------
# System parameters (matching run_scenarios.py exactly)
# ---------------------------------------------------------------------------
WEIGHTS    = [3]*3 + [2]*5 + [1]*4   # 12 agents: 3A + 5B + 4C
LAMBDAS    = [0.005]*3 + [0.008]*5 + [0.012]*4
K          = 18          # capacity threshold
T_MAX      = 120         # mission horizon (steps)
N_AGENTS   = len(WEIGHTS)

# Subsets whose total weight >= K  (precomputed once)
print(f"Precomputing {2**N_AGENTS:,} subsets for analytical benchmark...")
VALID_SUBSETS = []
for mask in range(2**N_AGENTS):
    w = sum(WEIGHTS[i] for i in range(N_AGENTS) if (mask >> i) & 1)
    if w >= K:
        VALID_SUBSETS.append(mask)
print(f"  -> {len(VALID_SUBSETS):,} valid subsets found (W >= {K})")


# ---------------------------------------------------------------------------
# Analytical reliability (exact closed-form)
# ---------------------------------------------------------------------------
def analytical_R(t: float) -> float:
    """
    P(sum of active weights >= K) under component independence.
    p_i(t) = exp(-lambda_i * t),  q_i(t) = 1 - p_i(t)
    """
    p = [math.exp(-lam * t) for lam in LAMBDAS]
    q = [1.0 - pi for pi in p]
    R = 0.0
    for mask in VALID_SUBSETS:
        prob = 1.0
        for i in range(N_AGENTS):
            if (mask >> i) & 1:
                prob *= p[i]
            else:
                prob *= q[i]
        R += prob
    return R


def analytical_MTTF(n_steps: int = T_MAX) -> float:
    """
    Discrete-time MTTF = sum_{t=0}^{T_MAX-1} R(t)   (per Algorithm 1 in paper)
    """
    return sum(analytical_R(t) for t in range(n_steps))


# ---------------------------------------------------------------------------
# Monte Carlo for S2 (tight cluster, independence copula)
# ---------------------------------------------------------------------------
def mc_single_run() -> int:
    """One S2 run. Returns step at first failure (or T_MAX if never fails)."""
    # Sample TTF from independent exponentials
    tau = [-math.log(np.random.rand()) / lam for lam in LAMBDAS]
    for step in range(T_MAX):
        active_weight = sum(
            WEIGHTS[i] for i in range(N_AGENTS) if step < tau[i]
        )
        if active_weight < K:
            return step
    return T_MAX


def mc_reliability_curve(N_rep: int, seed: int = 0) -> np.ndarray:
    """
    Returns survival curve R_hat(t), shape (T_MAX,).
    R_hat(t) = fraction of runs still surviving at step t.
    """
    np.random.seed(seed)
    # survived[r] = TTF of run r
    ttfs = np.array([mc_single_run() for _ in range(N_rep)], dtype=float)
    # R(t) = fraction of runs with TTF > t
    t_arr = np.arange(T_MAX)
    return np.array([(ttfs > t).mean() for t in t_arr])


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def main():
    os.makedirs("figures", exist_ok=True)

    # --- Analytical curve at all timesteps ---
    print("\nComputing analytical R(t) for t=0..119 ...")
    t_arr = np.arange(T_MAX)
    R_analytic = np.array([analytical_R(t) for t in t_arr])
    MTTF_analytic = analytical_MTTF()
    print(f"  Analytical MTTF = {MTTF_analytic:.4f} steps")

    # --- MC convergence study ---
    N_rep_values = [100, 500, 1000, 2500, 5000]
    t_check = 60     # reference time for convergence error
    seeds    = [42, 7, 13, 99, 0]  # one seed per N_rep for reproducibility

    records = []
    mc_curves = {}
    for N_rep, seed in zip(N_rep_values, seeds):
        print(f"  Running MC: N_rep={N_rep} ...", end=" ", flush=True)
        curve = mc_reliability_curve(N_rep, seed=seed)
        mc_curves[N_rep] = curve
        R_mc_check  = curve[t_check]
        R_ana_check = R_analytic[t_check]
        abs_err     = abs(R_mc_check - R_ana_check)
        mttf_mc     = curve.sum()   # discrete integral
        rel_err_mttf = abs(mttf_mc - MTTF_analytic) / MTTF_analytic * 100
        print(f"R_mc({t_check})={R_mc_check:.4f}, "
              f"|err|={abs_err:.4f}, "
              f"MTTF_mc={mttf_mc:.2f} ({rel_err_mttf:.2f}% err)")
        records.append(dict(
            N_rep=N_rep,
            R_mc_t60=round(R_mc_check, 5),
            R_analytic_t60=round(R_ana_check, 5),
            abs_error=round(abs_err, 5),
            MTTF_MC=round(mttf_mc, 4),
            MTTF_Analytic=round(MTTF_analytic, 4),
            MTTF_rel_err_pct=round(rel_err_mttf, 3),
        ))

    df = pd.DataFrame(records)
    df.to_csv("validation_results.csv", index=False)
    print("\nSaved: validation_results.csv")
    print(df.to_string(index=False))

    # --- Plot: two-panel figure ---
    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # Panel (a): Survival curve comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_arr, R_analytic, 'k-', lw=2.2, label='Analytical (exact)', zorder=5)
    colors = ['#c0392b', '#e67e22', '#27ae60', '#2980b9', '#8e44ad']
    styles = [':', '--', '-.', (0,(3,1,1,1)), '-']
    for (N_rep, color, ls) in zip(N_rep_values, colors, styles):
        ax1.plot(t_arr, mc_curves[N_rep], color=color, lw=1.3,
                 linestyle=ls, alpha=0.85, label=f'MC $N_{{rep}}={N_rep}$')
    ax1.axvline(t_check, color='gray', lw=0.9, ls=':', alpha=0.7,
                label=f'$t={t_check}$ (ref)')
    ax1.set_xlabel('Mission Step $t$', fontsize=11)
    ax1.set_ylabel('Survival Probability $R(t)$', fontsize=11)
    ax1.set_title('(a) Survival Curve: MC vs. Analytical', fontsize=11)
    ax1.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    ax1.set_xlim(0, T_MAX - 1)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Absolute error convergence at t=60
    ax2 = fig.add_subplot(gs[0, 1])
    errs = [r['abs_error'] for r in records]
    ax2.plot(N_rep_values, errs, 'ko-', lw=1.8, ms=7,
             markerfacecolor='#e74c3c', markeredgecolor='k', zorder=5)
    ax2.axhline(0.01, color='#27ae60', lw=1.2, ls='--', alpha=0.8,
                label='Threshold: 0.01')
    ax2.axhline(0.005, color='#2980b9', lw=1.2, ls=':', alpha=0.8,
                label='Threshold: 0.005')
    for x, y in zip(N_rep_values, errs):
        ax2.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=8,
                     color='#c0392b')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Replications $N_{rep}$', fontsize=11)
    ax2.set_ylabel(f'$|\\hat{{R}}_{{\\rm MC}}(t={t_check}) - R_{{\\rm analytic}}(t={t_check})|$',
                   fontsize=10)
    ax2.set_title('(b) MC Convergence to Analytical Benchmark', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xticks(N_rep_values)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.grid(True, alpha=0.3, which='both')

    plt.suptitle('Monte Carlo (MC) Engine Verification — Scenario S2 (Independence, Fixed Formation)',
                 fontsize=11, y=1.02)
    fig.savefig('figures/fig_validation_convergence.pdf',
                bbox_inches='tight', dpi=200)
    print("Saved: figures/fig_validation_convergence.pdf")
    plt.close(fig)

    # Summary
    final_err = records[-1]['abs_error']
    final_mttf_err = records[-1]['MTTF_rel_err_pct']
    print(f"\n[PASS] Verification complete: N_rep=5000 absolute error = {final_err:.5f} "
          f"({'< 0.01 OK' if final_err < 0.01 else '>= 0.01 (WARN)'})")
    print(f"   MTTF relative error = {final_mttf_err:.3f}%  "
          f"({'< 2% OK' if final_mttf_err < 2.0 else '>= 2% (check)'})")


if __name__ == "__main__":
    main()
