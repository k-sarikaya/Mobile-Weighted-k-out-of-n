"""Mission reliability experiments for the MW-k/n:G framework.

One simulation engine serves every analysis.  A trajectory is generated once
per replication and evaluated for every capacity threshold and edge model, so
threshold, edge-model and copula comparisons are paired rather than separate
Monte Carlo experiments; this is what the paired simultaneous confidence bands
require.

Usage:
    python run_experiments.py                # full suite, 5000 replications
    python run_experiments.py --smoke        # 100 replications
    python run_experiments.py --verify-only  # engine verification only
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, kendalltau, norm, t as t_dist


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_outputs"
MAX_STEPS = 120
W_MAX = 23
DEFAULT_K = 18
COPULAS = ("independence", "gaussian", "student_t", "clayton", "gumbel")
WORKERS = min(8, max(1, cpu_count() - 1))

# Nominal simultaneous coverage of the sup-t bootstrap bands.  The bootstrap
# statistic is max_k |t_k|, which already accounts for both tails, so the
# 1 - alpha quantile of that statistic yields 100(1 - alpha)% simultaneous
# coverage over the whole family of thresholds.
SUP_T_LEVEL = 0.95


def swarm_definition():
    # Three high-capacity, five medium-capacity, and four low-capacity agents.
    return np.array([3] * 3 + [2] * 5 + [1] * 4, dtype=float)


def copula_uniforms(rng, name, rho=0.6, df=3):
    n = 12
    if name == "independence":
        return rng.uniform(size=n)
    if name == "gaussian":
        cov = np.full((n, n), rho, dtype=float)
        np.fill_diagonal(cov, 1.0)
        return np.clip(norm.cdf(rng.multivariate_normal(np.zeros(n), cov)), 1e-10, 1 - 1e-10)
    if name == "student_t":
        cov = np.full((n, n), rho, dtype=float)
        np.fill_diagonal(cov, 1.0)
        z = rng.multivariate_normal(np.zeros(n), cov)
        scale = math.sqrt(df / rng.chisquare(df))
        return np.clip(t_dist.cdf(z * scale, df), 1e-10, 1 - 1e-10)
    if name == "clayton":
        tau = 2.0 / math.pi * math.asin(rho) if rho else 0.0
        theta = 2.0 * tau / max(1.0 - tau, 1e-12)
        if theta <= 0:
            return rng.uniform(size=n)
        v = max(rng.gamma(1.0 / theta, 1.0), 1e-12)
        return np.clip((1.0 + rng.exponential(size=n) / v) ** (-1.0 / theta), 1e-10, 1 - 1e-10)
    if name == "gumbel":
        tau = 2.0 / math.pi * math.asin(rho) if rho else 0.0
        theta = 1.0 / max(1.0 - tau, 1e-12)
        if theta <= 1.0:
            return rng.uniform(size=n)
        alpha = 1.0 / theta
        angle = rng.uniform(-math.pi / 2.0, math.pi / 2.0)
        w = rng.exponential()
        v = math.sin(alpha * (angle + math.pi / 2.0)) / max(math.cos(angle) ** (1.0 / alpha), 1e-12)
        v *= (math.cos(angle - alpha * (angle + math.pi / 2.0)) / w) ** ((1.0 - alpha) / alpha)
        v = max(v, 1e-12)
        return np.clip(np.exp(-(rng.exponential(size=n) / v) ** alpha), 1e-10, 1 - 1e-10)
    raise ValueError(f"Unknown copula: {name}")


def positions_and_soc(weights, mode, rng, max_steps=MAX_STEPS):
    n = len(weights)
    speed = rng.uniform(0.9, 1.1, size=n)
    angle_offset = rng.uniform(-0.1, 0.1, size=n)
    soc = np.ones(n)
    xy = np.zeros((max_steps, n, 2))
    active_from_battery = np.ones((max_steps, n), dtype=bool)
    latent_wind = 0.0
    # The realised latent stress path is returned so that a dependence model
    # can be driven by the same environmental process that already moves the
    # formation and drains the batteries, rather than by a separate device.
    stress = np.zeros(max_steps)

    for step in range(max_steps):
        if mode in ("S1", "S3"):
            if mode == "S3":
                # Persistent shared environmental stress.  The same latent
                # state affects trajectory expansion and battery drain.
                latent_wind = 0.85 * latent_wind + math.sqrt(1 - 0.85**2) * rng.normal()
                wind_speed = float(np.clip(10.0 + 2.5 * latent_wind, 5.0, 15.0))
                wind_factor = 1.0 + 0.01 * (wind_speed - 10.0)
                stress[step] = latent_wind
            else:
                wind_speed, wind_factor = 0.0, 1.0
            radius = (20.0 + step * 2.5) * speed * wind_factor
            angles = np.arange(n) * (2 * math.pi / n) + angle_offset
            xy[step, :, 0] = radius * np.cos(angles)
            xy[step, :, 1] = radius * np.sin(angles)
            velocity = 2.5 * speed
            draw = (0.004 + 0.00015 * velocity + 0.00007 * velocity**2 + 0.00025 * wind_speed) * speed
        else:
            angles = np.arange(n) * (2 * math.pi / n)
            xy[step, :, 0] = 40.0 * np.cos(angles)
            xy[step, :, 1] = 40.0 * np.sin(angles)
            velocity = 0.5 * speed
            draw = (0.003 + 0.0001 * velocity) * speed
        soc -= draw
        active_from_battery[step] = soc > 0.10
    return xy, active_from_battery, stress


def pdr(distance):
    return 1.0 / (1.0 + np.exp(0.05 * (distance - 250.0)))


def graph_metrics(xy, active, edge_rng, latency_rng, edge_model="bernoulli", q0=0.5):
    ids = np.flatnonzero(active)
    n = len(ids)
    if n == 0:
        return 0.0, math.inf, 0
    adj = {int(i): set() for i in ids}
    for a in range(n):
        for b in range(a + 1, n):
            d = float(np.linalg.norm(xy[ids[a]] - xy[ids[b]]))
            q = float(pdr(d))
            if edge_model == "deterministic":
                present = q >= q0
            else:
                present = edge_rng.random() <= q
            if present:
                i, j = int(ids[a]), int(ids[b])
                adj[i].add(j)
                adj[j].add(i)

    components = []
    unseen = set(adj)
    while unseen:
        root = unseen.pop()
        comp = {root}
        queue = [root]
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    comp.add(v)
                    queue.append(v)
        components.append(comp)
    largest = max(components, key=len)
    largest_weight = float(sum((3 if i < 3 else 2 if i < 8 else 1) for i in largest))

    max_latency = 0.0
    nodes = list(largest)
    for source in nodes:
        dist = {source: 0}
        queue = [source]
        for u in queue:
            for v in adj[u]:
                if v in largest and v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for target, hops in dist.items():
            if target != source:
                delays = latency_rng.lognormal(mean=2.0, sigma=0.5, size=hops)
                max_latency = max(max_latency, float(delays.sum()))
    return largest_weight, max_latency, len(largest)


def rho_from_tau(tau):
    """Invert Kendall's tau for the elliptical families."""
    return math.sin(math.pi * tau / 2.0)


# ---------------------------------------------------------------------------
# Environmental frailty arm -- FORMULATION ONLY, NOT USED BY ANY REPORTED
# RESULT.
#
# The functions whose names contain "frailty" -- frailty_lifetimes below,
# and further down _frailty_z_paths, calibrate_frailty_baseline,
# frailty_tau_from_sigma, solve_sigma_for_tau, frailty_job,
# frailty_marginal_diagnostic, frailty_calibration_table and
# run_frailty_sweep -- together implement the second time-varying dependence
# mechanism written down in Section 3.3, Eq. (22).  They are the only
# functions in this file with that status; everything else feeds a reported
# result.
#
# No figure, table, or number in the paper comes from this code, no CSV in
# experiment_outputs/ is produced by it, verify_claims.py makes no assertion
# about it, and main() calls none of it -- the functions are reachable only if
# called directly.  The reason is stated in the paper: the mechanism cannot be
# calibrated to a target Kendall's tau while holding the marginals fixed over
# the range of interest, so it is reported as a formulation and listed as
# future work (Section 7.3 and Section 8, item 4).  The evaluated
# time-varying arm is the phase-switching copula (Eq. (21),
# phase_copula_lifetimes), which does preserve the marginals exactly.
#
# The code is kept so the formulation is inspectable and so the calibration
# failure can be reproduced: frailty_marginal_diagnostic() records the sigma
# range over which the marginal error exceeds MARGINAL_ERROR_TOL.
# ---------------------------------------------------------------------------


def frailty_lifetimes(rng, stress, rates, sigma, w_persistent, baseline, max_steps):
    """Lifetimes driven by a shared stress process with a persistent component.

        Z(t) = sigma * ( sqrt(w) * W + sqrt(1 - w) * A(t) ),   Var Z(t) = sigma^2

    A(t) is the standardised AR(1) environmental path already simulated for the
    trajectory and the battery drain, and W ~ N(0,1) is drawn once per mission.
    The mixture matters: a purely mean-reverting A(t) averages out over a
    120-step horizon, so it induces almost no association between lifetimes.
    The persistent component W does not average out, so w controls how much
    dependence the mechanism can actually generate, while the A(t) part keeps
    the association time-varying within the mission.

    The conditional hazard is lambda_i(t | Z) = baseline_i * exp(Z(t)).  The
    baseline is NOT the nominal rate: it is recalibrated per (sigma, w) by
    calibrate_frailty_baseline so that the marginal survival matches the
    nominal exponential.  Without that step the arm would confound dependence
    with a shifted marginal, since by Jensen's inequality
    E[exp(-integral)] != exp(-E[integral]).
    """
    if sigma == 0.0:
        u = rng.uniform(size=len(rates))
        return -np.log1p(-u) / rates
    w_draw = rng.normal()
    z = sigma * (math.sqrt(w_persistent) * w_draw
                 + math.sqrt(1.0 - w_persistent) * stress)
    cumulative = np.cumsum(np.outer(np.exp(z), baseline), axis=0)
    threshold = rng.exponential(size=len(rates))
    exceeded = cumulative >= threshold
    any_exceeded = exceeded.any(axis=0)
    first = np.argmax(exceeded, axis=0).astype(float)
    return np.where(any_exceeded, first, np.inf)


def phase_copula_lifetimes(rng, rates, family, tau_early, tau_late, boundary, df=3):
    """Piecewise-constant dependence with exactly preserved marginals.

    A component that fails before the phase boundary draws its lifetime from a
    copula calibrated to tau_early; a component that survives the boundary
    draws a fresh residual lifetime from a copula calibrated to tau_late.
    Because the marginal lifetimes are exponential and therefore memoryless,
    the residual distribution after the boundary is again exponential with the
    same rate, so switching the copula at the boundary changes the dependence
    structure without perturbing any marginal.  This gives a phase-specific
    dependence parameter that does not require phase-resolved failure data.
    """
    u_early = copula_uniforms(rng, family, rho_from_tau(tau_early), df)
    t_early = -np.log1p(-u_early) / rates
    u_late = copula_uniforms(rng, family, rho_from_tau(tau_late), df)
    t_late = -np.log1p(-u_late) / rates
    return np.where(t_early <= boundary, t_early, boundary + t_late)


def simulate_trajectory(scenario, seed, max_steps=MAX_STEPS, edge_model="bernoulli", q0=0.5):
    streams = np.random.SeedSequence([20260822, seed]).spawn(4)
    rng = np.random.default_rng(streams[0])
    copula_rng = np.random.default_rng(streams[1])
    edge_rng = np.random.default_rng(streams[2])
    latency_rng = np.random.default_rng(streams[3])
    weights = swarm_definition()
    xy, battery_ok, stress = positions_and_soc(weights, scenario["mode"], rng, max_steps)
    if scenario["mode"] in ("S2", "S3"):
        rates = FAILURE_RATES
        kind = scenario.get("dependence", "copula")
        if kind == "frailty":
            failure_time = frailty_lifetimes(
                copula_rng, stress, rates, scenario.get("sigma", 0.0),
                scenario.get("w_persistent", FRAILTY_W),
                scenario.get("baseline", rates), max_steps)
        elif kind == "phase_copula":
            failure_time = phase_copula_lifetimes(
                copula_rng, rates, scenario.get("family", "gaussian"),
                scenario.get("tau_early", 0.41), scenario.get("tau_late", 0.41),
                scenario.get("boundary", max_steps // 2), scenario.get("df", 3))
        else:
            u = copula_uniforms(copula_rng, scenario["copula"], scenario.get("rho", 0.6), scenario.get("df", 3))
            failure_time = -np.log1p(-u) / rates
    else:
        failure_time = np.full(12, np.inf)

    total = np.zeros(max_steps)
    cc_weight = np.zeros(max_steps)
    delay = np.full(max_steps, np.inf)
    lcc_size = np.zeros(max_steps)
    node_ok = np.zeros(max_steps, dtype=bool)
    for step in range(max_steps):
        alive = battery_ok[step] & (step < failure_time)
        total[step] = float(weights[alive].sum())
        node_ok[step] = bool(alive.any())
        cc_weight[step], delay[step], lcc_size[step] = graph_metrics(xy[step], alive, edge_rng, latency_rng, edge_model, q0)
    return {"total_weight": total, "cc_weight": cc_weight, "delay": delay, "node_ok": node_ok}


def evaluate_trajectory(traj, k, latency_limit=100.0):
    a = traj["total_weight"] >= k
    b = traj["cc_weight"] >= k
    c = traj["delay"] <= latency_limit
    instant = a & b & c
    survival = np.minimum.accumulate(instant.astype(int)).astype(bool)
    failure = np.flatnonzero(~survival)
    ttf = int(failure[0]) if len(failure) else len(instant)
    return {"A": a, "B": b, "C": c, "instant": instant, "survival": survival, "ttf": ttf}


def summarize(values, horizon=MAX_STEPS):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": mean, "se": se, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}


def threshold_job(args):
    copula, run, max_steps = args
    scenario = {"mode": "S3", "copula": copula, "rho": 0.6, "df": 3}
    traj = simulate_trajectory(scenario, run, max_steps=max_steps)
    rows = []
    for k in range(2, W_MAX + 1):
        ev = evaluate_trajectory(traj, k)
        rows.append({"copula": copula, "run": run, "K": k, "kappa": k / W_MAX, "RMST": float(ev["survival"].sum()), "R_horizon": float(ev["survival"][-1]), "TTF": ev["ttf"]})
    return rows


def edge_job(args):
    model, q0, run, max_steps = args
    scenario = {"mode": "S3", "copula": "independence", "rho": 0.0, "df": 3}
    traj = simulate_trajectory(scenario, run, max_steps=max_steps, edge_model=model, q0=q0)
    ev = evaluate_trajectory(traj, DEFAULT_K)
    failure = ev["ttf"]
    if failure >= max_steps:
        binding = "none_within_horizon"
    else:
        states = (ev["A"][failure], ev["B"][failure], ev["C"][failure])
        binding = "capacity" if not states[0] else "connectivity" if not states[1] else "delay"
    return {"model": model, "q0": q0, "run": run, "RMST": ev["survival"].sum(), "R_horizon": ev["survival"][-1], "TTF": ev["ttf"], "binding": binding}


def main_scenario_job(args):
    label, scenario, run, max_steps = args
    traj = simulate_trajectory(scenario, run, max_steps=max_steps)
    ev = evaluate_trajectory(traj, DEFAULT_K)
    failure = ev["ttf"]
    if failure >= max_steps:
        binding = "none_within_horizon"
    else:
        states = (ev["A"][failure], ev["B"][failure], ev["C"][failure])
        binding = "capacity" if not states[0] else "connectivity" if not states[1] else "delay"
    return {"scenario": label, "run": run, "RMST": ev["survival"].sum(), "R_horizon": ev["survival"][-1], "TTF": failure, "binding": binding}


def run_threshold_sweep(n_runs, copulas=COPULAS, max_steps=MAX_STEPS):
    records = []
    jobs = [(copula, run, max_steps) for copula in copulas for run in range(n_runs)]
    with Pool(processes=WORKERS) as pool:
        run_records = [row for batch in pool.map(threshold_job, jobs) for row in batch]
    raw = pd.DataFrame(run_records)
    raw.to_csv(OUT / "threshold_sweep_runs.csv", index=False)
    for (copula, k), g in raw.groupby(["copula", "K"]):
        s = summarize(g["RMST"].to_numpy())
        records.append({"copula": copula, "K": k, "kappa": k / W_MAX, "RMST": s["mean"], "SE": s["se"], "CI_low": s["ci_low"], "CI_high": s["ci_high"]})
    summary = pd.DataFrame(records).sort_values(["K", "copula"])
    summary.to_csv(OUT / "threshold_sweep_summary.csv", index=False)
    return raw, summary


def run_edge_sensitivity(n_runs, max_steps=MAX_STEPS):
    # q0 is meaningful only for the deterministic arm.  The Bernoulli arm is
    # given the sentinel 0.5 so that the worker signature stays uniform, but it
    # is never read by the Bernoulli branch of graph_metrics and is written out
    # as an empty field so the published CSVs do not imply a threshold.
    jobs = [(model, 0.5 if np.isnan(q0) else float(q0), run, max_steps) for model, q0 in [("bernoulli", np.nan), ("deterministic", 0.3), ("deterministic", 0.5), ("deterministic", 0.7)] for run in range(n_runs)]
    with Pool(processes=WORKERS) as pool:
        rows = pool.map(edge_job, jobs)
    raw = pd.DataFrame(rows)
    raw.loc[raw["model"] == "bernoulli", "q0"] = np.nan
    raw.to_csv(OUT / "edge_model_sensitivity_runs.csv", index=False)
    summary_rows = []
    for (model, q0), group in raw.groupby(["model", "q0"], dropna=False):
        s = summarize(group["RMST"].to_numpy())
        summary_rows.append({"model": model, "q0": q0, "RMST": s["mean"], "RMST_SE": s["se"], "CI_low": s["ci_low"], "CI_high": s["ci_high"], "R_horizon": group["R_horizon"].mean(), "TTF": group["TTF"].mean()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "edge_model_sensitivity_summary.csv", index=False)
    base = raw[raw["model"] == "bernoulli"].set_index("run")["RMST"]
    differences = []
    for (model, q0), group in raw.groupby(["model", "q0"], dropna=False):
        if model == "bernoulli":
            continue
        delta = group.set_index("run")["RMST"] - base
        s = summarize(delta.to_numpy())
        differences.append({"model": model, "q0": q0, "paired_delta_RMST": s["mean"], "SE": s["se"], "CI_low": s["ci_low"], "CI_high": s["ci_high"]})
    pd.DataFrame(differences).to_csv(OUT / "edge_model_paired_differences.csv", index=False)
    raw.groupby(["model", "q0", "binding"]).size().reset_index(name="count").to_csv(OUT / "edge_model_binding_constraints.csv", index=False)
    return raw, summary


def run_main_scenarios(n_runs, max_steps=MAX_STEPS, threshold_raw=None):
    scenarios = [
        ("S1_Spatial", {"mode": "S1", "copula": "independence"}),
        ("S2_Temporal", {"mode": "S2", "copula": "independence"}),
    ]
    jobs = [(label, scenario, run, max_steps) for label, scenario in scenarios for run in range(n_runs)]
    with Pool(processes=WORKERS) as pool:
        rows = pool.map(main_scenario_job, jobs)
    raw = pd.DataFrame(rows)
    if threshold_raw is not None:
        label_map = {"independence": "S3_Joint_Independence", "gaussian": "S3_Joint_Gaussian", "student_t": "S3_Joint_Student_t", "clayton": "S3_Joint_Clayton", "gumbel": "S3_Joint_Gumbel"}
        default = threshold_raw[threshold_raw["K"] == DEFAULT_K].copy()
        default["scenario"] = default["copula"].map(label_map)
        default = default.rename(columns={"R_horizon": "R_horizon"})
        default["binding"] = "from_threshold_sweep"
        default = default[["scenario", "run", "RMST", "R_horizon", "TTF", "binding"]]
        raw = pd.concat([raw, default], ignore_index=True)
    raw.to_csv(OUT / "main_scenario_runs.csv", index=False)
    summary_rows = []
    for scenario, group in raw.groupby("scenario"):
        s = summarize(group["RMST"].to_numpy())
        summary_rows.append({"scenario": scenario, "RMST": s["mean"], "SE": s["se"], "CI_low": s["ci_low"], "CI_high": s["ci_high"], "R_horizon": group["R_horizon"].mean(), "TTF": group["TTF"].mean()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "main_scenario_summary.csv", index=False)
    raw.groupby(["scenario", "binding"]).size().reset_index(name="count").to_csv(OUT / "main_binding_constraints.csv", index=False)
    return raw, summary


VALIDATION_SEEDS = tuple(range(10))


def validate_copulas(n=10000, seeds=VALIDATION_SEEDS):
    """Sampler diagnostics repeated over independent seeds.

    A single seed makes a borderline Kolmogorov-Smirnov p-value hard to read:
    at n = 10000 an occasional p below 0.05 is expected even from an exact
    sampler.  Each family is therefore validated on several independent draws
    and the per-seed results are summarised, so the diagnostic reports a
    distribution rather than one realisation.
    """
    per_seed, rows = [], []
    for name in COPULAS:
        stats = []
        for s in seeds:
            rng = np.random.default_rng(
                np.random.SeedSequence([20260822, 9000 + COPULAS.index(name), int(s)]))
            sample = np.array([copula_uniforms(rng, name, 0.6, 3) for _ in range(n)])
            u1, u2 = sample[:, 0], sample[:, 1]
            tau = float(kendalltau(u1, u2).statistic)
            ks = kstest(u1, "uniform")
            tail = float(np.mean((u1 < 0.05) & (u2 < 0.05)) / 0.05)
            rec = {"copula": name, "seed": int(s), "n": int(n),
                   "empirical_kendall_tau": tau, "KS_stat": float(ks.statistic),
                   "KS_pvalue": float(ks.pvalue), "lower_tail_ratio_at_0.05": tail}
            per_seed.append(rec)
            stats.append(rec)
        d = pd.DataFrame(stats)
        rows.append({
            "copula": name,
            "n_seeds": len(seeds), "n_per_seed": int(n),
            "empirical_kendall_tau": d.empirical_kendall_tau.mean(),
            "kendall_tau_sd": d.empirical_kendall_tau.std(ddof=1),
            "KS_stat_mean": d.KS_stat.mean(),
            "KS_pvalue_min": d.KS_pvalue.min(),
            "KS_pvalue_median": d.KS_pvalue.median(),
            "KS_rejections_at_5pct": int((d.KS_pvalue < 0.05).sum()),
            "lower_tail_ratio_at_0.05": d["lower_tail_ratio_at_0.05"].mean(),
            "lower_tail_ratio_sd": d["lower_tail_ratio_at_0.05"].std(ddof=1),
        })
    pd.DataFrame(per_seed).to_csv(OUT / "copula_sampler_validation_by_seed.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "copula_sampler_validation.csv", index=False)
    return out


RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)
DF_GRID = (3, 5, 10, 30)

# --- Time-varying dependence arms -------------------------------------------
# Two mechanisms in which the dependence between component lifetimes is not a
# single fixed number.  The frailty arm makes the association a consequence of
# the shared environment; the phase arm lets it take different values in two
# mission phases.  Neither requires phase-resolved field data: the governing
# parameter is swept, exactly as Kendall's tau is swept in the copula arm.
FRAILTY_W = 0.5          # persistent share of the stress variance
# Largest marginal distortion tolerated in the frailty arm.  The per-(sigma, w)
# recalibration matches the survival function at the horizon; the residual
# sup-norm error over the interior of the grid grows with sigma, and beyond
# this tolerance the arm would no longer be a controlled comparison against
# the copula arm.  frailty_marginal_diagnostic() records the growth.
MARGINAL_ERROR_TOL = 0.06
# Dependence grid for the frailty arm.  It stops at tau = 0.15 because that is
# where the tolerance above binds (sigma ~ 1.0); the copula arm carries the
# comparison at higher tau, where the frailty marginals can no longer be held
# fixed.  FRAILTY_DIAGNOSTIC_SIGMAS spans the uncontrolled region so that the
# restriction is reported as a measurement rather than asserted.
FRAILTY_TAU_TARGETS = (0.0, 0.05, 0.10, 0.15)
FRAILTY_DIAGNOSTIC_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
PHASE_TAU_GRID = (0.0, 0.20, 0.41, 0.59)
PHASE_BOUNDARY = 40   # steps; the median first-passage time under S3 independence is 39


def _frailty_z_paths(sigma, w_persistent, n_paths, max_steps, seed):
    """Realisations of Z(t) = sigma (sqrt(w) W + sqrt(1-w) A(t))."""
    a = stress_paths(n_paths, max_steps, seed_base=seed)
    rng = np.random.default_rng(np.random.SeedSequence([20260822, seed + 1]))
    w_draw = rng.normal(size=(n_paths, 1))
    return sigma * (math.sqrt(w_persistent) * w_draw
                    + math.sqrt(1.0 - w_persistent) * a)


def calibrate_frailty_baseline(sigma, w_persistent, rates=None, horizon=MAX_STEPS,
                               n_paths=6000, seed=720000):
    """Baseline rates for which the frailty arm keeps the nominal marginals.

    Returns the baseline vector together with the residual marginal error.  The
    baseline solves  E[exp(-baseline_i * C(T))] = exp(-rate_i * T),  where
    C(t) = sum_{s<=t} exp(Z(s)); the expectation is over the shared stress
    paths.  This is the recalibration the manuscript flags as necessary once a
    frailty term is introduced, and without it the arm is not a controlled
    comparison against the copula arm.
    """
    rates = FAILURE_RATES if rates is None else rates
    if sigma == 0.0:
        return np.array(rates, dtype=float), 0.0
    z = _frailty_z_paths(sigma, w_persistent, n_paths, horizon, seed)
    cumulative = np.cumsum(np.exp(z), axis=1)          # (paths, steps)
    c_end = cumulative[:, -1]
    baseline = np.empty(len(rates))
    for i, rate in enumerate(rates):
        target = math.exp(-rate * horizon)
        lo, hi = 1e-8, max(rate * 50.0, 1.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(np.mean(np.exp(-mid * c_end))) > target:
                lo = mid
            else:
                hi = mid
        baseline[i] = 0.5 * (lo + hi)
    # Residual error over the whole grid, not just the matched endpoint.
    steps = np.arange(1, horizon + 1)
    worst = 0.0
    for i, rate in enumerate(rates):
        realised = np.mean(np.exp(-baseline[i] * cumulative), axis=0)
        nominal = np.exp(-rate * steps)
        worst = max(worst, float(np.max(np.abs(realised - nominal))))
    return baseline, worst


def frailty_tau_from_sigma(sigma, w_persistent, n_paths=3000, horizon=1200):
    """Kendall's tau between lifetimes induced by (sigma, w)."""
    if sigma == 0.0:
        return 0.0
    baseline, _ = calibrate_frailty_baseline(sigma, w_persistent)
    z = _frailty_z_paths(sigma, w_persistent, n_paths, horizon, seed=730000)
    cumulative = np.cumsum(np.exp(z), axis=1)[:, :, None] * baseline
    rng = np.random.default_rng(np.random.SeedSequence([20260822, 731000]))
    thresholds = rng.exponential(size=(n_paths, len(baseline)))
    exceeded = cumulative >= thresholds[:, None, :]
    observed = exceeded.any(axis=1)
    lifetimes = np.argmax(exceeded, axis=1).astype(float)
    keep = observed.all(axis=1)
    sub = lifetimes[keep]
    if len(sub) < 200:
        return float("nan")
    taus = [kendalltau(sub[:, i], sub[:, j]).statistic
            for i in range(12) for j in range(i + 1, 12)]
    return float(np.nanmean(taus))


def solve_sigma_for_tau(tau_target, w_persistent=FRAILTY_W, lo=0.0, hi=2.0, tol=0.004):
    """Stress scale that reproduces a target Kendall's tau.

    The upper bracket is 2.0 because the admissible grid is bounded by
    MARGINAL_ERROR_TOL long before that: tau(sigma=2) is already above every
    target in FRAILTY_TAU_TARGETS.
    """
    if tau_target == 0.0:
        return 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if frailty_tau_from_sigma(mid, w_persistent) < tau_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def frailty_job(args):
    sigma, w_persistent, baseline, tau, run, max_steps = args
    scenario = {"mode": "S3", "copula": "independence", "dependence": "frailty",
                "sigma": sigma, "w_persistent": w_persistent,
                "baseline": np.asarray(baseline)}
    ev = evaluate_trajectory(simulate_trajectory(scenario, run, max_steps), DEFAULT_K)
    return {"tau_target": tau, "sigma": sigma, "run": run,
            "RMST": float(ev["survival"].sum()),
            "R_horizon": float(ev["survival"][-1]), "TTF": ev["ttf"]}


def phase_job(args):
    tau_early, tau_late, boundary, run, max_steps = args
    scenario = {"mode": "S3", "copula": "gaussian", "dependence": "phase_copula",
                "family": "gaussian", "tau_early": tau_early, "tau_late": tau_late,
                "boundary": boundary}
    ev = evaluate_trajectory(simulate_trajectory(scenario, run, max_steps), DEFAULT_K)
    return {"tau_early": tau_early, "tau_late": tau_late, "boundary": boundary,
            "run": run, "RMST": float(ev["survival"].sum()),
            "R_horizon": float(ev["survival"][-1]), "TTF": ev["ttf"]}


def stress_paths(n_paths, max_steps, rho=0.85, seed_base=700000):
    """Standardised AR(1) stress paths matching positions_and_soc, vectorised.

    positions_and_soc advances the same recursion one step at a time inside the
    trajectory loop; reproducing it directly here avoids simulating positions
    and batteries when only the environmental path is needed.
    """
    rng = np.random.default_rng(np.random.SeedSequence([20260822, seed_base]))
    innovations = rng.normal(size=(n_paths, max_steps)) * math.sqrt(1 - rho ** 2)
    paths = np.empty((n_paths, max_steps))
    state = np.zeros(n_paths)
    for step in range(max_steps):
        state = rho * state + innovations[:, step]
        paths[:, step] = state
    return paths


def validate_phase_marginals(n=200000, pairs=((0.0, 0.59), (0.59, 0.0), (0.20, 0.41))):
    """Check that the phase switch leaves the exponential marginals intact.

    The memorylessness argument says the construction is exact, so this is a
    direct numerical confirmation rather than a tolerance test: for each
    (tau_early, tau_late) pair the recovered rate 1/mean(lifetime) is compared
    with the nominal rate of every component, and the worst relative error over
    the twelve components is reported.
    """
    rows = []
    for tau_early, tau_late in pairs:
        rng = np.random.default_rng(np.random.SeedSequence([20260822, 740000]))
        draws = np.empty((n, len(FAILURE_RATES)))
        for r in range(n):
            draws[r] = phase_copula_lifetimes(rng, FAILURE_RATES, "gaussian",
                                              tau_early, tau_late, PHASE_BOUNDARY)
        recovered = 1.0 / draws.mean(axis=0)
        rel = np.abs(recovered - FAILURE_RATES) / FAILURE_RATES
        rows.append({"tau_early": tau_early, "tau_late": tau_late, "n": n,
                     "boundary": PHASE_BOUNDARY,
                     "max_rel_rate_error": float(rel.max()),
                     "mean_rel_rate_error": float(rel.mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phase_marginal_validation.csv", index=False)
    return table


def frailty_marginal_diagnostic(sigmas=FRAILTY_DIAGNOSTIC_SIGMAS, w_persistent=FRAILTY_W):
    """How far the frailty marginals drift as the stress scale is raised.

    For each sigma this records the dependence the mechanism induces and the
    sup-norm distance between the recalibrated survival function and the
    nominal exponential, together with the ratio of the recalibrated baseline
    rates to the nominal ones.  It is the evidence for restricting the frailty
    sweep to FRAILTY_TAU_TARGETS: the recalibration matches survival at the
    horizon by construction, but the interior error grows with sigma, so at the
    stress levels needed to reach the copula arm's upper tau values the two
    arms would no longer share a marginal.
    """
    rates = FAILURE_RATES
    rows = []
    for sigma in sigmas:
        baseline, err = calibrate_frailty_baseline(sigma, w_persistent)
        ratio = np.asarray(baseline) / np.asarray(rates)
        rows.append({"sigma": sigma, "w_persistent": w_persistent,
                     "tau_induced": frailty_tau_from_sigma(sigma, w_persistent),
                     "max_marginal_abs_error": err,
                     "baseline_ratio_min": float(ratio.min()),
                     "baseline_ratio_max": float(ratio.max()),
                     "within_tolerance": bool(err <= MARGINAL_ERROR_TOL)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "frailty_marginal_diagnostic.csv", index=False)
    return table


def frailty_calibration_table(targets=FRAILTY_TAU_TARGETS, w_persistent=FRAILTY_W):
    """Solve sigma for each target tau and recalibrate the baselines.

    Each grid point is admitted only if the recalibrated marginal stays within
    MARGINAL_ERROR_TOL of the nominal exponential over the whole horizon, so
    every reported frailty result is a comparison at matched marginals and
    matched dependence.  A target that cannot meet the tolerance is dropped
    rather than reported with a distorted marginal.
    """
    rows = []
    for tau in targets:
        sigma = solve_sigma_for_tau(tau, w_persistent)
        baseline, marginal_err = calibrate_frailty_baseline(sigma, w_persistent)
        if marginal_err > MARGINAL_ERROR_TOL:
            print(f"[frailty] tau={tau:.3f} (sigma={sigma:.3f}) dropped: "
                  f"marginal error {marginal_err:.4f} > {MARGINAL_ERROR_TOL}")
            continue
        rows.append({"tau_target": tau, "sigma": sigma,
                     "tau_realised": frailty_tau_from_sigma(sigma, w_persistent),
                     "w_persistent": w_persistent,
                     "max_marginal_abs_error": marginal_err,
                     "baseline": baseline})
    table = pd.DataFrame(rows)
    table.drop(columns=["baseline"]).to_csv(OUT / "frailty_calibration.csv", index=False)
    return table


def run_frailty_sweep(n_runs, max_steps=MAX_STEPS, workers=None, calibration=None):
    """Entry point of the frailty arm -- NOT USED BY ANY REPORTED RESULT.

    main() never calls this, and no number in the paper depends on it; see the
    block comment above frailty_lifetimes for why the arm is reported as a
    formulation rather than an evaluated result.  Calling it writes
    frailty_sweep_runs.csv and frailty_sweep_summary.csv, which are not part of
    the deposited experiment_outputs/ set.
    """
    cal = frailty_calibration_table() if calibration is None else calibration
    jobs = []
    for _, r in cal.iterrows():
        jobs.extend([(r.sigma, r.w_persistent, tuple(r.baseline), r.tau_target, run, max_steps)
                     for run in range(n_runs)])
    with Pool(processes=workers or WORKERS) as pool:
        rows = pool.map(frailty_job, jobs, chunksize=200)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "frailty_sweep_runs.csv", index=False)

    base = raw[raw.tau_target == 0.0].set_index("run")["RMST"].sort_index()
    lookup = cal.set_index("tau_target")
    out = []
    for tau, g in raw.groupby("tau_target"):
        s = summarize(g["RMST"].to_numpy())
        d = (g.set_index("run")["RMST"].sort_index() - base).to_numpy()
        dse = d.std(ddof=1) / math.sqrt(len(d))
        out.append({"tau_target": tau, "sigma": float(lookup.loc[tau, "sigma"]),
                    "tau_realised": float(lookup.loc[tau, "tau_realised"]),
                    "max_marginal_abs_error": float(lookup.loc[tau, "max_marginal_abs_error"]),
                    "RMST": s["mean"], "SE": s["se"], "CI_low": s["ci_low"], "CI_high": s["ci_high"],
                    "R_horizon": g["R_horizon"].mean(),
                    "delta_RMST_vs_independence": float(d.mean()),
                    "delta_CI_low": float(d.mean() - 1.96 * dse),
                    "delta_CI_high": float(d.mean() + 1.96 * dse)})
    summary = pd.DataFrame(out)
    summary.to_csv(OUT / "frailty_sweep_summary.csv", index=False)
    return raw, summary


PHASE_BOUNDARY_GRID = (20, 40, 80)
PHASE_CORNERS = ((0.0, 0.0), (0.59, 0.0), (0.0, 0.59), (0.59, 0.59))


def run_phase_boundary_check(n_runs, boundaries=PHASE_BOUNDARY_GRID,
                             corners=PHASE_CORNERS, max_steps=MAX_STEPS, workers=None):
    """Is the early-over-late asymmetry an artefact of where the boundary sits?

    The main sweep fixes the phase boundary at the median first-passage time
    under independence.  That is a defensible choice but it is still a choice,
    so the corners of the grid are re-evaluated at an earlier and a later
    boundary.  The comparison of interest is the pair
    (tau_early, tau_late) = (0.59, 0) against (0, 0.59): if the ordering and
    the disjointness of the intervals survive all three boundaries, the
    asymmetry is a property of the mechanism rather than of t_1.

    Replications are paired within each boundary, since the (0, 0) arm is
    itself boundary-dependent at the realisation level even though its
    distribution is not.
    """
    jobs = [(te, tl, b, run, max_steps) for b in boundaries for (te, tl) in corners
            for run in range(n_runs)]
    with Pool(processes=workers or WORKERS) as pool:
        rows = pool.map(phase_job, jobs, chunksize=200)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "phase_boundary_runs.csv", index=False)

    out = []
    for b, gb in raw.groupby("boundary"):
        base = gb[(gb.tau_early == 0.0) & (gb.tau_late == 0.0)]
        base = base.set_index("run")["RMST"].sort_index()
        for (te, tl), g in gb.groupby(["tau_early", "tau_late"]):
            s = summarize(g["RMST"].to_numpy())
            d = (g.set_index("run")["RMST"].sort_index() - base).to_numpy()
            dse = d.std(ddof=1) / math.sqrt(len(d))
            out.append({"boundary": b, "tau_early": te, "tau_late": tl,
                        "RMST": s["mean"], "SE": s["se"],
                        "delta_RMST_vs_independence": float(d.mean()),
                        "delta_CI_low": float(d.mean() - 1.96 * dse),
                        "delta_CI_high": float(d.mean() + 1.96 * dse)})
    summary = pd.DataFrame(out)

    # The asymmetry test, one row per boundary.
    tests = []
    idx = summary.set_index(["boundary", "tau_early", "tau_late"])
    for b in boundaries:
        e = idx.loc[(b, 0.59, 0.0)]
        l = idx.loc[(b, 0.0, 0.59)]
        tests.append({"boundary": b,
                      "early_only_delta": float(e.delta_RMST_vs_independence),
                      "early_only_CI_low": float(e.delta_CI_low),
                      "early_only_CI_high": float(e.delta_CI_high),
                      "late_only_delta": float(l.delta_RMST_vs_independence),
                      "late_only_CI_low": float(l.delta_CI_low),
                      "late_only_CI_high": float(l.delta_CI_high),
                      "ratio_early_over_late": float(e.delta_RMST_vs_independence
                                                     / l.delta_RMST_vs_independence),
                      "intervals_disjoint": bool(e.delta_CI_low > l.delta_CI_high)})
    test = pd.DataFrame(tests)
    summary.to_csv(OUT / "phase_boundary_summary.csv", index=False)
    test.to_csv(OUT / "phase_boundary_asymmetry.csv", index=False)
    return raw, summary, test


def run_phase_sweep(n_runs, max_steps=MAX_STEPS, workers=None):
    jobs = [(te, tl, PHASE_BOUNDARY, run, max_steps)
            for te in PHASE_TAU_GRID for tl in PHASE_TAU_GRID
            for run in range(n_runs)]
    with Pool(processes=workers or WORKERS) as pool:
        rows = pool.map(phase_job, jobs, chunksize=200)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "phase_sweep_runs.csv", index=False)

    key = (raw.tau_early == 0.0) & (raw.tau_late == 0.0)
    base = raw[key].set_index("run")["RMST"].sort_index()
    out = []
    for (te, tl), g in raw.groupby(["tau_early", "tau_late"]):
        s = summarize(g["RMST"].to_numpy())
        d = (g.set_index("run")["RMST"].sort_index() - base).to_numpy()
        dse = d.std(ddof=1) / math.sqrt(len(d))
        out.append({"tau_early": te, "tau_late": tl, "RMST": s["mean"], "SE": s["se"],
                    "CI_low": s["ci_low"], "CI_high": s["ci_high"],
                    "R_horizon": g["R_horizon"].mean(),
                    "delta_RMST_vs_independence": float(d.mean()),
                    "delta_CI_low": float(d.mean() - 1.96 * dse),
                    "delta_CI_high": float(d.mean() + 1.96 * dse),
                    "boundary": PHASE_BOUNDARY})
    summary = pd.DataFrame(out)
    summary.to_csv(OUT / "phase_sweep_summary.csv", index=False)
    return raw, summary


def kendall_tau(rho):
    return 2.0 / math.pi * math.asin(rho)


def correlation_job(args):
    family, rho, df, run, max_steps = args
    scenario = {"mode": "S3", "copula": family, "rho": rho, "df": df}
    traj = simulate_trajectory(scenario, run, max_steps=max_steps)
    ev = evaluate_trajectory(traj, DEFAULT_K)
    return {"family": family, "rho": rho, "df": df, "run": run,
            "RMST": float(ev["survival"].sum()), "R_horizon": float(ev["survival"][-1]),
            "TTF": ev["ttf"]}


def correlation_sweep_configs():
    """Dependence-strength grid evaluated at the case-study threshold.

    Gaussian, Clayton and Gumbel at rho=0 reduce exactly to the independence
    copula, so those points reuse the independence arm rather than being
    recomputed.  The Student-t copula does not: at rho=0 it retains tail
    dependence through the shared chi-square mixing variable, so it is
    evaluated explicitly.
    """
    cfgs = []
    for family in ("gaussian", "clayton", "gumbel"):
        for rho in RHO_GRID:
            cfgs.append({"family": family, "rho": rho, "df": 3})
    # The Student-t family is evaluated on the full rho x nu grid so that the
    # interaction between dependence strength and tail weight can be tested,
    # rather than inferred from a single rho.
    for df in DF_GRID:
        for rho in RHO_GRID:
            cfgs.append({"family": "student_t", "rho": rho, "df": df})
    return cfgs


def run_correlation_sweep(n_runs, max_steps=MAX_STEPS, threshold_raw=None, workers=None):
    """RMST as a function of dependence strength at K = DEFAULT_K.

    Reuses the threshold sweep for the rho=0.6, df=3 configurations and the
    independence arm, both of which it has already produced on the same
    replication identifiers, so only the remaining grid points are simulated.
    """
    cached = {}
    if threshold_raw is not None:
        base = threshold_raw[threshold_raw["K"] == DEFAULT_K]
        for family, g in base.groupby("copula"):
            cached[family] = g.set_index("run")["RMST"].sort_index()

    indep = cached.get("independence")
    rows, jobs = [], []
    for cfg in correlation_sweep_configs():
        family, rho, df = cfg["family"], cfg["rho"], cfg["df"]
        reuse = None
        if rho == 0.6 and df == 3 and family in cached:
            reuse = cached[family]
        elif rho == 0.0 and family in ("gaussian", "clayton", "gumbel") and indep is not None:
            reuse = indep
        if reuse is not None:
            for run, val in reuse.items():
                rows.append({"family": family, "rho": rho, "df": df, "run": int(run),
                             "RMST": float(val), "R_horizon": np.nan, "TTF": float(val)})
        else:
            jobs.extend([(family, rho, df, run, max_steps) for run in range(n_runs)])

    if jobs:
        with Pool(processes=workers or WORKERS) as pool:
            rows.extend(pool.map(correlation_job, jobs, chunksize=200))

    if indep is not None:
        for run, val in indep.items():
            rows.append({"family": "independence", "rho": 0.0, "df": np.nan, "run": int(run),
                         "RMST": float(val), "R_horizon": np.nan, "TTF": float(val)})

    raw = pd.DataFrame(rows)
    raw["kendall_tau"] = raw["rho"].map(kendall_tau)
    raw.to_csv(OUT / "correlation_sweep_runs.csv", index=False)

    recs = []
    for (family, rho, df), g in raw.groupby(["family", "rho", "df"], dropna=False):
        st = summarize(g["RMST"].to_numpy())
        rec = {"family": family, "rho": rho, "df": df, "kendall_tau": kendall_tau(rho),
               "RMST": st["mean"], "SE": st["se"], "CI_low": st["ci_low"], "CI_high": st["ci_high"]}
        if indep is not None and family != "independence":
            delta = g.set_index("run")["RMST"].sort_index() - indep
            ds = summarize(delta.to_numpy())
            rec.update({"delta_RMST_vs_independence": ds["mean"], "delta_SE": ds["se"],
                        "delta_CI_low": ds["ci_low"], "delta_CI_high": ds["ci_high"]})
        recs.append(rec)
    summary = pd.DataFrame(recs).sort_values(["family", "df", "rho"])
    summary.to_csv(OUT / "correlation_sweep_summary.csv", index=False)
    return raw, summary


def validate_geometry(alphas=(2.0, 2.5, 3.0), steps=(0, 30, 60, 90, MAX_STEPS - 1)):
    """Geometric diagnostics for the link-model boundary condition.

    The prescribed trajectory is a homothety: every agent's position is a fixed
    vector scaled by a common time-dependent factor.  Under a homothety all
    pairwise distances scale together, so the ratio of received signal power to
    co-channel interference from the swarm's own transmitters is invariant in
    time for any path-loss exponent, while the signal-to-noise ratio degrades
    with separation.  This routine checks the homothety numerically and reports
    the resulting scale invariance.  It is a geometric identity check on the
    mobility model, not a mission simulation.
    """
    rng = np.random.default_rng(np.random.SeedSequence([20260822, 0]).spawn(4)[0])
    n = 12
    speed = rng.uniform(0.9, 1.1, size=n)
    angle_offset = rng.uniform(-0.1, 0.1, size=n)
    angles = np.arange(n) * (2 * math.pi / n) + angle_offset

    def layout(step):
        radius = (20.0 + step * 2.5) * speed
        return np.c_[radius * np.cos(angles), radius * np.sin(angles)]

    def mean_sir(pts, alpha):
        vals = []
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                sig = np.linalg.norm(pts[i] - pts[j]) ** (-alpha)
                interf = sum(np.linalg.norm(pts[l] - pts[j]) ** (-alpha)
                             for l in range(n) if l not in (i, j))
                vals.append(sig / interf)
        return float(np.mean(vals))

    def mean_pairwise(pts):
        return float(np.mean([np.linalg.norm(pts[i] - pts[j])
                              for i in range(n) for j in range(i + 1, n)]))

    ref = layout(0)
    rows = []
    for step in steps:
        pts = layout(step)
        ratio = pts / ref
        rows.append({
            "step": step,
            "mean_pairwise_distance_m": mean_pairwise(pts),
            "scale_vs_step0": mean_pairwise(pts) / mean_pairwise(ref),
            "homothety_residual": float(np.std(ratio) / abs(np.mean(ratio))),
            **{f"mean_SIR_alpha_{a}": mean_sir(pts, a) for a in alphas},
        })
    out = pd.DataFrame(rows)
    for a in alphas:
        col = f"mean_SIR_alpha_{a}"
        out[f"SIR_ratio_vs_step0_alpha_{a}"] = out[col] / out[col].iloc[0]
    rates = np.array([0.005] * 3 + [0.008] * 5 + [0.012] * 4)
    out["expected_active_transmitters"] = [float(np.exp(-rates * s).sum()) for s in steps]
    out.to_csv(OUT / "geometry_scale_invariance.csv", index=False)
    return out


def simultaneous_bands(raw, bootstrap_reps=1000):
    """Compute paired max-t bands across all threshold values.

    The same replication id is used for every copula and threshold, so the
    dependence curves are compared with paired resampling rather than with
    independent confidence intervals.
    """
    rng = np.random.default_rng(20260822 + 77)
    runs = np.sort(raw["run"].unique())
    n = len(runs)
    index = {run: i for i, run in enumerate(runs)}
    ks = np.sort(raw["K"].unique())
    copulas = list(COPULAS)
    cube = np.empty((len(copulas), n, len(ks)), dtype=float)
    for ci, name in enumerate(copulas):
        part = raw[raw["copula"] == name].pivot(index="run", columns="K", values="RMST")
        cube[ci] = part.loc[runs, ks].to_numpy()

    observed = cube.mean(axis=1)
    se = cube.std(axis=1, ddof=1) / math.sqrt(n)
    boot_max = np.zeros((len(copulas), bootstrap_reps))
    for b in range(bootstrap_reps):
        sample = rng.integers(0, n, size=n)
        boot_mean = cube[:, sample, :].mean(axis=1)
        boot_max[:, b] = np.max(np.abs((boot_mean - observed) / np.maximum(se, 1e-12)), axis=1)

    rows = []
    for ci, name in enumerate(copulas):
        # Two-sided sup-t band: the statistic is already max |t|, so the
        # 1 - alpha quantile gives 100(1 - alpha)% simultaneous coverage.
        crit = float(np.quantile(boot_max[ci], SUP_T_LEVEL))
        low = observed[ci] - crit * se[ci]
        high = observed[ci] + crit * se[ci]
        for j, k in enumerate(ks):
            rows.append({"quantity": "RMST", "copula": name, "K": int(k), "kappa": k / W_MAX,
                         "estimate": observed[ci, j], "SE": se[ci, j], "band_low": low[j],
                         "band_high": high[j], "critical_value": crit})

    base = cube[0]
    for ci, name in enumerate(copulas[1:], start=1):
        delta = cube[ci] - base
        observed_delta = delta.mean(axis=0)
        delta_se = delta.std(axis=0, ddof=1) / math.sqrt(n)
        max_values = np.empty(bootstrap_reps)
        for b in range(bootstrap_reps):
            sample = rng.integers(0, n, size=n)
            boot_delta = delta[sample].mean(axis=0)
            max_values[b] = np.max(np.abs((boot_delta - observed_delta) / np.maximum(delta_se, 1e-12)))
        crit = float(np.quantile(max_values, SUP_T_LEVEL))
        low = observed_delta - crit * delta_se
        high = observed_delta + crit * delta_se
        for j, k in enumerate(ks):
            rows.append({"quantity": "Delta_RMST_vs_independence", "copula": name, "K": int(k), "kappa": k / W_MAX,
                         "estimate": observed_delta[j], "SE": delta_se[j], "band_low": low[j],
                         "band_high": high[j], "critical_value": crit})
    bands = pd.DataFrame(rows)
    bands.to_csv(OUT / "threshold_sweep_simultaneous_bands.csv", index=False)
    return bands


# Categorical palette (Okabe-Ito subset) in fixed assignment order.  Hues are
# bound to the copula family, not to rank, so a figure that plots a subset keeps
# each family's colour.  Adjacent-pair CVD separation was checked for this order;
# line style and marker carry the identity a second time for print and for
# colour-vision deficiency.
FAMILY_STYLE = {
    "independence": {"color": "#0072B2", "marker": "o", "ls": "-", "label": "Independence"},
    "gaussian": {"color": "#D55E00", "marker": "s", "ls": "--", "label": "Gaussian"},
    "student_t": {"color": "#009E73", "marker": "^", "ls": "-.", "label": "Student-$t$"},
    "clayton": {"color": "#E69F00", "marker": "D", "ls": ":", "label": "Clayton"},
    "gumbel": {"color": "#CC79A7", "marker": "v", "ls": (0, (3, 1, 1, 1)), "label": "Gumbel"},
}


def _style(name):
    return FAMILY_STYLE.get(name, {"color": "#555555", "marker": "o", "ls": "-", "label": name})


def correlation_bands(raw, bootstrap_reps=1000):
    """Paired max-t bands across the dependence grid.

    Two curve sets are produced.  The family set fixes nu_t=3 and compares the
    four dependent families; the tail set fixes the Student-t family and
    compares degrees of freedom.  Each curve is banded simultaneously over its
    own tau grid, so a curve is separated from another only where its band and
    the comparison exclude one another.
    """
    rng = np.random.default_rng(20260822 + 91)
    curves = []
    for family, g in raw[raw["df"] == 3].groupby("family"):
        if family != "independence":
            curves.append((f"family:{family}", g))
    # Tail-weight contrasts are banded on the paired difference from the
    # heaviest-reference arm nu_t=30, because the levels themselves are almost
    # coincident and the contrast is what the panel has to resolve.
    ref = raw[(raw["family"] == "student_t") & (raw["df"] == 30)]
    ref_piv = ref.pivot_table(index="run", columns="rho", values="RMST")
    for df, g in raw[raw["family"] == "student_t"].groupby("df"):
        if int(df) == 30:
            continue
        piv = g.pivot_table(index="run", columns="rho", values="RMST")
        cols = np.sort(np.intersect1d(piv.columns.to_numpy(), ref_piv.columns.to_numpy()))
        diff = piv[cols] - ref_piv[cols]
        curves.append((f"student_t_dnu:{int(df)}", diff.reset_index().melt(
            id_vars="run", var_name="rho", value_name="RMST")))

    rows = []
    for label, g in curves:
        piv = g.pivot_table(index="run", columns="rho", values="RMST")
        rhos = np.sort(piv.columns.to_numpy())
        cube = piv[rhos].to_numpy()
        n = cube.shape[0]
        observed = cube.mean(axis=0)
        se = cube.std(axis=0, ddof=1) / math.sqrt(n)
        stats = np.empty(bootstrap_reps)
        for b in range(bootstrap_reps):
            boot = cube[rng.integers(0, n, size=n)].mean(axis=0)
            stats[b] = np.max(np.abs((boot - observed) / np.maximum(se, 1e-12)))
        crit = float(np.quantile(stats, SUP_T_LEVEL))
        kind, name = label.split(":", 1)
        for j, rho in enumerate(rhos):
            rows.append({"curve_set": kind, "series": name, "rho": float(rho),
                         "kendall_tau": kendall_tau(float(rho)), "estimate": observed[j],
                         "SE": se[j], "band_low": observed[j] - crit * se[j],
                         "band_high": observed[j] + crit * se[j], "critical_value": crit})
    bands = pd.DataFrame(rows)
    bands.to_csv(OUT / "correlation_sweep_simultaneous_bands.csv", index=False)
    return bands


def crossover_threshold(raw, bootstrap_reps=2000):
    """Threshold at which the paired dependence effect changes sign.

    The effect on RMST is negative at low capacity thresholds and positive at
    high ones, so it must pass through zero.  In a neighbourhood of that
    crossing no finite sample can resolve the sign, which is why the pointwise
    and simultaneous bands both contain zero there.  Rather than report the
    resulting interval as an absence of information, the crossing itself is
    estimated: the mean paired difference is interpolated linearly between the
    two attainable thresholds that bracket the sign change, and the whole
    procedure is bootstrapped over replication identifiers.
    """
    rng = np.random.default_rng(20260822 + 101)
    ks = np.sort(raw["K"].unique())
    base = raw[raw["copula"] == "independence"].pivot(index="run", columns="K", values="RMST")

    def crossing(matrix):
        mean = matrix.mean(axis=0)
        idx = np.where(np.diff(np.sign(mean)) > 0)[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        k0, k1, y0, y1 = ks[i], ks[i + 1], mean[i], mean[i + 1]
        return (k0 + (k1 - k0) * (-y0) / (y1 - y0)) / W_MAX

    rows = []
    for name in COPULAS:
        if name == "independence":
            continue
        piv = raw[raw["copula"] == name].pivot(index="run", columns="K", values="RMST")
        delta = (piv[ks] - base[ks]).to_numpy()
        observed = crossing(delta)
        boot = np.array([crossing(delta[rng.integers(0, len(delta), len(delta))])
                         for _ in range(bootstrap_reps)])
        # A resample in which the mean difference never changes sign yields no
        # crossing.  Those draws are dropped, so the interval is conditional on
        # a sign change existing; the share dropped is recorded alongside it.
        n_valid = int(np.count_nonzero(~np.isnan(boot)))
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({"copula": name, "kappa_star": observed, "kappa_star_CI_low": low,
                     "kappa_star_CI_high": high, "K_star": observed * W_MAX,
                     "bootstrap_reps": int(bootstrap_reps), "bootstrap_valid": n_valid,
                     "bootstrap_no_crossing_pct": 100.0 * (bootstrap_reps - n_valid) / bootstrap_reps})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "threshold_crossover.csv", index=False)
    return out


def design_zones(raw, bands, crossover):
    """Tabulate the dependence-effect zones used as the design guideline.

    A threshold belongs to a signed zone only when the simultaneous band for
    every dependent family excludes zero with a common sign; thresholds where
    any band contains zero form the unresolved region around the sign change.
    """
    ks = np.sort(raw["K"].unique())
    means = raw.groupby(["copula", "K"]).RMST.mean().unstack(0)
    base = means["independence"]
    fams = [c for c in COPULAS if c != "independence"]
    delta = means[fams].sub(base, axis=0)
    delta_bands = bands[bands["quantity"] == "Delta_RMST_vs_independence"]

    labels = {}
    for k in ks:
        sub = delta_bands[delta_bands["K"] == k]
        if sub.empty:
            continue
        if (sub["band_low"] > 0).all():
            labels[k] = "beneficial"
        elif (sub["band_high"] < 0).all():
            labels[k] = "detrimental"
        else:
            labels[k] = "unresolved"

    rows = []
    for zone in ("detrimental", "unresolved", "beneficial"):
        members = [k for k in ks if labels.get(k) == zone]
        if not members:
            continue
        sel = delta.loc[members]
        rel = sel.div(base.loc[members], axis=0) * 100.0
        rows.append({
            "zone": zone,
            "K_low": int(min(members)), "K_high": int(max(members)),
            "kappa_low": min(members) / W_MAX, "kappa_high": max(members) / W_MAX,
            "delta_RMST_min": float(sel.min().min()), "delta_RMST_max": float(sel.max().max()),
            "relative_min_pct": float(rel.min().min()), "relative_max_pct": float(rel.max().max()),
            "independence_assumption": {"detrimental": "non-conservative",
                                        "beneficial": "conservative",
                                        "unresolved": "undetermined"}[zone],
        })
    out = pd.DataFrame(rows)
    out["kappa_star_mean"] = float(crossover["kappa_star"].mean())
    out.to_csv(OUT / "design_zones.csv", index=False)
    return out


def make_dependence_figure(phase_summary, boundary_test):
    """Figure 10: the phase-switching copula and its boundary sensitivity.

    Panel (a) is the (tau_early, tau_late) map at the nominal boundary.  Panel
    (b) is the check that the early-over-late asymmetry in panel (a) is a
    property of the mechanism rather than of where the boundary was placed:
    the two single-phase corners are re-evaluated at an earlier and a later
    boundary, and the intervals are plotted side by side.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    grid = sorted(phase_summary.tau_early.unique())
    late = sorted(phase_summary.tau_late.unique())
    mat = np.full((len(grid), len(late)), np.nan)
    piv = phase_summary.set_index(["tau_early", "tau_late"])["delta_RMST_vs_independence"]
    for i, te in enumerate(grid):
        for j, tl in enumerate(late):
            if (te, tl) in piv.index:
                mat[i, j] = float(piv.loc[(te, tl)])
    im = axes[0].imshow(mat, origin="lower", cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(late)), [f"{v:.2f}" for v in late])
    axes[0].set_yticks(range(len(grid)), [f"{v:.2f}" for v in grid])
    for i in range(len(grid)):
        for j in range(len(late)):
            if not math.isnan(mat[i, j]):
                shade = "white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black"
                axes[0].text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                             fontsize=8, color=shade)
    axes[0].set(xlabel=r"$\tau_{\mathrm{late}}$", ylabel=r"$\tau_{\mathrm{early}}$",
                title=r"(a) Phase-switching copula: $\Delta$RMST (steps)")
    # No colorbar label: the panel title already names the quantity, and a
    # label here collides with panel (b)'s y-axis label.
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    t = boundary_test.sort_values("boundary")
    x = np.arange(len(t))
    width = 0.34
    for offset, key, colour, marker, label in (
            (-width / 2, "early_only", "#00634A", "o", r"early phase only ($\tau_{\mathrm{early}}=0.59$)"),
            (+width / 2, "late_only", "#8C4A00", "s", r"late phase only ($\tau_{\mathrm{late}}=0.59$)")):
        mid = t[f"{key}_delta"].to_numpy()
        lo = t[f"{key}_CI_low"].to_numpy()
        hi = t[f"{key}_CI_high"].to_numpy()
        axes[1].errorbar(x + offset, mid, yerr=[mid - lo, hi - mid], fmt=marker,
                         color=colour, markersize=6, capsize=4, elinewidth=1.1,
                         linestyle="none", label=label)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, [str(int(b)) for b in t.boundary])
    axes[1].set(xlabel=r"Phase boundary $t_1$ (steps)",
                ylabel=r"$\Delta$RMST vs. independence (steps)",
                title=r"(b) Boundary sensitivity of the phase asymmetry")
    axes[1].grid(alpha=0.22, linewidth=0.6, axis="y")
    # Headroom for a horizontal legend: at t_1 = 20 the late-phase point sits
    # near the top of the data range and a corner legend would cover it.
    axes[1].margins(y=0.30)
    axes[1].legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
                   handletextpad=0.4, columnspacing=1.4)

    fig.savefig(OUT / "Figure_10_time_varying_dependence.pdf", dpi=300)
    fig.savefig(OUT / "Figure_10_time_varying_dependence.png", dpi=300)
    plt.close(fig)


def make_correlation_figure(raw, summary, bands=None):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    indep = summary[summary.family == "independence"]
    base = float(indep.RMST.iloc[0]) if len(indep) else None

    grid = summary[(summary.df == 3) & (summary.family != "independence")]
    for family in ("gaussian", "student_t", "clayton", "gumbel"):
        g = grid[grid.family == family].sort_values("kendall_tau")
        if g.empty:
            continue
        st = _style(family)
        axes[0].plot(g.kendall_tau, g.RMST, color=st["color"], linestyle=st["ls"],
                     marker=st["marker"], markersize=5, linewidth=1.6, label=st["label"])
        if bands is not None:
            bd = bands[(bands.curve_set == "family") & (bands.series == family)].sort_values("kendall_tau")
            if not bd.empty:
                axes[0].fill_between(bd.kendall_tau, bd.band_low, bd.band_high,
                                     color=st["color"], alpha=0.12, linewidth=0)
    if base is not None:
        axes[0].axhline(base, color="#444444", linewidth=0.9, linestyle=(0, (4, 3)))
        axes[0].annotate("Independence", xy=(0.015, base), xytext=(0, 4),
                         textcoords="offset points", fontsize=8, color="#444444")
    axes[0].set(xlabel=r"Kendall's $\tau$", ylabel=r"RMST$(T)$ (steps)",
                title=r"(a) Dependence strength at $K=18$")
    axes[0].grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(fontsize=8, frameon=False)

    nu_shades = {3: "#00634A", 5: "#009E73", 10: "#7FD3B4"}
    nu_marks = {3: "^", 5: "o", 10: "s"}
    nu_ls = {3: "-", 5: "--", 10: "-."}
    tail = summary[summary.family == "student_t"]
    ref = tail[tail.df == 30].set_index("rho")["RMST"]
    for df in (3, 5, 10):
        g = tail[tail.df == df].sort_values("rho")
        if g.empty:
            continue
        y = g.set_index("rho")["RMST"] - ref
        axes[1].plot([kendall_tau(v) for v in y.index], y.values,
                     color=nu_shades[df], linestyle=nu_ls[df], marker=nu_marks[df],
                     markersize=5, linewidth=1.6, label=rf"$\nu_t={df}$")
        if bands is not None:
            bd = bands[(bands.curve_set == "student_t_dnu") & (bands.series == str(df))]
            bd = bd.sort_values("kendall_tau")
            if not bd.empty:
                axes[1].fill_between(bd.kendall_tau, bd.band_low, bd.band_high,
                                     color=nu_shades[df], alpha=0.13, linewidth=0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel=r"Kendall's $\tau$",
                ylabel=r"$\Delta$RMST vs. $\nu_t=30$ (steps)",
                title=r"(b) Student-$t$ tail weight, paired contrast")
    axes[1].grid(alpha=0.22, linewidth=0.6)
    axes[1].legend(fontsize=8, frameon=False)

    fig.savefig(OUT / "Figure_8_correlation_sweep.pdf", dpi=300)
    fig.savefig(OUT / "Figure_8_correlation_sweep.png", dpi=300)
    plt.close(fig)


def make_figures(raw, summary, bands=None):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    rmst_bands = bands[bands.quantity == "RMST"] if bands is not None else None
    for name in ("independence", "gaussian", "student_t", "clayton", "gumbel"):
        group = summary[summary.copula == name].sort_values("kappa")
        if group.empty:
            continue
        st = _style(name)
        axes[0].plot(group.kappa, group.RMST, color=st["color"], linestyle=st["ls"],
                     linewidth=1.6, label=st["label"])
        if rmst_bands is not None:
            band = rmst_bands[rmst_bands.copula == name].sort_values("kappa")
            axes[0].fill_between(band.kappa, band.band_low, band.band_high,
                                 color=st["color"], alpha=0.12, linewidth=0)
        else:
            axes[0].fill_between(group.kappa, group.CI_low, group.CI_high,
                                 color=st["color"], alpha=0.12, linewidth=0)
    axes[0].set(xlabel=r"Normalized threshold $\kappa=K/W_{\max}$", ylabel=r"RMST$(T)$ (steps)", title="(a) Finite-horizon survival")
    axes[0].grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    base = raw[raw.copula == "independence"].groupby("K").RMST.mean()
    delta_bands = bands[bands.quantity == "Delta_RMST_vs_independence"] if bands is not None else None
    for name in ("gaussian", "student_t", "clayton", "gumbel"):
        group = raw[raw.copula == name]
        if group.empty:
            continue
        st = _style(name)
        means = group.groupby("K").RMST.mean() - base
        axes[1].plot(means.index / W_MAX, means.values, color=st["color"],
                     linestyle=st["ls"], linewidth=1.6, label=st["label"])
        if delta_bands is not None:
            band = delta_bands[delta_bands.copula == name].sort_values("kappa")
            axes[1].fill_between(band.kappa, band.band_low, band.band_high,
                                 color=st["color"], alpha=0.12, linewidth=0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel=r"Normalized threshold $\kappa$", ylabel=r"$\Delta$RMST vs. independence", title="(b) Paired dependence effect")
    axes[1].grid(alpha=0.22, linewidth=0.6)
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "Figure_9_threshold_sensitivity.pdf", dpi=300)
    fig.savefig(OUT / "Figure_9_threshold_sensitivity.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------
# Monte Carlo engine verification against the closed-form benchmark
# ---------------------------------------------------------------
VERIFY_REPS = (100, 500, 1000, 2500, 5000)
VERIFY_STEP = 60
FAILURE_RATES = np.array([0.005] * 3 + [0.008] * 5 + [0.012] * 4)


def analytic_survival(k=DEFAULT_K, max_steps=MAX_STEPS):
    """Closed-form reliability of the degenerate S2 configuration.

    In S2 the swarm holds a tight fixed formation, so the connectivity and
    delay constraints are non-binding and the mission event reduces to the
    capacity constraint (Proposition 1).  With independent exponential
    lifetimes the survival probability is the sum over weight-feasible
    subsets of the agent set.
    """
    weights = swarm_definition()
    n = len(weights)
    masks = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)
    feasible = masks[masks @ weights >= k]
    steps = np.arange(max_steps)
    alive_p = np.exp(-np.outer(steps, FAILURE_RATES))
    curve = np.array([
        float(np.sum(np.prod(np.where(feasible, alive_p[s], 1.0 - alive_p[s]), axis=1)))
        for s in steps
    ])
    return steps, curve, int(feasible.shape[0]), int(masks.shape[0])


def verify_job(run):
    traj = simulate_trajectory({"mode": "S2", "copula": "independence"}, run)
    return evaluate_trajectory(traj, DEFAULT_K)["survival"].astype(np.int8)


def verify_engine(max_steps=MAX_STEPS, reps=VERIFY_REPS):
    """Compare the simulation engine against Eq. (analytic benchmark)."""
    OUT.mkdir(parents=True, exist_ok=True)
    steps, analytic, n_feasible, n_subsets = analytic_survival(max_steps=max_steps)
    with Pool(processes=WORKERS) as pool:
        surv = np.array(pool.map(verify_job, range(max(reps))))
    rmst_a = float(analytic.sum())
    curves, rows = {}, []
    for m in reps:
        mc = surv[:m].mean(axis=0)
        curves[m] = mc
        rows.append({
            "N_rep": m,
            "R_mc_t60": float(mc[VERIFY_STEP]),
            "R_analytic_t60": float(analytic[VERIFY_STEP]),
            "abs_error_t60": float(abs(mc[VERIFY_STEP] - analytic[VERIFY_STEP])),
            "max_abs_error": float(np.max(np.abs(mc - analytic))),
            "RMST_mc": float(mc.sum()),
            "RMST_analytic": rmst_a,
            "RMST_rel_error_pct": float(100.0 * abs(mc.sum() - rmst_a) / rmst_a),
            "n_feasible_subsets": n_feasible,
            "n_subsets": n_subsets,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "analytic_benchmark.csv", index=False)
    pd.DataFrame({"step": steps, "R_analytic": analytic,
                  **{f"R_mc_{m}": curves[m] for m in reps}}).to_csv(
        OUT / "analytic_benchmark_curves.csv", index=False)
    make_verification_figure(steps, analytic, curves, table)
    return table


def make_verification_figure(steps, analytic, curves, table):
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    shades = {100: "#CC79A7", 500: "#E69F00", 1000: "#009E73",
              2500: "#56B4E9", 5000: "#0072B2"}
    dashes = {100: ":", 500: (0, (3, 1, 1, 1)), 1000: "-.", 2500: "--", 5000: "-"}
    for m in sorted(curves):
        axes[0].plot(steps, curves[m], color=shades[m], linestyle=dashes[m],
                     linewidth=1.4, label=rf"MC, $N_{{\mathrm{{rep}}}}={m:,}$")
    axes[0].plot(steps, analytic, color="#000000", linewidth=2.0, alpha=0.85,
                 label="Analytical benchmark")
    axes[0].axvline(VERIFY_STEP, color="#777777", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].set(xlabel="Mission step $t$", ylabel=r"$R(t)$",
                title="(a) Survival curve vs. closed form (S2, independence)")
    axes[0].grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(table.N_rep, table.abs_error_t60, color="#0072B2", marker="o",
                 markersize=5, linewidth=1.6, label=rf"$|\hat R_{{MC}}({VERIFY_STEP})-R_{{an}}({VERIFY_STEP})|$")
    axes[1].plot(table.N_rep, table.max_abs_error, color="#D55E00", marker="s",
                 markersize=5, linewidth=1.6, linestyle="--",
                 label=r"$\sup_t |\hat R_{MC}(t)-R_{an}(t)|$")
    axes[1].axhline(0.01, color="#444444", linewidth=0.9, linestyle=(0, (4, 3)))
    axes[1].annotate("0.01", xy=(table.N_rep.iloc[0], 0.01), xytext=(2, 4),
                     textcoords="offset points", fontsize=8, color="#444444")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(xlabel=r"$N_{\mathrm{rep}}$", ylabel="Absolute error",
                title="(b) Convergence of the estimator")
    axes[1].grid(alpha=0.22, linewidth=0.6, which="both")
    axes[1].legend(fontsize=8, frameon=False)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure_2_verification.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_output(name):
    """Read a committed output file, transparently handling the gzipped copies.

    The four largest run-level CSVs are stored gzipped in the repository; a
    freshly executed run writes them uncompressed.  Either form is accepted.
    """
    plain, packed = OUT / name, OUT / f"{name}.gz"
    if plain.exists():
        return pd.read_csv(plain)
    if packed.exists():
        return pd.read_csv(packed, compression="gzip")
    raise FileNotFoundError(f"{plain} (or {packed}) not found; run the full suite first")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dependence-only", action="store_true")
    parser.add_argument("--boundary-only", action="store_true")
    args = parser.parse_args()
    n = 100 if args.smoke else args.runs
    OUT.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        raw = read_output("threshold_sweep_runs.csv")
        summary = pd.read_csv(OUT / "threshold_sweep_summary.csv")
        bands = simultaneous_bands(raw)
        make_figures(raw, summary, bands)
        print(bands.head())
        return
    if args.verify_only:
        print(verify_engine(max_steps=args.steps))
        return
    if args.boundary_only:
        # Re-uses the stored phase map; only the boundary corners are simulated.
        _, _, boundary_test = run_phase_boundary_check(n, max_steps=args.steps)
        make_dependence_figure(pd.read_csv(OUT / "phase_sweep_summary.csv"), boundary_test)
        print(boundary_test.to_string())
        return
    if args.dependence_only:
        print(validate_phase_marginals(20000 if args.smoke else 200000))
        _, phase_summary = run_phase_sweep(n, max_steps=args.steps)
        _, _, boundary_test = run_phase_boundary_check(n, max_steps=args.steps)
        make_dependence_figure(phase_summary, boundary_test)
        print(phase_summary)
        print(boundary_test)
        return
    if args.edge_only:
        _, edge_summary = run_edge_sensitivity(n, max_steps=args.steps)
        print(edge_summary)
        return
    raw, summary = run_threshold_sweep(n, max_steps=args.steps)
    edge_raw, edge_summary = run_edge_sensitivity(n, max_steps=args.steps)
    main_raw, main_summary = run_main_scenarios(n, max_steps=args.steps, threshold_raw=raw)
    validation = validate_copulas(1000 if args.smoke else 10000)
    benchmark = verify_engine(max_steps=args.steps)
    geometry = validate_geometry()
    bands = simultaneous_bands(raw)
    crossover = crossover_threshold(raw)
    zones = design_zones(raw, bands, crossover)
    make_figures(raw, summary, bands)
    # The dependence-strength sweep behind Figure 8 and the time-varying arms
    # behind Figure 10.  These belong in the default run: without them the
    # command does not reproduce every figure in the paper.
    corr_raw, corr_summary = run_correlation_sweep(n, max_steps=args.steps, threshold_raw=raw)
    corr_bands = correlation_bands(corr_raw)
    make_correlation_figure(corr_raw, corr_summary, corr_bands)
    phase_marginals = validate_phase_marginals(20000 if args.smoke else 200000)
    _, phase_summary = run_phase_sweep(n, max_steps=args.steps)
    _, _, boundary_test = run_phase_boundary_check(n, max_steps=args.steps)
    make_dependence_figure(phase_summary, boundary_test)
    metadata = {"runs": n, "steps": args.steps, "W_max": W_MAX, "K_values": list(range(2, W_MAX + 1)), "default_K": DEFAULT_K, "copulas": list(COPULAS), "edge_models": ["bernoulli", "deterministic_q0_0.3", "deterministic_q0_0.5", "deterministic_q0_0.7"], "seed_scheme": "SeedSequence([20260822, run_id])", "metric": "RMST(T)=sum first-passage survival indicators; R_horizon=survival at final epoch"}
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.head())
    print(edge_summary)
    print(main_summary)
    print(validation)
    print(benchmark)
    print(geometry)
    print(crossover)
    print(zones)
    print(corr_summary.head())
    print(phase_marginals)
    print(phase_summary.head())
    print(boundary_test)


if __name__ == "__main__":
    main()
