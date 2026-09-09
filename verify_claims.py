"""Cross-check every figure quoted in the time-varying dependence sections.

Each assertion pairs a number written in Section 3.3, Section 6.7, the caption
of Figure 10 or Concern 3 of the response letter with the CSV cell it came
from.  Run from the simulation directory; a silent exit means every quoted
value matches its source to the precision at which it is quoted.
"""
import sys
import pandas as pd

OUT = "experiment_outputs/"
fails = []


def check(label, quoted, actual, tol):
    if abs(quoted - actual) > tol:
        fails.append(f"{label}: quoted {quoted}, source {actual}")


phase = pd.read_csv(OUT + "phase_sweep_summary.csv").set_index(["tau_early", "tau_late"])
pm = pd.read_csv(OUT + "phase_marginal_validation.csv")
bnd = pd.read_csv(OUT + "phase_boundary_asymmetry.csv").set_index("boundary")
corr = pd.read_csv(OUT + "correlation_sweep_summary.csv")
gauss = corr[(corr.family == "gaussian") & (corr.df == 3)].set_index("rho")
indep = float(corr[corr.family == "independence"].RMST.iloc[0])

# --- Phase arm: stationary limit recovered -----------------------------------
check("phase (0,0) RMST", 42.48, float(phase.loc[(0.0, 0.0), "RMST"]), 0.005)
check("phase (0,0) SE", 0.30, float(phase.loc[(0.0, 0.0), "SE"]), 0.005)
check("independent copula RMST", 42.24, indep, 0.005)

# --- Phase arm: early/late asymmetry ----------------------------------------
check("early-only delta", 10.17, float(phase.loc[(0.59, 0.0), "delta_RMST_vs_independence"]), 0.005)
check("early-only CI low", 9.16, float(phase.loc[(0.59, 0.0), "delta_CI_low"]), 0.005)
check("early-only CI high", 11.18, float(phase.loc[(0.59, 0.0), "delta_CI_high"]), 0.005)
check("late-only delta", 6.33, float(phase.loc[(0.0, 0.59), "delta_RMST_vs_independence"]), 0.005)
check("late-only CI low", 5.85, float(phase.loc[(0.0, 0.59), "delta_CI_low"]), 0.005)
check("late-only CI high", 6.81, float(phase.loc[(0.0, 0.59), "delta_CI_high"]), 0.005)

lo_e = float(phase.loc[(0.59, 0.0), "delta_CI_low"])
hi_l = float(phase.loc[(0.0, 0.59), "delta_CI_high"])
if not lo_e > hi_l:
    fails.append(f"asymmetry not resolved: early CI low {lo_e} <= late CI high {hi_l}")
ratio = float(phase.loc[(0.59, 0.0), "delta_RMST_vs_independence"]) / \
    float(phase.loc[(0.0, 0.59), "delta_RMST_vs_independence"])
check("asymmetry ratio ~1.6", 1.6, ratio, 0.05)

# --- Phase arm: diagonal and maximum ----------------------------------------
check("diagonal 0.41 delta", 11.63, float(phase.loc[(0.41, 0.41), "delta_RMST_vs_independence"]), 0.005)
check("diagonal 0.41 CI low", 10.49, float(phase.loc[(0.41, 0.41), "delta_CI_low"]), 0.005)
check("diagonal 0.41 CI high", 12.77, float(phase.loc[(0.41, 0.41), "delta_CI_high"]), 0.005)
check("stationary Gaussian tau=0.41 delta", 13.06, float(gauss.loc[0.6, "RMST"]) - indep, 0.005)
check("max cell delta", 16.95, float(phase.loc[(0.59, 0.59), "delta_RMST_vs_independence"]), 0.005)
check("max cell CI low", 15.76, float(phase.loc[(0.59, 0.59), "delta_CI_low"]), 0.005)
check("max cell CI high", 18.14, float(phase.loc[(0.59, 0.59), "delta_CI_high"]), 0.005)

max_delta = float(phase.delta_RMST_vs_independence.max())
if max_delta > 18.1:
    fails.append(f"phase max {max_delta} exceeds the stationary 18.1-step envelope")

# every non-null cell resolved
unresolved = [(te, tl) for (te, tl), r in phase.iterrows()
              if not (te == 0.0 and tl == 0.0)
              and r.delta_CI_low <= 0.0 <= r.delta_CI_high]
if unresolved:
    fails.append(f"cells not excluding zero: {unresolved}")
n_cells = len(phase) - 1
if n_cells != 15:
    fails.append(f"quoted 'fifteen non-null cells' but grid has {n_cells}")

# --- Phase arm: marginal exactness ------------------------------------------
check("phase marginal worst error", 0.0046, float(pm.max_rel_rate_error.max()), 0.0001)
if float(pm.n.min()) != 200000:
    fails.append(f"quoted 200,000 draws, file says {int(pm.n.min())}")

# --- Boundary sensitivity: the three quoted pairs ---------------------------
QUOTED = {20: (5.65, 4.84, 6.45, 16.03, 15.14, 16.91),
          40: (10.17, 9.16, 11.18, 6.33, 5.85, 6.81),
          80: (16.32, 15.11, 17.52, 0.15, 0.08, 0.21)}
for b, (ed, elo, ehi, ld, llo, lhi) in QUOTED.items():
    check(f"t1={b} early delta", ed, float(bnd.loc[b, "early_only_delta"]), 0.005)
    check(f"t1={b} early CI low", elo, float(bnd.loc[b, "early_only_CI_low"]), 0.005)
    check(f"t1={b} early CI high", ehi, float(bnd.loc[b, "early_only_CI_high"]), 0.005)
    check(f"t1={b} late delta", ld, float(bnd.loc[b, "late_only_delta"]), 0.005)
    check(f"t1={b} late CI low", llo, float(bnd.loc[b, "late_only_CI_low"]), 0.005)
    check(f"t1={b} late CI high", lhi, float(bnd.loc[b, "late_only_CI_high"]), 0.005)

# The t1 = 40 corners must agree with the main phase map, same seeds.
check("t1=40 early corner matches the map",
      float(phase.loc[(0.59, 0.0), "delta_RMST_vs_independence"]),
      float(bnd.loc[40, "early_only_delta"]), 0.005)
check("t1=40 late corner matches the map",
      float(phase.loc[(0.0, 0.59), "delta_RMST_vs_independence"]),
      float(bnd.loc[0 + 40, "late_only_delta"]), 0.005)

# The claimed reversal at t1 = 20, and disjointness in the reversed direction.
if not bnd.loc[20, "late_only_delta"] > bnd.loc[20, "early_only_delta"]:
    fails.append("the ordering does not actually reverse at t1=20")
if not bnd.loc[20, "late_only_CI_low"] > bnd.loc[20, "early_only_CI_high"]:
    fails.append("the t1=20 reversal is not resolved (intervals overlap)")
for b in (40, 80):
    if not bnd.loc[b, "early_only_CI_low"] > bnd.loc[b, "late_only_CI_high"]:
        fails.append(f"the asymmetry at t1={b} is not resolved")

# Per-unit-time normalisation: the quoted rates and the claimed 1.8-55x range.
HORIZON = 120
RATES = {20: (0.28, 0.16), 40: (0.25, 0.08), 80: (0.20, 0.004)}
ratios = []
for b, (qe, ql) in RATES.items():
    e = float(bnd.loc[b, "early_only_delta"]) / b
    l = float(bnd.loc[b, "late_only_delta"]) / (HORIZON - b)
    check(f"t1={b} early rate", qe, e, 0.005)
    check(f"t1={b} late rate", ql, l, 0.005)
    if not e > l:
        fails.append(f"per-unit-time ordering fails at t1={b}: {e} vs {l}")
    ratios.append(e / l)
if not (1.7 <= min(ratios) <= 1.9):
    fails.append(f"quoted lower ratio 1.8 does not match {min(ratios)}")
if not (50 <= max(ratios) <= 60):
    fails.append(f"quoted upper ratio 55 does not match {max(ratios)}")

# --- Table 2 / Figure 8 values re-quoted in the letter ----------------------
top = corr[(corr.kendall_tau > 0.58) & (corr.family != "independence")].RMST
check("K=18 tau=0.59 max family RMST", 62.47, float(top.max()), 0.005)
check("K=18 tau=0.59 min family RMST", 60.16, float(top.min()), 0.005)

# --- Section 6.5 / 6.6 and Concern 4 of the letter --------------------------
# These bands moved when the sup-t quantile was corrected to 0.95 and when the
# correlation sweep was regenerated.  Both documents quote them, so both are
# checked here.
tb = pd.read_csv(OUT + "threshold_sweep_simultaneous_bands.csv")
delta = tb[tb.quantity == "Delta_RMST_vs_independence"].set_index(["K", "copula"])

check("K=12 Gaussian simultaneous low", -2.76, float(delta.loc[(12, "gaussian"), "band_low"]), 0.005)
check("K=12 Gaussian simultaneous high", 0.08, float(delta.loc[(12, "gaussian"), "band_high"]), 0.005)
if not (delta.loc[(12, "gaussian"), "band_low"] < 0 < delta.loc[(12, "gaussian"), "band_high"]):
    fails.append("the K=12 Gaussian band no longer contains zero, contradicting the text")

c14 = delta.loc[(14, "clayton")]
check("K=14 Clayton simultaneous low", -0.02, float(c14.band_low), 0.005)
check("K=14 Clayton simultaneous high", 2.62, float(c14.band_high), 0.005)
check("K=14 Clayton pointwise low", 0.33, float(c14.estimate - 1.96 * c14.SE), 0.005)
check("K=14 Clayton pointwise high", 2.27, float(c14.estimate + 1.96 * c14.SE), 0.005)
if not (c14.band_low < 0 < c14.band_high and c14.estimate - 1.96 * c14.SE > 0):
    fails.append("the K=14 pointwise-vs-simultaneous contrast no longer holds")

# Zone boundaries and the crossover estimates quoted in Table 4 and Section 6.6
zones = pd.read_csv(OUT + "design_zones.csv").set_index("zone")
check("detrimental K_high", 11, int(zones.loc["detrimental", "K_high"]), 0)
check("beneficial K_low", 15, int(zones.loc["beneficial", "K_low"]), 0)
check("detrimental kappa_high", 0.478, float(zones.loc["detrimental", "kappa_high"]), 0.0005)
check("beneficial kappa_low", 0.652, float(zones.loc["beneficial", "kappa_low"]), 0.0005)
for zone, lo, hi in (("detrimental", -15.6, -2.4), ("unresolved", -1.9, 1.3), ("beneficial", 2.8, 27.3)):
    check(f"{zone} delta min", lo, float(zones.loc[zone, "delta_RMST_min"]), 0.05)
    check(f"{zone} delta max", hi, float(zones.loc[zone, "delta_RMST_max"]), 0.05)

cross = pd.read_csv(OUT + "threshold_crossover.csv").set_index("copula")
for fam, k, lo, hi in (("gaussian", 0.577, 0.538, 0.611), ("student_t", 0.587, 0.553, 0.614),
                       ("clayton", 0.575, 0.548, 0.600), ("gumbel", 0.591, 0.556, 0.619)):
    check(f"kappa* {fam}", k, float(cross.loc[fam, "kappa_star"]), 0.0005)
    check(f"kappa* {fam} CI low", lo, float(cross.loc[fam, "kappa_star_CI_low"]), 0.0005)
    check(f"kappa* {fam} CI high", hi, float(cross.loc[fam, "kappa_star_CI_high"]), 0.0005)

# Student-t tail-weight contrasts quoted in Section 6.6 and in the letter
dnu = pd.read_csv(OUT + "correlation_sweep_simultaneous_bands.csv")
dnu = dnu[dnu.curve_set == "student_t_dnu"]
for nu, tau, est, lo, hi in ((3, 0.0, -0.48, -0.85, -0.12), (5, 0.0, -0.44, -0.71, -0.18),
                             (10, 0.0, -0.22, -0.37, -0.06), (3, 0.409666, 0.14, -0.17, 0.46)):
    row = dnu[(dnu.series.astype(str) == str(nu)) & ((dnu.kendall_tau - tau).abs() < 1e-3)]
    if row.empty:
        fails.append(f"no band row for nu={nu} at tau={tau}")
        continue
    r = row.iloc[0]
    check(f"nu={nu} tau={tau:.2f} estimate", est, float(r.estimate), 0.005)
    check(f"nu={nu} tau={tau:.2f} band low", lo, float(r.band_low), 0.005)
    check(f"nu={nu} tau={tau:.2f} band high", hi, float(r.band_high), 0.005)

# --- The flat figure copies the manuscript actually typesets ----------------
# The preamble resolves bare file names through ./ first, so a stale copy in the
# manuscript directory would be typeset in place of a freshly generated figure
# without any warning from LaTeX.  collect_figures accompanies the manuscript
# and is not part of the published simulation package, so its absence means
# there is no manuscript beside these outputs and the check does not apply.
try:
    import collect_figures  # noqa: E402
except ModuleNotFoundError:
    collect_figures = None

if collect_figures is not None and collect_figures.check() != 0:
    fails.append("the collected figure copies are missing or stale "
                 "(run: python collect_figures.py)")

if fails:
    print("MISMATCHES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all quoted values match their sources")
