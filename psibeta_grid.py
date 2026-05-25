"""
psi/beta robustness grid for the leisure-fertility model.

The CES-aggregator curvature parameters psi and beta are calibrated, not
estimated.  This script checks that the central qualitative result -- a fall
in the time cost of leisure chi lowers the TFR (a positive TFR-elasticity of
chi) -- is robust across a grid of (psi, beta).  At each grid point the model
is solved at the 1990 baseline (chi = 1.54) and at chi -10% (chi = 1.386),
holding Gamma_1 = 1.644 and Gamma_2 = 0.324, and the TFR-elasticity of chi is
reported.  Resumable: caches each grid point.  Run repeatedly, then 'report'.

Usage:  python psibeta_grid.py run     (solve uncached points for ~38s)
        python psibeta_grid.py report  (print the grid table)
"""
import warnings, os, sys, json, time
warnings.filterwarnings("ignore")
os.chdir("/sessions/determined-busy-cerf/mnt/population_growth"); sys.path.insert(0, os.getcwd())
import numpy as np
import leisure_fertility_model as mod
from scipy import integrate, interpolate

OUT = "/sessions/determined-busy-cerf/mnt/outputs"
CACHE = OUT + "/psibeta_cache"
os.makedirs(CACHE, exist_ok=True)
A_BINS = [15, 20, 25, 30, 35, 40, 45]
RB = (0.035, 0.105)
PSIS = [0.45, 0.50, 0.55]
BETAS = [0.60, 0.70, 0.80]
CHI_BASE, CHI_LO = 1.54, 1.386          # chi and chi -10%


def tfr(a, n, g):
    f = interpolate.interp1d(a, n, kind='linear', fill_value='extrapolate')
    return 5.0 * sum(integrate.quad(lambda s: f(s) * np.exp(-g * s), x, x + 5)[0] /
                     integrate.quad(lambda s: np.exp(-g * s), x, x + 5)[0] for x in A_BINS)


def solve_tfr(psi, beta, chi):
    p = mod.params.copy()
    p["psi"], p["beta"], p["chi"] = psi, beta, chi
    p["Gamma_1"], p["Gamma_2"] = 1.644, 0.324
    mod.clear_bvp_cache()
    r = mod.find_equilibrium_r(p, r_bounds=RB)
    sol = mod.get_sol_cached(r, p)
    if sol is None or not sol.success:
        raise RuntimeError("BVP failed")
    g = mod.find_g_n(sol, p)
    n = mod.n_profile_from_sol(sol, p)
    return float(tfr(sol.x, n, g))


def run():
    t0 = time.time()
    for psi in PSIS:
        for beta in BETAS:
            tag = f"psi{psi}_beta{beta}"
            fp = CACHE + f"/{tag}.json"
            if os.path.exists(fp):
                continue
            if time.time() - t0 > 38:
                print("time budget reached; rerun"); return
            rec = {"psi": psi, "beta": beta}
            try:
                tb = solve_tfr(psi, beta, CHI_BASE)
                tl = solve_tfr(psi, beta, CHI_LO)
                eps = ((tl - tb) / tb) / ((CHI_LO - CHI_BASE) / CHI_BASE)
                rec.update(TFR_base=tb, TFR_chi_lo=tl, eps_TFR_chi=eps, ok=True)
            except Exception as e:
                rec.update(ok=False, err=str(e))
            json.dump(rec, open(fp, "w"))
            print(f"{tag}: {rec}")
    print("ALL DONE" if all(os.path.exists(CACHE + f"/psi{p}_beta{b}.json")
                             for p in PSIS for b in BETAS) else "incomplete")


def report():
    print(f"{'psi':>6} {'beta':>6} {'TFR(chi=1.54)':>14} {'TFR(chi=1.386)':>16} {'eps_TFR_chi':>12}")
    allpos = True
    for psi in PSIS:
        for beta in BETAS:
            fp = CACHE + f"/psi{psi}_beta{beta}.json"
            if not os.path.exists(fp):
                print(f"{psi:>6} {beta:>6}   (not solved)"); continue
            d = json.load(open(fp))
            if not d.get("ok"):
                print(f"{psi:>6} {beta:>6}   FAIL: {d.get('err','')[:50]}"); continue
            print(f"{psi:>6} {beta:>6} {d['TFR_base']:>14.4f} {d['TFR_chi_lo']:>16.4f} "
                  f"{d['eps_TFR_chi']:>12.3f}")
            if d["eps_TFR_chi"] <= 0:
                allpos = False
    print(f"\nLeisure channel sign robust (eps_TFR_chi > 0 at all solved points): {allpos}")


if __name__ == "__main__":
    np.random.seed(123)
    (report if len(sys.argv) > 1 and sys.argv[1] == "report" else run)()
