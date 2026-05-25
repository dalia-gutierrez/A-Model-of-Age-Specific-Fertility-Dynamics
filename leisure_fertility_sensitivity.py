"""
leisure_fertility_sensitivity.py
================================
Sensitivity-analysis driver for leisure_fertility_model.

Solves the model at the baseline parameters (taken verbatim from
leisure_fertility_model.params) and at six perturbations:

    Gamma_1  +10%
    Gamma_2  +10%
    rho      -10%
    chi      -10%
    z        +10%
    psi      +10%

For each scenario the equilibrium (r*, g_n*) is found and the full age
profiles (c, l, h, b, o, n, work_time) are recorded.

Outputs (in the workspace folder):
    trajectories_v5_sensitivity.csv  -- one row per age grid point,
        with each variable's column suffixed by the scenario name,
        e.g. c_baseline, c_Gamma_1_up10, n_psi_up10, ...
    equilibrium_v5_sensitivity.csv -- one row per scenario, with r*,
        g_n*, the perturbed parameter, its baseline & new values.

The script does NOT modify leisure_fertility_model.py.  It imports its functions
and reuses the same BVP solver, only rebuilding the params dict per run
and clearing the BVP cache between scenarios so warm starts don't leak.
"""

import numpy as np
import pandas as pd

import leisure_fertility_model as mod


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each entry: (scenario_label, param_name, multiplier).  baseline is special.
SCENARIOS = [
    ("baseline",      None,        1.0),
    ("Gamma_1_up10",  "Gamma_1",   1.10),
    ("Gamma_2_up10",  "Gamma_2",   1.10),
    ("rho_dn10",      "rho",       0.90),
    ("chi_dn10",      "chi",       0.90),
    ("z_up10",        "z",         1.10),
    ("psi_up10",      "psi",       1.10),
    ("g_dn10",        "g",         0.90),
]

# Decomposition scenarios for the 1990 -> 2023 exercise (Section 4.4 of the
# working paper).  Each entry is (label, dict of parameter overrides).
# This is a three-channel decomposition: Gamma_1 (broad time cost of children),
# Gamma_2 (goods cost of children) and chi (time cost of leisure) all move from
# their 1990 to their 2023 calibrated values; pi is held at its biological value.
# All eight vertices of the (Gamma_1, Gamma_2, chi) cube are solved so the
# Shapley contributions are exact.  Scenario A reproduces TFR_1990 = 2.08;
# scenario E reproduces TFR_2023 = 1.62.
# 1990 values: Gamma_1 = 1.644, Gamma_2 = 0.324, chi = 1.54.
# 2023 values: Gamma_1 = 1.854, Gamma_2 = 0.278, chi = 1.045.
DECOMPOSITION_SCENARIOS = [
    ("decomp_A_1990",      {"Gamma_1": 1.644, "Gamma_2": 0.324, "chi": 1.54}),
    ("decomp_B_only_G1",   {"Gamma_1": 1.854, "Gamma_2": 0.324, "chi": 1.54}),
    ("decomp_C_only_G2",   {"Gamma_1": 1.644, "Gamma_2": 0.278, "chi": 1.54}),
    ("decomp_D_only_chi",  {"Gamma_1": 1.644, "Gamma_2": 0.324, "chi": 1.045}),
    ("decomp_BC_G1_G2",    {"Gamma_1": 1.854, "Gamma_2": 0.278, "chi": 1.54}),
    ("decomp_BD_G1_chi",   {"Gamma_1": 1.854, "Gamma_2": 0.324, "chi": 1.045}),
    ("decomp_CD_G2_chi",   {"Gamma_1": 1.644, "Gamma_2": 0.278, "chi": 1.045}),
    ("decomp_E_2023_full", {"Gamma_1": 1.854, "Gamma_2": 0.278, "chi": 1.045}),
]


def make_params(param_name, multiplier):
    """Return a fresh perturbed params dict (deep-copied from baseline)."""
    p = mod.params.copy()
    if param_name is not None:
        p[param_name] = mod.params[param_name] * multiplier
    return p


def make_params_overrides(overrides):
    """Return a fresh params dict with the given overrides applied."""
    p = mod.params.copy()
    for k, v in overrides.items():
        p[k] = v
    return p


def solve_scenario(label, param_name, multiplier, r_bounds=(0.050, 0.075)):
    """Solve one scenario.  Returns (r_eq, g_n_eq, sol, p) or raises."""
    p = make_params(param_name, multiplier)
    if param_name is None:
        print(f"\n=== Solving scenario '{label}' (baseline) ===")
    else:
        print(f"\n=== Solving scenario '{label}' "
              f"({param_name}: {mod.params[param_name]:.6g} -> {p[param_name]:.6g}) ===")

    # Fresh BVP cache so warm-starts from a different scenario don't bias the search.
    mod.clear_bvp_cache()

    # Pre-warm the cache from the baseline scenario's converged solution if present.
    # This avoids paying the homotopy cost again on every perturbation.
    if param_name is not None:
        seed_data = load_scenario_result("baseline")
        if seed_data is not None:
            try:
                _seed_bvp_from_baseline(seed_data, p)
            except Exception as e:
                print(f"  (warm-start from baseline failed: {e})")

    r_eq = mod.find_equilibrium_r(p, r_bounds=r_bounds)
    if r_eq is None:
        raise RuntimeError(f"find_equilibrium_r returned None for scenario '{label}'.")
    sol = mod.get_sol_cached(r_eq, p)
    if sol is None or not sol.success:
        raise RuntimeError(f"BVP failed at r*={r_eq} for scenario '{label}'.")
    g_n_eq = mod.find_g_n(sol, p)
    if g_n_eq is None:
        raise RuntimeError(f"find_g_n returned None for scenario '{label}'.")
    print(f"   r* = {r_eq:.6f},  g_n* = {g_n_eq:.6f}")
    return r_eq, g_n_eq, sol, p


def _seed_bvp_from_baseline(seed_data, p):
    """Inject baseline's converged solution into the underlying BVP cache so
    Strategy 1 (direct warm-start) succeeds on the first call."""
    from types import SimpleNamespace
    a_arr = np.asarray(seed_data["a"])
    y = np.array([seed_data["c"], seed_data["l"], seed_data["h"],
                  seed_data["b"], seed_data["o"]], dtype=float)
    fake_sol = SimpleNamespace(x=a_arr, y=y, success=True)
    r_seed = float(seed_data["r_star"])
    mod._bvp_cache[round(r_seed, 10)] = fake_sol


def trajectory_dict(sol, p):
    """Map BVP solution -> dict of age-indexed series for the CSV."""
    a_grid = sol.x
    c, l, h, b, o = sol.y
    n = mod.n_profile_from_sol(sol, p)
    work_time = 1.0 - l - p["Gamma_1"] * n - p["chi"] * o
    return {
        "a":          a_grid,
        "c":          c,
        "l":          l,
        "h":          h,
        "b":          b,
        "o":          o,
        "n":          n,
        "work_time":  work_time,
    }


# ---------------------------------------------------------------------------
# Persistence helpers (allow running scenarios one at a time across calls)
# ---------------------------------------------------------------------------
import json
import os
from scipy.interpolate import interp1d

CACHE_DIR = "sensitivity_cache"


def _cache_path(label):
    return os.path.join(CACHE_DIR, f"{label}.json")


def save_scenario_result(label, r_eq, g_n_eq, sol, p, param_name, mult):
    os.makedirs(CACHE_DIR, exist_ok=True)
    a = np.asarray(sol.x).tolist()
    y = np.asarray(sol.y).tolist()  # 5 x N
    n = mod.n_profile_from_sol(sol, p).tolist()
    payload = {
        "label":               label,
        "perturbed_parameter": param_name if param_name is not None else "(none)",
        "multiplier":          mult,
        "baseline_value":      mod.params.get(param_name) if param_name is not None else None,
        "new_value":           p.get(param_name) if param_name is not None else None,
        "r_star":              float(r_eq),
        "g_n_star":            float(g_n_eq),
        "a":                   a,
        "c":                   y[0],
        "l":                   y[1],
        "h":                   y[2],
        "b":                   y[3],
        "o":                   y[4],
        "n":                   n,
        "Gamma_1":             p["Gamma_1"],
        "chi":                 p["chi"],
    }
    with open(_cache_path(label), "w") as f:
        json.dump(payload, f)


def load_scenario_result(label):
    path = _cache_path(label)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def run_scenario_to_disk(label, param_name, mult):
    """Solve one scenario and persist its result.  Skip if already cached."""
    if load_scenario_result(label) is not None:
        print(f"[{label}] already cached, skipping.")
        return
    r_eq, g_n_eq, sol, p = solve_scenario(label, param_name, mult)
    save_scenario_result(label, r_eq, g_n_eq, sol, p, param_name, mult)
    print(f"[{label}] saved to {_cache_path(label)}")


def solve_decomp_scenario(label, overrides, r_bounds=(0.040, 0.090)):
    """Solve a decomposition scenario with a dict of parameter overrides.
    Used for the 1975->2015 counterfactual exercise described in Section 4.4
    of the working paper."""
    p = make_params_overrides(overrides)
    print(f"\n=== Solving decomposition scenario '{label}' "
          f"({', '.join(f'{k}={v:.4f}' for k,v in overrides.items())}) ===")
    mod.clear_bvp_cache()
    seed_data = load_scenario_result("baseline")
    if seed_data is not None:
        try:
            _seed_bvp_from_baseline(seed_data, p)
        except Exception as e:
            print(f"  (warm-start from baseline failed: {e})")
    r_eq = mod.find_equilibrium_r(p, r_bounds=r_bounds)
    if r_eq is None:
        raise RuntimeError(f"find_equilibrium_r returned None for scenario '{label}'.")
    sol = mod.get_sol_cached(r_eq, p)
    if sol is None or not sol.success:
        raise RuntimeError(f"BVP failed at r*={r_eq} for scenario '{label}'.")
    g_n_eq = mod.find_g_n(sol, p)
    print(f"   r* = {r_eq:.6f},  g_n* = {g_n_eq:.6f}")
    return r_eq, g_n_eq, sol, p


def run_decomp_scenario_to_disk(label, overrides):
    """Solve one decomposition scenario and persist its result."""
    if load_scenario_result(label) is not None:
        print(f"[{label}] already cached, skipping.")
        return
    r_eq, g_n_eq, sol, p = solve_decomp_scenario(label, overrides)
    # Store with "perturbed_parameter" = list of overridden params, "multiplier" = None
    save_scenario_result(label, r_eq, g_n_eq, sol, p,
                         param_name=",".join(overrides.keys()),
                         mult=None)
    print(f"[{label}] saved to {_cache_path(label)}")


def combine_to_csv():
    """Combine all cached scenario JSONs into the two output CSVs."""
    a_common = np.linspace(0.0, mod.params["T"], 200)
    traj_table = {"a": a_common}
    eq_rows = []

    for label, param_name, mult in SCENARIOS:
        data = load_scenario_result(label)
        if data is None:
            print(f"[combine] missing scenario '{label}' -- skipping")
            continue
        a_arr = np.asarray(data["a"])
        Gamma_1 = data["Gamma_1"]
        chi = data["chi"]
        for var in ("c", "l", "h", "b", "o", "n"):
            arr = np.asarray(data[var])
            f = interp1d(a_arr, arr, kind="linear", fill_value="extrapolate")
            traj_table[f"{var}_{label}"] = f(a_common)
        traj_table[f"work_time_{label}"] = (
            1.0 - traj_table[f"l_{label}"]
            - Gamma_1 * traj_table[f"n_{label}"]
            - chi    * traj_table[f"o_{label}"]
        )

        eq_rows.append({
            "scenario":            label,
            "perturbed_parameter": data["perturbed_parameter"],
            "multiplier":          data["multiplier"],
            "baseline_value":      data["baseline_value"],
            "new_value":           data["new_value"],
            "r_star":              data["r_star"],
            "g_n_star":            data["g_n_star"],
            "min_work_time":       float(np.min(traj_table[f"work_time_{label}"])),
            "max_n":               float(np.max(traj_table[f"n_{label}"])),
        })

    traj_df = pd.DataFrame(traj_table)
    eq_df = pd.DataFrame(eq_rows)
    traj_csv = "trajectories_v5_sensitivity.csv"
    eq_csv = "equilibrium_v5_sensitivity.csv"
    traj_df.to_csv(traj_csv, index=False)
    eq_df.to_csv(eq_csv, index=False)
    print(f"\nTrajectories saved: {traj_csv}  "
          f"({traj_df.shape[0]} rows x {traj_df.shape[1]} cols)")
    print(f"Equilibrium summary saved: {eq_csv}")
    print()
    print(eq_df[["scenario", "perturbed_parameter",
                 "baseline_value", "new_value",
                 "r_star", "g_n_star"]].to_string(index=False))


def main(argv=None):
    """CLI dispatcher.

    No args     -> solve every scenario that isn't cached yet, then combine.
    <label>     -> solve only that scenario (e.g. 'baseline', 'rho_dn10').
    combine     -> rebuild the CSVs from cached scenario JSONs.
    reset       -> wipe the cache directory.
    """
    import sys
    np.random.seed(123)
    argv = sys.argv if argv is None else argv

    if len(argv) >= 2 and argv[1] == "reset":
        import shutil
        if os.path.isdir(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        print(f"Removed {CACHE_DIR}")
        return

    if len(argv) >= 2 and argv[1] == "combine":
        combine_to_csv()
        return

    if len(argv) >= 2 and argv[1] == "decomp":
        # Run only the decomposition scenarios (1975->2015 exercise).
        for label, overrides in DECOMPOSITION_SCENARIOS:
            run_decomp_scenario_to_disk(label, overrides)
        return

    if len(argv) >= 2:
        target = argv[1]
        match = [s for s in SCENARIOS if s[0] == target]
        if match:
            label, pname, mult = match[0]
            run_scenario_to_disk(label, pname, mult)
            return
        dmatch = [s for s in DECOMPOSITION_SCENARIOS if s[0] == target]
        if dmatch:
            label, overrides = dmatch[0]
            run_decomp_scenario_to_disk(label, overrides)
            return
        raise SystemExit(f"Unknown scenario {target!r}. "
                         f"Choices: {[s[0] for s in SCENARIOS] + [s[0] for s in DECOMPOSITION_SCENARIOS]}")

    for label, pname, mult in SCENARIOS:
        run_scenario_to_disk(label, pname, mult)
    combine_to_csv()


if __name__ == "__main__":
    main()
