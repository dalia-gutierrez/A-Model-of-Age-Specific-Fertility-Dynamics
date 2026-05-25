"""
TFR_all.py
==========
Compute the Total Fertility Rate (TFR) for every scenario in the sensitivity-
analysis output.

Inputs (the new CSV outputs from leisure_fertility_sensitivity.py):
    trajectories_v5_sensitivity.csv   -- age grid `a` and columns
                                         n_<scenario> for each scenario.
    equilibrium_v5_sensitivity.csv    -- one row per scenario with r_star
                                         and g_n_star.

For each scenario the script:
  1. Pulls n_<scenario> from the trajectories file.
  2. Looks up g_n_star for that scenario in the equilibrium file
     (no more manual entry).
  3. Computes
        TFR = 10 * sum_{a in {15,20,25,30,35,40,45}}
                    ( int_a^{a+5} n(s) e^{-g_n s} ds
                      / int_a^{a+5} e^{-g_n s} ds )
     which is the stable-population weighted average of age-specific
     fertility over 5-year reproductive-age bins, summed over the seven
     bins covering [15, 50).  The factor 5 is the bin width (5 years),
     consistent with the standard TFR definition.

Output:
    TFR_per_scenario.csv -- one row per scenario with TFR, g_n_star,
                            and the perturbed parameter.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import integrate, interpolate

# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------
TRAJ_CSV = "trajectories_v5_sensitivity.csv"
EQ_CSV   = "equilibrium_v5_sensitivity.csv"
OUT_CSV  = "TFR_per_scenario.csv"

# Reproductive-age bins (5-year increments, ages 15-49).
A_BINS = [15, 20, 25, 30, 35, 40, 45]


def numerator(s, n_interp, g_n):
    return n_interp(s) * np.exp(-g_n * s)


def denominator(s, g_n):
    return np.exp(-g_n * s)


def tfr_one(a_grid, n_grid, g_n):
    """TFR for a single (n(a), g_n) pair using the same formula as the
    original TFR_all.py: 10 * sum over 5-year bins of E[n | bin] weighted
    by the stable-age weight e^{-g_n s}."""
    n_interp = interpolate.interp1d(a_grid, n_grid,
                                    kind='linear', fill_value='extrapolate')
    total = 0.0
    for a in A_BINS:
        num, _ = integrate.quad(lambda s: numerator(s, n_interp, g_n), a, a + 5)
        den, _ = integrate.quad(lambda s: denominator(s, g_n), a, a + 5)
        total += num / den
    return 5.0 * total


def main():
    if not os.path.exists(TRAJ_CSV):
        print(f"ERROR: {TRAJ_CSV} not found.  Run leisure_fertility_sensitivity.py first.",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(EQ_CSV):
        print(f"ERROR: {EQ_CSV} not found.  Run leisure_fertility_sensitivity.py first.",
              file=sys.stderr)
        sys.exit(1)

    traj = pd.read_csv(TRAJ_CSV)
    eq   = pd.read_csv(EQ_CSV)

    if 'a' not in traj.columns:
        raise ValueError(f"{TRAJ_CSV} must have an 'a' column for the age grid.")
    a_grid = traj['a'].to_numpy()

    # Map scenario -> g_n_star (and perturbed_parameter, baseline/new values).
    eq_index = eq.set_index('scenario')

    rows = []
    for scenario in eq_index.index:
        col = f"n_{scenario}"
        if col not in traj.columns:
            print(f"  [skip] {col} not in trajectories CSV")
            continue
        n_grid = traj[col].to_numpy()
        g_n = float(eq_index.loc[scenario, 'g_n_star'])
        r   = float(eq_index.loc[scenario, 'r_star'])

        tfr = tfr_one(a_grid, n_grid, g_n)
        rows.append({
            'scenario':            scenario,
            'perturbed_parameter': eq_index.loc[scenario, 'perturbed_parameter'],
            'baseline_value':      eq_index.loc[scenario, 'baseline_value'],
            'new_value':           eq_index.loc[scenario, 'new_value'],
            'g_n_star':            g_n,
            'r_star':              r,
            'TFR':                 tfr,
        })
        print(f"  {scenario:>15s}  g_n={g_n:.6f}  TFR={tfr:.4f}")

    if not rows:
        print("No matching scenarios found.", file=sys.stderr)
        sys.exit(2)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out_df)} scenarios)")
    print()
    print(out_df[['scenario', 'perturbed_parameter',
                  'g_n_star', 'TFR']].to_string(index=False))


if __name__ == "__main__":
    main()
