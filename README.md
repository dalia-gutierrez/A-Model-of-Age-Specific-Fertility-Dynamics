# A Model of Age Specific Fertility Dynamics
Python code that reproduces the working paper "A Model of Age-Specific Fertility Dynamics: The
Time Cost of Leisure and the Decline in Fertility", available **[here](https://www.dropbox.com/scl/fi/75pjqm451bs699pd98q90/Age-Specific-Fertility-Dynamics.pdf?rlkey=xp7u4eprfn81v3zkrmh1am7at&st=b31xmj6j&dl=0)**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-red)
![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Project Overview

This repository implements a continuous-time overlapping generations (OLG) model with endogenous fertility, human capital accumulation, and demographic dynamics. The model simulates age-specific fertility schedules (n(a)) and computes the Total Fertility Rate (TFR) and population growth rate (g_n) in general equilibrium.

This work is part of an ongoing academic research paper exploring the reasons for declining fertility.

---

## Authors
- **[Dalia Gutiérrez Valencia](www.linkedin.com/in/dalia-scherazada-gutiérrez-valencia-5b7202253)**
- Angélica Tan Jun Ríos
- P. Andrés Neumeyer

---

## Contents

- `leisure_fertility_model.py`       Equilibrium solver for the continuous-time
                                     overlapping-generations leisure-fertility
                                     model (Section 3 of the paper). Solves the
                                     household boundary-value problem, the
                                     cohort-renewal condition for the population
                                     growth rate g_n, and asset-market clearing
                                     for the interest rate r.
- `leisure_fertility_sensitivity.py` Driver. Solves the baseline, the seven
                                     +/-10% comparative-statics perturbations,
                                     and the four 1975->2023 decomposition
                                     scenarios (A, B, C, D).
- `TFR_all.py`                       Post-processing. Computes the total
                                     fertility rate for each scenario.
- `initial_guess_leisure.xlsx`,
  `resultados.xlsx`                  Initial guesses for the boundary-value-
                                     problem solver (warm starts). If absent the
                                     code falls back to a synthetic guess.

## Requirements

Python 3.9+ with numpy, scipy, pandas, openpyxl and matplotlib:

    pip install numpy scipy pandas openpyxl matplotlib

## How to reproduce the results

Run, in order, from inside this folder:

    python leisure_fertility_model.py        # baseline equilibrium: r*, g_n*, TFR*
    python leisure_fertility_sensitivity.py  # baseline + 7 perturbations + 4 decomposition scenarios
    python TFR_all.py                        # TFR for every scenario

Outputs written to this folder:

- `equilibrium_v5_sensitivity.csv`   r* and g_n* for every scenario.
- `trajectories_v5_sensitivity.csv`  Life-cycle profiles (c, l, h, b, o, n) per scenario.
- `TFR_per_scenario.csv`             TFR for every scenario.

These feed Table 1 (calibration), Table 2 (elasticities) and Table 3
(decomposition) of the paper, as well as the model-result figures.

## Notes

- The solver uses a continuation/homotopy method: a cold solve walks a ladder
  of the pi-floor parameter; warm solves reuse a cached or supplied guess.
- The Gamma_1 calibration draws on the American Time Use Survey (ATUS, 2003-2024)
  Activity Summary file. That public BLS microdata file is large (~250 MB) and is
  not bundled here; it can be downloaded from https://www.bls.gov/tus/.
