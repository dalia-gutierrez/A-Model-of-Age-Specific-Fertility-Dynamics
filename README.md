# A Model of Age Specific Fertility Dynamics
Python code that reproduces the working paper "A Model of Age-Specific Fertility Dynamics: The
Time Cost of Leisure and the Decline in Fertility", available **[here](https://www.dropbox.com/scl/fi/0jpqu0p2iz2vr0kt4z95u/Age-Specific-Fertility-Dynamics.pdf?rlkey=jrsvjp1pq3avgxki7puyenl5e&st=dlzzo3mq&dl=0)**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
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
                                     for the interest rate r. The default
                                     parameters are the 1990 calibration
                                     (Gamma_1=1.616, Gamma_2(0)=0.324, chi=1.51).
- `leisure_fertility_sensitivity.py` Driver. Solves the baseline, the seven
                                     +/-10% comparative-statics perturbations,
                                     and the eight 1990->2023 decomposition
                                     vertices (A, B, C, D, BC, BD, CD, E)
                                     spanning the cube in (Gamma_1, Gamma_2,
                                     chi); solving all eight makes the Shapley
                                     contributions in Table 3 exact.
- `TFR_all.py`                       Post-processing. Computes the total
                                     fertility rate for each scenario.
- `psibeta_grid.py`                  Robustness exercise: solves the model
                                     over a 3x3 grid of the CES curvature
                                     parameters (psi, beta), reports the TFR
                                     elasticity of chi at each grid point.
                                     Used in the limitations section to show
                                     the sign of the leisure channel is robust.
- `psibeta_grid_results.csv`         Output of `psibeta_grid.py`.
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
    python leisure_fertility_sensitivity.py  # baseline + 7 perturbations + 8 decomposition vertices
    python TFR_all.py                        # TFR for every scenario
    python psibeta_grid.py run               # psi/beta robustness grid; rerun until 'ALL DONE'
    python psibeta_grid.py report            # tabulate the grid

Outputs written to this folder:

- `equilibrium_v5_sensitivity.csv`   r* and g_n* for every scenario.
- `trajectories_v5_sensitivity.csv`  Life-cycle profiles (c, l, h, b, o, n) per scenario.
- `TFR_per_scenario.csv`             TFR for every scenario.
- `psibeta_grid_results.csv`         psi/beta grid output.

These feed Table 1 (calibration), Table 2 (elasticities) and Table 3
(decomposition) of the paper, as well as the model-result figures.

## Notes

- The model is anchored to 1990 as the pre-decline steady state and decomposed
  against 2023; the post-1990 plateau is the empirical regularity the paper
  asks the model to explain (see Section 4 of the paper).
- The solver uses a continuation/homotopy method: a cold solve walks a ladder
  of the pi-floor parameter; warm solves reuse a cached or supplied guess.
- The Gamma_1 calibration draws on the time-use diaries of Bianchi, Robinson
  and Milkie (2006, Tables 4.1 and 5.1, the 1985 and 1995 U.S. surveys
  averaged to bracket 1990) for the 1990 anchor, and on the American Time Use
  Survey (ATUS, 2003-2024) Activity Summary file for 2023. The ATUS file is
  a public BLS microdata product (~250 MB), not bundled here; download it
  from https://www.bls.gov/tus/. The Gamma_1 formula uses a fixed
  intensive-margin completed-family-size denominator N=1.80, held constant
  across calibration years (see Section 4 for the rationale).
- The Gamma_2 calibration uses the USDA "Expenditures on a Child by Families"
  series: the 1995 annual report (data from the 1990-92 Consumer Expenditure
  Survey) for the 1990 anchor and the 2017 release for 2023; both totals are
  the USDA birth-through-age-17 figure (college is excluded from both years
  for consistency).
