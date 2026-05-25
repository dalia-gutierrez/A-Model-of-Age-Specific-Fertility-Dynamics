# Age-Specific-Fertility-rates-A-theoretical-model
Python code that reproduces the working paper "A Model of Age-Specific Fertility Dynamics: Child Costs, Human Capital and Sexual Desire in U.S. Fertility Decline", available at: [https://www.dropbox.com/scl/fi/cm42dejn4zil9xsj1g6ir/A-Model-of-Age-Specific-Fertility-Dynamics.pdf?rlkey=0qujgs2se1ci9o9yd5rrbhyls&st=tcs52b6a&dl=0](https://www.dropbox.com/scl/fi/75pjqm451bs699pd98q90/Age-Specific-Fertility-Dynamics.pdf?rlkey=xp7u4eprfn81v3zkrmh1am7at&st=b31xmj6j&dl=0)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-red)
![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Project Overview

This repository implements a continuous-time overlapping generations (OLG) model with endogenous fertility, human capital accumulation, and demographic dynamics. The model simulates age-specific fertility schedules (n(a)) and computes the Total Fertility Rate (TFR) and population growth rate (g_n) in general equilibrium.

The core innovation is solving a boundary value problem (BVP) for each cohort using scipy.solve_bvp, combined with a continuation method over a regularization parameter (pitol) and a fixed-point iteration to close the model via aggregate consistency conditions. Currently, an inital guess is provided such that the homothopy method is not active.

This work is part of an ongoing academic research paper (co-authored with two collaborators) exploring the reasons for declining fertility.

---

## Authors
- **[Dalia Gutiérrez Valencia](www.linkedin.com/in/dalia-scherazada-gutiérrez-valencia-5b7202253)**
- Angélica Tan Jun Ríos
- P. Andrés Neumeyer

---

## Key Features

| Feature | Description |
- Cohort-level BVP: Solves a 6-dimensional ODE system over lifetime [0, T=70] for consumption, labor, fertility, human capital, bonds, and cumulative births.
- Endogenous Fertility: Fertility n(a) responds to wages, interest rates, child-rearing costs.
- General Equilibrium: Finds equilibrium interest rate r* and population growth g_n* such that bond market clears and population is stationary in growing frame.
- TFR Calculation: Post-processing script (Calculate_TFR.py) computes TFR from simulated n(a) using 5-year age bins and user-input g_n.
- Parallel Computation: Uses multiprocessing to scan over boundary condition parameter bc_epsilon to match terminal fertility condition.
- Robust Numerics: Split integration, interpolation, continuation in pitol, and adaptive grid handling. |

---

## Model Outputs

After convergence:

- Equilibrium interest rate r ≈ 0.04–0.06
- Population growth rate g_n ≈ 0.01–0.02
- TFR ≈ 1.5–2.3 (depending on calibration)
- Full lifecycle profiles: c(a), l(a), n(a), h(a), b(a), N_tilde(a)

Saved to:
- resultados.xlsx → Lifecycle trajectories (input to TFR calculator)
- results_plot.png → Visualization of all variables

---

## Quick Start

- run main.py
- run Calculate_TFR.py
