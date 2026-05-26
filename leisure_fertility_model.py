"""
leisure_fertility_model.py -- equilibrium solver for the leisure-fertility model.

Model features:
  * New utility: U_tilde = xi_n * pi^kappa * n^((1-psi)/beta) + xi_o * o^(1-psi)
    with two new parameters kappa and beta (v1 implicitly had kappa=psi, beta=1).
  * Fertility n is solved ALGEBRAICALLY from the simultaneous FOCs for n and o,
    so the BVP has 5 states (c, l, h, b, o) instead of 6 (no n_dot equation).
  * Leisure ODE follows PDF eq. 13 with explicit U1, Delta_1, Delta_2 (eqs. 6-8).
  * BC for o(0) is enforced via the equivalent c-cost FOC relation:
        c(0) = beta * (w0*h0*Gamma1 + Gamma2_0) * n(0) + w0*h0*chi*o(0)
    (this is the simultaneous solution of the n-FOC and o-FOC at a=0).

Algorithm:
  1. Inner BVP at given r: solve_bvp on 5 states.
  2. From BVP solution compute n(a) algebraically -> integrate to get g_n
     fixed point (1 = int_0^T n(a) exp(-g_n a) da).
  3. Outer root: find r* such that aggregate B(t) =
        int_0^T exp((g+g_n)(T-a)) * b(a) da = 0.
"""

from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_bvp
from scipy.stats import lognorm
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# pitol_values is the ladder used to homotope from a "noisy" pi profile (large
# floor) down to the target pi profile (tiny floor).  Cold solves walk the
# whole ladder; warm solves use only the final value.
pitol_values = [5.0, 1.0, 0.2, 0.05, 0.01, 0.003, 0.001, 0.0004]

params = {
    # --- Pi (sex-productivity) ---
    'pitol': pitol_values[-1],
    'sd': 0.25,
    'mean_pi': 3.3,
    # --- Wages, costs, growth ---
    # Baseline = 1990 calibration.  Gamma_1 = 1.616 is the broad-measure time
    # cost of children: the mother/father average of (primary childcare + core
    # housework) ~ 18.15 h/week around 1990, from Bianchi-Robinson-Milkie (2006)
    # Tables 4.1 and 5.1 (the 1985 and 1995 diary surveys, averaged to bracket
    # 1990), mapped by Gamma_1 = 52*W*D / (H * N) with D = 18,
    # H = 5840 non-sleep h/yr, N = 1.80 (intensive-margin completed family
    # size for U.S. ever-mothers, held fixed across years so that the
    # year-to-year movement in Gamma_1 reflects the time-use measurement only).
    # The same broad measure is 20.47 h/week in 2021-2024 (ATUS), giving
    # Gamma_1 = 1.823 for 2023.
    # Gamma_2 = 0.324 is the 1990 goods cost: USDA 1990-92 CES (crc1995),
    # middle-income per-child total 145320 / group income 44800, times w0=0.1.
    # 2023 goods cost is Gamma_2 = 0.278 (USDA 2015 birth-to-17 data, no college).
    'w0': 0.1, 'Gamma_1': 1.616, 'Gamma_2': 0.324,
    'g': 0.02, 'g0': 0.02,
    # --- Lifecycle / discounting ---
    'T': 70, 'rho': 0.02,
    # --- Human capital tech ---
    'z': 0.1, 'h_0': 1.0, 'ddelta': 0.05,
    'alpha': 0.25, 'phi': 0.6,
    # --- Utility curvatures ---
    'psi': 0.5,
    'kappa': 0.7,   # NEW: exponent on pi in n-utility (replaces old "psi" there) 0.5 before
    'beta': 0.7,    # NEW: curvature of n in utility, n^((1-psi)/beta)
    # --- Time costs / utility weights ---
    # chi = 1.51 is the 1990-calibrated time cost of leisure: the residual that
    # reproduces TFR_1990 = 2.08 given Gamma_1 = 1.616, Gamma_2 = 0.324.  The
    # 2023 residual is chi = 1.015 (reproduces TFR_2023 = 1.62); chi falls 33%.
    'chi': 1.51, 'xi_n': 4.0, 'xi_o': 0.2,
    # --- Solver ---
    'bc_epsilon': 1e-4,
}

# Initial guess sources -- tried in order.  resultados.xlsx is v1's last
# converged solution (5 of its 6 columns map cleanly to v2's state).
guess_file_candidates = ["resultados.xlsx", "initial_guess_leisure.xlsx"]

# State ordering: y = (c, l, h, b, o).  n is computed algebraically.
N_STATES = 5
IDX_C, IDX_L, IDX_H, IDX_B, IDX_O = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Pi (age-specific sex productivity) and its log-derivative
# ---------------------------------------------------------------------------

def pi(a, params):
    return lognorm.pdf(a, params['sd'], scale=np.exp(params['mean_pi'])) + params['pitol']


def dlnpidt(a, params):
    """d/da log(pi)  via centered finite differences."""
    h = 1e-6
    a_lo = np.maximum(a - h, 0.0)
    a_hi = a + h
    return (np.log(pi(a_hi, params)) - np.log(pi(a_lo, params))) / (a_hi - a_lo)


# ---------------------------------------------------------------------------
# Algebraic n from the simultaneous FOCs (PDF eqs 16-18 implications).
#
# n-FOC:   xi_n * pi^kappa * n^((1-psi-beta)/beta) / (beta*Utilde) = (w*h*Gamma_1+Gamma_2)/c
# o-FOC:   xi_o * o^(-psi) / Utilde                              = w*h*chi/c
# Ratio:   n^((1-psi-beta)/beta) / (beta * xi_o/xi_n * pi^(-kappa) * o^(-psi))
#                                                                = (w*h*Gamma_1+Gamma_2)/(w*h*chi)
#  =>      n^((psi+beta-1)/beta) = xi_n * pi^kappa * w*h*chi * o^psi
#                                    / (beta * xi_o * (w*h*Gamma_1+Gamma_2))
#
# Note: w(t) and Gamma_2(t) both grow at rate g, so the time-trending factors
# CANCEL in the ratio.  The algebraic n therefore depends only on the constant
# w0, Gamma_2 (=Gamma_2 at calendar time 0), o, h, pi -- not on calendar time.
# ---------------------------------------------------------------------------

def algebraic_n(o, h, pi_a, params):
    """Algebraic n from PDF eq. (6).

    PDF eq. (6) (April 2026 PDF, after the FOC was added):
        n = [ beta * (xi_o/xi_n) * pi^(-kappa) * (w0*h*Gamma1 + Gamma2) / (w0*h*chi) ]^( beta/(1-psi-beta) )
            * o^( psi*beta / (beta + psi - 1) )

    Equivalently (by inverting the inner bracket and using the identity
    beta/(1-psi-beta) = -beta/(psi+beta-1)):
        n = (A * o^psi)^( beta/(psi+beta-1) )
    with
        A = (xi_n * pi^kappa * w0*h*chi) / (beta * xi_o * (w0*h*Gamma1 + Gamma2)).

    The two forms are identical; the second form is what's coded below
    (it is numerically more stable for large psi+beta-1).  Requires
    psi + beta > 1 (so the exponent is well-defined and strictly positive).
    """
    psi, beta, kappa = params['psi'], params['beta'], params['kappa']
    chi, Gamma1 = params['chi'], params['Gamma_1']
    xi_n, xi_o = params['xi_n'], params['xi_o']
    w0, Gamma2 = params['w0'], params['Gamma_2']
    if psi + beta <= 1.0:
        raise ValueError(
            f"Need psi+beta > 1 for eq. (6) to be valid; got "
            f"psi={psi}, beta={beta}, psi+beta={psi+beta}."
        )
    cost_n = w0 * h * Gamma1 + Gamma2     # cost per child in bonds at calendar time 0
    A = xi_n * pi_a ** kappa * w0 * h * chi / (beta * xi_o * cost_n)
    expo = beta / (psi + beta - 1.0)
    base = A * o ** psi
    base = np.maximum(base, 1e-30)
    return base ** expo


def verify_n_matches_pdf_eq6(params, n_test_points=20):
    """Internal sanity check: compare algebraic_n vs the verbatim PDF eq. (6) form.

    Returns the worst relative discrepancy.  Should be ~machine epsilon.
    """
    psi, beta, kappa = params['psi'], params['beta'], params['kappa']
    chi, Gamma1 = params['chi'], params['Gamma_1']
    xi_n, xi_o = params['xi_n'], params['xi_o']
    w0, Gamma2 = params['w0'], params['Gamma_2']
    rng = np.random.default_rng(0)
    o = rng.uniform(0.001, 0.5, n_test_points)
    h = rng.uniform(1.0, 3.0, n_test_points)
    pi_a = rng.uniform(0.001, 5.0, n_test_points)
    inner = beta * (xi_o/xi_n) * pi_a**(-kappa) * (w0*h*Gamma1+Gamma2) / (w0*h*chi)
    n_eq6 = inner**(beta/(1.0 - psi - beta)) * o**(psi*beta/(beta+psi-1.0))
    n_mine = algebraic_n(o, h, pi_a, params)
    return float(np.abs((n_mine - n_eq6) / n_eq6).max())


# ---------------------------------------------------------------------------
# ODE system (5 states; n algebraic).
# ---------------------------------------------------------------------------

def model_odes(a, y, tau, r, params):
    c = np.maximum(y[IDX_C], 1e-10)
    l = np.maximum(y[IDX_L], 1e-10)
    h = np.maximum(y[IDX_H], params['h_0'])
    b = y[IDX_B]
    o = np.maximum(y[IDX_O], 1e-10)

    psi, beta, kappa = params['psi'], params['beta'], params['kappa']
    chi, Gamma1 = params['chi'], params['Gamma_1']
    xi_n, xi_o = params['xi_n'], params['xi_o']
    w0, Gamma2_0 = params['w0'], params['Gamma_2']
    rho, gp = params['rho'], params['g']
    z, alpha, phi = params['z'], params['alpha'], params['phi']
    h0p, ddelta = params['h_0'], params['ddelta']

    pi_a = pi(a, params)
    dlnpi_da = dlnpidt(a, params)

    # --- Algebraic n from FOC (constant w0, Gamma_2_0 due to BGP cancellation) ---
    n = algebraic_n(o, h, pi_a, params)
    n = np.maximum(n, 1e-12)

    # --- Utility components ---
    # U1 = xi_n * pi^kappa * n^((1-psi)/beta)  is the n-component of U_tilde.
    U1 = xi_n * pi_a ** kappa * n ** ((1.0 - psi) / beta)
    U_o = xi_o * o ** (1.0 - psi)
    U_tilde = U1 + U_o
    U_tilde = np.maximum(U_tilde, 1e-30)

    # --- Time-varying coefficients used only in the bond ODE ---
    t_cal = a + tau
    w_t = w0 * np.exp(gp * t_cal)
    Gamma2_t = Gamma2_0 * np.exp(params['g0'] * t_cal)

    # --- Time allocation residual (work_time) ---
    work_time = 1.0 - l - Gamma1 * n - chi * o

    # --- ODEs ---
    # (4) c-Euler
    dcdt = (r - rho) * c

    # (5) human capital
    dhdt = z * h ** phi * l ** alpha - ddelta * (h - h0p)

    # (12) education-time ODE  (term1 - term2 - term3)
    term1 = l / (1.0 - alpha) * (
        ddelta + r - gp + (1.0 - phi) * ddelta * (1.0 - h0p / h)
    )
    term2 = z * h ** (phi - 1.0) * l ** (alpha + 1.0) / (1.0 - alpha)
    term3 = z * alpha * h ** (phi - 1.0) * l ** alpha / (1.0 - alpha) * work_time
    dldt = term1 - term2 - term3

    # (14) bond accumulation -- uses time-varying w(t), Gamma_2(t)
    dbdt = r * b + w_t * h * work_time - c - Gamma2_t * n

    # --- Delta_1 (PDF eq 7).  Uses constant w0, Gamma_2_0 inside the FOC
    # derivative because the time-varying factor exp(g*t) cancels in the ratio
    # w*h*chi / (w*h*Gamma_1 + Gamma_2). ---
    cost_n_0 = w0 * h * Gamma1 + Gamma2_0
    # bracket = d/da [ w0*h*chi / (w0*h*Gamma_1+Gamma_2_0) ] (only h depends on a)
    bracket = w0 * chi * dhdt * Gamma2_0 / cost_n_0 ** 2
    coeff_pi = kappa * beta / (psi + beta - 1.0)         # = -kappa*beta/(1-psi-beta)
    coeff_h  = (1.0 - psi) / (psi + beta - 1.0)
    Delta_1 = (
        coeff_pi * U1 * dlnpi_da
        + coeff_h * U1 * cost_n_0 / (w0 * h * chi) * bracket
    )

    # --- Delta_2 (PDF eq 8) ---
    Delta_2 = (
        psi * (1.0 - psi) / (psi + beta - 1.0) * U1
        + xi_o * (1.0 - psi) * o ** (1.0 - psi)
    )

    # --- (13) leisure-time ODE ---
    numer_o = (r - rho - gp) - dhdt / h - Delta_1 / U_tilde
    denom_o = Delta_2 / U_tilde + psi
    # safeguard
    denom_o = np.where(np.abs(denom_o) < 1e-12, 1e-12 * np.sign(denom_o + 1e-30), denom_o)
    dodt = o * numer_o / denom_o

    return np.vstack([dcdt, dldt, dhdt, dbdt, dodt])


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def bc(ya, yb, tau, r, params):
    """5 BCs:
       (i)   h(0) = h0                                      (PDF eq 16)
       (ii)  b(0) = 0                                       (PDF eq 15a)
       (iii) l(T) = 0                                       (PDF eq 17)
       (iv)  b(T) = 0                                       (PDF eq 15b)
       (v)   c(0) = beta*(w0*h0*G1+G2)*n(0) + w0*h0*chi*o(0)   <=>
             o(0) = (xi_o*c(0)/(U_tilde(0)*w0*h0*chi))^(1/psi)   (PDF eq 18)
    """
    psi, beta = params['psi'], params['beta']
    chi, Gamma1 = params['chi'], params['Gamma_1']
    w0, Gamma2_0 = params['w0'], params['Gamma_2']
    h0p = params['h_0']

    o0 = ya[IDX_O]
    pi_0 = pi(0.0, params)
    n0 = algebraic_n(o0, h0p, pi_0, params)
    cost_n_0 = w0 * h0p * Gamma1 + Gamma2_0
    foc_residual = ya[IDX_C] - beta * cost_n_0 * n0 - w0 * h0p * chi * o0

    return np.array([
        ya[IDX_H] - h0p,        # h(0) = h0
        ya[IDX_B],              # b(0) = 0
        yb[IDX_L],              # l(T) = 0
        yb[IDX_B],              # b(T) = 0
        foc_residual,           # implicit FOC for o(0) <-> c(0)
    ])


# ---------------------------------------------------------------------------
# Initial guess  (5-state version of the file initial_guess_leisure.xlsx,
# which has 6 columns: c, l, n, h, b, o.  We drop n.)
# ---------------------------------------------------------------------------

def load_excel_guess(filepath, params, n_points=100):
    df = pd.read_excel(filepath)
    a_data = df.iloc[:, 0].to_numpy()
    cols = list(df.columns[1:])
    # File order: c, l, n, h, b, o.  We want: c, l, h, b, o (drop n).
    wanted = [cols[0], cols[1], cols[3], cols[4], cols[5]]
    a_eval = np.linspace(0, params['T'], n_points)
    y_guess = np.zeros((N_STATES, n_points))
    for i, name in enumerate(wanted):
        fn = interp1d(a_data, df[name].to_numpy(),
                      kind='linear', fill_value='extrapolate')
        y_guess[i] = fn(a_eval)
    # Sanitize: enforce positivity and clip o (stale guesses often have o>1).
    y_guess[IDX_C] = np.maximum(y_guess[IDX_C], 1e-4)
    y_guess[IDX_L] = np.clip(y_guess[IDX_L], 1e-6, 0.5)
    y_guess[IDX_H] = np.maximum(y_guess[IDX_H], params['h_0'])
    y_guess[IDX_O] = np.clip(y_guess[IDX_O], 1e-6, 0.3)
    return y_guess, a_eval


def load_first_available_guess(params, n_points=100):
    import os
    for fname in guess_file_candidates:
        if os.path.exists(fname):
            try:
                return load_excel_guess(fname, params, n_points)
            except Exception as e:
                print(f"  (failed to load {fname}: {e})")
    return None, None


def make_synth_guess(params, r, n_points=100):
    """Synthetic, BC-consistent initial guess (used when no Excel guess works)."""
    a = np.linspace(0, params['T'], n_points)
    T = params['T']

    # Pick c0 small enough that work_time stays positive; refined later by BVP.
    c0 = 0.05
    growth = max(r - params['rho'], 0.0)
    c = c0 * np.exp(growth * a)

    # Education profile peaks young, hits zero at T (terminal BC).
    l = np.clip(0.4 * (1.0 - a / T) ** 1.5, 1e-8, 0.9)

    # Human capital grows then plateaus.
    h = params['h_0'] * (1.0 + 0.5 * a / T)

    # Bonds: zero at endpoints, hump in the middle.
    b = np.zeros(n_points)

    # Leisure: small at birth, growing with age.
    o = np.clip(0.05 + 0.4 * a / T, 1e-6, 0.8)

    return np.vstack([c, l, h, b, o])


# ---------------------------------------------------------------------------
# BVP solve with cache + continuation
# ---------------------------------------------------------------------------

_bvp_cache = {}     # key: rounded r -> sol (or None)


def clear_bvp_cache():
    global _bvp_cache
    _bvp_cache = {}


def _interp_sol(sol, n_points=100, T=70):
    a_eval = np.linspace(0, T, n_points)
    y = np.zeros((sol.y.shape[0], n_points))
    for i in range(sol.y.shape[0]):
        fn = interp1d(sol.x, sol.y[i], kind='linear', fill_value='extrapolate')
        y[i] = fn(a_eval)
    return y


def solve_cohort(tau, r, params, n_points=100, y_guess=None):
    a_eval = np.linspace(0, params['T'], n_points)
    if y_guess is None:
        loaded, _ = load_first_available_guess(params, n_points=n_points)
        y_guess = loaded if loaded is not None else make_synth_guess(params, r, n_points=n_points)
    return solve_bvp(
        fun=lambda a, y: model_odes(a, y, tau, r, params),
        bc=lambda ya, yb: bc(ya, yb, tau, r, params),
        x=a_eval, y=y_guess,
        max_nodes=100000, verbose=0,
    )


def _nearest_r_and_sol(r):
    successful = {k: v for k, v in _bvp_cache.items()
                  if v is not None and v.success}
    if not successful:
        return None, None
    nearest_key = min(successful.keys(), key=lambda k: abs(k - r))
    return nearest_key, successful[nearest_key]


def _r_continuation(tau, r_target, params, r_start, sol_start, n_steps=10):
    r_path = np.linspace(r_start, r_target, n_steps + 1)[1:]
    p = params.copy()
    p['pitol'] = pitol_values[-1]
    current_sol = sol_start
    for r_step in r_path:
        ck = round(float(r_step), 10)
        if ck in _bvp_cache and _bvp_cache[ck] is not None and _bvp_cache[ck].success:
            current_sol = _bvp_cache[ck]
            continue
        y_guess = _interp_sol(current_sol, n_points=100, T=p['T'])
        sol = solve_cohort(tau, float(r_step), p, y_guess=y_guess)
        if not sol.success:
            return None
        _bvp_cache[ck] = sol
        current_sol = sol
    return current_sol


def _pitol_ladder(tau, r, params, y_seed=None):
    """Walk down the pitol ladder, warm-starting each step."""
    seed = y_seed if y_seed is not None else make_synth_guess(params, r)
    sol = None
    for pitol_val in pitol_values:
        p = params.copy()
        p['pitol'] = pitol_val
        if sol is None:
            sol = solve_cohort(tau, r, p, y_guess=seed)
        else:
            y_prev = _interp_sol(sol, n_points=100, T=p['T'])
            sol = solve_cohort(tau, r, p, y_guess=y_prev)
        if not sol.success:
            print(f"  [pitol={pitol_val:.4f}] failed: {sol.message}")
            return None
    return sol


def _beta_kappa_homotopy(tau, r, params, y_seed):
    """Walk (beta, kappa) from (1, psi) -> (target, target), warm-starting at each step.

    At beta=1, kappa=psi the new utility reduces to v1's structure (no fractional
    powers in the algebraic n FOC), giving a well-conditioned starting BVP.
    """
    psi_t = params['psi']
    beta_target = params['beta']
    kappa_target = params['kappa']
    n_steps = 5
    sol = None
    for f in np.linspace(0.0, 1.0, n_steps + 1):
        beta_step = (1.0 - f) * 1.0 + f * beta_target
        kappa_step = (1.0 - f) * psi_t + f * kappa_target
        p = params.copy()
        p['beta'] = beta_step
        p['kappa'] = kappa_step
        guess = y_seed if sol is None else _interp_sol(sol, n_points=100, T=p['T'])
        sol = solve_cohort(tau, r, p, y_guess=guess)
        if not sol.success:
            print(f"    [homotopy] failed at beta={beta_step:.3f}, "
                  f"kappa={kappa_step:.3f}: {sol.message}")
            return None
    return sol


def get_sol_cached(r, params):
    key = round(r, 10)
    if key in _bvp_cache:
        return _bvp_cache[key]

    nearest_r, warm = _nearest_r_and_sol(r)

    # Strategy 1: direct warm start (cached or from disk file)
    if warm is not None:
        y_warm = _interp_sol(warm, n_points=100, T=params['T'])
        src_msg = f"r={nearest_r:.6f}"
    else:
        y_warm, _ = load_first_available_guess(params)
        src_msg = "file guess"

    if y_warm is not None:
        sol = solve_cohort(0, r, params, y_guess=y_warm)
        if sol.success:
            print(f"  [direct] r={r:.6f} OK (from {src_msg})")
            _bvp_cache[key] = sol
            return sol

    # Strategy 2: r-continuation from nearest cached r (uses target params)
    if warm is not None:
        sol = _r_continuation(0, r, params, nearest_r, warm, n_steps=15)
        if sol is not None:
            print(f"  [r-continuation] r={r:.6f} OK from {nearest_r:.6f}")
            _bvp_cache[key] = sol
            return sol

    # Strategy 3: beta-kappa homotopy (start at v1-equivalent params, walk to target)
    print(f"  [beta/kappa homotopy] r={r:.6f}")
    y_seed = y_warm if y_warm is not None else make_synth_guess(params, r)
    sol = _beta_kappa_homotopy(0, r, params, y_seed)
    if sol is not None and sol.success:
        _bvp_cache[key] = sol
        return sol

    # Strategy 4: pitol ladder (last resort)
    print(f"  [pitol ladder] r={r:.6f}")
    sol = _pitol_ladder(0, r, params, y_seed=y_seed)
    _bvp_cache[key] = sol
    return sol


# ---------------------------------------------------------------------------
# Equilibrium: g_n from N fixed point, r from B(t)=0
# ---------------------------------------------------------------------------

def n_profile_from_sol(sol, params):
    """Recover n(a) algebraically from the BVP solution."""
    a_grid = sol.x
    h_grid = sol.y[IDX_H]
    o_grid = sol.y[IDX_O]
    pi_grid = pi(a_grid, params)
    return algebraic_n(o_grid, h_grid, pi_grid, params)


def fixed_point_g_n(g_n, sol, params):
    """1 - integral_0^T n(a) exp(-g_n a) da."""
    n_grid = n_profile_from_sol(sol, params)
    integrand = n_grid * np.exp(-g_n * sol.x)
    # 1/2 = female share of newborns: only women bear children, N counts both sexes.
    return 1.0 - 0.5 * np.trapezoid(integrand, sol.x)


def find_g_n(sol, params, brackets=((-0.1, 1.0), (-0.5, 2.0), (-2.0, 5.0))):
    for lo, hi in brackets:
        f_lo = fixed_point_g_n(lo, sol, params)
        f_hi = fixed_point_g_n(hi, sol, params)
        if f_lo * f_hi < 0:
            res = root_scalar(lambda g: fixed_point_g_n(g, sol, params),
                              bracket=(lo, hi), method='brentq',
                              xtol=1e-8, rtol=1e-8)
            return res.root
    return None


def agg_bonds(r, params):
    sol = get_sol_cached(r, params)
    if sol is None or not sol.success:
        return np.nan
    g_n = find_g_n(sol, params)
    if g_n is None:
        return np.nan
    a_vals = sol.x
    b_vals = sol.y[IDX_B]
    integrand = np.exp((g_n + params['g']) * (params['T'] - a_vals)) * b_vals
    return np.trapezoid(integrand, a_vals)


def find_equilibrium_r(params, r_bounds=(0.05, 0.07)):
    f_a = agg_bonds(r_bounds[0], params)
    f_b = agg_bonds(r_bounds[1], params)
    print(f"agg_bonds(r={r_bounds[0]})={f_a:.6f}, agg_bonds(r={r_bounds[1]})={f_b:.6f}")
    if np.isnan(f_a) or np.isnan(f_b):
        raise ValueError("agg_bonds returned NaN -- BVP did not converge at endpoint.")
    if f_a * f_b >= 0:
        raise ValueError(f"No sign change in agg_bonds over r_bounds={r_bounds}")
    res = root_scalar(lambda r: agg_bonds(r, params),
                      bracket=r_bounds, method='brentq',
                      xtol=1e-8, rtol=1e-8)
    return res.root


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    np.random.seed(123)
    clear_bvp_cache()
    print("=" * 72)
    print("leisure-fertility model -- equilibrium solver")
    print("=" * 72)
    print(f"Parameters: psi={params['psi']}, kappa={params['kappa']}, "
          f"beta={params['beta']}, rho={params['rho']}, g={params['g']}")
    print(f"            xi_n={params['xi_n']}, xi_o={params['xi_o']}, "
          f"chi={params['chi']}, T={params['T']}")
    print(f"Constraint check: psi+beta = {params['psi']+params['beta']:.3f}  "
          f"(need > 1 for eq.(6); ok)")
    rel_n_err = verify_n_matches_pdf_eq6(params)
    print(f"Sanity: algebraic_n vs PDF eq.(6) max relative diff = {rel_n_err:.2e}")
    print()

    try:
        r_eq = find_equilibrium_r(params)
    except Exception as e:
        print(f"\n*** Failed to find equilibrium r: {e} ***")
        return

    sol = get_sol_cached(r_eq, params)
    if sol is None or not sol.success:
        print(f"\n*** BVP failed at r*={r_eq:.6f} ***")
        return

    g_n_eq = find_g_n(sol, params)

    # ----- Print results -----
    print()
    print("=" * 72)
    print("EQUILIBRIUM FOUND")
    print("=" * 72)
    print(f"  Interest rate           r* = {r_eq:.6f}")
    print(f"  Population growth     g_n* = {g_n_eq:.6f}")
    print(f"  (Wage / consumption growth   g  = {params['g']:.6f})")
    print()

    # ----- Time-allocation diagnostics -----
    a_grid = sol.x
    c_s, l_s, h_s, b_s, o_s = sol.y
    n_s = n_profile_from_sol(sol, params)
    work_time = 1.0 - l_s - params['Gamma_1'] * n_s - params['chi'] * o_s
    print("Time-allocation diagnostics:")
    print(f"  min work_time (1-l-G1*n-chi*o) = {work_time.min():.4f}  "
          "(should be in [0,1])")
    print(f"  max(o)  = {o_s.max():.4f}")
    print(f"  max(l)  = {l_s.max():.4f}")
    print(f"  max(n)  = {n_s.max():.4f}")
    if work_time.min() < 0:
        frac = float(np.mean(work_time < 0)) * 100
        print(f"  *** WARNING: work-time constraint violated at {frac:.1f}% of grid. ***")
    print()

    # ----- Plot -----
    plt.figure(figsize=(14, 10))
    panels = [
        ("Consumo (c)",  c_s),
        ("Edu time (l)", l_s),
        ("Capital humano (h)", h_s),
        ("Bonos (b)",    b_s),
        ("Ocio (o)",     o_s),
        ("Fertilidad (n)", n_s),
    ]
    for i, (title, y) in enumerate(panels):
        plt.subplot(4, 2, i + 1)
        plt.plot(a_grid, y, label='tau=0')
        plt.title(title)
        plt.xlabel('a')
        plt.legend()
    plt.subplot(4, 2, 7)
    plt.plot(a_grid, work_time, label='work time', color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.axhline(1, color='orange', linestyle='--', linewidth=0.8)
    plt.title('Work time (1 - l - G1 n - chi o)')
    plt.xlabel('a')
    plt.legend()
    plt.subplot(4, 2, 8)
    plt.plot(a_grid, pi(a_grid, params), label='pi(a)', color='purple')
    plt.title('pi(a) -- sex productivity')
    plt.xlabel('a')
    plt.legend()
    plt.suptitle(f"r*={r_eq:.5f},  g_n*={g_n_eq:.5f},  "
                 f"kappa={params['kappa']}, beta={params['beta']}", y=1.0)
    plt.tight_layout()
    plot_file = "results_plot_v2.png"
    plt.savefig(plot_file, dpi=120, bbox_inches='tight')
    print(f"Plot saved: {plot_file}")

    data = {
        'a':                  a_grid,
        'Consumo (c)':        c_s,
        'Edu time (l)':       l_s,
        'Fertilidad (n)':     n_s,
        'Capital humano (h)': h_s,
        'Bonos (b)':          b_s,
        'Ocio (o)':           o_s,
        'Work time':          work_time,
    }
    df = pd.DataFrame(data)
    out_xlsx = "resultados_v2.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"Resultados exportados: {out_xlsx}")

    print()
    print(f"SUMMARY  r*={r_eq:.6f}  g_n*={g_n_eq:.6f}")
    return r_eq, g_n_eq, sol



if __name__ == "__main__":
    main()
