import math
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

from core.reaction.kinetics import first_order_rate, nth_order_rate, arrhenius_k


# ---------------------------------------------------------------------------
# Kept for backward compatibility
# ---------------------------------------------------------------------------

def simulate_first_order_batch(k: float, ca0: float, t_final: float):
    def ode(_t, y):
        return [first_order_rate(k, y[0])]

    t_eval = np.linspace(0, t_final, 200)
    sol = solve_ivp(ode, (0, t_final), [ca0], t_eval=t_eval)
    return sol.t, sol.y[0]


# ---------------------------------------------------------------------------
# Tab 1 — Ideal Reactors
# ---------------------------------------------------------------------------

def simulate_batch_reactor(k: float, n: float, ca0: float, t_final: float):
    """Batch: dCA/dt = -k*CA^n.  Returns t, CA, X arrays."""
    def ode(_t, y):
        return [nth_order_rate(k, y[0], n)]

    t_eval = np.linspace(0, t_final, 300)
    sol = solve_ivp(ode, (0, t_final), [ca0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    ca = np.maximum(sol.y[0], 0.0)
    x = (ca0 - ca) / ca0 if ca0 > 0 else np.zeros_like(ca)
    return sol.t, ca, x


def simulate_cstr(k: float, n: float, ca0: float, tau_max: float, n_points: int = 300):
    """CSTR steady-state: CA0 - CA = k*CA^n * tau.
    Sweeps tau from 0 to tau_max.  Returns tau, CA, X arrays."""
    tau_array = np.linspace(0.0, tau_max, n_points)
    ca_array = np.empty(n_points)
    ca_array[0] = ca0

    for i, tau in enumerate(tau_array[1:], 1):
        def f(ca):
            return ca + k * (max(ca, 0.0) ** n) * tau - ca0

        try:
            ca_array[i] = brentq(f, 0.0, ca0 + 1e-10, xtol=1e-12)
        except Exception:
            ca_array[i] = max(ca_array[i - 1] - 1e-6, 0.0)

    x_array = (ca0 - ca_array) / ca0 if ca0 > 0 else np.zeros(n_points)
    return tau_array, ca_array, x_array


def simulate_pfr(k: float, n: float, ca0: float, v0: float, v_final: float):
    """PFR: dCA/dV = -k*CA^n / v0.  Returns V, CA, X arrays."""
    def ode(_v, y):
        return [-k * (max(y[0], 0.0) ** n) / v0]

    v_eval = np.linspace(0, v_final, 300)
    sol = solve_ivp(ode, (0, v_final), [ca0], t_eval=v_eval, rtol=1e-8, atol=1e-10)
    ca = np.maximum(sol.y[0], 0.0)
    x = (ca0 - ca) / ca0 if ca0 > 0 else np.zeros_like(ca)
    return sol.t, ca, x


# ---------------------------------------------------------------------------
# Tab 2 — Arrhenius / Temperature Effects
# ---------------------------------------------------------------------------

def compute_arrhenius_curve(A: float, Ea: float, T_min: float, T_max: float, n_points: int = 300):
    """Returns T_array (K), k_array."""
    T_array = np.linspace(T_min, T_max, n_points)
    k_array = np.array([arrhenius_k(A, Ea, T) for T in T_array])
    return T_array, k_array


# ---------------------------------------------------------------------------
# Tab 3 — Series & Parallel Reactions
# ---------------------------------------------------------------------------

def simulate_series_reactions(
    k1: float, k2: float, ca0: float, cb0: float, cc0: float, t_final: float
):
    """A -> B -> C (series). Returns t, CA, CB, CC."""
    def ode(_t, y):
        ca, cb, _ = y
        return [
            -k1 * max(ca, 0.0),
            k1 * max(ca, 0.0) - k2 * max(cb, 0.0),
            k2 * max(cb, 0.0),
        ]

    t_eval = np.linspace(0, t_final, 300)
    sol = solve_ivp(ode, (0, t_final), [ca0, cb0, cc0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def simulate_parallel_reactions(k1: float, k2: float, ca0: float, t_final: float):
    """A -> B (k1), A -> C (k2) (parallel). Returns t, CA, CB, CC."""
    def ode(_t, y):
        ca = max(y[0], 0.0)
        return [-(k1 + k2) * ca, k1 * ca, k2 * ca]

    t_eval = np.linspace(0, t_final, 300)
    sol = solve_ivp(ode, (0, t_final), [ca0, 0.0, 0.0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ---------------------------------------------------------------------------
# Tab 4 — Reactor Sizing & Levenspiel
# ---------------------------------------------------------------------------

def compute_reactor_sizing(
    k: float, n: float, ca0: float, v0: float, x_target: float, n_points: int = 300
):
    """Compute CSTR and PFR volumes vs conversion X for nth-order reaction.

    Returns:
        X_array, V_cstr_array, V_pfr_array,
        V_cstr_target, V_pfr_target,
        inv_ra_array  (Levenspiel function: 1 / (-rA))
    """
    FA0 = v0 * ca0
    X_array = np.linspace(0.0, min(x_target * 1.02, 0.9999), n_points)

    def minus_ra(X: float) -> float:
        ca = ca0 * (1.0 - X)
        return k * (max(ca, 1e-30) ** n)

    # CSTR: V = FA0 * X / (-rA_exit)
    V_cstr = np.array([FA0 * X / minus_ra(X) if X > 0 else 0.0 for X in X_array])

    # PFR: V = FA0 * integral(dX / (-rA))
    V_pfr = np.zeros(n_points)
    for i, X in enumerate(X_array):
        if X <= 0:
            continue
        try:
            val, _ = quad(lambda x: FA0 / minus_ra(x), 0.0, X, limit=200)
            V_pfr[i] = val
        except Exception:
            V_pfr[i] = V_pfr[i - 1] if i > 0 else 0.0

    # Target volumes
    V_cstr_target = FA0 * x_target / minus_ra(x_target)
    try:
        V_pfr_target, _ = quad(lambda x: FA0 / minus_ra(x), 0.0, x_target, limit=200)
    except Exception:
        V_pfr_target = 0.0

    inv_ra = np.array([1.0 / minus_ra(X) for X in X_array])

    return X_array, V_cstr, V_pfr, V_cstr_target, V_pfr_target, inv_ra


# ---------------------------------------------------------------------------
# Tab 5 — Chemical Equilibrium
# ---------------------------------------------------------------------------

def compute_equilibrium(
    dH_rxn_J: float,
    dG_rxn_J: float,
    stoich_reac: list[float],
    stoich_prod: list[float],
    n_init: list[float],
    P_bar: float = 1.0,
    T_ref_K: float = 298.15,
    T_min_K: float = 300.0,
    T_max_K: float = 1200.0,
    n_points: int = 300,
):
    """Van't Hoff equilibrium calculation for a gas-phase reaction.

    Assumes ΔH_rxn constant (reasonable over moderate T range).
    Δn = (sum products stoich) - (sum reactants stoich).
    Returns: T_arr, Kp_arr, Keq_at_Tref, extent_at_Tref.
    """
    R = 8.314
    Keq_ref = math.exp(-dG_rxn_J / (R * T_ref_K))
    delta_n = sum(stoich_prod) - sum(stoich_reac)

    T_arr = np.linspace(T_min_K, T_max_K, n_points)
    Kp_arr = np.array([
        Keq_ref * math.exp(-dH_rxn_J / R * (1.0 / T - 1.0 / T_ref_K))
        for T in T_arr
    ])

    # Extent of reaction at T_ref for a simple A <-> B type (all equal stoich)
    # Using generic extent calculation via mole fractions
    n_total_init = sum(n_init)
    # For simplicity compute at T_ref
    def _mole_frac(xi, i, is_prod):
        dn = xi * (stoich_prod[i] if is_prod else -stoich_reac[i])
        return max(dn / (n_total_init + xi * delta_n), 0.0)

    # Numerical extent: simplified for single reaction
    extent_arr = np.zeros(n_points)
    for idx, (T, Kp) in enumerate(zip(T_arr, Kp_arr)):
        # Kp = prod(y_i^v_i) * (P/P0)^delta_n, P0=1 bar
        # Solve numerically for extent xi in [0, min(n_i/v_i)]
        max_xi = min(n / s for n, s in zip(n_init, stoich_reac) if s > 0)
        def obj(xi):
            n_tot = n_total_init + xi * delta_n
            if n_tot <= 0:
                return float("inf")
            # Reactant mole fracs
            yr = [(n_init[i] - xi * stoich_reac[i]) / n_tot for i in range(len(stoich_reac))]
            yp = [(xi * stoich_prod[i]) / n_tot for i in range(len(stoich_prod))]
            if any(y <= 0 for y in yr) or any(y < 0 for y in yp):
                return float("inf")
            Kp_calc = (
                np.prod([y ** s for y, s in zip(yp, stoich_prod)]) /
                np.prod([y ** s for y, s in zip(yr, stoich_reac)])
            ) * (P_bar) ** delta_n
            return Kp_calc - Kp
        try:
            xi_eq = brentq(obj, 1e-10, max_xi * 0.9999, xtol=1e-12)
        except Exception:
            xi_eq = 0.0
        extent_arr[idx] = xi_eq

    return T_arr, Kp_arr, Keq_ref, extent_arr


# ---------------------------------------------------------------------------
# Tab 6 — Non-isothermal Reactor (Batch or PFR with energy balance)
# ---------------------------------------------------------------------------

def simulate_nonisothermal_batch(
    k0: float,
    Ea_J: float,
    n: float,
    ca0: float,
    T0_K: float,
    dH_rxn_J_per_mol: float,
    Cp_J_per_mol_K: float,
    rho_mol_per_L: float,
    UA_W_per_K: float,
    Tc_K: float,
    t_final: float,
    n_points: int = 300,
):
    """Coupled mass + energy balance for non-isothermal batch reactor.

    dCA/dt = -k(T)*CA^n
    dT/dt  = (-dH_rxn * (-rA) - UA*(T-Tc)) / (rho * Cp)
    Returns t, CA, X, T arrays.
    """
    R = 8.314

    def ode(_t, y):
        ca, T = y
        ca = max(ca, 0.0)
        k = k0 * math.exp(-Ea_J / (R * T)) if T > 0 else 0.0
        ra = k * (ca ** n)
        dca_dt = -ra
        dT_dt = ((-dH_rxn_J_per_mol) * ra - UA_W_per_K * (T - Tc_K)) / (
            rho_mol_per_L * 1000 * Cp_J_per_mol_K  # rho in mol/L -> mol/m³ cancel
        )
        return [dca_dt, dT_dt]

    t_eval = np.linspace(0, t_final, n_points)
    sol = solve_ivp(ode, (0, t_final), [ca0, T0_K], t_eval=t_eval,
                    rtol=1e-6, atol=1e-8, method="Radau")
    ca = np.maximum(sol.y[0], 0.0)
    T_arr = sol.y[1]
    x = (ca0 - ca) / ca0 if ca0 > 0 else np.zeros_like(ca)
    return sol.t, ca, x, T_arr


def simulate_nonisothermal_pfr(
    k0: float,
    Ea_J: float,
    n: float,
    ca0: float,
    T0_K: float,
    v0: float,
    dH_rxn_J_per_mol: float,
    Cp_J_per_mol_K: float,
    rho_mol_per_L: float,
    UA_W_per_K_per_L: float,
    Tc_K: float,
    V_final: float,
    n_points: int = 300,
):
    """Coupled mass + energy balance for non-isothermal PFR.

    dCA/dV = -k(T)*CA^n / v0
    dT/dV  = ((-dH_rxn)*k(T)*CA^n - UA*(T-Tc)) / (rho*Cp*v0)
    Returns V, CA, X, T arrays.
    """
    R = 8.314
    rho_SI = rho_mol_per_L * 1000  # mol/m³  (1 L = 0.001 m³, so mol/L * 1000 = mol/m³)

    def ode(_v, y):
        ca, T = y
        ca = max(ca, 0.0)
        k = k0 * math.exp(-Ea_J / (R * T)) if T > 0 else 0.0
        ra = k * (ca ** n)
        dca_dv = -ra / v0
        dT_dv = ((-dH_rxn_J_per_mol) * ra - UA_W_per_K_per_L * (T - Tc_K)) / (
            rho_SI * Cp_J_per_mol_K * v0
        )
        return [dca_dv, dT_dv]

    v_eval = np.linspace(0, V_final, n_points)
    sol = solve_ivp(ode, (0, V_final), [ca0, T0_K], t_eval=v_eval,
                    rtol=1e-6, atol=1e-8, method="Radau")
    ca = np.maximum(sol.y[0], 0.0)
    T_arr = sol.y[1]
    x = (ca0 - ca) / ca0 if ca0 > 0 else np.zeros_like(ca)
    return sol.t, ca, x, T_arr


# ---------------------------------------------------------------------------
# Tab 7 — Residence Time Distribution (RTD)
# ---------------------------------------------------------------------------

def compute_rtd_tanks_in_series(tau: float, N: int, t_final: float, n_points: int = 300):
    """E(t) and F(t) for N tanks-in-series model (plug-flow approaches as N→∞).

    E(t) = (N/tau)^N * t^(N-1) * exp(-N*t/tau) / (N-1)!
    Returns t, E, F arrays.
    """
    t_arr = np.linspace(0, t_final, n_points)
    factor = (N / tau) ** N / math.factorial(N - 1)
    E_arr = factor * t_arr ** (N - 1) * np.exp(-N * t_arr / tau)
    F_arr = np.cumsum(E_arr) * (t_arr[1] - t_arr[0]) if n_points > 1 else np.zeros(n_points)
    F_arr = np.clip(F_arr, 0.0, 1.0)
    return t_arr, E_arr, F_arr


def compute_rtd_dispersion(tau: float, Pe: float, t_final: float, n_points: int = 300):
    """E(t) for the axial dispersion model (closed-closed boundary).

    E(theta) ≈ sqrt(Pe/(4*pi*theta)) * exp(-Pe*(1-theta)^2 / (4*theta))
    where theta = t/tau.
    Returns t, E, F arrays.
    """
    t_arr = np.linspace(1e-4, t_final, n_points)
    theta = t_arr / tau
    E_theta = np.sqrt(Pe / (4 * np.pi * theta)) * np.exp(
        -Pe * (1 - theta) ** 2 / (4 * theta)
    )
    E_arr = E_theta / tau  # convert from E(θ) to E(t)
    dt = t_arr[1] - t_arr[0]
    F_arr = np.clip(np.cumsum(E_arr) * dt, 0.0, 1.0)
    return t_arr, E_arr, F_arr
