"""Fluid Dynamics solvers for four tabs:

  1. Pipe Flow       — Reynolds number, friction factor (Colebrook), Darcy-Weisbach ΔP
  2. Bernoulli       — Energy balance, head loss, orifice flow
  3. Pump Sizing     — Pump head, power, efficiency, operating-point curve
  4. Compressible    — Mach number, isentropic relations, choked-flow limit
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# 1. Pipe Flow
# ---------------------------------------------------------------------------

def reynolds_number(rho: float, v: float, D: float, mu: float) -> float:
    if mu <= 0:
        raise ValueError("Dynamic viscosity μ must be > 0.")
    return rho * v * D / mu


def friction_factor_colebrook(Re: float, eps_D: float) -> float:
    """Darcy friction factor via Colebrook-White (implicit)."""
    if Re <= 0:
        raise ValueError("Re must be > 0.")
    if Re < 2300:
        return 64.0 / Re   # laminar

    # Swamee-Jain explicit approximation as initial guess
    def colebrook(f):
        return (1.0 / math.sqrt(f)
                + 2.0 * math.log10(eps_D / 3.7 + 2.51 / (Re * math.sqrt(f))))

    try:
        f = brentq(colebrook, 1e-6, 0.1, xtol=1e-12)
    except Exception:
        # Swamee-Jain explicit fallback
        f = 0.25 / (math.log10(eps_D / 3.7 + 5.74 / Re ** 0.9)) ** 2
    return float(f)


def pipe_flow_analysis(
    rho: float, v: float, D: float,
    mu: float, L: float, eps: float = 0.0,
) -> dict:
    """Full pipe-flow analysis: Re, regime, f, ΔP, head loss."""
    Re = reynolds_number(rho, v, D, mu)
    if Re < 2300:
        regime = "Laminar"
    elif Re < 4000:
        regime = "Transitional"
    else:
        regime = "Turbulent"
    eps_D = eps / D if D > 0 else 0.0
    f = friction_factor_colebrook(Re, eps_D)
    dP = f * (L / D) * (rho * v ** 2 / 2.0)
    h_L = dP / (rho * 9.81)
    return {
        "Re": Re, "regime": regime, "f": f,
        "dP_Pa": dP, "h_L_m": h_L,
        "v": v, "D": D, "L": L, "rho": rho, "mu": mu, "eps": eps,
    }


def moody_chart_data(eps_D_values: list[float]) -> dict:
    """Return f vs Re data for multiple relative roughness values (Moody chart)."""
    Re_lam = np.linspace(100, 2300, 50)
    f_lam = 64.0 / Re_lam
    Re_turb = np.logspace(math.log10(4000), 8, 300)
    curves = {}
    for eps_D in eps_D_values:
        label = f"ε/D = {eps_D:.1e}" if eps_D > 0 else "Smooth (ε/D = 0)"
        curves[label] = np.array([friction_factor_colebrook(Re, eps_D) for Re in Re_turb])
    return {"Re_lam": Re_lam, "f_lam": f_lam, "Re_turb": Re_turb, "curves": curves}


# ---------------------------------------------------------------------------
# 2. Bernoulli / Energy Balance
# ---------------------------------------------------------------------------

def bernoulli_check(
    P1: float, v1: float, z1: float,
    P2: float, v2: float, z2: float,
    rho: float, g: float = 9.81,
) -> dict:
    """
    Compute total head at each point and head loss h_L = H1 - H2.
    A positive h_L means energy is lost between 1 and 2 (e.g. friction).
    A negative h_L means a pump is adding energy.
    """
    H1 = P1 / (rho * g) + v1 ** 2 / (2 * g) + z1
    H2 = P2 / (rho * g) + v2 ** 2 / (2 * g) + z2
    h_L = H1 - H2
    return {
        "H1": H1, "H2": H2, "h_L": h_L,
        "P1": P1, "v1": v1, "z1": z1,
        "P2": P2, "v2": v2, "z2": z2,
        "rho": rho,
    }


def orifice_flow(Cd: float, D_orifice: float, dP: float, rho: float) -> dict:
    """Q = Cd * A * sqrt(2 ΔP / ρ)"""
    if Cd <= 0 or Cd > 1:
        raise ValueError("Discharge coefficient Cd must be between 0 and 1.")
    A = math.pi * D_orifice ** 2 / 4.0
    Q = Cd * A * math.sqrt(2.0 * dP / rho)
    v = Q / A
    return {"Q_m3s": Q, "v_ms": v, "A_m2": A,
            "Cd": Cd, "D_orifice": D_orifice, "dP": dP, "rho": rho}


# ---------------------------------------------------------------------------
# 3. Pump Sizing
# ---------------------------------------------------------------------------

def pump_sizing(Q: float, rho: float, H: float, eta: float, g: float = 9.81) -> dict:
    """
    Hydraulic power = ρ g Q H
    Shaft power     = P_hyd / η
    """
    if eta <= 0 or eta > 1:
        raise ValueError("Pump efficiency η must be between 0 and 1.")
    P_hydraulic = rho * g * Q * H
    P_shaft = P_hydraulic / eta
    return {
        "Q_m3s": Q, "H_m": H, "eta": eta,
        "P_hydraulic_W": P_hydraulic,
        "P_shaft_W": P_shaft,
        "rho": rho,
    }


def pump_system_curve(
    H_static: float, K_friction: float,
    Q_max: float, n: int = 100,
) -> dict:
    """System curve: H_sys = H_static + K_friction * Q²"""
    Q_arr = np.linspace(0, Q_max, n)
    H_sys = H_static + K_friction * Q_arr ** 2
    return {"Q": Q_arr, "H_sys": H_sys}


def pump_operating_curve(
    H_shutoff: float, Q_BEP: float, H_BEP: float,
    Q_max: float, n: int = 100,
) -> dict:
    """Pump head curve (parabolic): H = H_shutoff - a·Q²"""
    a = (H_shutoff - H_BEP) / (Q_BEP ** 2) if Q_BEP > 0 else 0.0
    Q_arr = np.linspace(0, Q_max, n)
    H_arr = np.maximum(H_shutoff - a * Q_arr ** 2, 0.0)
    return {"Q": Q_arr, "H_pump": H_arr, "H_shutoff": H_shutoff,
            "Q_BEP": Q_BEP, "H_BEP": H_BEP, "a": a}


def pump_operating_point(
    H_shutoff: float, Q_BEP: float, H_BEP: float,
    H_static: float, K_friction: float,
) -> dict:
    """Find intersection of pump curve and system curve."""
    a = (H_shutoff - H_BEP) / (Q_BEP ** 2) if Q_BEP > 0 else 0.0

    def f(Q):
        H_pump = H_shutoff - a * Q ** 2
        H_sys = H_static + K_friction * Q ** 2
        return H_pump - H_sys

    Q_op = None
    try:
        # Pump head equals zero or shutoff head
        Q_right = math.sqrt((H_shutoff - H_static) / (a + K_friction)) if (a + K_friction) > 0 else Q_BEP * 2
        Q_op = brentq(f, 0.0, Q_right * 1.5, xtol=1e-10)
    except Exception:
        pass

    if Q_op is not None and Q_op > 0:
        H_op = H_shutoff - a * Q_op ** 2
    else:
        Q_op, H_op = float("nan"), float("nan")

    return {"Q_op": Q_op, "H_op": H_op}


# ---------------------------------------------------------------------------
# 4. Compressible Flow
# ---------------------------------------------------------------------------

def mach_from_velocity(v: float, gamma: float, R_gas: float, T: float) -> float:
    """M = v / a,  a = sqrt(γ R T)"""
    a = math.sqrt(gamma * R_gas * T)
    return v / a


def isentropic_relations(M: float, gamma: float) -> dict:
    """
    Stagnation ratios:
      T0/T  = 1 + (γ-1)/2 · M²
      P0/P  = (T0/T)^(γ/(γ-1))
      ρ0/ρ  = (T0/T)^(1/(γ-1))
      A/A*  = (1/M) · [(2/(γ+1)) · (1 + (γ-1)/2 · M²)]^((γ+1)/(2(γ-1)))
    """
    T0_T = 1 + (gamma - 1) / 2 * M ** 2
    P0_P = T0_T ** (gamma / (gamma - 1))
    rho0_rho = T0_T ** (1 / (gamma - 1))
    A_Astar = ((1.0 / M)
               * ((2 / (gamma + 1)) * T0_T)
               ** ((gamma + 1) / (2 * (gamma - 1))))
    return {
        "M": M, "gamma": gamma,
        "T0_T": T0_T, "P0_P": P0_P, "rho0_rho": rho0_rho,
        "A_Astar": A_Astar,
        "choked": M >= 1.0,
        "regime": ("Subsonic" if M < 1 else ("Sonic" if M == 1 else "Supersonic")),
    }


def normal_shock(M1: float, gamma: float) -> dict:
    """Normal shock relations across a stationary shock at M1 > 1."""
    if M1 <= 1.0:
        raise ValueError("Normal shock requires M1 > 1.")
    g = gamma
    M2 = math.sqrt((M1 ** 2 + 2 / (g - 1)) / (2 * g / (g - 1) * M1 ** 2 - 1))
    P2_P1 = (2 * g * M1 ** 2 - (g - 1)) / (g + 1)
    T2_T1 = P2_P1 * (2 + (g - 1) * M1 ** 2) / ((g + 1) * M1 ** 2)
    rho2_rho1 = (g + 1) * M1 ** 2 / (2 + (g - 1) * M1 ** 2)
    P02_P01 = (P2_P1 * ((2 + (g - 1) * M2 ** 2) / (2 + (g - 1) * M1 ** 2))
               ** (g / (g - 1)))
    return {
        "M1": M1, "M2": M2, "gamma": gamma,
        "P2_P1": P2_P1, "T2_T1": T2_T1,
        "rho2_rho1": rho2_rho1, "P02_P01": P02_P01,
    }


def isentropic_profile(
    gamma: float, M_max: float = 3.0, n: int = 300
) -> dict:
    """Arrays for plotting isentropic ratios vs Mach number."""
    M_arr = np.linspace(0.01, M_max, n)
    T0_T = 1 + (gamma - 1) / 2 * M_arr ** 2
    P0_P = T0_T ** (gamma / (gamma - 1))
    A_Astar = ((1.0 / M_arr)
               * ((2 / (gamma + 1)) * T0_T)
               ** ((gamma + 1) / (2 * (gamma - 1))))
    return {"M": M_arr, "T0_T": T0_T, "P0_P": P0_P, "A_Astar": A_Astar}
