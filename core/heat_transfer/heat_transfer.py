"""Heat Transfer solvers for four tabs:

  1. Conduction   — Fourier's law, flat/cylindrical walls, composite resistance
  2. Convection   — Newton's law of cooling, pipe-flow correlations
  3. Heat Exchangers — LMTD (co/counter) and NTU-effectiveness
  4. Radiation    — Stefan-Boltzmann, grey-body, Planck spectrum
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# 1. Conduction
# ---------------------------------------------------------------------------

def conduction_flat_wall(k: float, A: float, T1: float, T2: float, L: float) -> dict:
    """Q = k A (T1 - T2) / L"""
    if L <= 0:
        raise ValueError("Thickness L must be > 0.")
    Q = k * A * (T1 - T2) / L
    R = L / (k * A)
    return {"Q_W": Q, "R_KW": R, "dT": T1 - T2, "L": L, "k": k, "A": A}


def conduction_composite_wall(
    layers: list[dict],   # each: {"L": float, "k": float, "label": str}
    A: float,
    T_in: float,
    T_out: float,
) -> dict:
    """Composite flat wall with N layers in series."""
    R_layers = [lay["L"] / (lay["k"] * A) for lay in layers]
    R_total = sum(R_layers)
    Q = (T_in - T_out) / R_total if R_total > 1e-20 else 0.0
    dT_layers = [Q * R for R in R_layers]
    T_profile = [T_in]
    for dT in dT_layers:
        T_profile.append(T_profile[-1] - dT)
    return {
        "Q_W": Q,
        "R_total": R_total,
        "R_layers": R_layers,
        "dT_layers": dT_layers,
        "T_profile": T_profile,
        "layers": layers,
        "A": A,
    }


def conduction_cylinder(
    k: float, L: float, r1: float, r2: float, T1: float, T2: float
) -> dict:
    """Q = 2π k L (T1 - T2) / ln(r2/r1)"""
    if r2 <= r1:
        raise ValueError("Outer radius r2 must be greater than inner radius r1.")
    ln_r = math.log(r2 / r1)
    Q = 2 * math.pi * k * L * (T1 - T2) / ln_r
    R = ln_r / (2 * math.pi * k * L)
    return {"Q_W": Q, "R_KW": R, "dT": T1 - T2, "k": k, "L": L, "r1": r1, "r2": r2}


# ---------------------------------------------------------------------------
# 2. Convection
# ---------------------------------------------------------------------------

def convection_newton(h: float, A: float, T_surface: float, T_fluid: float) -> dict:
    """Q = h A (T_s - T_f)"""
    Q = h * A * (T_surface - T_fluid)
    return {"Q_W": Q, "h": h, "A": A, "dT": T_surface - T_fluid}


def _nusselt_dittus_boelter(Re: float, Pr: float, heating: bool) -> float:
    if Re < 10_000:
        raise ValueError("Dittus-Boelter requires Re > 10 000 (turbulent flow).")
    n = 0.4 if heating else 0.3
    return 0.023 * (Re ** 0.8) * (Pr ** n)


def _nusselt_gnielinski(Re: float, Pr: float) -> float:
    if Re < 3_000:
        raise ValueError("Gnielinski correlation requires Re > 3 000.")
    f = (0.790 * math.log(Re) - 1.64) ** (-2)   # Petukhov friction factor
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


def pipe_flow_convection(
    Re: float,
    Pr: float,
    k_fluid: float,
    D: float,
    correlation: str = "Dittus-Boelter",
    heating: bool = True,
) -> dict:
    """Compute Nusselt number and h for internal pipe flow."""
    if correlation == "Gnielinski":
        Nu = _nusselt_gnielinski(Re, Pr)
    else:
        Nu = _nusselt_dittus_boelter(Re, Pr, heating)
    h = Nu * k_fluid / D
    return {"Nu": Nu, "h": h, "Re": Re, "Pr": Pr, "k_fluid": k_fluid, "D": D,
            "correlation": correlation}


def h_vs_Re_data(
    Re_min: float, Re_max: float, Pr: float, k_fluid: float, D: float,
    correlation: str = "Dittus-Boelter", heating: bool = True, n: int = 200,
) -> dict:
    """Return Re array and corresponding h values for a plot."""
    Re_arr = np.logspace(math.log10(max(Re_min, 1)), math.log10(Re_max), n)
    h_arr = []
    for Re in Re_arr:
        try:
            res = pipe_flow_convection(Re, Pr, k_fluid, D, correlation, heating)
            h_arr.append(res["h"])
        except ValueError:
            h_arr.append(float("nan"))
    return {"Re": Re_arr, "h": np.array(h_arr), "Pr": Pr, "k_fluid": k_fluid, "D": D}


# ---------------------------------------------------------------------------
# 3. Heat Exchangers
# ---------------------------------------------------------------------------

def _lmtd(T_h_in: float, T_h_out: float, T_c_in: float, T_c_out: float,
          flow: str = "counter") -> float:
    if flow == "counter":
        dT1, dT2 = T_h_in - T_c_out, T_h_out - T_c_in
    else:
        dT1, dT2 = T_h_in - T_c_in, T_h_out - T_c_out
    if dT1 <= 0 or dT2 <= 0:
        raise ValueError("Temperature cross detected — LMTD not valid for this configuration.")
    if abs(dT1 - dT2) < 1e-10:
        return dT1
    return (dT1 - dT2) / math.log(dT1 / dT2)


def heat_exchanger_lmtd(
    T_h_in: float, T_h_out: float,
    T_c_in: float, T_c_out: float,
    U: float, A: float,
    flow: str = "counter",
) -> dict:
    """Q = U A LMTD.  If A=0, calculates required A from Q (energy balance)."""
    LMTD = _lmtd(T_h_in, T_h_out, T_c_in, T_c_out, flow)
    Q = U * A * LMTD
    return {
        "Q_W": Q, "LMTD": LMTD, "U": U, "A": A, "flow": flow,
        "T_h_in": T_h_in, "T_h_out": T_h_out,
        "T_c_in": T_c_in, "T_c_out": T_c_out,
    }


def heat_exchanger_ntu(
    C_hot: float, C_cold: float,
    U: float, A: float,
    T_h_in: float, T_c_in: float,
    hx_type: str = "counter",
) -> dict:
    """NTU-effectiveness method.  C = m_dot * cp  [W/K]."""
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_r = C_min / C_max if C_max > 0 else 0.0
    NTU = U * A / C_min if C_min > 0 else 0.0

    if hx_type == "parallel":
        eps = (1 - math.exp(-NTU * (1 + C_r))) / (1 + C_r)
    elif hx_type == "crossflow_unmixed":
        eps = (1 - math.exp((NTU ** 0.22 / C_r) * (math.exp(-C_r * NTU ** 0.78) - 1))
               if C_r > 1e-10 else 1 - math.exp(-NTU))
    else:   # counter-flow
        if abs(C_r - 1.0) < 1e-8:
            eps = NTU / (NTU + 1)
        else:
            eps = ((1 - math.exp(-NTU * (1 - C_r)))
                   / (1 - C_r * math.exp(-NTU * (1 - C_r))))

    Q_max = C_min * (T_h_in - T_c_in)
    Q = eps * Q_max
    T_h_out = T_h_in - Q / C_hot if C_hot > 0 else T_h_in
    T_c_out = T_c_in + Q / C_cold if C_cold > 0 else T_c_in
    return {
        "NTU": NTU, "C_r": C_r, "eps": eps,
        "Q_W": Q, "Q_max": Q_max,
        "T_h_out": T_h_out, "T_c_out": T_c_out,
        "C_min": C_min, "C_max": C_max,
        "U": U, "A": A,
    }


# ---------------------------------------------------------------------------
# 4. Radiation
# ---------------------------------------------------------------------------

_SIGMA = 5.670374419e-8   # W/(m² K⁴)


def radiation_blackbody(T: float, A: float = 1.0) -> dict:
    """Total emissive power: Q = σ A T⁴"""
    if T < 0:
        raise ValueError("Temperature must be ≥ 0 K.")
    Q = _SIGMA * A * T ** 4
    lam_max = 2897.8e-6 / T if T > 0 else float("inf")   # Wien displacement [m]
    return {"Q_W": Q, "T": T, "A": A, "sigma": _SIGMA,
            "lambda_max_um": lam_max * 1e6}


def radiation_grey_body(
    T1: float, T2: float, eps: float, A: float, F12: float = 1.0
) -> dict:
    """Net radiation: Q = σ ε A F₁₂ (T1⁴ − T2⁴)"""
    if not (0 < eps <= 1):
        raise ValueError("Emissivity must be between 0 and 1.")
    Q = _SIGMA * eps * A * F12 * (T1 ** 4 - T2 ** 4)
    return {"Q_W": Q, "T1": T1, "T2": T2, "eps": eps, "A": A, "F12": F12}


def radiation_planck_spectrum(
    T: float,
    lam_min_um: float = 0.1,
    lam_max_um: float = 100.0,
    n: int = 500,
) -> dict:
    """Planck spectral emissive power E_λ [W/(m³)] vs wavelength [μm]."""
    if T <= 0:
        raise ValueError("Temperature must be > 0 K.")
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    lam = np.linspace(lam_min_um * 1e-6, lam_max_um * 1e-6, n)
    E = (2 * h * c ** 2) / (lam ** 5 * (np.exp(h * c / (k * T * lam)) - 1))
    peak_lam_um = float(lam[int(np.argmax(E))] * 1e6)
    wien_lam_um = 2897.8 / T   # Wien displacement law
    return {
        "lam_um": lam * 1e6,
        "E": E,
        "T": T,
        "peak_lam_um": peak_lam_um,
        "wien_lam_um": wien_lam_um,
    }
