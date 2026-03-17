"""Electrochemistry calculations.

Covers:
  - Nernst equation (cell potential vs. reaction quotient & temperature)
  - Butler-Volmer kinetics (i-η curve, Tafel slopes)
  - Faraday's law (mass deposited / dissolved)
  - Fuel cell polarization curve (H₂/O₂ PEM cell)
  - Corrosion rate estimation (mixed potential theory)
"""
from __future__ import annotations

import math

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
_F = 96485.0   # C/mol  (Faraday constant)
_R = 8.314     # J/(mol·K)


# ── Nernst equation ────────────────────────────────────────────────────────────

def nernst_equation(E0: float, n: int, Q: float, T_K: float = 298.15) -> dict:
    """
    E = E° − (RT / nF) · ln(Q)

    E0  : standard cell potential [V]
    n   : electrons transferred
    Q   : reaction quotient (dimensionless activity product)
    T_K : temperature [K]
    """
    if n <= 0:
        return {"error": "n must be a positive integer"}
    if Q <= 0:
        return {"error": "Q must be positive (product of activities)"}

    correction = (_R * T_K / (n * _F)) * math.log(Q)
    E = E0 - correction

    T_range = np.linspace(273.15, 373.15, 200)
    E_vs_T = E0 - (_R * T_range / (n * _F)) * math.log(Q)

    return {
        "E0": float(E0),
        "E": float(E),
        "n": int(n),
        "Q": float(Q),
        "T_K": float(T_K),
        "correction_V": float(correction),
        "T_range_K": T_range.tolist(),
        "E_vs_T": E_vs_T.tolist(),
    }


# ── Butler-Volmer kinetics ─────────────────────────────────────────────────────

def butler_volmer(
    i0: float,
    alpha: float,
    T_K: float = 298.15,
    eta_max: float = 0.5,
) -> dict:
    """
    i = i₀ · [exp(α·F·η/RT) − exp(−(1−α)·F·η/RT)]

    i0    : exchange current density [A/cm²]
    alpha : anodic transfer coefficient (0–1)
    T_K   : temperature [K]
    """
    eta = np.linspace(-eta_max, eta_max, 800)
    FRT = _F / (_R * T_K)
    i = i0 * (np.exp(alpha * FRT * eta) - np.exp(-(1.0 - alpha) * FRT * eta))

    # Tafel slopes [mV/decade]
    ba = 2.303 * _R * T_K / (alpha * _F) * 1000
    bc = 2.303 * _R * T_K / ((1.0 - alpha) * _F) * 1000

    return {
        "eta": eta.tolist(),
        "i": i.tolist(),
        "i0": float(i0),
        "alpha": float(alpha),
        "T_K": float(T_K),
        "ba_mV_dec": float(ba),
        "bc_mV_dec": float(bc),
    }


# ── Faraday's law ──────────────────────────────────────────────────────────────

def faraday_law(
    current: float,      # [A]
    time_s: float,       # [s]
    M_molar: float,      # molar mass of deposited species [g/mol]
    n: int,              # electrons per formula unit
    current_eff: float = 1.0,  # current efficiency (0–1)
) -> dict:
    """m = (I · t · η_c · M) / (n · F)"""
    Q = current * time_s
    mass_g = Q * current_eff * M_molar / (n * _F)
    moles = mass_g / M_molar

    # Cumulative mass vs time
    t_range = np.linspace(0, time_s, 300)
    mass_vs_t = current * t_range * current_eff * M_molar / (n * _F)

    return {
        "Q_C": float(Q),
        "mass_g": float(mass_g),
        "moles": float(moles),
        "current_A": float(current),
        "time_s": float(time_s),
        "current_eff": float(current_eff),
        "t_range": t_range.tolist(),
        "mass_vs_t": mass_vs_t.tolist(),
    }


# ── Fuel cell polarization ─────────────────────────────────────────────────────

def fuel_cell_polarization(
    T_K: float = 353.15,        # operating temperature [K]
    i_max: float = 1.5,         # maximum current density to plot [A/cm²]
    i0_cathode: float = 1e-6,   # cathode exchange current density [A/cm²]
    R_ohmic: float = 0.1,       # cell ohmic resistance [Ω·cm²]
    alpha_c: float = 0.5,       # cathode transfer coefficient
    i_limit: float = 1.8,       # limiting current density [A/cm²]
) -> dict:
    """
    PEM H₂/O₂ polarization curve:
      V = E_rev − η_act − η_ohm − η_conc

    E_rev   : reversible cell voltage (corrected for T)
    η_act   : activation loss  (ORR at cathode dominates)
    η_ohm   : ohmic loss
    η_conc  : mass-transport / concentration loss
    """
    E_rev = 1.229 - 8.5e-4 * (T_K - 298.15)   # simplified linear T correction
    eta_th = 237_100 / 285_830                  # ΔG/ΔH  (thermodynamic efficiency)

    i = np.linspace(1e-4, min(i_max, i_limit * 0.97), 500)
    FRT = _F / (_R * T_K)

    eta_act = (_R * T_K / (alpha_c * _F)) * np.log(i / i0_cathode)
    eta_ohm = i * R_ohmic
    with np.errstate(divide="ignore", invalid="ignore"):
        eta_conc = -(_R * T_K / (2 * _F)) * np.log(1.0 - i / i_limit)
        eta_conc = np.where(np.isfinite(eta_conc), eta_conc, np.nan)

    V_cell = E_rev - eta_act - eta_ohm - eta_conc
    V_cell = np.where(V_cell > 0.0, V_cell, np.nan)
    P_density = i * V_cell   # W/cm²

    return {
        "i": i.tolist(),
        "V_cell": V_cell.tolist(),
        "P_density": P_density.tolist(),
        "eta_act": eta_act.tolist(),
        "eta_ohm": eta_ohm.tolist(),
        "eta_conc": eta_conc.tolist(),
        "E_rev": float(E_rev),
        "eta_th": float(eta_th),
        "T_K": float(T_K),
    }


# ── Corrosion rate ─────────────────────────────────────────────────────────────

def corrosion_rate(
    i_corr: float,   # corrosion current density [µA/cm²]
    M_molar: float,  # molar mass of metal [g/mol]
    n: int,          # valence (electrons per atom)
    rho: float,      # density [g/cm³]
    area: float = 1.0,  # exposed area [cm²]
) -> dict:
    """
    CR [mm/yr] = (i_corr · M) / (n · F · ρ) × unit conversions
    """
    i_A = i_corr * 1e-6  # µA/cm² → A/cm²
    mass_rate = i_A * M_molar / (n * _F)   # g/cm²/s
    pen_rate_cm_s = mass_rate / rho        # cm/s
    sec_yr = 365.25 * 24 * 3600
    CR_mm_yr = pen_rate_cm_s * sec_yr * 10.0   # mm/yr
    CR_mpy = CR_mm_yr / 0.0254                  # mils per year
    mass_loss_yr = mass_rate * area * sec_yr    # g/yr

    if CR_mm_yr < 0.1:
        category = "Outstanding  (< 0.1 mm/yr)"
    elif CR_mm_yr < 0.5:
        category = "Excellent    (0.1 – 0.5 mm/yr)"
    elif CR_mm_yr < 1.0:
        category = "Good         (0.5 – 1.0 mm/yr)"
    elif CR_mm_yr < 5.0:
        category = "Fair         (1 – 5 mm/yr)"
    else:
        category = "Poor         (> 5 mm/yr)"

    return {
        "i_corr_uA": float(i_corr),
        "CR_mm_yr": float(CR_mm_yr),
        "CR_mpy": float(CR_mpy),
        "mass_loss_g_yr": float(mass_loss_yr),
        "category": category,
    }
