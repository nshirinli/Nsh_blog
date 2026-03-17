"""Bioprocess engineering calculations.

Covers:
  - Monod / Andrews growth kinetics
  - Batch bioreactor simulation (ODE)
  - Chemostat steady-state analysis
  - Oxygen transfer rate & KLa
  - Thermal sterilization (Del-factor / D-z approach)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


def _monod(S: float, mu_max: float, Ks: float) -> float:
    return mu_max * S / (Ks + S) if (Ks + S) > 1e-12 else 0.0


# ── Growth kinetics ────────────────────────────────────────────────────────────

def growth_kinetics(
    mu_max: float,
    Ks: float,
    Ki: float | None = None,
    S_max: float = 10.0,
) -> dict:
    """Return μ vs [S] data for Monod and (optionally) Andrews models."""
    S = np.linspace(0.001, S_max, 400)
    mu_monod = mu_max * S / (Ks + S)
    result: dict = {
        "S": S.tolist(),
        "mu_monod": mu_monod.tolist(),
        "mu_max": float(mu_max),
        "Ks": float(Ks),
    }
    if Ki is not None and Ki > 0:
        mu_andrews = mu_max * S / (Ks + S + S ** 2 / Ki)
        result["mu_andrews"] = mu_andrews.tolist()
        result["Ki"] = float(Ki)
        S_peak = math.sqrt(Ks * Ki)
        result["S_peak"] = float(S_peak)
        result["mu_peak"] = float(mu_max * S_peak / (Ks + S_peak + S_peak ** 2 / Ki))
    return result


# ── Batch bioreactor ───────────────────────────────────────────────────────────

def batch_bioreactor(
    S0: float,
    X0: float,
    P0: float,
    mu_max: float,
    Ks: float,
    Yxs: float,
    Yps: float,
    ms: float,
    t_end: float,
) -> dict:
    """Simulate batch bioreactor: Monod kinetics with maintenance & product formation."""

    def odes(t, y):
        X, S, P = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0)
        mu = _monod(S, mu_max, Ks)
        dX = mu * X
        dS = -(mu / Yxs + ms) * X
        dP = Yps * (mu / Yxs) * X
        return [dX, dS, dP]

    def substrate_zero(t, y):
        return y[1]
    substrate_zero.terminal = True
    substrate_zero.direction = -1

    sol = solve_ivp(
        odes, [0, t_end], [X0, S0, P0],
        method="RK45", max_step=t_end / 500,
        events=substrate_zero,
    )
    t, (X, S, P) = sol.t, sol.y
    S = np.maximum(S, 0.0)
    productivity = float(P[-1] / t[-1]) if t[-1] > 1e-9 else 0.0
    return {
        "t": t.tolist(),
        "X": X.tolist(),
        "S": S.tolist(),
        "P": P.tolist(),
        "X_final": float(X[-1]),
        "S_final": float(S[-1]),
        "P_final": float(P[-1]),
        "productivity_g_L_h": productivity,
        "t_total": float(t[-1]),
    }


# ── Chemostat ──────────────────────────────────────────────────────────────────

def chemostat_analysis(
    Sin: float,
    mu_max: float,
    Ks: float,
    Yxs: float,
    Yps: float,
) -> dict:
    """Sweep D from zero to washout; return steady-state profiles."""
    D_washout = mu_max * Sin / (Ks + Sin)
    D_opt = mu_max * (1.0 - math.sqrt(Ks / (Ks + Sin)))
    D = np.linspace(0.005 * D_washout, 0.97 * D_washout, 400)
    S_ss = Ks * D / (mu_max - D)
    X_ss = np.maximum(Yxs * (Sin - S_ss), 0.0)
    P_ss = np.maximum(Yps * (Sin - S_ss), 0.0)
    productivity = D * P_ss
    return {
        "D": D.tolist(),
        "S_ss": S_ss.tolist(),
        "X_ss": X_ss.tolist(),
        "P_ss": P_ss.tolist(),
        "productivity": productivity.tolist(),
        "D_washout": float(D_washout),
        "D_opt": float(D_opt),
    }


def chemostat_point(
    D: float,
    Sin: float,
    mu_max: float,
    Ks: float,
    Yxs: float,
    Yps: float,
) -> dict:
    """Steady-state at a single dilution rate D."""
    D_washout = mu_max * Sin / (Ks + Sin)
    if D >= D_washout:
        return {"washout": True, "D_washout": float(D_washout)}
    S_ss = Ks * D / (mu_max - D)
    X_ss = max(Yxs * (Sin - S_ss), 0.0)
    P_ss = max(Yps * (Sin - S_ss), 0.0)
    return {
        "washout": False,
        "S_ss": float(S_ss),
        "X_ss": float(X_ss),
        "P_ss": float(P_ss),
        "productivity": float(D * P_ss),
        "D_washout": float(D_washout),
    }


# ── Oxygen transfer ────────────────────────────────────────────────────────────

def oxygen_transfer(
    OUR: float,     # O2 uptake rate [mmol O2/L/h]
    C_star: float,  # dissolved O2 saturation [mg/L]
    C_L: float,     # actual dissolved O2 [mg/L]
    V: float,       # reactor volume [L]
) -> dict:
    """Calculate KLa requirement and related OTR metrics."""
    OUR_mg = OUR * 32.0  # mmol/L/h → mg O2/L/h (MW O2 = 32)
    driving = C_star - C_L
    KLa = OUR_mg / driving if abs(driving) > 1e-9 else float("nan")
    DO_pct = 100.0 * C_L / C_star if C_star > 1e-9 else float("nan")
    # van't Riet correlation estimate: KLa ≈ 0.026·(P/V)^0.4·Vs^0.5, Vs=0.01 m/s
    try:
        P_over_V = (KLa / (0.026 * 0.01 ** 0.5)) ** (1 / 0.4)
    except Exception:
        P_over_V = float("nan")
    return {
        "OUR_mmol_L_h": float(OUR),
        "OUR_mg_L_h": float(OUR_mg),
        "C_star": float(C_star),
        "C_L": float(C_L),
        "DO_pct": float(DO_pct),
        "driving_force": float(driving),
        "KLa": float(KLa),
        "total_O2_g_h": float(OUR_mg * V / 1000.0),
        "P_over_V_kW_m3": float(P_over_V),
    }


# ── Sterilization ──────────────────────────────────────────────────────────────

def sterilization(
    T_steril: float,   # sterilization temperature [°C]
    D_121: float,      # D-value at 121°C [min]
    z: float,          # z-value [°C]
    N0: float,         # initial contamination [organisms]
    N_target: float,   # target contamination (e.g. 1e-3)
    t_hold: float,     # hold time at T_steril [min]
) -> dict:
    """Thermal sterilization using D-value / z-value approach."""
    D_T = D_121 * 10.0 ** ((121.0 - T_steril) / z)
    nabla_req = math.log(N0 / N_target)          # ln-based Del factor
    nabla_actual = math.log(10) * t_hold / D_T   # Del factor achieved
    t_req = nabla_req * D_T / math.log(10)
    N_final = N0 * 10.0 ** (-(t_hold / D_T))

    T_range = np.linspace(100.0, 145.0, 200)
    D_range = D_121 * 10.0 ** ((121.0 - T_range) / z)
    t_req_range = nabla_req * D_range / math.log(10)

    return {
        "T_steril": float(T_steril),
        "D_at_T": float(D_T),
        "z": float(z),
        "nabla_required": float(nabla_req),
        "nabla_actual": float(nabla_actual),
        "t_required_min": float(t_req),
        "t_hold_min": float(t_hold),
        "safe": bool(nabla_actual >= nabla_req),
        "margin": float(nabla_actual - nabla_req),
        "N_final": float(N_final),
        "T_range": T_range.tolist(),
        "t_req_curve": t_req_range.tolist(),
    }
