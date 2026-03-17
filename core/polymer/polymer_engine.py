"""Polymer engineering calculations.

Covers:
  - Molecular weight statistics (Mn, Mw, Mz, PDI)
  - Flory-Huggins solution thermodynamics
  - Mark-Houwink-Sakurada viscosity equation
  - Glass transition: Fox equation & WLF shift factor
  - Free-radical polymerization kinetics (QSSA)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


# ── Molecular weight statistics ────────────────────────────────────────────────

def mw_statistics(Ni: list[float], Mi: list[float]) -> dict:
    """Compute Mn, Mw, Mz, and dispersity from a discrete MW distribution."""
    Ni_arr = np.array(Ni, dtype=float)
    Mi_arr = np.array(Mi, dtype=float)

    sum_N = np.sum(Ni_arr)
    sum_NM = np.sum(Ni_arr * Mi_arr)
    sum_NM2 = np.sum(Ni_arr * Mi_arr ** 2)
    sum_NM3 = np.sum(Ni_arr * Mi_arr ** 3)

    Mn = sum_NM / sum_N
    Mw = sum_NM2 / sum_NM
    Mz = sum_NM3 / sum_NM2
    PDI = Mw / Mn

    wi = Ni_arr * Mi_arr / sum_NM
    ni = Ni_arr / sum_N

    return {
        "Mn": float(Mn),
        "Mw": float(Mw),
        "Mz": float(Mz),
        "PDI": float(PDI),
        "Mi": Mi_arr.tolist(),
        "ni": ni.tolist(),   # mole fraction distribution
        "wi": wi.tolist(),   # weight fraction distribution
    }


# ── Flory-Huggins ──────────────────────────────────────────────────────────────

def flory_huggins(chi: float, r: float) -> dict:
    """
    Flory-Huggins mixing free energy for polymer solution.

    ΔGmix / (n·R·T) = φ₁·ln(φ₁) + (φ₂/r)·ln(φ₂) + χ·φ₁·φ₂

    φ₁ = solvent volume fraction, φ₂ = polymer volume fraction,
    r  = degree of polymerization.
    """
    phi2 = np.linspace(1e-4, 1 - 1e-4, 600)
    phi1 = 1.0 - phi2

    dG = phi1 * np.log(phi1) + (phi2 / r) * np.log(phi2) + chi * phi1 * phi2
    dS = -(phi1 * np.log(phi1) + (phi2 / r) * np.log(phi2))
    dH = chi * phi1 * phi2

    # Critical parameters
    chi_c = 0.5 * (1.0 + 1.0 / math.sqrt(r)) ** 2
    phi2_c = 1.0 / (1.0 + math.sqrt(r))

    # Spinodal: d²G/dφ₂² = 1/φ₁ + 1/(r·φ₂) − 2χ = 0
    d2G = 1.0 / phi1 + 1.0 / (r * phi2) - 2.0 * chi
    spinodal = []
    for i in range(len(d2G) - 1):
        if d2G[i] * d2G[i + 1] < 0:
            spinodal.append(float(phi2[i]))

    return {
        "phi2": phi2.tolist(),
        "dG": dG.tolist(),
        "dS": dS.tolist(),
        "dH": dH.tolist(),
        "chi": float(chi),
        "r": float(r),
        "chi_critical": float(chi_c),
        "phi2_critical": float(phi2_c),
        "spinodal": spinodal,
        "miscible": bool(chi < chi_c),
    }


# ── Mark-Houwink-Sakurada ──────────────────────────────────────────────────────

def mark_houwink(
    K: float,
    alpha: float,
    M_range: tuple[float, float] = (1e3, 1e6),
    M_known: float | None = None,
) -> dict:
    """[η] = K·M^α  (Mark-Houwink-Sakurada equation)."""
    M = np.logspace(math.log10(M_range[0]), math.log10(M_range[1]), 400)
    eta = K * M ** alpha
    result: dict = {
        "M": M.tolist(),
        "eta_intrinsic": eta.tolist(),
        "K": float(K),
        "alpha": float(alpha),
    }
    if M_known is not None and M_known > 0:
        result["M_known"] = float(M_known)
        result["eta_at_M"] = float(K * M_known ** alpha)
    return result


# ── Glass transition ───────────────────────────────────────────────────────────

def glass_transition_fox(components: list[dict]) -> dict:
    """
    Fox equation for copolymer/blend glass transition:
        1/Tg = Σ (wᵢ / Tgᵢ)
    Each component: {"w": weight_fraction, "Tg": Tg_in_K}
    """
    if not components:
        return {}
    inv_Tg = sum(c["w"] / c["Tg"] for c in components)
    Tg = 1.0 / inv_Tg if inv_Tg > 1e-12 else float("nan")
    return {
        "Tg_K": float(Tg),
        "Tg_C": float(Tg - 273.15),
        "components": components,
    }


def wlf_shift(T: float, T_ref: float, C1: float = 17.44, C2: float = 51.6) -> dict:
    """
    WLF equation: log₁₀(aT) = −C1·(T − T_ref) / (C2 + (T − T_ref))
    Universal constants: C1 = 17.44, C2 = 51.6 (T_ref = Tg).
    """
    dT = T - T_ref
    denom = C2 + dT
    if abs(denom) < 1e-9:
        return {"error": "C2 + (T − Tref) ≈ 0  (singular point)"}
    log_aT = -C1 * dT / denom
    aT = 10.0 ** log_aT

    # Scan T from T_ref to T_ref + 150
    T_range = np.linspace(T_ref, T_ref + 150.0, 300)
    dT_range = T_range - T_ref
    log_aT_range = -C1 * dT_range / (C2 + dT_range)

    return {
        "T": float(T),
        "T_ref": float(T_ref),
        "C1": float(C1),
        "C2": float(C2),
        "log_aT": float(log_aT),
        "aT": float(aT),
        "T_range": T_range.tolist(),
        "log_aT_range": log_aT_range.tolist(),
    }


# ── Free-radical polymerization ────────────────────────────────────────────────

def free_radical_kinetics(
    kp: float,    # propagation rate constant [L/mol/s]
    kt: float,    # termination rate constant  [L/mol/s]
    kd: float,    # initiator decomp. rate     [1/s]
    f: float,     # initiator efficiency (0–1)
    I0: float,    # initial initiator concentration [mol/L]
    M0: float,    # initial monomer concentration   [mol/L]
    t_end: float, # reaction duration               [s]
) -> dict:
    """
    QSSA on radical: [M·] = (f·kd·[I] / kt)^0.5
    Rp = kp·[M]·[M·]
    DPn ≈ kp·[M] / (2·(kt·f·kd·[I])^0.5)  (combination termination)
    """

    def odes(t, y):
        M, I = max(y[0], 0.0), max(y[1], 0.0)
        M_rad = math.sqrt(f * kd * I / kt) if kt > 0 and (f * kd * I) >= 0 else 0.0
        dM = -kp * M * M_rad
        dI = -kd * I
        return [dM, dI]

    sol = solve_ivp(odes, [0, t_end], [M0, I0], method="RK45", max_step=t_end / 500)
    t, (M, I) = sol.t, sol.y
    M = np.maximum(M, 0.0)
    I = np.maximum(I, 0.0)

    M_rad = np.sqrt(np.maximum(f * kd * I / kt, 0.0))
    Rp = kp * M * M_rad
    conversion = (M0 - M) / M0
    with np.errstate(divide="ignore", invalid="ignore"):
        DPn = np.where(M_rad > 1e-30, kp * M / (2.0 * kt * M_rad), np.nan)

    return {
        "t": t.tolist(),
        "M": M.tolist(),
        "I": I.tolist(),
        "Rp": Rp.tolist(),
        "conversion": conversion.tolist(),
        "DPn": DPn.tolist(),
        "conversion_final": float(conversion[-1]),
        "Rp_initial": float(Rp[0]),
        "M_final": float(M[-1]),
    }
