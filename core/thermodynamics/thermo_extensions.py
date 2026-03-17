"""Extended thermodynamics calculations:
  - Enthalpy & entropy from Cp polynomial (NASA/DIPPR form)
  - Kirchhoff's law: ΔH_rxn(T)
  - Adiabatic flame temperature
  - Activity coefficients (Margules, van Laar)
  - Non-ideal VLE with activity coefficients
  - Psychrometrics (humid air)
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


R = 8.314  # J/(mol·K)
_MW_WATER = 18.015
_MW_AIR   = 28.97


# ---------------------------------------------------------------------------
# 1. Enthalpy & entropy from Cp polynomial
#    Cp [J/(mol·K)] = a + b*T + c*T^2 + d*T^3   (T in K)
# ---------------------------------------------------------------------------

def cp_at_T(a: float, b: float, c: float, d: float, T_K: float) -> float:
    """Cp [J/(mol·K)] at temperature T_K."""
    return a + b*T_K + c*T_K**2 + d*T_K**3


def enthalpy_change(a: float, b: float, c: float, d: float,
                    T1_K: float, T2_K: float) -> float:
    """ΔH = ∫Cp dT from T1 to T2 [J/mol]."""
    return (a*(T2_K - T1_K)
            + b/2*(T2_K**2 - T1_K**2)
            + c/3*(T2_K**3 - T1_K**3)
            + d/4*(T2_K**4 - T1_K**4))


def entropy_change(a: float, b: float, c: float, d: float,
                   T1_K: float, T2_K: float) -> float:
    """ΔS = ∫(Cp/T) dT from T1 to T2 [J/(mol·K)]."""
    return (a*math.log(T2_K/T1_K)
            + b*(T2_K - T1_K)
            + c/2*(T2_K**2 - T1_K**2)
            + d/3*(T2_K**3 - T1_K**3))


def gibbs_change(a: float, b: float, c: float, d: float,
                 T1_K: float, T2_K: float, dH_ref: float, T_ref: float = 298.15) -> dict:
    """Compute ΔH, ΔS, ΔG for heating from T1 to T2.
    dH_ref is an optional reference enthalpy offset (e.g. ΔHf° at T_ref).
    Returns dict with dH_J, dS_J_K, dG_J, Cp_T1, Cp_T2.
    """
    dH = enthalpy_change(a, b, c, d, T1_K, T2_K) + dH_ref
    dS = entropy_change(a, b, c, d, T1_K, T2_K)
    T_avg = (T1_K + T2_K) / 2
    dG = dH - T2_K * dS
    return {
        "dH_J": dH,
        "dS_J_K": dS,
        "dG_J": dG,
        "Cp_T1": cp_at_T(a, b, c, d, T1_K),
        "Cp_T2": cp_at_T(a, b, c, d, T2_K),
    }


def cp_curve(a: float, b: float, c: float, d: float,
             T_min_K: float, T_max_K: float, n: int = 200) -> tuple:
    """Return (T_arr, Cp_arr) over [T_min, T_max]."""
    T_arr = np.linspace(T_min_K, T_max_K, n)
    Cp_arr = a + b*T_arr + c*T_arr**2 + d*T_arr**3
    return T_arr, Cp_arr


def enthalpy_curve(a: float, b: float, c: float, d: float,
                   T_ref_K: float, T_max_K: float, n: int = 200) -> tuple:
    """Return (T_arr, H_arr) where H(T) = ∫Cp dT from T_ref to T [kJ/mol]."""
    T_arr = np.linspace(T_ref_K, T_max_K, n)
    H_arr = np.array([enthalpy_change(a, b, c, d, T_ref_K, T) / 1000 for T in T_arr])
    return T_arr, H_arr


# ---------------------------------------------------------------------------
# 2. Kirchhoff's law — ΔH_rxn(T2) from ΔH_rxn(T1) and ΔCp
#    ΔHrxn(T2) = ΔHrxn(T1) + ∫ΔCp dT  from T1 to T2
#    ΔCp = Σ(product Cp) - Σ(reactant Cp)  (stoich-weighted)
# ---------------------------------------------------------------------------

def kirchhoff_dHrxn(
    dHrxn_ref_J: float,
    T_ref_K: float,
    T_calc_K: float,
    delta_a: float,
    delta_b: float,
    delta_c: float,
    delta_d: float,
) -> dict:
    """Kirchhoff correction: ΔH_rxn(T) = ΔH_rxn(T_ref) + ∫ΔCp dT.

    delta_a..d = Σ(stoich_prod * a_prod) - Σ(stoich_reac * a_reac), same for b, c, d.
    """
    dH_corr = enthalpy_change(delta_a, delta_b, delta_c, delta_d, T_ref_K, T_calc_K)
    dHrxn_T = dHrxn_ref_J + dH_corr

    T_arr = np.linspace(min(T_ref_K, T_calc_K), max(T_ref_K, T_calc_K), 200)
    dHrxn_arr = np.array([
        (dHrxn_ref_J + enthalpy_change(delta_a, delta_b, delta_c, delta_d, T_ref_K, T)) / 1000
        for T in T_arr
    ])
    return {
        "dHrxn_ref_J": dHrxn_ref_J,
        "dHrxn_T_J": dHrxn_T,
        "dHrxn_T_kJ": dHrxn_T / 1000,
        "T_arr": T_arr,
        "dHrxn_arr_kJ": dHrxn_arr,
        "T_ref": T_ref_K,
        "T_calc": T_calc_K,
        "dH_corr_J": dH_corr,
    }


# ---------------------------------------------------------------------------
# 3. Adiabatic Flame Temperature (constant-pressure, complete combustion)
#    Reactants at T_in → Products at T_ad
#    ΔH_rxn(T_ref) + ∫Cp_reac*(T_in - T_ref) = ∫Cp_prod*(T_ad - T_ref)
#
#    Simplified: compute for general user-specified ΔH_comb and Cp_products
# ---------------------------------------------------------------------------

def adiabatic_flame_temperature(
    dH_comb_J: float,       # heat of combustion (negative, exothermic)
    T_reactants_K: float,   # inlet T of reactants
    Cp_products_J_molK: float,  # average Cp of combustion products (J/mol/K)
    n_products: float = 1.0,    # total moles of products per mole fuel
    T_ref_K: float = 298.15,
) -> dict:
    """Estimate adiabatic flame temperature.

    Energy balance:
      -ΔH_comb = n_prod * Cp_prod * (T_ad - T_ref) - Cp_reac*(T_reac - T_ref)
    Simplified for user-specified average Cp_products (assumed constant).
    """
    # Heat released at T_ref
    Q_released = -dH_comb_J   # positive for exothermic

    # Sensible heat of reactants above T_ref  (approx Cp_products as similar)
    Q_reactants = Cp_products_J_molK * n_products * (T_reactants_K - T_ref_K)

    # T_ad = T_ref + (Q_released + Q_reactants) / (n_prod * Cp_prod)
    denom = n_products * Cp_products_J_molK
    if denom <= 0:
        raise ValueError("n_products * Cp_products must be > 0")

    T_ad = T_ref_K + (Q_released + Q_reactants) / denom

    # Sweep over equivalence ratios
    phi_arr = np.linspace(0.5, 2.0, 200)
    T_ad_arr = np.zeros(len(phi_arr))
    for i, phi in enumerate(phi_arr):
        # Rich: excess fuel absorbs heat; lean: excess air absorbs heat
        eff_Q = Q_released * min(phi, 1.0) / phi if phi > 0 else 0.0
        # Approximate: dilution by excess air lowers T
        Cp_eff = Cp_products_J_molK * n_products * (1.0 + max(0, 1/phi - 1) * 1.4)
        T_ad_arr[i] = T_ref_K + (eff_Q * phi + Q_reactants) / max(Cp_eff, 1.0)

    return {
        "T_ad_K": T_ad,
        "T_ad_C": T_ad - 273.15,
        "Q_released_J": Q_released,
        "phi_arr": phi_arr,
        "T_ad_phi_arr": T_ad_arr,
        "T_reactants_K": T_reactants_K,
        "dH_comb_J": dH_comb_J,
    }


# ---------------------------------------------------------------------------
# 4. Activity coefficients — Margules & van Laar for binary systems
# ---------------------------------------------------------------------------

def margules_one_suffix(A: float, x1_arr: np.ndarray) -> tuple:
    """One-suffix (symmetric) Margules: ln γ1 = A*x2^2, ln γ2 = A*x1^2."""
    x2_arr = 1.0 - x1_arr
    ln_g1 = A * x2_arr**2
    ln_g2 = A * x1_arr**2
    return np.exp(ln_g1), np.exp(ln_g2)


def margules_two_suffix(A12: float, A21: float, x1_arr: np.ndarray) -> tuple:
    """Two-suffix (asymmetric) Margules:
    ln γ1 = x2^2 * [A12 + 2*(A21-A12)*x1]
    ln γ2 = x1^2 * [A21 + 2*(A12-A21)*x2]
    """
    x2_arr = 1.0 - x1_arr
    ln_g1 = x2_arr**2 * (A12 + 2*(A21 - A12)*x1_arr)
    ln_g2 = x1_arr**2 * (A21 + 2*(A12 - A21)*x2_arr)
    return np.exp(ln_g1), np.exp(ln_g2)


def van_laar(A12: float, A21: float, x1_arr: np.ndarray) -> tuple:
    """van Laar activity coefficients:
    ln γ1 = A12 / (1 + A12*x1/(A21*x2))^2
    ln γ2 = A21 / (1 + A21*x2/(A12*x1))^2
    """
    x2_arr = 1.0 - x1_arr
    eps = 1e-12
    denom1 = (1.0 + A12 * x1_arr / (A21 * x2_arr + eps))**2
    denom2 = (1.0 + A21 * x2_arr / (A12 * x1_arr + eps))**2
    ln_g1 = A12 / denom1
    ln_g2 = A21 / denom2
    return np.exp(np.clip(ln_g1, -20, 20)), np.exp(np.clip(ln_g2, -20, 20))


def nonideal_vle_pxy(
    A12: float,
    A21: float,
    model: str,
    Psat1_mmHg: float,
    Psat2_mmHg: float,
    n_pts: int = 100,
) -> dict:
    """Compute P-x-y diagram for non-ideal binary with activity coefficients.

    y1 = γ1 * x1 * Psat1 / P_total
    P_total = γ1*x1*Psat1 + γ2*x2*Psat2

    Returns x1_arr, y1_bubble, P_bubble.
    """
    x1_arr = np.linspace(1e-4, 1 - 1e-4, n_pts)
    x2_arr = 1.0 - x1_arr

    if model == "Margules (1-suffix)":
        g1, g2 = margules_one_suffix(A12, x1_arr)
    elif model == "Margules (2-suffix)":
        g1, g2 = margules_two_suffix(A12, A21, x1_arr)
    else:  # van Laar
        g1, g2 = van_laar(A12, A21, x1_arr)

    P_arr = g1 * x1_arr * Psat1_mmHg + g2 * x2_arr * Psat2_mmHg
    y1_arr = g1 * x1_arr * Psat1_mmHg / np.where(P_arr > 0, P_arr, 1e-15)

    # Check for azeotrope
    diff = y1_arr - x1_arr
    azeotrope = None
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            x_az = float(x1_arr[i] + (x1_arr[i+1] - x1_arr[i]) *
                         abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1])))
            P_az = float(np.interp(x_az, x1_arr, P_arr))
            azeotrope = {"x_az": x_az, "P_az_mmHg": P_az}
            break

    return {
        "x1": x1_arr,
        "y1": y1_arr,
        "P_mmHg": P_arr,
        "g1": g1,
        "g2": g2,
        "azeotrope": azeotrope,
        "model": model,
    }


def gE_excess(
    A12: float, A21: float, model: str, T_K: float, n_pts: int = 100
) -> dict:
    """Compute GE/RT = x1*ln(γ1) + x2*ln(γ2) — excess Gibbs energy."""
    x1_arr = np.linspace(1e-4, 1 - 1e-4, n_pts)
    x2_arr = 1.0 - x1_arr

    if model == "Margules (1-suffix)":
        g1, g2 = margules_one_suffix(A12, x1_arr)
    elif model == "Margules (2-suffix)":
        g1, g2 = margules_two_suffix(A12, A21, x1_arr)
    else:
        g1, g2 = van_laar(A12, A21, x1_arr)

    GE_RT = x1_arr * np.log(g1) + x2_arr * np.log(g2)
    GE_J  = GE_RT * R * T_K

    return {"x1": x1_arr, "GE_RT": GE_RT, "GE_J": GE_J}


# ---------------------------------------------------------------------------
# 5. Psychrometrics — humid air at standard atmospheric pressure
# ---------------------------------------------------------------------------

_P_ATM_Pa = 101325.0

def _psat_water_Pa(T_K: float) -> float:
    """Antoine vapor pressure of water [Pa] using standard constants."""
    T_C = T_K - 273.15
    log_p_mmHg = 8.07131 - 1730.63 / (T_C + 233.426)
    return 10**log_p_mmHg * 133.322  # mmHg → Pa


def humidity_ratio(T_db_K: float, T_wb_K: float, P_Pa: float = _P_ATM_Pa) -> float:
    """Humidity ratio ω [kg water / kg dry air] from wet-bulb temperature.
    Sprung formula for psychrometer.
    """
    Pws_db = _psat_water_Pa(T_db_K)
    Pws_wb = _psat_water_Pa(T_wb_K)
    # Saturation humidity at wet bulb
    W_wb = 0.622 * Pws_wb / (P_Pa - Pws_wb)
    # Sprung: W = W_wb - A*(T_db - T_wb)  where A ≈ 6.6e-4 (°C-1, ventilated)
    A = 6.6e-4
    W = W_wb - A * (T_db_K - T_wb_K)
    return max(W, 0.0)


def relative_humidity(W: float, T_db_K: float, P_Pa: float = _P_ATM_Pa) -> float:
    """Relative humidity [0–1] from humidity ratio W and dry-bulb T."""
    Pws = _psat_water_Pa(T_db_K)
    Pw = W * P_Pa / (0.622 + W)
    return min(Pw / Pws, 1.0)


def dew_point(W: float, P_Pa: float = _P_ATM_Pa) -> float:
    """Dew-point temperature [K] given humidity ratio W."""
    Pw = W * P_Pa / (0.622 + W)
    # Invert Antoine for water (in Pa → mmHg)
    Pw_mmHg = Pw / 133.322
    if Pw_mmHg <= 0:
        return 233.426 + 273.15
    T_C = 1730.63 / (8.07131 - math.log10(Pw_mmHg)) - 233.426
    return T_C + 273.15


def enthalpy_humid_air(T_db_K: float, W: float) -> float:
    """Specific enthalpy of humid air [kJ/kg dry air].
    h = Cp_air*(T-0°C) + W*(hg0 + Cp_steam*(T-0°C))
    Cp_air ≈ 1.006, Cp_steam ≈ 1.805 kJ/(kg·K), hg0 = 2501 kJ/kg.
    """
    T_C = T_db_K - 273.15
    return 1.006 * T_C + W * (2501.0 + 1.805 * T_C)


def specific_volume(T_db_K: float, W: float, P_Pa: float = _P_ATM_Pa) -> float:
    """Specific volume [m³/kg dry air] of humid air."""
    R_air = 287.05  # J/(kg·K)
    R_mix = R_air * (1 + W / 0.622) / (1 + W)  # approx
    return R_mix * T_db_K / P_Pa * (1 + W) / 1.0


def psychrometric_chart_data(
    T_min_C: float = -10.0,
    T_max_C: float = 50.0,
    P_Pa: float = _P_ATM_Pa,
    n_pts: int = 200,
) -> dict:
    """Generate data for psychrometric chart:
    - Saturation curve (RH = 100%)
    - RH lines: 10%, 20%, ..., 90%
    - Constant wet-bulb lines
    Returns dict with T_arr (°C), W_sat, rh_lines, wb_lines.
    """
    T_arr_C = np.linspace(T_min_C, T_max_C, n_pts)
    T_arr_K = T_arr_C + 273.15

    # Saturation curve
    Pws = np.array([_psat_water_Pa(T) for T in T_arr_K])
    W_sat = 0.622 * Pws / (P_Pa - Pws)

    # Constant RH lines
    rh_lines = {}
    for rh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        W_rh = rh * 0.622 * Pws / (P_Pa - rh * Pws)
        rh_lines[rh] = W_rh

    # Constant wet-bulb lines (T_wb = -5, 5, 15, 25, 35 °C)
    wb_lines = {}
    for T_wb_C in [-5, 5, 15, 25, 35]:
        T_wb_K = T_wb_C + 273.15
        W_wb_line = np.array([humidity_ratio(T, T_wb_K, P_Pa) for T in T_arr_K])
        W_wb_line = np.clip(W_wb_line, 0.0, 0.1)
        wb_lines[T_wb_C] = W_wb_line

    return {
        "T_C": T_arr_C,
        "W_sat": W_sat,
        "rh_lines": rh_lines,
        "wb_lines": wb_lines,
    }


def calc_psychro_state(
    T_db_C: float,
    T_wb_C: float,
    P_Pa: float = _P_ATM_Pa,
) -> dict:
    """Calculate full psychrometric state from dry-bulb + wet-bulb temperatures."""
    T_db_K = T_db_C + 273.15
    T_wb_K = T_wb_C + 273.15
    W = humidity_ratio(T_db_K, T_wb_K, P_Pa)
    RH = relative_humidity(W, T_db_K, P_Pa)
    T_dp_K = dew_point(W, P_Pa)
    h = enthalpy_humid_air(T_db_K, W)
    v = specific_volume(T_db_K, W, P_Pa)
    Pw = W * P_Pa / (0.622 + W)

    return {
        "T_db_C": T_db_C,
        "T_wb_C": T_wb_C,
        "T_dp_C": T_dp_K - 273.15,
        "W_kg_kg": W,
        "W_g_kg": W * 1000,
        "RH_pct": RH * 100,
        "h_kJ_kg": h,
        "v_m3_kg": v,
        "Pw_Pa": Pw,
    }
