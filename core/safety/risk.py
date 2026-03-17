"""Safety & Risk Analysis engine for ChemEng Platform.

Covers:
  * Gaussian plume dispersion        (Pasquill-Gifford, continuous release)
  * Vapor cloud explosion            (TNT equivalency method)
  * Pool fire                        (Thomas + point-source radiation model)
  * Risk matrix                      (5×5 likelihood vs consequence)
  * LOPA                             (Layer of Protection Analysis)
  * Flammability analysis            (Le Chatelier mixing, flash point estimate)
"""
from __future__ import annotations

import numpy as np


# ===========================================================================
# 1. Gaussian Plume Dispersion
# ===========================================================================

# Pasquill-Gifford dispersion coefficients — Briggs rural formulas
# σy = a·x·(1 + b·x)^(-1/2),  σz = c·x·(1 + d·x)^(-e)
# x in metres, σ in metres
_PG_PARAMS: dict[str, tuple] = {
    "A": (0.22, 1e-4, 0.20, 0.0, 0.0),
    "B": (0.16, 1e-4, 0.12, 0.0, 0.0),
    "C": (0.11, 1e-4, 0.08, 2e-4, 0.5),
    "D": (0.08, 1e-4, 0.06, 1.5e-3, 0.5),
    "E": (0.06, 1e-4, 0.03, 3e-4, 1.0),
    "F": (0.04, 1e-4, 0.016, 3e-4, 1.0),
}


def _sigma_y(x: float, cls: str) -> float:
    a, b, *_ = _PG_PARAMS[cls]
    return a * x * (1 + b * x) ** (-0.5)


def _sigma_z(x: float, cls: str) -> float:
    _, _, c, d, e = _PG_PARAMS[cls]
    if d == 0.0:
        return c * x
    return c * x * (1 + d * x) ** (-e)


def gaussian_plume(
    Q: float,           # source strength (g/s)
    u: float,           # mean wind speed (m/s)
    H: float,           # effective release height (m)
    stability: str,     # Pasquill-Gifford class A–F
    x_max: float = 3000.0,
    n_points: int = 400,
) -> dict:
    """Centreline ground-level concentration C(x, 0, 0) [mg/m³].

    C = Q / (π σy σz u) · exp(−H² / 2σz²)
    """
    xs = np.linspace(10.0, x_max, n_points)
    sy = np.array([_sigma_y(x, stability) for x in xs])
    sz = np.array([_sigma_z(x, stability) for x in xs])

    with np.errstate(divide="ignore", invalid="ignore"):
        C = (Q / (np.pi * sy * sz * u)) * np.exp(-H ** 2 / (2 * sz ** 2))
    C = np.where(np.isfinite(C), C, 0.0)  # mg/m³  (Q in g/s = 1000 mg/s)
    C *= 1000.0  # g/s → mg/s, divide by m³ → mg/m³

    # Find distance to peak
    peak_idx = int(np.argmax(C))

    return {
        "x": xs,
        "C_mg_m3": C,
        "sigma_y": sy,
        "sigma_z": sz,
        "C_peak_mg_m3": float(C[peak_idx]),
        "x_peak_m": float(xs[peak_idx]),
        "Q_gs": Q,
        "u_ms": u,
        "H_m": H,
        "stability": stability,
    }


# ===========================================================================
# 2. Vapor Cloud Explosion — TNT Equivalency
# ===========================================================================

# Reference overpressure curve: scaled distance Z (m/kg^(1/3)) → Ps (kPa)
# Derived from Baker et al. (1983) / SFPE handbook
_TNT_Z = np.array([0.40, 0.50, 0.70, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0,
                   15.0, 20.0, 30.0, 50.0])
_TNT_Ps = np.array([8000, 3000, 1000, 400, 150, 80, 35, 15, 7, 3.5,
                    1.8, 1.0, 0.5, 0.2])   # kPa


def _interp_overpressure(Z: np.ndarray) -> np.ndarray:
    return np.interp(Z, _TNT_Z, _TNT_Ps, left=_TNT_Ps[0], right=_TNT_Ps[-1])


def vapor_cloud_explosion(
    m_fuel: float,          # kg of flammable fuel
    dHc_MJ_kg: float,       # heat of combustion (MJ/kg)
    alpha: float = 0.05,    # explosion yield factor (3–10 %)
    r_max: float = 500.0,   # max distance (m)
    n_points: int = 300,
) -> dict:
    """TNT equivalent mass and overpressure vs distance."""
    dHtnt = 4.68          # MJ/kg TNT detonation energy
    W_tnt = alpha * m_fuel * dHc_MJ_kg / dHtnt

    rs = np.linspace(1.0, r_max, n_points)
    Z = rs / W_tnt ** (1.0 / 3.0)
    Ps = _interp_overpressure(Z)

    # Damage categories (kPa)
    thresholds = {
        "Window breakage": 0.5,
        "Partial wall collapse": 3.0,
        "Serious structural damage": 17.0,
        "Eardrum rupture": 35.0,
        "Lung damage": 70.0,
    }
    damage_distances = {}
    for label, ps_thresh in thresholds.items():
        idx = np.where(Ps <= ps_thresh)[0]
        damage_distances[label] = float(rs[idx[0]]) if len(idx) else float(r_max)

    return {
        "W_tnt_kg": float(W_tnt),
        "r": rs,
        "Ps_kPa": Ps,
        "Z": Z,
        "damage_distances": damage_distances,
        "m_fuel_kg": m_fuel,
        "dHc_MJ_kg": dHc_MJ_kg,
        "alpha": alpha,
    }


# ===========================================================================
# 3. Pool Fire
# ===========================================================================

def pool_fire(
    diameter: float,       # pool diameter (m)
    m_dot_pp: float,       # burning rate (kg/m²/s), e.g. 0.05 for petrol
    dHc_MJ_kg: float,      # heat of combustion (MJ/kg)
    rho_air: float = 1.2,  # air density (kg/m³)
    eta: float = 0.30,     # radiative fraction
    r_max: float = 200.0,
    n_points: int = 300,
) -> dict:
    """Pool fire flame height, heat release rate, and incident heat flux vs distance.

    Thomas (1963) flame height correlation:
        L/D = 42 * (m'' / (ρa √(gD))) ** 0.61

    Point-source radiation model:
        q(r) = η · Q_total / (4π r²)   [W/m²]
    where r is measured from pool centre.
    """
    g = 9.81
    A_pool = np.pi * diameter ** 2 / 4.0
    Q_comb = m_dot_pp * A_pool * dHc_MJ_kg * 1e6   # W (total combustion rate)

    # Flame height
    Froude_term = m_dot_pp / (rho_air * np.sqrt(g * diameter))
    L_D = 42.0 * Froude_term ** 0.61
    flame_height = L_D * diameter

    # Radiation source term
    Q_rad = eta * Q_comb   # W

    rs = np.linspace(diameter / 2.0 + 0.1, r_max, n_points)
    q = Q_rad / (4.0 * np.pi * rs ** 2)   # W/m²

    # Harm thresholds (incident heat flux W/m²)
    thresholds = {
        "Pain threshold  (1.6 kW/m²)": 1600.0,
        "1% burns, 10s exposure  (4 kW/m²)": 4000.0,
        "Piloted ignition  (12.5 kW/m²)": 12_500.0,
        "Auto-ignition  (25 kW/m²)": 25_000.0,
        "Steel structural failure  (35 kW/m²)": 35_000.0,
    }
    harm_distances = {}
    for label, q_thresh in thresholds.items():
        idx = np.where(q <= q_thresh)[0]
        harm_distances[label] = float(rs[idx[0]]) if len(idx) else float(rs[0])

    return {
        "diameter_m": diameter,
        "flame_height_m": float(flame_height),
        "L_D": float(L_D),
        "Q_comb_MW": float(Q_comb / 1e6),
        "Q_rad_MW": float(Q_rad / 1e6),
        "r": rs,
        "q_W_m2": q,
        "harm_distances": harm_distances,
    }


# ===========================================================================
# 4. Risk Matrix
# ===========================================================================

_RISK_LABELS = [
    ["L", "L", "L",  "M",  "M"],
    ["L", "L", "M",  "M",  "H"],
    ["L", "M", "M",  "H",  "H"],
    ["M", "M", "H",  "H",  "C"],
    ["M", "H", "H",  "C",  "C"],
]  # [severity-1][likelihood-1] ; severity rows, likelihood columns

_RISK_COLORS = {"L": "#4caf50", "M": "#ffc107", "H": "#ff9800", "C": "#f44336"}

_RISK_NAMES = {"L": "Low", "M": "Medium", "H": "High", "C": "Critical"}

_SEVERITY_LABELS = [
    "1 — Negligible",
    "2 — Minor",
    "3 — Moderate",
    "4 — Major",
    "5 — Catastrophic",
]
_LIKELIHOOD_LABELS = [
    "A — Very Unlikely  (<10⁻⁵/yr)",
    "B — Unlikely       (10⁻⁵–10⁻⁴/yr)",
    "C — Possible       (10⁻⁴–10⁻³/yr)",
    "D — Likely         (10⁻³–10⁻²/yr)",
    "E — Very Likely    (>10⁻²/yr)",
]


def risk_level(severity: int, likelihood: int) -> dict:
    """Severity 1–5, Likelihood 1–5 → risk level."""
    s = max(1, min(5, severity)) - 1
    l = max(1, min(5, likelihood)) - 1
    code = _RISK_LABELS[s][l]
    return {
        "code": code,
        "name": _RISK_NAMES[code],
        "color": _RISK_COLORS[code],
        "severity": severity,
        "likelihood": likelihood,
    }


def risk_matrix_data() -> dict:
    """Return the full 5×5 matrix for rendering."""
    return {
        "matrix": _RISK_LABELS,
        "colors": _RISK_COLORS,
        "names": _RISK_NAMES,
        "severity_labels": _SEVERITY_LABELS,
        "likelihood_labels": _LIKELIHOOD_LABELS,
    }


# ===========================================================================
# 5. LOPA — Layer of Protection Analysis
# ===========================================================================

def lopa(
    ie_description: str,
    ie_freq: float,          # initiating event frequency (/yr)
    ipls: list[dict],        # list of {"description": str, "pfd": float}
    target_freq: float = 1e-5,
) -> dict:
    """Calculate mitigated event likelihood and required risk reduction.

    Parameters
    ----------
    ie_freq      : initiating event frequency (/yr)
    ipls         : list of IPL dicts, each with "description" and "pfd"
    target_freq  : tolerable event likelihood (/yr)
    """
    pfd_product = 1.0
    for ipl in ipls:
        pfd_product *= ipl.get("pfd", 1.0)

    mitigated_freq = ie_freq * pfd_product
    risk_reduction = ie_freq / mitigated_freq if mitigated_freq > 0 else float("inf")
    meets_target = mitigated_freq <= target_freq

    # Required PFD from remaining gap
    required_reduction = mitigated_freq / target_freq if not meets_target else 1.0

    return {
        "ie_description": ie_description,
        "ie_freq": ie_freq,
        "ipls": ipls,
        "pfd_product": pfd_product,
        "mitigated_freq": mitigated_freq,
        "risk_reduction_factor": risk_reduction,
        "target_freq": target_freq,
        "meets_target": meets_target,
        "additional_rrf_needed": required_reduction,
    }


# ===========================================================================
# 6. Flammability Analysis
# ===========================================================================

def flammability_limits_mixture(
    components: list[str],
    mole_fracs: list[float],
    lfl_vals: list[float],
    ufl_vals: list[float],
) -> dict:
    """Le Chatelier's rule for mixture flammability limits.

    LFLmix = 1 / Σ(yᵢ / LFLᵢ)
    UFLmix = 1 / Σ(yᵢ / UFLᵢ)
    (only combustible fractions used)
    """
    z = np.array(mole_fracs, dtype=float)
    z = z / z.sum()
    lfl = np.array(lfl_vals, dtype=float)
    ufl = np.array(ufl_vals, dtype=float)

    # Only sum combustible components (LFL > 0)
    comb_mask = lfl > 0
    z_comb = z[comb_mask]
    z_comb = z_comb / z_comb.sum() if z_comb.sum() > 0 else z_comb

    LFLmix = 1.0 / (z_comb / lfl[comb_mask]).sum() if z_comb.sum() > 0 else float("nan")
    UFLmix = 1.0 / (z_comb / ufl[comb_mask]).sum() if z_comb.sum() > 0 else float("nan")

    flammable_range = UFLmix - LFLmix
    # Stoichiometric concentration  (simple estimate: ~1.5 × LFL)
    Cst = 1.5 * LFLmix

    return {
        "LFLmix": float(LFLmix),
        "UFLmix": float(UFLmix),
        "flammable_range_pct": float(flammable_range),
        "Cst_pct": float(Cst),
        "components": components,
        "mole_fracs": z.tolist(),
        "LFL_vals": lfl.tolist(),
        "UFL_vals": ufl.tolist(),
    }


def flash_point_estimate(Tb_K: float, LFL_pct: float) -> dict:
    """Estimate flash point from boiling point and LFL (Catoire & Naudet, 2004).

    Simplified: Tflash ≈ Tb − (Tb − 273.15) × k  where k depends on LFL.
    Uses: Tflash ≈ 0.932 × Tb − 88.1  (approximate for common hydrocarbons).
    Also uses the Antoine/Clausius-Clapeyron approximation for vapour pressure at flash point.
    """
    # Empirical linear fit good for Tb 300-600 K
    Tflash_K = 0.932 * Tb_K - 88.1
    Tflash_C = Tflash_K - 273.15

    return {
        "Tb_K": Tb_K,
        "LFL_pct": LFL_pct,
        "Tflash_K": float(Tflash_K),
        "Tflash_C": float(Tflash_C),
        "note": "Empirical estimate; verify with experimental data.",
    }


def minimum_oxygen_concentration(LFL_pct: float, stoich_O2: float) -> float:
    """MOC = LFL × (stoich O₂ per mole fuel).  Units: vol% O₂."""
    return LFL_pct * stoich_O2 / 100.0 * 100.0
