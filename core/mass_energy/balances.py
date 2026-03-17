"""Mass & Energy Balances engine for ChemEng Platform.

Covers:
  * Stream property calculation  (MW, component flows, enthalpy)
  * Adiabatic mixer  (two feed streams → outlet T)
  * Stream splitter  (one feed → N outlets via split fractions)
  * Overall material balance  (single reactor, with conversion)
  * Energy balance  (sensible heat, latent heat, heat of reaction)
  * Recycle loop  (single-pass conversion, purge, overall conversion)
  * Composition converter  (mass frac ↔ mole frac)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Stream Properties
# ---------------------------------------------------------------------------

def stream_properties(
    T: float,
    P: float,
    F_total: float,
    components: list[str],
    mole_fracs: list[float],
    mol_weights: list[float],
) -> dict:
    """Calculate stream properties from mole-fraction specification.

    Parameters
    ----------
    T         : Temperature (K)
    P         : Pressure (bar)
    F_total   : Total molar flow (mol/s)
    components: list of component names
    mole_fracs: list of mole fractions (must sum ≈ 1)
    mol_weights: molecular weights (g/mol)

    Returns
    -------
    dict with per-component and total stream data.
    """
    z = np.array(mole_fracs, dtype=float)
    MW = np.array(mol_weights, dtype=float)

    if abs(z.sum() - 1.0) > 1e-4:
        z = z / z.sum()  # normalise silently

    MW_avg = float((z * MW).sum())
    F_total_kgs = F_total * MW_avg / 1000.0   # kg/s

    F_comp = z * F_total                         # mol/s per component
    F_comp_kgs = F_comp * MW / 1000.0           # kg/s per component
    w_comp = (F_comp_kgs / F_total_kgs) if F_total_kgs > 0 else z.copy()

    return {
        "T_K": T,
        "P_bar": P,
        "F_total_mols": F_total,
        "F_total_kgs": F_total_kgs,
        "MW_avg": MW_avg,
        "components": components,
        "mole_fracs": z.tolist(),
        "mass_fracs": w_comp.tolist(),
        "F_mols": F_comp.tolist(),
        "F_kgs": F_comp_kgs.tolist(),
        "mol_weights": MW.tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Adiabatic Mixer
# ---------------------------------------------------------------------------

def adiabatic_mixer(
    F1: float, T1: float, Cp1: float,
    F2: float, T2: float, Cp2: float,
    components_1: list[str] | None = None,
    mole_fracs_1: list[float] | None = None,
    components_2: list[str] | None = None,
    mole_fracs_2: list[float] | None = None,
) -> dict:
    """Two-stream adiabatic mixer using overall Cp (J/mol/K or J/kg/K).

    Tout = (F1·Cp1·T1 + F2·Cp2·T2) / (F1·Cp1 + F2·Cp2)
    """
    Cp1F1T1 = F1 * Cp1 * T1
    Cp2F2T2 = F2 * Cp2 * T2
    denom = F1 * Cp1 + F2 * Cp2
    T_out = (Cp1F1T1 + Cp2F2T2) / denom if denom != 0 else float("nan")
    F_out = F1 + F2

    # If compositions provided, mix them proportionally
    mixed_fracs: list[float] = []
    mixed_comps: list[str] = []
    if mole_fracs_1 and mole_fracs_2 and components_1 and components_2:
        all_comps = list(dict.fromkeys(list(components_1) + list(components_2)))
        z1 = {c: f for c, f in zip(components_1, mole_fracs_1)}
        z2 = {c: f for c, f in zip(components_2, mole_fracs_2)}
        mixed_fracs = [
            (F1 * z1.get(c, 0) + F2 * z2.get(c, 0)) / F_out
            for c in all_comps
        ]
        mixed_comps = all_comps

    return {
        "T_out_K": T_out,
        "F_out": F_out,
        "Q_duty_W": 0.0,   # adiabatic
        "mixed_fracs": mixed_fracs,
        "mixed_comps": mixed_comps,
        "F1": F1, "T1": T1, "F2": F2, "T2": T2,
    }


# ---------------------------------------------------------------------------
# 3. Stream Splitter
# ---------------------------------------------------------------------------

def stream_splitter(
    F_in: float,
    mole_fracs_in: list[float],
    components: list[str],
    split_fracs: list[float],
) -> dict:
    """Split a stream into N outlets using fixed split fractions.

    split_fracs[i] = fraction of total feed going to outlet i (must sum to 1).
    Composition of each outlet equals feed composition (ideal splitter).
    """
    sf = np.array(split_fracs, dtype=float)
    if abs(sf.sum() - 1.0) > 1e-6:
        sf = sf / sf.sum()

    outlets = []
    for i, frac in enumerate(sf):
        outlets.append({
            "label": f"Outlet {i+1}",
            "F_mols": float(F_in * frac),
            "split_frac": float(frac),
            "mole_fracs": list(mole_fracs_in),
            "components": list(components),
        })

    return {
        "F_in": F_in,
        "n_outlets": len(outlets),
        "outlets": outlets,
        "mole_fracs_in": mole_fracs_in,
        "components": components,
    }


# ---------------------------------------------------------------------------
# 4. Overall Material Balance  (single reactor)
# ---------------------------------------------------------------------------

def reactor_material_balance(
    F_feed: float,
    z_feed: list[float],
    components: list[str],
    key_reactant_idx: int,
    stoich_coeffs: list[float],   # negative = reactant, positive = product
    conversion: float,
) -> dict:
    """Overall steady-state material balance around a single reactor.

    Uses stoichiometry-based approach.

    Parameters
    ----------
    F_feed          : total molar feed flow (mol/s)
    z_feed          : mole fractions of feed components
    components      : component names (must align with stoich_coeffs)
    key_reactant_idx: index of key reactant in components
    stoich_coeffs   : stoichiometric coefficients (+ = product, - = reactant)
    conversion      : fractional conversion of key reactant (0–1)
    """
    z = np.array(z_feed, dtype=float)
    z = z / z.sum()
    nu = np.array(stoich_coeffs, dtype=float)

    F_in = z * F_feed
    F_reactant_in = F_in[key_reactant_idx]
    extent = conversion * F_reactant_in / abs(nu[key_reactant_idx])

    F_out = F_in + nu * extent
    F_out = np.maximum(F_out, 0.0)
    F_total_out = F_out.sum()
    z_out = F_out / F_total_out if F_total_out > 0 else F_out

    return {
        "F_in": F_in.tolist(),
        "F_out": F_out.tolist(),
        "z_out": z_out.tolist(),
        "F_total_in": float(F_feed),
        "F_total_out": float(F_total_out),
        "extent_of_reaction": float(extent),
        "conversion": conversion,
        "components": components,
        "stoich_coeffs": stoich_coeffs,
    }


# ---------------------------------------------------------------------------
# 5. Energy Balance
# ---------------------------------------------------------------------------

def sensible_heat(mass_or_moles: float, Cp: float, T_in: float, T_out: float) -> dict:
    """Q = m·Cp·ΔT"""
    dT = T_out - T_in
    Q = mass_or_moles * Cp * dT
    return {"Q_J": Q, "dT_K": dT, "Cp": Cp, "amount": mass_or_moles}


def latent_heat(mass_or_moles: float, lambda_: float) -> dict:
    """Q = m·λ  (latent heat of phase change)"""
    Q = mass_or_moles * lambda_
    return {"Q_J": Q, "lambda_Jkg": lambda_, "amount": mass_or_moles}


def heat_of_reaction(extent: float, dH_rxn: float) -> dict:
    """Q = ξ·ΔHrxn  (ξ = extent of reaction in mol)"""
    Q = extent * dH_rxn
    return {"Q_J": Q, "dH_rxn_Jmol": dH_rxn, "extent_mol": extent}


def combined_energy_balance(
    sensible_items: list[dict],   # list of {amount, Cp, T_in, T_out}
    latent_items: list[dict],     # list of {amount, lambda_}
    reaction_items: list[dict],   # list of {extent, dH_rxn}
) -> dict:
    """Sum all energy contributions and return total duty."""
    Q_total = 0.0
    breakdown = []

    for item in sensible_items:
        r = sensible_heat(item["amount"], item["Cp"], item["T_in"], item["T_out"])
        Q_total += r["Q_J"]
        breakdown.append({"type": "sensible", **r})

    for item in latent_items:
        r = latent_heat(item["amount"], item["lambda_"])
        Q_total += r["Q_J"]
        breakdown.append({"type": "latent", **r})

    for item in reaction_items:
        r = heat_of_reaction(item["extent"], item["dH_rxn"])
        Q_total += r["Q_J"]
        breakdown.append({"type": "reaction", **r})

    return {"Q_total_J": Q_total, "breakdown": breakdown}


# ---------------------------------------------------------------------------
# 6. Recycle Loop
# ---------------------------------------------------------------------------

def recycle_loop(
    F_fresh: float,
    single_pass_conversion: float,
    purge_fraction: float,
    n_components: int = 1,
) -> dict:
    """Steady-state recycle loop analysis.

    Assumptions: single component, single-pass conversion X_sp,
    purge fraction p (fraction of recycle that is purged).

    At steady state:
      Overall conversion  X_ov = 1 − p(1 − X_sp) / (p + X_sp(1 − p))
      Recycle ratio       R    = (1 − X_sp)(1 − p) / (p + X_sp(1 − p))
      F_recycle           = R · F_fresh
      F_purge             = p · F_recycle
    """
    X = single_pass_conversion
    p = purge_fraction

    denom = p + X * (1 - p)
    X_ov = 1 - p * (1 - X) / denom if denom > 0 else float("nan")
    R = (1 - X) * (1 - p) / denom if denom > 0 else float("nan")

    F_recycle = R * F_fresh
    F_purge = p * F_recycle
    F_reactor_in = F_fresh + F_recycle
    F_reactor_out = F_reactor_in * (1 - X)   # unreacted fraction leaving reactor
    F_product = F_fresh * X_ov               # at steady state

    return {
        "X_single_pass": X,
        "X_overall": float(X_ov),
        "recycle_ratio": float(R),
        "F_fresh": F_fresh,
        "F_recycle": float(F_recycle),
        "F_purge": float(F_purge),
        "F_reactor_in": float(F_reactor_in),
        "F_product": float(F_product),
        "purge_fraction": p,
    }


# ---------------------------------------------------------------------------
# 7. Composition Converter
# ---------------------------------------------------------------------------

def mass_to_mole_fractions(
    mass_fracs: list[float], mol_weights: list[float]
) -> dict:
    """Convert mass fractions → mole fractions."""
    w = np.array(mass_fracs, dtype=float)
    MW = np.array(mol_weights, dtype=float)
    if abs(w.sum() - 1.0) > 1e-4:
        w = w / w.sum()
    n = w / MW
    z = n / n.sum()
    MW_avg = float(1.0 / (w / MW).sum())
    return {"mole_fracs": z.tolist(), "mass_fracs": w.tolist(), "MW_avg": MW_avg}


def mole_to_mass_fractions(
    mole_fracs: list[float], mol_weights: list[float]
) -> dict:
    """Convert mole fractions → mass fractions."""
    z = np.array(mole_fracs, dtype=float)
    MW = np.array(mol_weights, dtype=float)
    if abs(z.sum() - 1.0) > 1e-4:
        z = z / z.sum()
    m = z * MW
    w = m / m.sum()
    MW_avg = float((z * MW).sum())
    return {"mole_fracs": z.tolist(), "mass_fracs": w.tolist(), "MW_avg": MW_avg}
