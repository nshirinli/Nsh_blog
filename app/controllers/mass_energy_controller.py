"""Mass & Energy Balances controller — formats results for the UI."""
from __future__ import annotations

import numpy as np

from core.mass_energy.balances import (
    stream_properties,
    adiabatic_mixer,
    stream_splitter,
    reactor_material_balance,
    combined_energy_balance,
    recycle_loop,
    mass_to_mole_fractions,
    mole_to_mass_fractions,
)


def _fmt(val: float, unit: str = "") -> str:
    s = f"{val:.5g}"
    return f"{s} {unit}".strip()


class MassEnergyController:

    # ------------------------------------------------------------------
    # Tab 1 — Stream Properties
    # ------------------------------------------------------------------

    def run_stream(
        self,
        T: float, P: float, F_total: float,
        components: list[str],
        mole_fracs: list[float],
        mol_weights: list[float],
    ) -> tuple[str, dict]:
        try:
            r = stream_properties(T, P, F_total, components, mole_fracs, mol_weights)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Stream Properties",
            "═" * 52,
            f"Temperature     : {r['T_K']:.2f} K  ({r['T_K']-273.15:.2f} °C)",
            f"Pressure        : {r['P_bar']:.4g} bar",
            f"Total flow      : {r['F_total_mols']:.5g} mol/s  "
            f"({r['F_total_kgs']*3600:.4g} kg/h)",
            f"Average MW      : {r['MW_avg']:.4f} g/mol",
            "",
            f"{'Component':<18}  {'MW':>6}  {'Mole fr':>8}  "
            f"{'Mass fr':>8}  {'F (mol/s)':>12}  {'F (kg/h)':>10}",
            "─" * 72,
        ]
        for i, comp in enumerate(r["components"]):
            lines.append(
                f"  {comp:<16}  {r['mol_weights'][i]:>6.3f}  "
                f"{r['mole_fracs'][i]:>8.4f}  {r['mass_fracs'][i]:>8.4f}  "
                f"{r['F_mols'][i]:>12.4g}  {r['F_kgs'][i]*3600:>10.4g}"
            )
        return "\n".join(lines), {"type": "stream", "result": r}

    # ------------------------------------------------------------------
    # Tab 2 — Adiabatic Mixer
    # ------------------------------------------------------------------

    def run_mixer(
        self,
        F1: float, T1: float, Cp1: float,
        F2: float, T2: float, Cp2: float,
        comp_names: list[str] | None = None,
        z1: list[float] | None = None,
        z2: list[float] | None = None,
    ) -> tuple[str, dict]:
        try:
            r = adiabatic_mixer(F1, T1, Cp1, F2, T2, Cp2,
                                comp_names, z1, comp_names, z2)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Adiabatic Mixer",
            "═" * 48,
            f"Feed 1 : F={F1:.4g} mol/s,  T={T1:.2f} K,  Cp={Cp1:.4g} J/(mol·K)",
            f"Feed 2 : F={F2:.4g} mol/s,  T={T2:.2f} K,  Cp={Cp2:.4g} J/(mol·K)",
            "",
            f"Outlet flow         : {r['F_out']:.5g} mol/s",
            f"Outlet temperature  : {r['T_out_K']:.4f} K  ({r['T_out_K']-273.15:.2f} °C)",
            f"Duty (adiabatic)    : 0  W",
        ]
        if r["mixed_fracs"]:
            lines += ["", "Outlet composition:"]
            for comp, zz in zip(r["mixed_comps"], r["mixed_fracs"]):
                lines.append(f"  {comp:<18}  z = {zz:.4f}")
        return "\n".join(lines), {"type": "mixer", "result": r}

    # ------------------------------------------------------------------
    # Tab 3 — Splitter
    # ------------------------------------------------------------------

    def run_splitter(
        self,
        F_in: float,
        mole_fracs: list[float],
        components: list[str],
        split_fracs: list[float],
    ) -> tuple[str, dict]:
        try:
            r = stream_splitter(F_in, mole_fracs, components, split_fracs)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Stream Splitter",
            "═" * 48,
            f"Inlet flow  : {F_in:.5g} mol/s",
            f"Outlets     : {r['n_outlets']}",
            "",
        ]
        for o in r["outlets"]:
            lines.append(f"  {o['label']}: {o['F_mols']:.5g} mol/s  "
                         f"(split fraction = {o['split_frac']:.4f})")
        lines += ["", "All outlets carry the same composition as the feed."]
        return "\n".join(lines), {"type": "splitter", "result": r}

    # ------------------------------------------------------------------
    # Tab 4 — Material Balance
    # ------------------------------------------------------------------

    def run_material_balance(
        self,
        F_feed: float,
        z_feed: list[float],
        components: list[str],
        key_idx: int,
        stoich: list[float],
        conversion: float,
    ) -> tuple[str, dict]:
        try:
            r = reactor_material_balance(F_feed, z_feed, components,
                                         key_idx, stoich, conversion)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Overall Material Balance  (Single Reactor)",
            "═" * 56,
            f"Key reactant    : {components[key_idx]}",
            f"Conversion      : {conversion:.1%}",
            f"Extent of rxn   : {r['extent_of_reaction']:.5g} mol/s",
            "",
            f"Total feed flow : {r['F_total_in']:.5g} mol/s",
            f"Total product   : {r['F_total_out']:.5g} mol/s",
            "",
            f"{'Component':<18}  {'Stoich':>6}  {'F_in (mol/s)':>14}  "
            f"{'F_out (mol/s)':>14}  {'z_out':>8}",
            "─" * 68,
        ]
        for i, comp in enumerate(components):
            lines.append(
                f"  {comp:<16}  {r['stoich_coeffs'][i]:>+6.2f}  "
                f"{r['F_in'][i]:>14.4g}  {r['F_out'][i]:>14.4g}  "
                f"{r['z_out'][i]:>8.4f}"
            )
        return "\n".join(lines), {"type": "material_balance", "result": r}

    # ------------------------------------------------------------------
    # Tab 5 — Energy Balance
    # ------------------------------------------------------------------

    def run_energy_balance(
        self,
        sensible_items: list[dict],
        latent_items: list[dict],
        reaction_items: list[dict],
    ) -> tuple[str, dict]:
        try:
            r = combined_energy_balance(sensible_items, latent_items, reaction_items)
        except Exception as exc:
            return f"Error: {exc}", {}

        Q = r["Q_total_J"]
        lines = [
            "Energy Balance",
            "═" * 52,
        ]
        for i, b in enumerate(r["breakdown"], 1):
            t = b["type"]
            q = b["Q_J"]
            if t == "sensible":
                lines.append(
                    f"  [{i}] Sensible heat:  m={b['amount']:.4g},  "
                    f"Cp={b['Cp']:.4g},  ΔT={b['dT_K']:.4g} K  →  Q={q:.4g} J"
                )
            elif t == "latent":
                lines.append(
                    f"  [{i}] Latent heat:    m={b['amount']:.4g},  "
                    f"λ={b['lambda_Jkg']:.4g} J/unit  →  Q={q:.4g} J"
                )
            elif t == "reaction":
                lines.append(
                    f"  [{i}] Heat of rxn:    ξ={b['extent_mol']:.4g} mol,  "
                    f"ΔHrxn={b['dH_rxn_Jmol']:.4g} J/mol  →  Q={q:.4g} J"
                )
        lines += [
            "─" * 52,
            f"Total duty  Q = {Q:.5g} J",
            f"            Q = {Q/1000:.5g} kJ",
            f"            Q = {Q/1e6:.5g} MJ",
            "",
            ("Endothermic (heat input required)" if Q > 0
             else "Exothermic (heat removed / released)"),
        ]
        return "\n".join(lines), {
            "type": "energy_balance",
            "result": r,
            "Q_J": Q,
        }

    # ------------------------------------------------------------------
    # Tab 6 — Recycle Loop
    # ------------------------------------------------------------------

    def run_recycle(
        self,
        F_fresh: float,
        X_sp: float,
        purge_frac: float,
    ) -> tuple[str, dict]:
        try:
            r = recycle_loop(F_fresh, X_sp, purge_frac)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Recycle Loop Analysis",
            "═" * 48,
            f"Fresh feed              : {F_fresh:.5g} mol/s",
            f"Single-pass conversion  : {X_sp:.1%}",
            f"Purge fraction          : {purge_frac:.3f}  (of recycle stream)",
            "",
            f"Overall conversion      : {r['X_overall']:.4f}  ({r['X_overall']*100:.2f} %)",
            f"Recycle ratio           : {r['recycle_ratio']:.4f}",
            f"Recycle flow            : {r['F_recycle']:.5g} mol/s",
            f"Purge flow              : {r['F_purge']:.5g} mol/s",
            f"Reactor inlet flow      : {r['F_reactor_in']:.5g} mol/s",
            f"Net product flow        : {r['F_product']:.5g} mol/s",
        ]
        return "\n".join(lines), {"type": "recycle", "result": r}

    # ------------------------------------------------------------------
    # Tab 7 — Composition Converter
    # ------------------------------------------------------------------

    def run_composition_convert(
        self,
        mode: str,           # "mass→mole" or "mole→mass"
        fracs: list[float],
        mol_weights: list[float],
        components: list[str],
    ) -> tuple[str, dict]:
        try:
            if mode == "mass→mole":
                r = mass_to_mole_fractions(fracs, mol_weights)
                from_label, to_label = "Mass fraction", "Mole fraction"
                from_fracs, to_fracs = r["mass_fracs"], r["mole_fracs"]
            else:
                r = mole_to_mass_fractions(fracs, mol_weights)
                from_label, to_label = "Mole fraction", "Mass fraction"
                from_fracs, to_fracs = r["mole_fracs"], r["mass_fracs"]
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            f"Composition Converter  ({mode})",
            "═" * 52,
            f"Average molecular weight : {r['MW_avg']:.4f} g/mol",
            "",
            f"{'Component':<18}  {'MW (g/mol)':>10}  "
            f"{from_label:>14}  {to_label:>14}",
            "─" * 62,
        ]
        for i, comp in enumerate(components):
            lines.append(
                f"  {comp:<16}  {mol_weights[i]:>10.4f}  "
                f"{from_fracs[i]:>14.6f}  {to_fracs[i]:>14.6f}"
            )
        lines.append("─" * 62)
        lines.append(
            f"  {'Sum':<16}  {'':>10}  "
            f"{sum(from_fracs):>14.6f}  {sum(to_fracs):>14.6f}"
        )
        return "\n".join(lines), {
            "type": "composition", "result": r,
            "components": components, "mol_weights": mol_weights,
            "mole_fracs": r["mole_fracs"], "mass_fracs": r["mass_fracs"],
        }
