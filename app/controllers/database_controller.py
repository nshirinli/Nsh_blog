"""Database Controller — wraps core/database/compounds.py for the UI."""
from __future__ import annotations

import numpy as np

from core.database.compounds import (
    list_compounds,
    list_categories,
    search,
    get_compound,
    get_properties_text,
    compute_vapor_pressure,
    compute_cp,
    compare_property,
)


class DatabaseController:

    def all_compounds(self) -> list[dict]:
        return list_compounds()

    def all_categories(self) -> list[str]:
        return list_categories()

    def search(self, query: str, category: str | None = None) -> list[dict]:
        if not query.strip() and category is None:
            return list_compounds()
        if not query.strip():
            return [c for c in list_compounds() if c["category"] == category]
        return search(query, category if category else None)

    def get_text(self, name: str) -> str:
        return get_properties_text(name)

    # ------------------------------------------------------------------
    # Vapor-pressure curve
    # ------------------------------------------------------------------

    def vapor_pressure_curve(
        self, name: str, T_min_C: float = -20.0, T_max_C: float = 150.0, n: int = 200
    ) -> tuple[str, dict]:
        c = get_compound(name)
        if c is None:
            return f"Compound '{name}' not found.", {}

        T_arr = np.linspace(T_min_C, T_max_C, n)
        try:
            P_arr = np.array([compute_vapor_pressure(name, T) for T in T_arr])
        except Exception as exc:
            return f"Error computing vapor pressure: {exc}", {}

        # Convert mmHg → kPa
        P_kPa = P_arr * 0.133322

        Tb_C = c["Tb"] - 273.15
        P_at_Tb = compute_vapor_pressure(name, Tb_C)

        lines = [
            f"Vapor Pressure — {name} ({c['formula']})",
            "",
            f"Antoine: log₁₀(P/mmHg) = {c['antoine'][0]:.5f} − {c['antoine'][1]:.3f} / (T + {c['antoine'][2]:.3f})",
            f"Range plotted: {T_min_C:.1f} °C → {T_max_C:.1f} °C",
            "",
            f"Normal boiling point Tb = {Tb_C:.2f} °C  (P = 760 mmHg = 101.325 kPa)",
            f"P at Tb from Antoine: {P_at_Tb:.2f} mmHg  ({P_at_Tb*0.133322:.3f} kPa)",
        ]
        plot_data = {
            "type": "vapor_pressure",
            "name": name,
            "formula": c["formula"],
            "T_C": T_arr,
            "P_mmHg": P_arr,
            "P_kPa": P_kPa,
            "Tb_C": Tb_C,
        }
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Cp vs T curve
    # ------------------------------------------------------------------

    def cp_curve(
        self, name: str, T_min_K: float = 200.0, T_max_K: float = 800.0, n: int = 200
    ) -> tuple[str, dict]:
        c = get_compound(name)
        if c is None:
            return f"Compound '{name}' not found.", {}

        T_arr = np.linspace(T_min_K, T_max_K, n)
        Cp_arr = np.array([compute_cp(name, T) for T in T_arr])

        a, b, cc, d = c["cp"]
        Cp_298 = compute_cp(name, 298.15)

        lines = [
            f"Heat Capacity (Cp) — {name} ({c['formula']})",
            "",
            f"Polynomial: Cp = a + bT + cT² + dT³  [J/(mol·K)]",
            f"  a = {a:.4g},  b = {b:.4g},  c = {cc:.4g},  d = {d:.4g}",
            "",
            f"Cp at 298.15 K (25°C):  {Cp_298:.4f} J/(mol·K)  =  {Cp_298/c['MW']*1000:.4f} J/(kg·K)",
            f"Range plotted: {T_min_K:.1f} K → {T_max_K:.1f} K",
        ]
        plot_data = {
            "type": "cp_curve",
            "name": name,
            "formula": c["formula"],
            "T_K": T_arr,
            "Cp": Cp_arr,
            "Cp_298": Cp_298,
        }
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Property comparison
    # ------------------------------------------------------------------

    def compare(self, names: list[str], prop: str) -> tuple[str, dict]:
        PROP_LABELS = {
            "MW": "Molecular Weight (g/mol)",
            "Tb": "Normal Boiling Point (K)",
            "Tm": "Melting Point (K)",
            "Tc": "Critical Temperature (K)",
            "Pc": "Critical Pressure (bar)",
            "Vc": "Critical Volume (cm³/mol)",
            "omega": "Acentric Factor",
            "dHf": "ΔHf° at 298 K (kJ/mol)",
            "dGf": "ΔGf° at 298 K (kJ/mol)",
            "mu25": "Viscosity at 25°C (mPa·s)",
            "rho25": "Density at 25°C (kg/m³)",
            "kth": "Thermal Conductivity at 25°C (W/(m·K))",
        }
        label = PROP_LABELS.get(prop, prop)
        values = compare_property(names, prop)

        lines = [f"Property Comparison — {label}", ""]
        for name, val in values.items():
            val_str = f"{val:.4g}" if val is not None else "N/A"
            lines.append(f"  {name:<30} {val_str}")

        plot_data = {
            "type": "comparison",
            "prop": prop,
            "label": label,
            "names": list(values.keys()),
            "values": [values[n] for n in values],
        }
        return "\n".join(lines), plot_data
