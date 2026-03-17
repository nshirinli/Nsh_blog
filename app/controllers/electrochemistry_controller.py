from __future__ import annotations

from core.electrochemistry.electrochem_engine import (
    nernst_equation,
    butler_volmer,
    faraday_law,
    fuel_cell_polarization,
    corrosion_rate,
)


class ElectrochemController:

    # ── Tab 1: Cell Potential / Nernst ───────────────────────────────────────

    def run_nernst(
        self, E0: float, n: int, Q: float, T_C: float
    ) -> tuple[str, dict]:
        T_K = T_C + 273.15
        data = nernst_equation(E0, n, Q, T_K)
        if "error" in data:
            return f"Error: {data['error']}", {}
        lines = [
            "── Nernst Equation ──",
            f"Standard potential E°  = {E0:.4f} V",
            f"Electrons transferred n = {n}",
            f"Temperature            = {T_C:.1f} °C  ({T_K:.2f} K)",
            f"Reaction quotient Q    = {Q:.4g}",
            f"RT/nF correction       = {data['correction_V'] * 1000:.2f} mV",
            "",
            f"Cell potential E       = {data['E']:.4f} V",
            "",
            "  Q < 1  →  E > E°  (reaction favors products at std. state)",
            "  Q > 1  →  E < E°",
            "  Q = 1  →  E = E°  (standard conditions)",
        ]
        return "\n".join(lines), {"type": "nernst", **data}

    # ── Tab 2: Butler-Volmer ──────────────────────────────────────────────────

    def run_butler_volmer(
        self, i0: float, alpha: float, T_C: float, eta_max: float = 0.5
    ) -> tuple[str, dict]:
        T_K = T_C + 273.15
        data = butler_volmer(i0, alpha, T_K, eta_max)
        lines = [
            "── Butler-Volmer Kinetics ──",
            f"Exchange current density i₀ = {i0:.4e} A/cm²",
            f"Anodic transfer coefficient α = {alpha:.4f}",
            f"Temperature                   = {T_C:.1f} °C",
            "",
            f"Anodic Tafel slope  ba = {data['ba_mV_dec']:.2f} mV/decade",
            f"Cathodic Tafel slope bc = {data['bc_mV_dec']:.2f} mV/decade",
            "",
            "Tafel region: high |η| where one exponential dominates",
            "Linear region: low |η| ≈ i = i₀·(F·η)/(RT)  (charge-transfer control)",
        ]
        return "\n".join(lines), {"type": "butler_volmer", **data}

    # ── Tab 3: Faraday's Law ──────────────────────────────────────────────────

    def run_faraday(
        self,
        current: float, time_h: float,
        M_molar: float, n: int,
        current_eff: float,
    ) -> tuple[str, dict]:
        data = faraday_law(current, time_h * 3600.0, M_molar, n, current_eff)
        lines = [
            "── Faraday's Law ──",
            f"Current              = {current:.4g} A",
            f"Time                 = {time_h:.4g} h  ({data['time_s']:.4g} s)",
            f"Molar mass M         = {M_molar:.4g} g/mol",
            f"Electrons n          = {n}",
            f"Current efficiency ηc = {current_eff * 100:.1f} %",
            "",
            f"Charge passed  Q     = {data['Q_C']:.4g} C",
            f"Mass deposited m     = {data['mass_g']:.4g} g",
            f"Moles deposited      = {data['moles']:.4g} mol",
        ]
        return "\n".join(lines), {"type": "faraday", **data}

    # ── Tab 4: Fuel Cell ──────────────────────────────────────────────────────

    def run_fuel_cell(
        self,
        T_C: float, i_max: float,
        i0_cathode: float, R_ohmic: float,
        alpha_c: float, i_limit: float,
    ) -> tuple[str, dict]:
        T_K = T_C + 273.15
        data = fuel_cell_polarization(T_K, i_max, i0_cathode, R_ohmic, alpha_c, i_limit)
        lines = [
            "── Fuel Cell Polarization Curve (PEM H₂/O₂) ──",
            f"Operating temperature    = {T_C:.1f} °C",
            f"Reversible potential     = {data['E_rev']:.4f} V",
            f"Thermodynamic efficiency = {data['eta_th'] * 100:.2f} %  (ΔG/ΔH)",
            "",
            "Voltage losses included:",
            f"  Activation (cathode ORR): i₀ = {i0_cathode:.2e} A/cm²,  α = {alpha_c:.2f}",
            f"  Ohmic:                    R  = {R_ohmic:.4g} Ω·cm²",
            f"  Concentration:            iL = {i_limit:.4g} A/cm²",
        ]
        return "\n".join(lines), {"type": "fuel_cell", **data}

    # ── Tab 5: Corrosion ──────────────────────────────────────────────────────

    def run_corrosion(
        self,
        i_corr: float, M_molar: float, n: int,
        rho: float, area: float,
    ) -> tuple[str, dict]:
        data = corrosion_rate(i_corr, M_molar, n, rho, area)
        lines = [
            "── Corrosion Rate Estimation ──",
            f"Corrosion current density = {i_corr:.4g} µA/cm²",
            f"Molar mass M              = {M_molar:.4g} g/mol",
            f"Valence n                 = {n}",
            f"Density ρ                 = {rho:.4g} g/cm³",
            f"Exposed area              = {area:.4g} cm²",
            "",
            f"Corrosion rate   = {data['CR_mm_yr']:.4g} mm/yr",
            f"                 = {data['CR_mpy']:.4g} mpy",
            f"Mass loss        = {data['mass_loss_g_yr']:.4g} g/yr",
            "",
            f"Rating: {data['category']}",
            "",
            "Common metals (typical i_corr in passive film):",
            "  Steel (mild)   : 0.1 – 10  µA/cm²",
            "  Stainless 316L : 0.001 – 0.1 µA/cm²",
            "  Aluminium      : 0.01 – 1   µA/cm²",
            "  Copper         : 0.05 – 5   µA/cm²",
        ]
        return "\n".join(lines), {"type": "corrosion", **data}
