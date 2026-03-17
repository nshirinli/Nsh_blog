from __future__ import annotations

from core.bioprocess.kinetics import (
    growth_kinetics,
    batch_bioreactor,
    chemostat_analysis,
    chemostat_point,
    oxygen_transfer,
    sterilization,
)


class BioprocessController:

    # ── Tab 1: Growth Kinetics ────────────────────────────────────────────────

    def run_growth_kinetics(
        self,
        mu_max: float,
        Ks: float,
        model: str = "monod",
        Ki: float | None = None,
        S_max: float = 10.0,
    ) -> tuple[str, dict]:
        data = growth_kinetics(mu_max, Ks, Ki if model == "Andrews" else None, S_max)
        lines = [
            "── Microbial Growth Kinetics ──",
            f"Model           : {model}",
            f"μ_max           = {mu_max:.4g} h⁻¹",
            f"Ks              = {Ks:.4g} g/L",
            f"μ at [S]=Ks     = {mu_max/2:.4g} h⁻¹   (= μ_max / 2)",
        ]
        if data.get("mu_andrews"):
            lines += [
                "",
                "── Andrews (Substrate Inhibition) ──",
                f"Ki              = {data['Ki']:.4g} g/L",
                f"Optimal [S]     = {data['S_peak']:.4g} g/L",
                f"μ_peak          = {data['mu_peak']:.4g} h⁻¹",
            ]
        return "\n".join(lines), {"type": "growth_kinetics", **data}

    # ── Tab 2: Batch Bioreactor ───────────────────────────────────────────────

    def run_batch(
        self,
        S0: float, X0: float, P0: float,
        mu_max: float, Ks: float,
        Yxs: float, Yps: float, ms: float,
        t_end: float,
    ) -> tuple[str, dict]:
        data = batch_bioreactor(S0, X0, P0, mu_max, Ks, Yxs, Yps, ms, t_end)
        lines = [
            "── Batch Bioreactor Simulation ──",
            f"Simulation time = {data['t_total']:.4g} h",
            f"X_final         = {data['X_final']:.4f} g/L  (biomass)",
            f"S_final         = {data['S_final']:.4f} g/L  (substrate)",
            f"P_final         = {data['P_final']:.4f} g/L  (product)",
            f"Volumetric prod.= {data['productivity_g_L_h']:.4f} g/L/h",
        ]
        return "\n".join(lines), {"type": "batch", **data}

    # ── Tab 3: Chemostat ──────────────────────────────────────────────────────

    def run_chemostat(
        self,
        Sin: float,
        mu_max: float, Ks: float,
        Yxs: float, Yps: float,
        D: float | None = None,
    ) -> tuple[str, dict]:
        sweep = chemostat_analysis(Sin, mu_max, Ks, Yxs, Yps)
        lines = [
            "── Chemostat (Continuous Bioreactor) ──",
            f"D_washout       = {sweep['D_washout']:.4g} h⁻¹",
            f"D_optimal       = {sweep['D_opt']:.4g} h⁻¹   (max productivity)",
        ]
        if D is not None:
            pt = chemostat_point(D, Sin, mu_max, Ks, Yxs, Yps)
            if pt.get("washout"):
                lines += ["", f"D = {D:.4g} h⁻¹ → WASHOUT (D ≥ D_washout)"]
            else:
                lines += [
                    "",
                    f"At D = {D:.4g} h⁻¹:",
                    f"  S*           = {pt['S_ss']:.4f} g/L",
                    f"  X*           = {pt['X_ss']:.4f} g/L",
                    f"  P*           = {pt['P_ss']:.4f} g/L",
                    f"  Productivity = {pt['productivity']:.4f} g/L/h",
                ]
        data = {"type": "chemostat", "D_point": D, **sweep}
        return "\n".join(lines), data

    # ── Tab 4: Oxygen Transfer ────────────────────────────────────────────────

    def run_oxygen_transfer(
        self, OUR: float, C_star: float, C_L: float, V: float
    ) -> tuple[str, dict]:
        data = oxygen_transfer(OUR, C_star, C_L, V)
        lines = [
            "── Oxygen Transfer Rate (OTR) ──",
            f"OUR              = {data['OUR_mmol_L_h']:.4g} mmol O₂/L/h",
            f"OUR              = {data['OUR_mg_L_h']:.4g} mg O₂/L/h",
            f"O₂ saturation C* = {data['C_star']:.4g} mg/L",
            f"Actual DO    CL  = {data['C_L']:.4g} mg/L  ({data['DO_pct']:.1f}% sat.)",
            f"Driving force    = {data['driving_force']:.4g} mg/L",
            f"KLa required     = {data['KLa']:.4g} h⁻¹",
            f"Total O₂ demand  = {data['total_O2_g_h']:.4g} g O₂/h",
            f"Est. P/V (kW/m³) = {data['P_over_V_kW_m3']:.4g}",
            "",
            "Typical KLa ranges:",
            "  Shake flasks : 10 – 100  h⁻¹",
            "  Stirred tanks: 100 – 400 h⁻¹",
            "  Airlifts     : 50  – 200 h⁻¹",
        ]
        return "\n".join(lines), {"type": "oxygen_transfer", **data}

    # ── Tab 5: Sterilization ──────────────────────────────────────────────────

    def run_sterilization(
        self,
        T_steril: float, D_121: float, z: float,
        N0: float, N_target: float, t_hold: float,
    ) -> tuple[str, dict]:
        data = sterilization(T_steril, D_121, z, N0, N_target, t_hold)
        safe_str = "✓  SAFE — sterilization criterion met" if data["safe"] else "✗  INSUFFICIENT — extend time or raise temperature"
        lines = [
            "── Thermal Sterilization (D-z Method) ──",
            f"Sterilization T  = {T_steril:.1f} °C",
            f"D-value @ 121°C  = {D_121:.4g} min",
            f"D-value @ {T_steril:.0f}°C   = {data['D_at_T']:.4g} min",
            f"z-value          = {z:.4g} °C",
            "",
            f"∇ required   = ln(N₀/N_target) = {data['nabla_required']:.4f}",
            f"∇ achieved   = {data['nabla_actual']:.4f}",
            f"Margin       = {data['margin']:+.4f} ∇-units",
            f"t required   = {data['t_required_min']:.4g} min  at {T_steril:.0f}°C",
            f"t entered    = {data['t_hold_min']:.4g} min",
            f"Expected N_final ≈ {data['N_final']:.3e} organisms",
            "",
            safe_str,
        ]
        return "\n".join(lines), {"type": "sterilization", **data}
