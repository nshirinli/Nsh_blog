from __future__ import annotations

from core.polymer.polymer_engine import (
    mw_statistics,
    flory_huggins,
    mark_houwink,
    glass_transition_fox,
    wlf_shift,
    free_radical_kinetics,
)


class PolymerController:

    # ── Tab 1: Molecular Weight ───────────────────────────────────────────────

    def run_mw_stats(self, Ni: list[float], Mi: list[float]) -> tuple[str, dict]:
        data = mw_statistics(Ni, Mi)
        lines = [
            "── Molecular Weight Statistics ──",
            f"Number-average  Mn  = {data['Mn']:,.1f} g/mol",
            f"Weight-average  Mw  = {data['Mw']:,.1f} g/mol",
            f"Z-average       Mz  = {data['Mz']:,.1f} g/mol",
            f"Dispersity      Đ   = {data['PDI']:.4f}",
            "",
            "Dispersity guide:",
            "  Đ ≈ 1.0  — monodisperse (ideal living polymerization)",
            "  Đ = 1–2  — narrow (ATRP, RAFT, anionic polymerization)",
            "  Đ ≈ 2.0  — most-probable distribution (free-radical)",
            "  Đ > 2    — broad / blended / branched system",
        ]
        return "\n".join(lines), {"type": "mw_distribution", **data}

    # ── Tab 2: Flory-Huggins ──────────────────────────────────────────────────

    def run_flory_huggins(self, chi: float, r: float) -> tuple[str, dict]:
        data = flory_huggins(chi, r)
        status = "Miscible" if data["miscible"] else "Immiscible / Phase-separating"
        lines = [
            "── Flory-Huggins Solution Thermodynamics ──",
            f"χ parameter        = {chi:.4f}",
            f"Degree of polym. r = {r:.0f}",
            f"χ_critical         = {data['chi_critical']:.4f}",
            f"φ₂_critical        = {data['phi2_critical']:.4f}",
            f"Miscibility status : {status}",
        ]
        if data["spinodal"]:
            sp = data["spinodal"]
            lines.append(
                f"Spinodal compositions: φ₂ ≈ {', '.join(f'{x:.3f}' for x in sp)}"
            )
        lines += [
            "",
            "χ > χ_c → phase separation may occur",
            "χ < 0  → exothermic mixing (strong interactions)",
        ]
        return "\n".join(lines), {"type": "flory_huggins", **data}

    # ── Tab 3: Mark-Houwink ───────────────────────────────────────────────────

    def run_mark_houwink(
        self,
        K: float, alpha: float,
        M_min: float, M_max: float,
        M_known: float | None = None,
    ) -> tuple[str, dict]:
        data = mark_houwink(K, alpha, (M_min, M_max), M_known)
        lines = [
            "── Mark-Houwink-Sakurada ──",
            f"K     = {K:.4e}  [(dL/g)·(mol/g)^α]",
            f"α     = {alpha:.4f}",
        ]
        if M_known is not None:
            lines += [
                "",
                f"At M = {M_known:.4g} g/mol:",
                f"  [η] = {data['eta_at_M']:.4g} dL/g",
            ]
        lines += [
            "",
            "α-value guide:",
            "  0 – 0.5 : poor solvent / theta condition",
            "  ≈ 0.5   : theta (ideal) solvent",
            "  0.5–0.8 : good solvent (flexible coil)",
            "  0.8–1.0 : rod-like / semi-rigid chains",
            "  > 1.0   : polyelectrolytes / excluded volume",
        ]
        return "\n".join(lines), {"type": "mark_houwink", **data}

    # ── Tab 4: Glass Transition ───────────────────────────────────────────────

    def run_glass_transition(self, components: list[dict]) -> tuple[str, dict]:
        data = glass_transition_fox(components)
        lines = [
            "── Fox Equation — Copolymer / Blend Tg ──",
            f"Tg (blend)  = {data['Tg_K']:.2f} K  =  {data['Tg_C']:.2f} °C",
            "",
            "Components:",
        ]
        for c in components:
            lines.append(
                f"  w = {c['w']:.4f},  Tg = {c['Tg']:.1f} K  ({c['Tg'] - 273.15:.1f} °C)"
            )
        return "\n".join(lines), {"type": "fox_equation", **data}

    def run_wlf(self, T_C: float, Tg_C: float, C1: float, C2: float) -> tuple[str, dict]:
        data = wlf_shift(T_C, Tg_C, C1, C2)
        if "error" in data:
            return f"Error: {data['error']}", {}
        lines = [
            "── Williams-Landel-Ferry (WLF) Equation ──",
            f"T           = {T_C:.1f} °C",
            f"T_ref (Tg)  = {Tg_C:.1f} °C",
            f"C₁          = {C1:.4g}",
            f"C₂          = {C2:.4g} °C",
            "",
            f"log₁₀(aT)  = {data['log_aT']:.4f}",
            f"aT          = {data['aT']:.4e}",
            "",
            "WLF valid for T_ref ≤ T ≤ T_ref + 100 °C",
            "Universal constants: C₁=17.44, C₂=51.6 (T_ref=Tg)",
        ]
        return "\n".join(lines), {"type": "wlf", **data}

    # ── Tab 5: Free-Radical Polymerization ───────────────────────────────────

    def run_free_radical(
        self,
        kp: float, kt: float, kd: float, f: float,
        I0: float, M0: float, t_end: float,
    ) -> tuple[str, dict]:
        data = free_radical_kinetics(kp, kt, kd, f, I0, M0, t_end)
        lines = [
            "── Free-Radical Polymerization Kinetics ──",
            f"kp = {kp:.4g} L/mol/s    (propagation)",
            f"kt = {kt:.4g} L/mol/s    (termination)",
            f"kd = {kd:.4g} s⁻¹       (initiator decomp.)",
            f"f  = {f:.4g}            (initiator efficiency)",
            "",
            f"Final conversion   X  = {data['conversion_final'] * 100:.2f} %",
            f"Final [M]             = {data['M_final']:.4g} mol/L",
            f"Initial Rp            = {data['Rp_initial']:.4g} mol/L/s",
        ]
        return "\n".join(lines), {"type": "free_radical", **data}
