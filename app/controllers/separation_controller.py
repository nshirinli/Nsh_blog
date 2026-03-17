from __future__ import annotations

import math

from core.separation.distillation import (
    solve_mccabe_thiele,
    solve_kremser,
    solve_flash,
    solve_extraction,
    solve_adsorption_isotherm,
    solve_membrane_separation,
)


class SeparationController:

    # ------------------------------------------------------------------
    # Tab 1 — McCabe-Thiele
    # ------------------------------------------------------------------

    def run_mccabe_thiele(
        self,
        alpha: float,
        R: float,
        xD: float,
        xB: float,
        zF: float,
        q: float,
    ) -> tuple[str, dict]:
        result = solve_mccabe_thiele(alpha=alpha, R=R, xD=xD, xB=xB, zF=zF, q=q)

        R_min = result["R_min"]
        r_ratio = R / R_min if R_min > 1e-10 else float("inf")
        r_ratio_str = f"{r_ratio:.3f}" if not math.isinf(r_ratio) else "∞"

        lines = [
            "McCabe-Thiele  (binary distillation, constant α)",
            "",
            f"Relative volatility  α  = {alpha:.4g}",
            f"Feed composition     zF = {zF:.4g}",
            f"Distillate           xD = {xD:.4g}",
            f"Bottoms              xB = {xB:.4g}",
            f"Reflux ratio         R  = {R:.4g}",
            f"Feed condition       q  = {q:.4g}",
            "",
            f"Minimum reflux  R_min = {R_min:.4f}",
            f"R / R_min             = {r_ratio_str}",
            "",
            f"Theoretical stages  N = {result['n_stages']}",
            f"Optimal feed tray     = {result['feed_stage']}",
            "  (counted from top; total condenser = stage 0)",
        ]
        return "\n".join(lines), result

    # ------------------------------------------------------------------
    # Tab 2 — Kremser (Absorption)
    # ------------------------------------------------------------------

    def run_kremser(
        self,
        A: float,
        m: float,
        y_in: float,
        y_out: float,
        x_in: float,
    ) -> tuple[str, dict]:
        result = solve_kremser(A=A, m=m, y_in=y_in, y_out=y_out, x_in=x_in)

        N = result["N"]
        N_str = f"{N:.4f}" if N != float("inf") else "∞"
        N_ceil = result["N_ceil"]
        N_ceil_str = str(int(N_ceil)) if N_ceil != float("inf") else "∞"

        lines = [
            "Kremser Equation  (dilute absorber, linear equilibrium)",
            "",
            f"Absorption factor  A    = {A:.4g}   (A = L / (m·G))",
            f"Equilibrium slope  m    = {m:.4g}   (y* = m·x)",
            f"Gas  inlet   y_in  = {y_in:.4g}",
            f"Gas  outlet  y_out = {y_out:.4g}",
            f"Liquid inlet x_in  = {x_in:.4g}",
            "",
            f"Liquid outlet  x_out = {result['x_out']:.6f}",
            f"Absorption efficiency = {result['absorb_eff'] * 100:.2f}%",
            "",
            f"Theoretical stages  N      = {N_str}",
            f"Actual stages (η=1)  N_act = {N_ceil_str}",
        ]
        return "\n".join(lines), result

    # ------------------------------------------------------------------
    # Tab 3 — Rachford-Rice Flash
    # ------------------------------------------------------------------

    def run_flash(
        self,
        z_text: str,
        k_text: str,
        labels: list[str] | None = None,
    ) -> tuple[str, dict]:
        z_vals = [float(v) for v in z_text.strip().split()]
        k_vals = [float(v) for v in k_text.strip().split()]

        if len(z_vals) != len(k_vals):
            raise ValueError(
                f"Number of z values ({len(z_vals)}) must equal "
                f"number of K values ({len(k_vals)})."
            )
        if not (2 <= len(z_vals) <= 10):
            raise ValueError("Enter between 2 and 10 components.")
        if abs(sum(z_vals) - 1.0) > 0.02:
            raise ValueError(
                f"Feed fractions must sum to 1.0 (got {sum(z_vals):.4f})."
            )

        result = solve_flash(z_vals, k_vals)
        n = result["n_comp"]

        if labels is None or len(labels) != n:
            labels = [f"C{i+1}" for i in range(n)]

        result["labels"] = labels

        lines = [
            "Rachford-Rice Isothermal Flash",
            "",
            f"Components:    {n}",
            f"Vapor fraction β = V/F = {result['beta']:.6f}",
            "",
            f"{'Comp':>5}  {'z_i':>8}  {'K_i':>8}  {'x_i (liq)':>10}  {'y_i (vap)':>10}",
            "─" * 52,
        ]
        for i, (lb, z, K, x, y) in enumerate(
            zip(labels, result["z"], result["K"], result["x"], result["y"])
        ):
            lines.append(
                f"  {lb:>4}  {z:>8.4f}  {K:>8.4f}  {x:>10.6f}  {y:>10.6f}"
            )

        lines.extend([
            "",
            f"  Σ x_i = {sum(result['x']):.6f}   (should be 1.000)",
            f"  Σ y_i = {sum(result['y']):.6f}   (should be 1.000)",
        ])
        return "\n".join(lines), result

    # ------------------------------------------------------------------
    # Tab 4 — Liquid-Liquid Extraction
    # ------------------------------------------------------------------

    def run_extraction(
        self,
        z_feed: float,
        K_D: float,
        S_over_F: float,
        n_stages: int,
        mode: str,
    ) -> tuple[str, dict]:
        result = solve_extraction(z_feed, K_D, S_over_F, n_stages, mode)
        E = result["E_total"] * 100

        lines = [
            f"Liquid-Liquid Extraction — {mode.title()}",
            "",
            f"Feed solute  z_feed  = {z_feed:.4f}",
            f"Distribution coeff  K_D = {K_D:.4g}   (y* = K_D · x)",
            f"Solvent ratio  S/F  = {S_over_F:.4g}",
            f"Equilibrium stages  N  = {n_stages}",
            "",
            f"Raffinate after stage N:  x_N = {result['x_stages'][-1]:.6f}",
            f"Overall extraction efficiency = {E:.2f}%",
            "",
            "Stage-by-stage raffinate composition:",
            f"  {'Stage':>5}   {'x (raffinate)':>14}",
            "  " + "─" * 24,
        ]
        for i, xv in enumerate(result["x_stages"]):
            stage_label = "Feed" if i == 0 else f"  {i}"
            lines.append(f"  {stage_label:>5}   {xv:>14.6f}")

        return "\n".join(lines), result

    # ------------------------------------------------------------------
    # Tab 5 — Adsorption Isotherms
    # ------------------------------------------------------------------

    def run_adsorption(
        self,
        model: str,
        C_max: float,
        params: dict,
    ) -> tuple[str, dict]:
        result = solve_adsorption_isotherm(model, C_max, params)

        lines = [f"Adsorption Isotherm — {model}", ""]

        if model == "Langmuir":
            lines += [
                f"q_max = {params['q_max']:.4g} mg/g",
                f"K_L   = {params['K_L']:.4g} L/mg",
                f"q at C_max ({C_max:.4g}) = {float(result['q'][-1]):.4f} mg/g",
                f"q at half-saturation = {result['q_half']:.4f} mg/g",
                f"Separation factor RL = {result['RL']:.4f}",
                "  (RL < 1 → favorable, RL = 0 → irreversible)",
            ]
        elif model == "Freundlich":
            lines += [
                f"K_F = {params['K_F']:.4g},  1/n = {1.0/params['n_F']:.4f}",
                f"q at C_max ({C_max:.4g}) = {float(result['q'][-1]):.4f}",
                f"n_F = {params['n_F']:.4g}  (n>1 → favorable, n<1 → unfavorable)",
            ]
        elif model == "BET":
            lines += [
                f"q_m    = {params['q_m']:.4g} (monolayer capacity)",
                f"K_BET  = {params['K_BET']:.4g}",
                f"Cs     = {params['Cs']:.4g} (saturation concentration)",
            ]
        elif model == "Temkin":
            lines += [
                f"A_T = {params['A_T']:.4g},  b = {params['b']:.4g}",
                f"T   = {params.get('T_K', 298.15):.2f} K",
            ]

        return "\n".join(lines), result

    # ------------------------------------------------------------------
    # Tab 6 — Membrane Separation
    # ------------------------------------------------------------------

    def run_membrane(
        self,
        P_A: float,
        P_B: float,
        thickness_um: float,
        p_feed_bar: float,
        p_perm_bar: float,
        z_A_feed: float,
    ) -> tuple[str, dict]:
        import numpy as np
        stage_cut = np.linspace(0.001, 0.999, 200)
        result = solve_membrane_separation(
            P_A, P_B, thickness_um, p_feed_bar, p_perm_bar, z_A_feed, stage_cut
        )
        sel = result["selectivity"]
        y_A_zero = float(result["y_A"][0])

        lines = [
            "Membrane Gas Separation (Solution-Diffusion Model)",
            "",
            f"Component A permeability  P_A = {P_A:.4g} Barrer",
            f"Component B permeability  P_B = {P_B:.4g} Barrer",
            f"Ideal selectivity  α* = P_A/P_B = {sel:.4f}",
            f"Membrane thickness  l = {thickness_um:.4g} μm",
            f"Feed pressure    p_f = {p_feed_bar:.4g} bar",
            f"Permeate pressure p_p = {p_perm_bar:.4g} bar",
            f"Pressure ratio   φ   = p_p/p_f = {p_perm_bar/p_feed_bar:.4f}",
            "",
            f"Feed composition  z_A = {z_A_feed:.4f}",
            f"Permeate y_A (θ→0)   = {y_A_zero:.4f}",
            f"Enrichment ratio      = {y_A_zero/z_A_feed:.4f}",
            "",
            f"Approx. total flux (θ→0):  {result['J_total']:.4g} mol/(m²·s)",
        ]
        return "\n".join(lines), result
