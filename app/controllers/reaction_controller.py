from __future__ import annotations

import numpy as np

from core.reaction.kinetics import arrhenius_k
from core.reaction.reactors import (
    simulate_first_order_batch,
    simulate_batch_reactor,
    simulate_cstr,
    simulate_pfr,
    simulate_series_reactions,
    simulate_parallel_reactions,
    compute_arrhenius_curve,
    compute_reactor_sizing,
    compute_equilibrium,
    simulate_nonisothermal_batch,
    simulate_nonisothermal_pfr,
    compute_rtd_tanks_in_series,
    compute_rtd_dispersion,
)


class ReactionController:
    # ------------------------------------------------------------------
    # Legacy (kept for backward compatibility)
    # ------------------------------------------------------------------

    def run_demo_simulation(self, reactor: str, k: float, ca0: float, t_final: float):
        t, ca = simulate_first_order_batch(k=k, ca0=ca0, t_final=t_final)
        conversion = (ca0 - ca[-1]) / ca0 if ca0 != 0 else 0.0
        summary = (
            f"Reactor Type: {reactor}\n"
            f"Final Concentration: {ca[-1]:.6f}\n"
            f"Conversion: {conversion:.4f}"
        )
        return summary, t, ca

    # ------------------------------------------------------------------
    # Tab 1 — Ideal Reactors
    # ------------------------------------------------------------------

    def run_ideal_reactor(
        self,
        reactor_type: str,
        n: float,
        k: float,
        ca0: float,
        t_or_v: float,
        v0: float = 1.0,
    ) -> tuple[str, dict]:
        if reactor_type == "Batch":
            t, ca, x = simulate_batch_reactor(k, n, ca0, t_or_v)
            lines = [
                f"Reactor:    Batch",
                f"Order n:    {n}",
                f"k:          {k:.4g} (conc^(1-n)/s)",
                f"CA0:        {ca0:.4g} mol/L",
                f"Final time: {t_or_v:.4g} s",
                "",
                f"CA(final) = {ca[-1]:.6f} mol/L",
                f"Conversion X = {x[-1]:.4f}  ({x[-1]*100:.2f}%)",
            ]
            plot_data = {"type": "batch_pfr", "xdata": t, "ca": ca, "x": x,
                         "xlabel": "Time (s)"}

        elif reactor_type == "CSTR":
            tau, ca, x = simulate_cstr(k, n, ca0, t_or_v)
            v_final = v0 * t_or_v
            lines = [
                f"Reactor:    CSTR (steady-state)",
                f"Order n:    {n}",
                f"k:          {k:.4g}",
                f"CA0:        {ca0:.4g} mol/L",
                f"Max τ:      {t_or_v:.4g} s",
                f"v0:         {v0:.4g} L/s  →  V = {v_final:.4g} L at τ_max",
                "",
                f"At τ_max:",
                f"  CA   = {ca[-1]:.6f} mol/L",
                f"  X    = {x[-1]:.4f}  ({x[-1]*100:.2f}%)",
            ]
            plot_data = {"type": "batch_pfr", "xdata": tau, "ca": ca, "x": x,
                         "xlabel": "Residence time τ (s)"}

        elif reactor_type == "PFR":
            v_arr, ca, x = simulate_pfr(k, n, ca0, v0, t_or_v)
            lines = [
                f"Reactor:    PFR",
                f"Order n:    {n}",
                f"k:          {k:.4g}",
                f"CA0:        {ca0:.4g} mol/L",
                f"Volume:     {t_or_v:.4g} L",
                f"v0:         {v0:.4g} L/s  →  τ = {t_or_v/v0:.4g} s",
                "",
                f"At exit:",
                f"  CA   = {ca[-1]:.6f} mol/L",
                f"  X    = {x[-1]:.4f}  ({x[-1]*100:.2f}%)",
            ]
            plot_data = {"type": "batch_pfr", "xdata": v_arr, "ca": ca, "x": x,
                         "xlabel": "Reactor volume V (L)"}

        else:
            raise ValueError(f"Unknown reactor type: {reactor_type}")

        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 2 — Arrhenius
    # ------------------------------------------------------------------

    def run_arrhenius(
        self,
        A: float,
        Ea_J: float,
        T_calc_K: float,
        T_min_K: float,
        T_max_K: float,
    ) -> tuple[str, dict]:
        k_at_T = arrhenius_k(A, Ea_J, T_calc_K)
        T_arr, k_arr = compute_arrhenius_curve(A, Ea_J, T_min_K, T_max_K)
        R = 8.314
        lines = [
            "Arrhenius:  k(T) = A · exp(-Ea / RT)",
            "",
            f"A:          {A:.4g}",
            f"Ea:         {Ea_J:.4g} J/mol  ({Ea_J/1000:.4g} kJ/mol)",
            f"Ea/R:       {Ea_J/R:.2f} K",
            "",
            f"T calc:     {T_calc_K:.2f} K  ({T_calc_K - 273.15:.2f} °C)",
            f"k(T):       {k_at_T:.6g}",
        ]
        plot_data = {"T": T_arr, "k": k_arr, "T_calc": T_calc_K, "k_calc": k_at_T}
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 3 — Series & Parallel Reactions
    # ------------------------------------------------------------------

    def run_series_parallel(
        self,
        reaction_type: str,
        k1: float,
        k2: float,
        ca0: float,
        t_final: float,
        cb0: float = 0.0,
        cc0: float = 0.0,
    ) -> tuple[str, dict]:
        if reaction_type == "Series (A→B→C)":
            t, ca, cb, cc = simulate_series_reactions(k1, k2, ca0, cb0, cc0, t_final)
        else:
            t, ca, cb, cc = simulate_parallel_reactions(k1, k2, ca0, t_final)

        x_a = (ca0 - float(ca[-1])) / ca0 if ca0 > 0 else 0.0
        yield_b = float(cb[-1]) / ca0 if ca0 > 0 else 0.0
        cc_final = float(cc[-1])
        cb_final = float(cb[-1])
        sel_str = f"{cb_final / cc_final:.4f}" if cc_final > 1e-12 else "∞ (no C formed)"

        lines = [
            f"Reaction:   {reaction_type}",
            f"k1 = {k1:.4g},  k2 = {k2:.4g}",
            f"CA0 = {ca0:.4g} mol/L,  t_final = {t_final:.4g} s",
            "",
            "Final concentrations:",
            f"  CA = {ca[-1]:.6f} mol/L",
            f"  CB = {cb[-1]:.6f} mol/L",
            f"  CC = {cc[-1]:.6f} mol/L",
            "",
            f"Conversion of A:  X_A  = {x_a:.4f}  ({x_a*100:.2f}%)",
            f"Yield of B:       Y_B  = {yield_b:.4f}  ({yield_b*100:.2f}%)",
            f"Selectivity B/C:  S_BC = {sel_str}",
        ]
        plot_data = {"t": t, "ca": ca, "cb": cb, "cc": cc, "type": reaction_type}
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 4 — Reactor Sizing & Levenspiel
    # ------------------------------------------------------------------

    def run_reactor_sizing(
        self,
        k: float,
        n: float,
        ca0: float,
        v0: float,
        x_target: float,
    ) -> tuple[str, dict]:
        X_arr, V_cstr, V_pfr, V_cstr_tgt, V_pfr_tgt, inv_ra = compute_reactor_sizing(
            k, n, ca0, v0, x_target
        )
        ratio = V_cstr_tgt / V_pfr_tgt if V_pfr_tgt > 1e-12 else float("inf")
        ratio_str = f"{ratio:.4f}" if not np.isinf(ratio) else "∞"

        lines = [
            "Reactor Sizing  (-rA = k · CA^n,  constant density)",
            "",
            f"k = {k:.4g},  n = {n:.2g}",
            f"CA0 = {ca0:.4g} mol/L,  v0 = {v0:.4g} L/s",
            f"FA0 = {v0*ca0:.4g} mol/s",
            f"Target conversion X = {x_target:.4f}  ({x_target*100:.2f}%)",
            "",
            "Required reactor volumes:",
            f"  CSTR: V = {V_cstr_tgt:.4f} L",
            f"  PFR:  V = {V_pfr_tgt:.4f} L",
            "",
            f"Volume ratio V_CSTR / V_PFR = {ratio_str}",
            "(CSTR > PFR for n > 0 positive-order reactions)",
        ]
        plot_data = {
            "X": X_arr,
            "V_cstr": V_cstr,
            "V_pfr": V_pfr,
            "inv_ra": inv_ra,
            "V_cstr_tgt": V_cstr_tgt,
            "V_pfr_tgt": V_pfr_tgt,
            "x_target": x_target,
        }
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 5 — Chemical Equilibrium
    # ------------------------------------------------------------------

    def run_equilibrium(
        self,
        dH_kJ: float,
        dG_kJ: float,
        stoich_reac: list[float],
        stoich_prod: list[float],
        n_init: list[float],
        P_bar: float,
        T_ref_K: float,
        T_min_K: float,
        T_max_K: float,
    ) -> tuple[str, dict]:
        import math
        R = 8.314
        dH_J = dH_kJ * 1000
        dG_J = dG_kJ * 1000
        Keq_ref = math.exp(-dG_J / (R * T_ref_K))

        T_arr, Kp_arr, Keq_ref, extent_arr = compute_equilibrium(
            dH_J, dG_J, stoich_reac, stoich_prod, n_init,
            P_bar, T_ref_K, T_min_K, T_max_K
        )
        delta_n = sum(stoich_prod) - sum(stoich_reac)
        reaction_type = "Exothermic" if dH_kJ < 0 else "Endothermic"

        lines = [
            "Chemical Equilibrium — Van't Hoff Analysis",
            "",
            f"ΔH_rxn = {dH_kJ:.4g} kJ/mol  ({reaction_type})",
            f"ΔG_rxn = {dG_kJ:.4g} kJ/mol  (at {T_ref_K:.2f} K)",
            f"Δn_gas = {delta_n:.3g}",
            f"P      = {P_bar:.4g} bar",
            "",
            f"Keq at {T_ref_K:.2f} K = {Keq_ref:.6g}",
            f"ln(Keq) = {math.log(Keq_ref):.4f}",
            "",
            "Equilibrium extent at selected temperatures:",
            f"  {'T (K)':>8}  {'Kp':>12}  {'ξ_eq (mol)':>12}",
            "  " + "─" * 36,
        ]
        for T, Kp, xi in zip(T_arr[::30], Kp_arr[::30], extent_arr[::30]):
            lines.append(f"  {T:>8.2f}  {Kp:>12.4g}  {xi:>12.4f}")

        plot_data = {
            "T": T_arr, "Kp": Kp_arr, "extent": extent_arr,
            "dH_kJ": dH_kJ, "T_ref": T_ref_K, "Keq_ref": Keq_ref,
        }
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 6 — Non-isothermal Reactor
    # ------------------------------------------------------------------

    def run_nonisothermal(
        self,
        reactor_type: str,
        k0: float,
        Ea_kJ: float,
        n: float,
        ca0: float,
        T0_C: float,
        v0: float,
        dH_kJ: float,
        Cp_J_mol_K: float,
        rho_mol_L: float,
        UA: float,
        Tc_C: float,
        t_or_V: float,
    ) -> tuple[str, dict]:
        Ea_J = Ea_kJ * 1000
        T0_K = T0_C + 273.15
        Tc_K = Tc_C + 273.15
        dH_J = dH_kJ * 1000

        if reactor_type == "Batch":
            t_arr, ca, x, T_arr = simulate_nonisothermal_batch(
                k0, Ea_J, n, ca0, T0_K, dH_J, Cp_J_mol_K, rho_mol_L, UA, Tc_K, t_or_V
            )
            xaxis_label = "Time (s)"
        else:
            t_arr, ca, x, T_arr = simulate_nonisothermal_pfr(
                k0, Ea_J, n, ca0, T0_K, v0, dH_J, Cp_J_mol_K, rho_mol_L, UA, Tc_K, t_or_V
            )
            xaxis_label = "Volume (L)"

        T_C_arr = T_arr - 273.15
        lines = [
            f"Non-isothermal {reactor_type} Reactor",
            "",
            f"k0 = {k0:.4g},  Ea = {Ea_kJ:.4g} kJ/mol,  n = {n:.2g}",
            f"CA0 = {ca0:.4g} mol/L,  T0 = {T0_C:.2f} °C",
            f"ΔH_rxn = {dH_kJ:.4g} kJ/mol,  Cp = {Cp_J_mol_K:.4g} J/(mol·K)",
            f"UA = {UA:.4g} W/K,  Tc = {Tc_C:.2f} °C",
            "",
            f"Exit CA    = {ca[-1]:.6f} mol/L",
            f"Exit X     = {x[-1]:.4f}  ({x[-1]*100:.2f}%)",
            f"Exit T     = {T_C_arr[-1]:.2f} °C  ({T_arr[-1]:.2f} K)",
            f"Max T      = {T_C_arr.max():.2f} °C  (hot spot at "
            f"{t_arr[T_C_arr.argmax()]:.3g} {xaxis_label.split()[0].lower()})",
        ]
        plot_data = {
            "xdata": t_arr, "ca": ca, "x": x, "T_C": T_C_arr,
            "xlabel": xaxis_label, "reactor": reactor_type,
        }
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 7 — RTD Analysis
    # ------------------------------------------------------------------

    def run_rtd(
        self,
        model: str,
        tau: float,
        N_or_Pe: float,
        t_final: float,
    ) -> tuple[str, dict]:
        import math
        if model == "Tanks-in-Series":
            N = max(1, int(round(N_or_Pe)))
            t_arr, E_arr, F_arr = compute_rtd_tanks_in_series(tau, N, t_final)
            sigma2 = tau**2 / N
            sigma2_theta = 1.0 / N
            model_desc = f"N = {N} tanks"
        else:  # Dispersion
            Pe = N_or_Pe
            t_arr, E_arr, F_arr = compute_rtd_dispersion(tau, Pe, t_final)
            sigma2_theta = 2.0/Pe + 8.0/Pe**2  # closed vessel approx
            sigma2 = sigma2_theta * tau**2
            model_desc = f"Pe = {Pe:.4g}"

        mean_t = float(np.trapz(t_arr * E_arr, t_arr))

        lines = [
            f"RTD Analysis — {model}  ({model_desc})",
            "",
            f"Mean residence time τ    = {tau:.4g} s",
            f"E(t) integral  ≈ {float(np.trapz(E_arr, t_arr)):.4f}  (should be 1.0)",
            f"Mean time from E(t):  <t> = {mean_t:.4f} s",
            f"Variance  σ²             = {sigma2:.4g} s²",
            f"σ²/τ²                    = {sigma2_theta:.4f}",
            "",
            "Interpretation:",
        ]
        if model == "Tanks-in-Series":
            N = int(round(N_or_Pe))
            if N == 1:
                lines.append("  N=1 → Perfect CSTR (exponential E(t))")
            elif N >= 20:
                lines.append(f"  N={N} → Approaches plug-flow behavior")
            else:
                lines.append(f"  N={N} tanks → Intermediate mixing")
        else:
            Pe = N_or_Pe
            if Pe < 1:
                lines.append("  Pe < 1 → High dispersion, near perfect mixing")
            elif Pe > 100:
                lines.append("  Pe > 100 → Low dispersion, near plug flow")
            else:
                lines.append(f"  Pe = {Pe:.2g} → Intermediate dispersion")

        plot_data = {"t": t_arr, "E": E_arr, "F": F_arr, "tau": tau, "model": model}
        return "\n".join(lines), plot_data
