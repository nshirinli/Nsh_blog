from __future__ import annotations

import numpy as np

from core.process_control.dynamics import (
    bode_fopdt,
    bode_pid_loop,
    fopdt_ramp_response,
    fopdt_step_response,
    fit_fopdt_from_prm,
    pid_simulation,
    second_order_response,
    ziegler_nichols_closed_loop,
    ziegler_nichols_open_loop,
)


class ProcessControlController:

    # ------------------------------------------------------------------
    # Process Dynamics
    # ------------------------------------------------------------------

    def run_fopdt_step(
        self,
        Kp: float,
        tau: float,
        theta: float,
        delta_u: float = 1.0,
        t_final: float | None = None,
    ) -> tuple[str, dict]:
        data = fopdt_step_response(Kp, tau, theta, delta_u, t_final)
        lines = [
            "FOPDT Step Response",
            f"  G(s) = {Kp:.4g} · exp(-{theta:.4g}s) / ({tau:.4g}s + 1)",
            f"  Step size  Δu = {delta_u:.4g}",
            "",
            f"  Steady-state gain  y_ss = {data['y_ss']:.4f}",
            f"  63.2% response time  t63 = {data['t63']:.4f} s",
            f"  28.3% response time  t28 = {data['t28']:.4f} s",
        ]
        return "\n".join(lines), data

    def run_fopdt_ramp(
        self,
        Kp: float,
        tau: float,
        theta: float,
        ramp_rate: float = 1.0,
        t_final: float | None = None,
    ) -> tuple[str, dict]:
        data = fopdt_ramp_response(Kp, tau, theta, ramp_rate, t_final)
        lines = [
            "FOPDT Ramp Response",
            f"  G(s) = {Kp:.4g} · exp(-{theta:.4g}s) / ({tau:.4g}s + 1)",
            f"  Ramp rate = {ramp_rate:.4g} units/time",
            "",
            f"  Kp = {Kp:.4g},  τ = {tau:.4g} s,  θ = {theta:.4g} s",
        ]
        return "\n".join(lines), data

    def run_second_order(
        self,
        Kp: float,
        tau_n: float,
        zeta: float,
        delta_u: float = 1.0,
        t_final: float | None = None,
    ) -> tuple[str, dict]:
        data = second_order_response(Kp, tau_n, zeta, delta_u, t_final)
        regime = (
            "underdamped" if zeta < 1.0 - 1e-4
            else "critically damped" if abs(zeta - 1.0) < 1e-4
            else "overdamped"
        )
        lines = [
            "Second-Order Step Response",
            f"  G(s) = {Kp:.4g} / (τ_n²s² + 2ζτ_ns + 1)",
            f"  τ_n = {tau_n:.4g} s,  ζ = {zeta:.4g}  → {regime}",
            f"  Step size  Δu = {delta_u:.4g}",
            "",
            f"  Steady-state  y_ss = {data['y_ss']:.4f}",
            f"  Overshoot     = {data['overshoot_pct']:.2f}%",
            f"  Rise time (10→90%) = {data['t_rise']:.4f} s" if not np.isnan(data['t_rise']) else "  Rise time: n/a",
        ]
        return "\n".join(lines), data

    # ------------------------------------------------------------------
    # PID Simulation
    # ------------------------------------------------------------------

    def run_pid_simulation(
        self,
        Kp: float,
        tau: float,
        theta: float,
        Kc: float,
        Ti: float,
        Td: float = 0.0,
        setpoint: float = 1.0,
        disturbance: float = 0.0,
        dist_time: float | None = None,
        t_final: float | None = None,
    ) -> tuple[str, dict]:
        data = pid_simulation(Kp, tau, theta, Kc, Ti, Td, setpoint,
                              disturbance, dist_time, t_final)
        settle_str = (
            f"{data['settling_time']:.3f} s"
            if not np.isnan(data["settling_time"])
            else "not reached"
        )
        lines = [
            "PID Closed-Loop Simulation (FOPDT Process)",
            "",
            f"Process: Kp={Kp:.4g}, τ={tau:.4g} s, θ={theta:.4g} s",
            f"Controller: Kc={Kc:.4g}, Ti={Ti:.4g} s, Td={Td:.4g} s",
            f"Setpoint = {setpoint:.4g}",
            f"Disturbance = {disturbance:.4g} at t = {dist_time or 'auto'} s",
            "",
            f"IAE             = {data['IAE']:.4f}",
            f"ISE             = {data['ISE']:.4f}",
            f"Overshoot       = {data['overshoot_pct']:.2f}%",
            f"Settling time   = {settle_str}",
        ]
        return "\n".join(lines), data

    # ------------------------------------------------------------------
    # Controller Tuning
    # ------------------------------------------------------------------

    def run_tuning_open_loop(
        self,
        Kp: float,
        tau: float,
        theta: float,
    ) -> tuple[str, dict]:
        rules = ziegler_nichols_open_loop(Kp, tau, theta)
        lines = [
            "Controller Tuning — Open-Loop (Process Reaction Curve)",
            f"  Process: Kp={Kp:.4g}, τ={tau:.4g} s, θ={theta:.4g} s",
            f"  θ/τ = {theta/tau:.4f}",
            "",
            "Ziegler-Nichols:",
        ]
        for key in ["P", "PI", "PID"]:
            if key in rules:
                r = rules[key]
                s = f"  {key}: Kc={r['Kc']:.4g}"
                if "Ti" in r:
                    s += f",  Ti={r['Ti']:.4g} s"
                if "Td" in r:
                    s += f",  Td={r['Td']:.4g} s"
                lines.append(s)

        lines.append("")
        lines.append("ITAE (Kaya 1999):")
        for key in ["ITAE_PI", "ITAE_PID"]:
            if key in rules:
                r = rules[key]
                s = f"  {key}: Kc={r['Kc']:.4g},  Ti={r['Ti']:.4g} s"
                if "Td" in r:
                    s += f",  Td={r['Td']:.4g} s"
                lines.append(s)

        lines.append("")
        lines.append("IMC-based PI:")
        r = rules["IMC_PI"]
        lines.append(f"  Kc={r['Kc']:.4g},  Ti={r['Ti']:.4g} s  (λ={r['lambda']:.4g} s)")

        return "\n".join(lines), rules

    def run_tuning_closed_loop(
        self,
        Kcu: float,
        Pu: float,
    ) -> tuple[str, dict]:
        rules = ziegler_nichols_closed_loop(Kcu, Pu)
        lines = [
            "Controller Tuning — Closed-Loop (Ultimate Gain Method)",
            f"  Kcu = {Kcu:.4g},  Pu = {Pu:.4g} s",
            "",
            "Ziegler-Nichols:",
        ]
        for key in ["P", "PI", "PID"]:
            r = rules[key]
            s = f"  {key}: Kc={r['Kc']:.4g}"
            if "Ti" in r:
                s += f",  Ti={r['Ti']:.4g} s"
            if "Td" in r:
                s += f",  Td={r['Td']:.4g} s"
            lines.append(s)
        return "\n".join(lines), rules

    # ------------------------------------------------------------------
    # Bode / Frequency Response
    # ------------------------------------------------------------------

    def run_bode_process(
        self,
        Kp: float,
        tau: float,
        theta: float,
    ) -> tuple[str, dict]:
        data = bode_fopdt(Kp, tau, theta)
        gm_str = f"{data['GM_dB']:.2f} dB" if not np.isinf(data['GM_dB']) else "∞"
        lines = [
            "Bode Plot — FOPDT Process",
            f"  Kp={Kp:.4g}, τ={tau:.4g} s, θ={theta:.4g} s",
            "",
            f"  Gain crossover ω_gc  = {data['omega_gc']:.4g} rad/s",
            f"  Phase margin    PM   = {data['PM_deg']:.2f}°",
            f"  Phase crossover ω_pc = {data['omega_pc']:.4g} rad/s"
            if not np.isnan(data['omega_pc']) else "  Phase crossover: none",
            f"  Gain margin     GM   = {gm_str}",
        ]
        return "\n".join(lines), data

    def run_bode_loop(
        self,
        Kp: float,
        tau: float,
        theta: float,
        Kc: float,
        Ti: float,
        Td: float = 0.0,
    ) -> tuple[str, dict]:
        data = bode_pid_loop(Kp, tau, theta, Kc, Ti, Td)
        gm_str = f"{data['GM_dB']:.2f} dB" if not np.isinf(data['GM_dB']) else "∞"
        lines = [
            "Bode Plot — PID Closed-Loop",
            f"  Process: Kp={Kp:.4g}, τ={tau:.4g} s, θ={theta:.4g} s",
            f"  PID: Kc={Kc:.4g}, Ti={Ti:.4g} s, Td={Td:.4g} s",
            "",
            f"  Gain crossover  ω_gc = {data['omega_gc']:.4g} rad/s",
            f"  Phase margin    PM   = {data['PM_deg']:.2f}°",
            f"  Gain margin     GM   = {gm_str}",
        ]
        return "\n".join(lines), data
