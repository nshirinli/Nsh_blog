"""Fluid Dynamics controller — formats solver results for the UI."""
from __future__ import annotations

from core.fluid_dynamics.fluid_dynamics import (
    pipe_flow_analysis,
    moody_chart_data,
    bernoulli_check,
    orifice_flow,
    pump_sizing,
    pump_operating_curve,
    pump_system_curve,
    pump_operating_point,
    isentropic_relations,
    isentropic_profile,
    normal_shock,
    mach_from_velocity,
)


class FluidDynamicsController:

    # ------------------------------------------------------------------
    # Tab 1 — Pipe Flow
    # ------------------------------------------------------------------

    def run_pipe_flow(self, rho, v, D, mu, L, eps=0.0) -> tuple[str, dict]:
        r = pipe_flow_analysis(rho=rho, v=v, D=D, mu=mu, L=L, eps=eps)
        moody = moody_chart_data([0.0, 1e-4, 1e-3, 0.01])
        moody["Re_calc"] = r["Re"]
        moody["f_calc"] = r["f"]

        lines = [
            "Pipe Flow Analysis  (Darcy-Weisbach)",
            "",
            f"Fluid density     ρ   = {rho:.4g} kg/m³",
            f"Mean velocity     v   = {v:.4g} m/s",
            f"Pipe diameter     D   = {D:.4g} m",
            f"Dyn. viscosity    μ   = {mu:.4g} Pa·s",
            f"Pipe length       L   = {L:.4g} m",
            f"Roughness         ε   = {eps:.4g} m   (ε/D = {eps/D:.4g})",
            "",
            f"Reynolds number   Re  = {r['Re']:.4g}   [{r['regime']}]",
            f"Friction factor   f   = {r['f']:.5f}   (Darcy)",
            f"Pressure drop     ΔP  = {r['dP_Pa']:.5g} Pa   "
            f"({r['dP_Pa']/1000:.4g} kPa)",
            f"Head loss         h_L = {r['h_L_m']:.4g} m",
            f"Velocity head     v²/2g = {v**2/(2*9.81):.4g} m",
        ]
        return "\n".join(lines), moody

    # ------------------------------------------------------------------
    # Tab 2 — Bernoulli / Energy Balance
    # ------------------------------------------------------------------

    def run_bernoulli(self, P1, v1, z1, P2, v2, z2, rho) -> tuple[str, dict]:
        r = bernoulli_check(P1=P1, v1=v1, z1=z1, P2=P2, v2=v2, z2=z2, rho=rho)
        sign = "loss (friction)" if r["h_L"] >= 0 else "gain (pump work)"
        lines = [
            "Bernoulli / Energy Balance",
            "",
            f"Fluid density  ρ = {rho:.4g} kg/m³",
            "",
            "Point 1:",
            f"  P1 = {P1:.4g} Pa   v1 = {v1:.4g} m/s   z1 = {z1:.4g} m",
            f"  Total head H1 = {r['H1']:.4f} m",
            "",
            "Point 2:",
            f"  P2 = {P2:.4g} Pa   v2 = {v2:.4g} m/s   z2 = {z2:.4g} m",
            f"  Total head H2 = {r['H2']:.4f} m",
            "",
            f"Head difference  H1 − H2 = {r['h_L']:.4f} m  [{sign}]",
        ]
        return "\n".join(lines), r

    def run_orifice(self, Cd, D_orifice, dP, rho) -> tuple[str, dict]:
        r = orifice_flow(Cd=Cd, D_orifice=D_orifice, dP=dP, rho=rho)
        Q_L_s = r["Q_m3s"] * 1000
        lines = [
            "Orifice Plate / Flow Meter",
            "",
            f"Discharge coefficient  Cd = {Cd:.4g}",
            f"Orifice diameter       D  = {D_orifice:.4g} m",
            f"Pressure drop          ΔP = {dP:.4g} Pa",
            f"Fluid density          ρ  = {rho:.4g} kg/m³",
            "",
            f"Orifice area           A  = {r['A_m2']:.5g} m²",
            f"Mean jet velocity      v  = {r['v_ms']:.4g} m/s",
            f"Volumetric flow rate   Q  = {r['Q_m3s']:.5g} m³/s",
            f"                           = {Q_L_s:.4g} L/s",
        ]
        return "\n".join(lines), r

    # ------------------------------------------------------------------
    # Tab 3 — Pump Sizing
    # ------------------------------------------------------------------

    def run_pump_sizing(
        self, Q, rho, H, eta,
        H_shutoff, Q_BEP, H_BEP, H_static, K_friction,
    ) -> tuple[str, dict]:
        r = pump_sizing(Q=Q, rho=rho, H=H, eta=eta)
        pump_curve = pump_operating_curve(
            H_shutoff=H_shutoff, Q_BEP=Q_BEP, H_BEP=H_BEP,
            Q_max=Q_BEP * 2,
        )
        sys_curve = pump_system_curve(
            H_static=H_static, K_friction=K_friction,
            Q_max=Q_BEP * 2,
        )
        op_point = pump_operating_point(
            H_shutoff=H_shutoff, Q_BEP=Q_BEP, H_BEP=H_BEP,
            H_static=H_static, K_friction=K_friction,
        )
        plot_data = {
            "pump": pump_curve,
            "system": sys_curve,
            "op_point": op_point,
        }
        lines = [
            "Pump Sizing",
            "",
            f"Required flow rate   Q   = {Q:.4g} m³/s  ({Q*1000:.4g} L/s)",
            f"Required head        H   = {H:.4g} m",
            f"Fluid density        ρ   = {rho:.4g} kg/m³",
            f"Pump efficiency      η   = {eta*100:.2f}%",
            "",
            f"Hydraulic power  P_hyd  = {r['P_hydraulic_W']:.5g} W  "
            f"({r['P_hydraulic_W']/1000:.4g} kW)",
            f"Shaft power      P_shaft = {r['P_shaft_W']:.5g} W  "
            f"({r['P_shaft_W']/1000:.4g} kW)",
            "",
            "Operating Point (pump curve ∩ system curve):",
        ]
        Q_op, H_op = op_point["Q_op"], op_point["H_op"]
        if Q_op == Q_op and Q_op > 0:   # not NaN
            lines.extend([
                f"  Q_op = {Q_op:.5g} m³/s  ({Q_op*1000:.4g} L/s)",
                f"  H_op = {H_op:.5g} m",
            ])
        else:
            lines.append("  (no intersection found in given range)")
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 4 — Compressible Flow
    # ------------------------------------------------------------------

    def run_isentropic(self, M, gamma) -> tuple[str, dict]:
        r = isentropic_relations(M=M, gamma=gamma)
        profile = isentropic_profile(gamma=gamma, M_max=max(3.0, M * 1.5))
        profile["M_calc"] = M
        profile["T0_T_calc"] = r["T0_T"]
        profile["P0_P_calc"] = r["P0_P"]
        profile["A_Astar_calc"] = r["A_Astar"]

        lines = [
            "Isentropic Flow Relations",
            "",
            f"Mach number  M  = {M:.4g}   [{r['regime']}]",
            f"Specific heat ratio  γ = {gamma:.4g}",
            "",
            f"T0/T   = {r['T0_T']:.5f}   (stagnation/static temperature ratio)",
            f"P0/P   = {r['P0_P']:.5f}   (stagnation/static pressure ratio)",
            f"ρ0/ρ   = {r['rho0_rho']:.5f}   (stagnation/static density ratio)",
            f"A/A*   = {r['A_Astar']:.5f}   (area / sonic throat area)",
            "",
            "Choked?" + ("  YES  (M ≥ 1)" if r["choked"] else "  No"),
        ]
        return "\n".join(lines), profile

    def run_normal_shock(self, M1, gamma) -> tuple[str, dict]:
        r = normal_shock(M1=M1, gamma=gamma)
        lines = [
            "Normal Shock Relations",
            "",
            f"Upstream Mach   M1 = {M1:.4g}",
            f"γ               = {gamma:.4g}",
            "",
            f"Downstream Mach M2     = {r['M2']:.5f}",
            f"Pressure ratio  P2/P1  = {r['P2_P1']:.5f}",
            f"Temperature ratio T2/T1 = {r['T2_T1']:.5f}",
            f"Density ratio   ρ2/ρ1  = {r['rho2_rho1']:.5f}",
            f"Stagnation pressure loss P02/P01 = {r['P02_P01']:.5f}",
        ]
        return "\n".join(lines), r
