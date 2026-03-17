"""Heat Transfer controller — formats solver results for the UI."""
from __future__ import annotations

from core.heat_transfer.heat_transfer import (
    conduction_flat_wall,
    conduction_composite_wall,
    conduction_cylinder,
    convection_newton,
    pipe_flow_convection,
    h_vs_Re_data,
    heat_exchanger_lmtd,
    heat_exchanger_ntu,
    radiation_blackbody,
    radiation_grey_body,
    radiation_planck_spectrum,
)


class HeatTransferController:

    # ------------------------------------------------------------------
    # Tab 1 — Conduction
    # ------------------------------------------------------------------

    def run_flat_wall(self, k, A, T1, T2, L) -> tuple[str, dict]:
        r = conduction_flat_wall(k=k, A=A, T1=T1, T2=T2, L=L)
        lines = [
            "Flat-Wall Conduction  (Fourier's Law)",
            "",
            f"Thermal conductivity  k  = {k:.4g} W/(m·K)",
            f"Area                  A  = {A:.4g} m²",
            f"Thickness             L  = {L:.4g} m",
            f"Hot side temperature  T1 = {T1:.4g} K",
            f"Cold side temperature T2 = {T2:.4g} K",
            "",
            f"Temperature difference ΔT = {r['dT']:.4g} K",
            f"Thermal resistance     R  = {r['R_KW']:.5g} K/W",
            f"Heat flow rate         Q  = {r['Q_W']:.5g} W",
        ]
        return "\n".join(lines), r

    def run_composite_wall(self, layers, A, T_in, T_out) -> tuple[str, dict]:
        r = conduction_composite_wall(layers=layers, A=A, T_in=T_in, T_out=T_out)
        lines = [
            "Composite Flat Wall  (Series Thermal Resistances)",
            "",
            f"Area   A    = {A:.4g} m²",
            f"T_in        = {T_in:.4g} K",
            f"T_out       = {T_out:.4g} K",
            "",
            f"Total thermal resistance R_total = {r['R_total']:.5g} K/W",
            f"Total heat flow rate     Q       = {r['Q_W']:.5g} W",
            "",
            "Layer breakdown:",
        ]
        for i, (lay, R, dT, T) in enumerate(
            zip(layers, r["R_layers"], r["dT_layers"], r["T_profile"][:-1])
        ):
            lines.append(
                f"  {lay['label']:20s}  L={lay['L']:.4g} m  k={lay['k']:.4g} W/(m·K)"
                f"  R={R:.4g} K/W  ΔT={dT:.4g} K  "
                f"T: {T:.2f} → {T - dT:.2f} K"
            )
        return "\n".join(lines), r

    def run_cylinder(self, k, L, r1, r2, T1, T2) -> tuple[str, dict]:
        r = conduction_cylinder(k=k, L=L, r1=r1, r2=r2, T1=T1, T2=T2)
        lines = [
            "Cylindrical-Wall Conduction",
            "",
            f"Thermal conductivity k  = {k:.4g} W/(m·K)",
            f"Cylinder length      L  = {L:.4g} m",
            f"Inner radius         r1 = {r1:.4g} m",
            f"Outer radius         r2 = {r2:.4g} m",
            f"Inner temperature    T1 = {T1:.4g} K",
            f"Outer temperature    T2 = {T2:.4g} K",
            "",
            f"ln(r2/r1)              = {r2/r1:.5g}  (ratio)",
            f"Thermal resistance  R  = {r['R_KW']:.5g} K/W",
            f"Heat flow rate      Q  = {r['Q_W']:.5g} W",
        ]
        return "\n".join(lines), r

    # ------------------------------------------------------------------
    # Tab 2 — Convection
    # ------------------------------------------------------------------

    def run_newton_cooling(self, h, A, T_s, T_f) -> tuple[str, dict]:
        r = convection_newton(h=h, A=A, T_surface=T_s, T_fluid=T_f)
        lines = [
            "Newton's Law of Cooling",
            "",
            f"Heat transfer coeff  h  = {h:.4g} W/(m²·K)",
            f"Surface area         A  = {A:.4g} m²",
            f"Surface temperature  Ts = {T_s:.4g} K",
            f"Fluid temperature    Tf = {T_f:.4g} K",
            "",
            f"Temperature difference ΔT = {r['dT']:.4g} K",
            f"Heat transfer rate     Q  = {r['Q_W']:.5g} W",
        ]
        return "\n".join(lines), r

    def run_pipe_convection(
        self, Re, Pr, k_fluid, D, correlation, heating
    ) -> tuple[str, dict]:
        r = pipe_flow_convection(
            Re=Re, Pr=Pr, k_fluid=k_fluid, D=D,
            correlation=correlation, heating=heating
        )
        plot_data = h_vs_Re_data(
            Re_min=max(Re * 0.1, 1000),
            Re_max=Re * 5,
            Pr=Pr, k_fluid=k_fluid, D=D,
            correlation=correlation, heating=heating,
        )
        plot_data["Re_calc"] = Re
        plot_data["h_calc"] = r["h"]
        lines = [
            f"Pipe-Flow Convection  ({correlation})",
            "",
            f"Reynolds number Re = {Re:.4g}",
            f"Prandtl number  Pr = {Pr:.4g}",
            f"Fluid conductivity k = {k_fluid:.4g} W/(m·K)",
            f"Pipe diameter      D = {D:.4g} m",
            "",
            f"Nusselt number  Nu = {r['Nu']:.4f}",
            f"Heat transfer coeff h = {r['h']:.4f} W/(m²·K)",
        ]
        return "\n".join(lines), plot_data

    # ------------------------------------------------------------------
    # Tab 3 — Heat Exchangers
    # ------------------------------------------------------------------

    def run_lmtd(self, T_h_in, T_h_out, T_c_in, T_c_out, U, A, flow) -> tuple[str, dict]:
        r = heat_exchanger_lmtd(
            T_h_in=T_h_in, T_h_out=T_h_out,
            T_c_in=T_c_in, T_c_out=T_c_out,
            U=U, A=A, flow=flow,
        )
        lines = [
            f"Heat Exchanger — LMTD Method  ({flow.title()}-flow)",
            "",
            f"Hot fluid:   T_in = {T_h_in:.4g} K → T_out = {T_h_out:.4g} K",
            f"Cold fluid:  T_in = {T_c_in:.4g} K → T_out = {T_c_out:.4g} K",
            f"Overall U = {U:.4g} W/(m²·K)",
            f"Area      A = {A:.4g} m²",
            "",
            f"LMTD     = {r['LMTD']:.4f} K",
            f"Q        = {r['Q_W']:.5g} W  ({r['Q_W']/1000:.4g} kW)",
        ]
        if flow == "counter":
            dT1, dT2 = T_h_in - T_c_out, T_h_out - T_c_in
        else:
            dT1, dT2 = T_h_in - T_c_in, T_h_out - T_c_out
        lines.extend(["", f"ΔT₁ = {dT1:.4g} K   ΔT₂ = {dT2:.4g} K"])
        return "\n".join(lines), r

    def run_ntu(self, C_hot, C_cold, U, A, T_h_in, T_c_in, hx_type) -> tuple[str, dict]:
        r = heat_exchanger_ntu(
            C_hot=C_hot, C_cold=C_cold, U=U, A=A,
            T_h_in=T_h_in, T_c_in=T_c_in, hx_type=hx_type,
        )
        lines = [
            f"Heat Exchanger — NTU-Effectiveness  ({hx_type.replace('_', ' ').title()})",
            "",
            f"C_hot  = {C_hot:.4g} W/K     C_cold = {C_cold:.4g} W/K",
            f"C_min  = {r['C_min']:.4g} W/K   C_max  = {r['C_max']:.4g} W/K",
            f"C_r    = {r['C_r']:.4f}",
            f"NTU    = {r['NTU']:.4f}",
            f"U      = {U:.4g} W/(m²·K)   A = {A:.4g} m²",
            "",
            f"Effectiveness ε = {r['eps']:.4f}  ({r['eps']*100:.2f}%)",
            f"Q_max           = {r['Q_max']:.5g} W",
            f"Q               = {r['Q_W']:.5g} W  ({r['Q_W']/1000:.4g} kW)",
            "",
            f"Outlet temperatures:",
            f"  Hot   T_out = {r['T_h_out']:.4f} K",
            f"  Cold  T_out = {r['T_c_out']:.4f} K",
        ]
        return "\n".join(lines), r

    # ------------------------------------------------------------------
    # Tab 4 — Radiation
    # ------------------------------------------------------------------

    def run_blackbody(self, T, A) -> tuple[str, dict]:
        r = radiation_blackbody(T=T, A=A)
        plot_data = radiation_planck_spectrum(T=T)
        plot_data["T"] = T
        plot_data["A"] = A
        plot_data["Q_total"] = r["Q_W"]
        lines = [
            "Blackbody Radiation  (Stefan-Boltzmann Law)",
            "",
            f"Temperature  T = {T:.4g} K",
            f"Area         A = {A:.4g} m²",
            f"σ            = {r['sigma']:.6e} W/(m²·K⁴)",
            "",
            f"Total emissive power   Q = {r['Q_W']:.5g} W",
            f"Wien peak wavelength   λ_max = {r['lambda_max_um']:.4g} μm",
        ]
        return "\n".join(lines), plot_data

    def run_grey_body(self, T1, T2, eps, A, F12) -> tuple[str, dict]:
        r = radiation_grey_body(T1=T1, T2=T2, eps=eps, A=A, F12=F12)
        lines = [
            "Grey-Body Net Radiation  Q = σ ε A F₁₂ (T1⁴ − T2⁴)",
            "",
            f"Surface 1 temperature  T1 = {T1:.4g} K",
            f"Surface 2 temperature  T2 = {T2:.4g} K",
            f"Emissivity             ε  = {eps:.4g}",
            f"Area                   A  = {A:.4g} m²",
            f"View factor            F₁₂ = {F12:.4g}",
            "",
            f"Net heat transfer rate  Q = {r['Q_W']:.5g} W",
            f"  (positive → from surface 1 to surface 2)",
        ]
        return "\n".join(lines), r
