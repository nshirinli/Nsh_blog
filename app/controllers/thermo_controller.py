from __future__ import annotations
from core.thermodynamics.component_data import get_component, list_component_names

from core.thermodynamics.eos import eos_z_curve, solve_eos_state
from core.thermodynamics.ideal_gas import (
    compressibility_factor,
    ideal_gas_residual,
    pressure_temperature_curve,
    solve_ideal_gas,
)
from core.thermodynamics.raoult import (
    bubble_pressure_binary,
    bubble_temperature_binary,
    dew_pressure_binary,
    dew_temperature_binary,
    txy_curve_binary,
)
from core.thermodynamics.units import (
    pressure_from_mmhg,
    pressure_from_pascal,
    pressure_to_mmhg,
    pressure_to_pascal,
    temperature_from_celsius,
    temperature_from_kelvin,
    temperature_to_celsius,
    temperature_to_kelvin,
    volume_from_m3,
    volume_to_m3,
)
from core.thermodynamics.vapour_pressure import (
    antoine_curve,
    antoine_pressure,
    antoine_temperature,
    clausius_clapeyron_line,
    estimate_heat_of_vaporization,
)
from core.thermodynamics.thermo_extensions import (
    cp_curve,
    enthalpy_curve,
    enthalpy_change,
    entropy_change,
    kirchhoff_dHrxn,
    adiabatic_flame_temperature,
    nonideal_vle_pxy,
    gE_excess,
    calc_psychro_state,
    psychrometric_chart_data,
)


class ThermoController:
    def list_available_components(self) -> list[str]:
        return list_component_names()

    def get_component_payload(self, component_name: str) -> dict:
        component = get_component(component_name)

        antoine = component["antoine"]
        critical = component["critical"]

        message = (
            "Thermodynamics • Component Database\n\n"
            f"Component: {component['name']}\n"
            f"Formula: {component['formula']}\n"
            f"Molecular weight: {component['mw_g_per_mol']:.5f} g/mol\n"
            f"Normal boiling point: {component['normal_boiling_point_C']:.3f} °C\n\n"
            "Antoine constants\n"
            f"A = {antoine['A']}\n"
            f"B = {antoine['B']}\n"
            f"C = {antoine['C']}\n"
            f"Valid range: {antoine['Tmin_C']} to {antoine['Tmax_C']} °C\n"
            f"Pressure basis: {antoine['pressure_unit']}\n\n"
            "Critical properties\n"
            f"Tc = {critical['Tc_K']} K\n"
            f"Pc = {critical['Pc_bar']} bar\n"
            f"ω = {critical['omega']}\n"
            f"Zc = {critical['Zc']}"
        )

        return {
            "message": message,
            "name": component["name"],
            "formula": component["formula"],
            "mw_g_per_mol": component["mw_g_per_mol"],
            "normal_boiling_point_C": component["normal_boiling_point_C"],
            "antoine": antoine,
            "critical": critical,
        }
    
    def evaluate_ideal_gas(
        self,
        temperature_value: float | None,
        temperature_unit: str,
        pressure_value: float | None,
        pressure_unit: str,
        volume_value: float | None,
        volume_unit: str,
        moles_value: float | None,
    ) -> dict:
        T = None if temperature_value is None else temperature_to_kelvin(temperature_value, temperature_unit)
        P = None if pressure_value is None else pressure_to_pascal(pressure_value, pressure_unit)
        V = None if volume_value is None else volume_to_m3(volume_value, volume_unit)
        n = moles_value

        missing_count = sum(value is None for value in [T, P, V, n])
        if missing_count > 1:
            raise ValueError("Enter 3 known variables and leave at most 1 blank.")

        if missing_count == 1:
            solved = solve_ideal_gas(T=T, P=P, V=V, n=n)
            T = solved["T"]
            P = solved["P"]
            V = solved["V"]
            n = solved["n"]
            mode = f"Solved for {solved['solved_for']}"
        else:
            mode = "Consistency check"

        residual = ideal_gas_residual(T=T, P=P, V=V, n=n)
        z_value = compressibility_factor(T=T, P=P, V=V, n=n)

        message = (
            "Thermodynamics • Ideal Gas\n\n"
            f"Mode: {mode}\n"
            f"T = {T:.6f} K ({temperature_from_kelvin(T, '°C'):.6f} °C)\n"
            f"P = {P:.6f} Pa ({pressure_from_pascal(P, 'bar'):.6f} bar)\n"
            f"V = {V:.6f} m³ ({volume_from_m3(V, 'L'):.6f} L)\n"
            f"n = {n:.6f} mol\n\n"
            f"Residual (PV - nRT) = {residual:.8f}\n"
            f"Compressibility factor Z = {z_value:.8f}\n\n"
            "For a perfectly consistent ideal-gas state, the residual should be near zero and Z should be near 1."
        )

        return {
            "message": message,
            "T_K": T,
            "P_Pa": P,
            "V_m3": V,
            "n_mol": n,
            "residual": residual,
            "Z": z_value,
        }

    def generate_ideal_gas_pressure_curve(
        self,
        volume_value: float,
        volume_unit: str,
        moles_value: float,
        temperature_min_value: float,
        temperature_max_value: float,
        temperature_unit: str,
        output_pressure_unit: str,
    ) -> dict:
        V = volume_to_m3(volume_value, volume_unit)
        T_min = temperature_to_kelvin(temperature_min_value, temperature_unit)
        T_max = temperature_to_kelvin(temperature_max_value, temperature_unit)

        temperatures_K, pressures_Pa = pressure_temperature_curve(
            V=V,
            n=moles_value,
            T_min=T_min,
            T_max=T_max,
        )

        temperatures_plot = [temperature_from_kelvin(T, temperature_unit) for T in temperatures_K]
        pressures_plot = [pressure_from_pascal(P, output_pressure_unit) for P in pressures_Pa]

        return {
            "x": temperatures_plot,
            "y": pressures_plot,
            "x_label": f"Temperature ({temperature_unit})",
            "y_label": f"Pressure ({output_pressure_unit})",
            "title": "Ideal Gas Pressure–Temperature Curve",
        }

    def calculate_antoine_pressure(
        self,
        A: float,
        B: float,
        C: float,
        temperature_value: float,
        temperature_unit: str,
        output_pressure_unit: str,
    ) -> dict:
        T_c = temperature_to_celsius(temperature_value, temperature_unit)
        p_mmhg = antoine_pressure(A, B, C, T_c)
        p_out = pressure_from_mmhg(p_mmhg, output_pressure_unit)

        message = (
            "Thermodynamics • Antoine Vapor Pressure\n\n"
            f"T = {T_c:.6f} °C ({temperature_from_celsius(T_c, 'K'):.6f} K)\n"
            f"P_sat = {p_mmhg:.6f} mmHg\n"
            f"P_sat = {p_out:.6f} {output_pressure_unit}\n\n"
            "Note: Antoine constants are assumed to use T in °C and pressure in mmHg."
        )

        return {
            "message": message,
            "pressure_mmhg": p_mmhg,
            "pressure_out": p_out,
            "temperature_celsius": T_c,
        }

    def calculate_antoine_temperature(
        self,
        A: float,
        B: float,
        C: float,
        pressure_value: float,
        pressure_unit: str,
        output_temperature_unit: str,
    ) -> dict:
        p_mmhg = pressure_to_mmhg(pressure_value, pressure_unit)
        T_c = antoine_temperature(A, B, C, p_mmhg)
        T_out = temperature_from_celsius(T_c, output_temperature_unit)

        message = (
            "Thermodynamics • Inverse Antoine Calculation\n\n"
            f"P_sat = {p_mmhg:.6f} mmHg\n"
            f"T = {T_c:.6f} °C\n"
            f"T = {T_out:.6f} {output_temperature_unit}\n\n"
            "Note: Antoine constants are assumed to use T in °C and pressure in mmHg."
        )

        return {
            "message": message,
            "temperature_celsius": T_c,
            "temperature_out": T_out,
        }

    def generate_antoine_curve(
        self,
        A: float,
        B: float,
        C: float,
        temperature_min_value: float,
        temperature_max_value: float,
        temperature_unit: str,
        output_pressure_unit: str,
    ) -> dict:
        T_min_c = temperature_to_celsius(temperature_min_value, temperature_unit)
        T_max_c = temperature_to_celsius(temperature_max_value, temperature_unit)

        temperatures_c, pressures_mmhg = antoine_curve(
            A=A,
            B=B,
            C=C,
            T_min_celsius=T_min_c,
            T_max_celsius=T_max_c,
        )

        temperatures_plot = [temperature_from_celsius(T, temperature_unit) for T in temperatures_c]
        pressures_plot = [pressure_from_mmhg(P, output_pressure_unit) for P in pressures_mmhg]

        return {
            "x": temperatures_plot,
            "y": pressures_plot,
            "x_label": f"Temperature ({temperature_unit})",
            "y_label": f"Vapor Pressure ({output_pressure_unit})",
            "title": "Antoine Vapor Pressure Curve",
        }

    def estimate_heat_of_vap(
        self,
        T1_value: float,
        T1_unit: str,
        P1_value: float,
        P1_unit: str,
        T2_value: float,
        T2_unit: str,
        P2_value: float,
        P2_unit: str,
    ) -> dict:
        T1_K = temperature_to_kelvin(T1_value, T1_unit)
        T2_K = temperature_to_kelvin(T2_value, T2_unit)
        P1_Pa = pressure_to_pascal(P1_value, P1_unit)
        P2_Pa = pressure_to_pascal(P2_value, P2_unit)

        delta_h = estimate_heat_of_vaporization(
            T1_kelvin=T1_K,
            P1_pascal=P1_Pa,
            T2_kelvin=T2_K,
            P2_pascal=P2_Pa,
        )

        inv_t, ln_p = clausius_clapeyron_line(
            T1_kelvin=T1_K,
            P1_pascal=P1_Pa,
            T2_kelvin=T2_K,
            P2_pascal=P2_Pa,
        )

        message = (
            "Thermodynamics • Clausius–Clapeyron Estimate\n\n"
            f"Point 1: T = {T1_K:.6f} K, P = {P1_Pa:.6f} Pa\n"
            f"Point 2: T = {T2_K:.6f} K, P = {P2_Pa:.6f} Pa\n\n"
            f"Estimated ΔHvap = {delta_h:.6f} J/mol\n"
            f"Estimated ΔHvap = {delta_h / 1000.0:.6f} kJ/mol"
        )

        return {
            "message": message,
            "delta_h_j_per_mol": delta_h,
            "x": inv_t,
            "y": ln_p,
            "x_label": "1 / T (1/K)",
            "y_label": "ln(Pa)",
            "title": "Clausius–Clapeyron Line",
        }

    def calculate_binary_bubble_pressure(
        self,
        A1: float,
        B1: float,
        C1: float,
        A2: float,
        B2: float,
        C2: float,
        temperature_value: float,
        temperature_unit: str,
        x1: float,
        output_pressure_unit: str,
    ) -> dict:
        T_c = temperature_to_celsius(temperature_value, temperature_unit)

        result = bubble_pressure_binary(
            x1=x1,
            T_celsius=T_c,
            constants_1=(A1, B1, C1),
            constants_2=(A2, B2, C2),
        )

        p_out = pressure_from_mmhg(result["P_bubble_mmhg"], output_pressure_unit)

        message = (
            "Thermodynamics • Binary Bubble Pressure (Raoult's Law)\n\n"
            f"T = {T_c:.6f} °C\n"
            f"x1 = {x1:.6f}, x2 = {1 - x1:.6f}\n"
            f"P_sat,1 = {result['P_sat_1_mmhg']:.6f} mmHg\n"
            f"P_sat,2 = {result['P_sat_2_mmhg']:.6f} mmHg\n\n"
            f"P_bubble = {result['P_bubble_mmhg']:.6f} mmHg\n"
            f"P_bubble = {p_out:.6f} {output_pressure_unit}\n"
            f"y1 = {result['y1']:.6f}, y2 = {result['y2']:.6f}"
        )

        return {
            "message": message,
            "pressure_out": p_out,
            "pressure_mmhg": result["P_bubble_mmhg"],
            "y1": result["y1"],
            "y2": result["y2"],
        }

    def calculate_binary_dew_pressure(
        self,
        A1: float,
        B1: float,
        C1: float,
        A2: float,
        B2: float,
        C2: float,
        temperature_value: float,
        temperature_unit: str,
        y1: float,
        output_pressure_unit: str,
    ) -> dict:
        T_c = temperature_to_celsius(temperature_value, temperature_unit)

        result = dew_pressure_binary(
            y1=y1,
            T_celsius=T_c,
            constants_1=(A1, B1, C1),
            constants_2=(A2, B2, C2),
        )

        p_out = pressure_from_mmhg(result["P_dew_mmhg"], output_pressure_unit)

        message = (
            "Thermodynamics • Binary Dew Pressure (Raoult's Law)\n\n"
            f"T = {T_c:.6f} °C\n"
            f"y1 = {y1:.6f}, y2 = {1 - y1:.6f}\n"
            f"P_sat,1 = {result['P_sat_1_mmhg']:.6f} mmHg\n"
            f"P_sat,2 = {result['P_sat_2_mmhg']:.6f} mmHg\n\n"
            f"P_dew = {result['P_dew_mmhg']:.6f} mmHg\n"
            f"P_dew = {p_out:.6f} {output_pressure_unit}\n"
            f"x1 = {result['x1']:.6f}, x2 = {result['x2']:.6f}"
        )

        return {
            "message": message,
            "pressure_out": p_out,
            "pressure_mmhg": result["P_dew_mmhg"],
            "x1": result["x1"],
            "x2": result["x2"],
        }

    def calculate_binary_bubble_temperature(
        self,
        A1: float,
        B1: float,
        C1: float,
        A2: float,
        B2: float,
        C2: float,
        pressure_value: float,
        pressure_unit: str,
        x1: float,
        output_temperature_unit: str,
    ) -> dict:
        P_mmhg = pressure_to_mmhg(pressure_value, pressure_unit)

        result = bubble_temperature_binary(
            x1=x1,
            P_mmhg=P_mmhg,
            constants_1=(A1, B1, C1),
            constants_2=(A2, B2, C2),
        )

        T_out = temperature_from_celsius(result["T_bubble_celsius"], output_temperature_unit)

        message = (
            "Thermodynamics • Binary Bubble Temperature (Raoult's Law)\n\n"
            f"P = {P_mmhg:.6f} mmHg\n"
            f"x1 = {x1:.6f}, x2 = {1 - x1:.6f}\n\n"
            f"T_bubble = {result['T_bubble_celsius']:.6f} °C\n"
            f"T_bubble = {T_out:.6f} {output_temperature_unit}\n"
            f"y1 = {result['y1']:.6f}, y2 = {result['y2']:.6f}"
        )

        return {
            "message": message,
            "temperature_out": T_out,
            "temperature_celsius": result["T_bubble_celsius"],
            "y1": result["y1"],
            "y2": result["y2"],
        }

    def calculate_binary_dew_temperature(
        self,
        A1: float,
        B1: float,
        C1: float,
        A2: float,
        B2: float,
        C2: float,
        pressure_value: float,
        pressure_unit: str,
        y1: float,
        output_temperature_unit: str,
    ) -> dict:
        P_mmhg = pressure_to_mmhg(pressure_value, pressure_unit)

        result = dew_temperature_binary(
            y1=y1,
            P_mmhg=P_mmhg,
            constants_1=(A1, B1, C1),
            constants_2=(A2, B2, C2),
        )

        T_out = temperature_from_celsius(result["T_dew_celsius"], output_temperature_unit)

        message = (
            "Thermodynamics • Binary Dew Temperature (Raoult's Law)\n\n"
            f"P = {P_mmhg:.6f} mmHg\n"
            f"y1 = {y1:.6f}, y2 = {1 - y1:.6f}\n\n"
            f"T_dew = {result['T_dew_celsius']:.6f} °C\n"
            f"T_dew = {T_out:.6f} {output_temperature_unit}\n"
            f"x1 = {result['x1']:.6f}, x2 = {result['x2']:.6f}"
        )

        return {
            "message": message,
            "temperature_out": T_out,
            "temperature_celsius": result["T_dew_celsius"],
            "x1": result["x1"],
            "x2": result["x2"],
        }

    def generate_txy_curve(
        self,
        A1: float,
        B1: float,
        C1: float,
        A2: float,
        B2: float,
        C2: float,
        pressure_value: float,
        pressure_unit: str,
        output_temperature_unit: str,
    ) -> dict:
        P_mmhg = pressure_to_mmhg(pressure_value, pressure_unit)

        result = txy_curve_binary(
            P_mmhg=P_mmhg,
            constants_1=(A1, B1, C1),
            constants_2=(A2, B2, C2),
        )

        temperatures_plot = [
            temperature_from_celsius(T, output_temperature_unit)
            for T in result["T_bubble_celsius"]
        ]

        return {
            "x_bubble": result["x1_bubble"],
            "x_dew": result["y1_dew"],
            "y": temperatures_plot,
            "title": f"T-x-y Diagram at {pressure_value} {pressure_unit}",
            "x_label": "Mole fraction of component 1",
            "y_label": f"Temperature ({output_temperature_unit})",
        }

    def calculate_eos_state(
        self,
        eos_name: str,
        temperature_value: float,
        pressure_value: float,
        pressure_unit: str,
        critical_temperature_value: float,
        critical_pressure_value: float,
        critical_pressure_unit: str,
        acentric_factor: float,
    ) -> dict:
        T = temperature_value
        P = pressure_to_pascal(pressure_value, pressure_unit)
        Tc = critical_temperature_value
        Pc = pressure_to_pascal(critical_pressure_value, critical_pressure_unit)

        state = solve_eos_state(
            eos_name=eos_name,
            T=T,
            P=P,
            Tc=Tc,
            Pc=Pc,
            omega=acentric_factor,
        )

        roots_text = ", ".join(f"{root:.6f}" for root in state["roots"])

        message_lines = [
            f"Thermodynamics • {state['model']} EOS",
            "",
            f"T = {T:.6f} K",
            f"P = {P:.6f} Pa ({pressure_from_pascal(P, 'bar'):.6f} bar)",
            f"Tc = {Tc:.6f} K",
            f"Pc = {Pc:.6f} Pa ({pressure_from_pascal(Pc, 'bar'):.6f} bar)",
            f"ω = {acentric_factor:.6f}",
            "",
            f"A = {state['A']:.6f}",
            f"B = {state['B']:.6f}",
            f"Real Z roots: {roots_text}",
            f"Liquid root Z = {state['z_liquid']:.6f}",
            f"Vapor root Z = {state['z_vapor']:.6f}",
            "",
            f"Liquid molar volume = {state['v_liquid_m3_per_mol']:.8e} m³/mol",
            f"Vapor molar volume = {state['v_vapor_m3_per_mol']:.8e} m³/mol",
        ]

        if state["model"] == "Peng–Robinson":
            if state.get("phi_vapor") is not None:
                message_lines.extend(
                    [
                        "",
                        f"φ_vapor = {state['phi_vapor']:.6f}",
                        f"φ_liquid = {state['phi_liquid']:.6f}",
                        f"ln φ_vapor = {state['ln_phi_vapor']:.6f}",
                        f"ln φ_liquid = {state['ln_phi_liquid']:.6f}",
                    ]
                )
            else:
                message_lines.extend(
                    [
                        "",
                        "Fugacity coefficient could not be evaluated for this root selection.",
                    ]
                )

        return {
            "message": "\n".join(message_lines),
            "state": state,
        }

    def generate_eos_z_curve(
        self,
        eos_name: str,
        temperature_value: float,
        critical_temperature_value: float,
        critical_pressure_value: float,
        critical_pressure_unit: str,
        acentric_factor: float,
        pressure_min_value: float,
        pressure_max_value: float,
        pressure_unit: str,
    ) -> dict:
        T = temperature_value
        Tc = critical_temperature_value
        Pc = pressure_to_pascal(critical_pressure_value, critical_pressure_unit)
        P_min = pressure_to_pascal(pressure_min_value, pressure_unit)
        P_max = pressure_to_pascal(pressure_max_value, pressure_unit)

        pressures_Pa, z_vapor, z_liquid = eos_z_curve(
            eos_name=eos_name,
            T=T,
            Tc=Tc,
            Pc=Pc,
            omega=acentric_factor,
            P_min=P_min,
            P_max=P_max,
        )

        pressures_plot = [pressure_from_pascal(P, pressure_unit) for P in pressures_Pa]

        return {
            "x": pressures_plot,
            "z_vapor": z_vapor,
            "z_liquid": z_liquid,
            "title": f"{eos_name} Compressibility Curve",
            "x_label": f"Pressure ({pressure_unit})",
            "y_label": "Compressibility factor Z",
        }

    # -------------------------------------------------------------------
    # Enthalpy / Entropy (Cp polynomial)
    # -------------------------------------------------------------------

    def run_enthalpy_entropy(
        self,
        a: float, b: float, c: float, d: float,
        T1_K: float, T2_K: float,
    ) -> tuple[str, dict]:
        dH = enthalpy_change(a, b, c, d, T1_K, T2_K)
        dS = entropy_change(a, b, c, d, T1_K, T2_K)
        dG = dH - T2_K * dS
        T_arr, Cp_arr = cp_curve(a, b, c, d, T1_K, T2_K)
        _, H_arr = enthalpy_curve(a, b, c, d, T1_K, T2_K)

        lines = [
            "Enthalpy / Entropy from Cp Polynomial",
            "",
            f"Cp(T) = {a:.6g} + {b:.6g}T + {c:.6g}T² + {d:.6g}T³  [J/(mol·K)]",
            f"T₁ = {T1_K:.2f} K,  T₂ = {T2_K:.2f} K",
            "",
            f"ΔH = {dH/1000:.4f} kJ/mol",
            f"ΔS = {dS:.4f} J/(mol·K)",
            f"ΔG = ΔH − T₂·ΔS = {dG/1000:.4f} kJ/mol",
            "",
            f"Cp(T₁) = {a + b*T1_K + c*T1_K**2 + d*T1_K**3:.4f} J/(mol·K)",
            f"Cp(T₂) = {a + b*T2_K + c*T2_K**2 + d*T2_K**3:.4f} J/(mol·K)",
        ]
        return "\n".join(lines), {
            "T_arr": T_arr, "Cp_arr": Cp_arr, "H_arr": H_arr,
            "dH_J": dH, "dS_J_K": dS, "dG_J": dG,
        }

    def run_kirchhoff(
        self,
        dHrxn_ref_kJ: float,
        T_ref_K: float,
        T_calc_K: float,
        da: float, db: float, dc: float, dd: float,
    ) -> tuple[str, dict]:
        result = kirchhoff_dHrxn(
            dHrxn_ref_J=dHrxn_ref_kJ * 1000,
            T_ref_K=T_ref_K,
            T_calc_K=T_calc_K,
            delta_a=da, delta_b=db, delta_c=dc, delta_d=dd,
        )
        lines = [
            "Kirchhoff's Law — ΔH_rxn(T)",
            "",
            f"ΔH_rxn(T_ref = {T_ref_K:.2f} K) = {dHrxn_ref_kJ:.4f} kJ/mol",
            f"ΔCp = {da:.6g} + {db:.6g}T + {dc:.6g}T² + {dd:.6g}T³",
            "",
            f"ΔH_rxn(T = {T_calc_K:.2f} K) = {result['dHrxn_T_kJ']:.4f} kJ/mol",
            f"Correction  ΔH_corr = {result['dH_corr_J']/1000:.4f} kJ/mol",
        ]
        return "\n".join(lines), result

    # -------------------------------------------------------------------
    # Activity Coefficients / Non-ideal VLE
    # -------------------------------------------------------------------

    def run_activity_vle(
        self,
        model: str,
        A12: float,
        A21: float,
        Psat1_mmHg: float,
        Psat2_mmHg: float,
        T_K: float,
    ) -> tuple[str, dict]:
        pxy = nonideal_vle_pxy(A12, A21, model, Psat1_mmHg, Psat2_mmHg)
        ge  = gE_excess(A12, A21, model, T_K)

        az = pxy["azeotrope"]
        az_str = (
            f"  x_az = {az['x_az']:.4f},  P_az = {az['P_az_mmHg']:.2f} mmHg"
            if az else "  None detected"
        )

        lines = [
            f"Non-ideal VLE — {model}",
            "",
            f"A₁₂ = {A12:.4g},  A₂₁ = {A21:.4g}",
            f"P_sat,1 = {Psat1_mmHg:.4g} mmHg,  P_sat,2 = {Psat2_mmHg:.4g} mmHg",
            f"T = {T_K:.2f} K  (for G^E calculation)",
            "",
            "Azeotrope:",
            az_str,
            "",
            f"P_bubble at x1=0.5 ≈ {float(pxy['P_mmHg'][49]):.2f} mmHg",
            f"γ₁ at x1=0.5 ≈ {float(pxy['g1'][49]):.4f}",
            f"γ₂ at x1=0.5 ≈ {float(pxy['g2'][49]):.4f}",
        ]
        return "\n".join(lines), {"pxy": pxy, "ge": ge}

    # -------------------------------------------------------------------
    # Psychrometrics
    # -------------------------------------------------------------------

    def run_psychrometrics(
        self,
        T_db_C: float,
        T_wb_C: float,
    ) -> tuple[str, dict]:
        state = calc_psychro_state(T_db_C, T_wb_C)
        chart = psychrometric_chart_data(T_min_C=-10, T_max_C=50)

        lines = [
            "Psychrometrics — Humid Air State",
            "",
            f"Dry-bulb T    = {state['T_db_C']:.2f} °C",
            f"Wet-bulb T    = {state['T_wb_C']:.2f} °C",
            f"Dew-point T   = {state['T_dp_C']:.2f} °C",
            "",
            f"Humidity ratio  ω = {state['W_g_kg']:.2f} g/kg  ({state['W_kg_kg']:.5f} kg/kg)",
            f"Relative humidity = {state['RH_pct']:.1f}%",
            f"Enthalpy          = {state['h_kJ_kg']:.3f} kJ/kg dry air",
            f"Specific volume   = {state['v_m3_kg']:.4f} m³/kg dry air",
            f"Partial press H₂O = {state['Pw_Pa']:.2f} Pa",
        ]
        return "\n".join(lines), {"state": state, "chart": chart}

    # -------------------------------------------------------------------
    # Adiabatic Flame Temperature
    # -------------------------------------------------------------------

    def run_adiabatic_flame(
        self,
        dH_comb_kJ: float,
        T_reactants_C: float,
        Cp_products_J_molK: float,
        n_products: float,
    ) -> tuple[str, dict]:
        result = adiabatic_flame_temperature(
            dH_comb_J=dH_comb_kJ * 1000,
            T_reactants_K=T_reactants_C + 273.15,
            Cp_products_J_molK=Cp_products_J_molK,
            n_products=n_products,
        )
        lines = [
            "Adiabatic Flame Temperature",
            "",
            f"Heat of combustion  ΔH_comb = {dH_comb_kJ:.4f} kJ/mol",
            f"Reactants inlet T   = {T_reactants_C:.2f} °C  ({T_reactants_C+273.15:.2f} K)",
            f"Products Cp (avg)   = {Cp_products_J_molK:.4f} J/(mol·K)",
            f"Moles of products   = {n_products:.4g} per mol fuel",
            "",
            f"Adiabatic flame temperature  T_ad = {result['T_ad_K']:.2f} K  ({result['T_ad_C']:.2f} °C)",
            f"Heat released  Q = {result['Q_released_J']/1000:.4f} kJ/mol",
            "",
            "See plot for T_ad vs. equivalence ratio φ.",
        ]
        return "\n".join(lines), result