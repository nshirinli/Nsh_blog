import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.thermo_controller import ThermoController

st.set_page_config(layout="wide", page_title="Thermodynamics - ChemEng")
st.title("🌡 Thermodynamics")

ctrl = ThermoController()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Ideal Gas", "Vapor Pressure", "VLE / Bubble-Dew",
    "EOS / Real Gas", "Enthalpy & Entropy", "Activity Coeff.",
    "Psychrometrics", "Adiabatic Flame"
])

# Tab 1 - Ideal Gas
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Ideal Gas Law")
            T_val = st.number_input("Temperature", value=298.15, step=1.0)
            T_unit = st.selectbox("T unit", ["K", "C", "F"])
            P_val = st.number_input("Pressure", value=101325.0, step=100.0)
            P_unit = st.selectbox("P unit", ["Pa", "bar", "atm", "kPa"])
            V_val = st.number_input("Volume", value=24.5, step=0.1)
            V_unit = st.selectbox("V unit", ["L", "m3"])
            n_val = st.number_input("Moles (n)", value=1.0, step=0.1)
            st.markdown("**PV Curve Settings**")
            T_min_curve = st.number_input("T min (curve)", value=200.0, step=10.0)
            T_max_curve = st.number_input("T max (curve)", value=600.0, step=10.0)
            out_P_unit = st.selectbox("Output P unit", ["Pa", "bar", "atm", "kPa"])
            submitted = st.form_submit_button("Calculate & Plot", use_container_width=True)
        if submitted:
            try:
                result = ctrl.evaluate_ideal_gas(T_val, T_unit, P_val, P_unit, V_val, V_unit, n_val)
                msg = result.get("message", "")
                curve = ctrl.generate_ideal_gas_pressure_curve(V_val, V_unit, n_val, T_min_curve, T_max_curve, T_unit, out_P_unit)
                st.session_state["res_t1"] = (msg, curve)
            except Exception as e:
                st.session_state["res_t1"] = (f"Error: {e}", {})
    with col_out:
        if "res_t1" in st.session_state:
            msg, data = st.session_state["res_t1"]
            if msg:
                st.code(msg, language=None)
            if data and "T" in data and "P" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T"], data["P"], color="steelblue", linewidth=2)
                ax.set_xlabel(f"Temperature ({T_unit})")
                ax.set_ylabel(f"Pressure ({out_P_unit})")
                ax.set_title("Ideal Gas: Pressure vs Temperature")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Vapor Pressure
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Antoine Equation")
            A = st.number_input("Antoine A", value=8.07131, step=0.001, format="%.5f")
            B = st.number_input("Antoine B", value=1730.63, step=0.01)
            C = st.number_input("Antoine C", value=233.426, step=0.001, format="%.3f")
            T_vp = st.number_input("Temperature", value=100.0, step=1.0)
            T_unit_vp = st.selectbox("T unit", ["C", "K"])
            P_unit_vp = st.selectbox("P unit", ["mmHg", "Pa", "bar", "atm"])
            T_min_vp = st.number_input("T min (curve)", value=25.0, step=1.0)
            T_max_vp = st.number_input("T max (curve)", value=150.0, step=1.0)
            submitted = st.form_submit_button("Calculate & Plot", use_container_width=True)
        if submitted:
            try:
                point = ctrl.calculate_antoine_pressure(A, B, C, T_vp, T_unit_vp, P_unit_vp)
                curve = ctrl.generate_antoine_curve(A, B, C, T_min_vp, T_max_vp, T_unit_vp, P_unit_vp)
                msg = point.get("message", "")
                st.session_state["res_t2"] = (msg, curve)
            except Exception as e:
                st.session_state["res_t2"] = (f"Error: {e}", {})
    with col_out:
        if "res_t2" in st.session_state:
            msg, data = st.session_state["res_t2"]
            if msg:
                st.code(msg, language=None)
            if data and "T_values" in data and "P_values" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T_values"], data["P_values"], color="darkorange", linewidth=2)
                ax.set_xlabel(f"Temperature ({T_unit_vp})")
                ax.set_ylabel(f"Vapor Pressure ({P_unit_vp})")
                ax.set_title("Antoine Vapor Pressure Curve")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - VLE / Bubble-Dew
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("VLE T-x-y Diagram")
            st.markdown("**Component 1 Antoine Constants**")
            A1 = st.number_input("A1", value=8.07131, step=0.001, format="%.5f")
            B1 = st.number_input("B1", value=1730.63, step=0.01)
            C1 = st.number_input("C1", value=233.426, step=0.001, format="%.3f")
            st.markdown("**Component 2 Antoine Constants**")
            A2 = st.number_input("A2", value=7.96681, step=0.001, format="%.5f")
            B2 = st.number_input("B2", value=1668.21, step=0.01)
            C2 = st.number_input("C2", value=228.0, step=0.001, format="%.3f")
            P_vle = st.number_input("Pressure", value=760.0, step=1.0)
            P_unit_vle = st.selectbox("P unit", ["mmHg", "Pa", "bar", "atm"])
            T_unit_vle = st.selectbox("T unit", ["C", "K"])
            submitted = st.form_submit_button("Generate T-x-y", use_container_width=True)
        if submitted:
            try:
                data = ctrl.generate_txy_curve(A1, B1, C1, A2, B2, C2, P_vle, P_unit_vle, T_unit_vle)
                st.session_state["res_t3"] = ("T-x-y curve generated.", data)
            except Exception as e:
                st.session_state["res_t3"] = (f"Error: {e}", {})
    with col_out:
        if "res_t3" in st.session_state:
            msg, data = st.session_state["res_t3"]
            if msg:
                st.code(msg, language=None)
            if data and "x" in data and "T_bubble" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["x"], data["T_bubble"], label="Bubble", color="steelblue", linewidth=2)
                ax.plot(data["y"], data["T_dew"], label="Dew", color="tomato", linewidth=2, linestyle="--")
                ax.set_xlabel("Mole fraction of component 1")
                ax.set_ylabel(f"Temperature ({T_unit_vle})")
                ax.set_title("T-x-y Diagram")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - EOS / Real Gas
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Equation of State")
            eos_name = st.selectbox("EOS", ["van der Waals", "Redlich-Kwong", "Soave-RK", "Peng-Robinson"])
            T_eos = st.number_input("Temperature (K)", value=400.0, step=1.0)
            P_eos = st.number_input("Pressure", value=50.0, step=1.0)
            P_unit_eos = st.selectbox("P unit", ["bar", "Pa", "atm", "kPa"])
            Tc = st.number_input("Critical Temp Tc (K)", value=304.2, step=0.1)
            Pc = st.number_input("Critical Pressure Pc", value=73.8, step=0.1)
            Pc_unit = st.selectbox("Pc unit", ["bar", "Pa", "atm", "kPa"])
            omega = st.number_input("Acentric factor ω", value=0.228, step=0.001, format="%.3f")
            P_min_eos = st.number_input("P min (curve)", value=1.0, step=1.0)
            P_max_eos = st.number_input("P max (curve)", value=200.0, step=10.0)
            submitted = st.form_submit_button("Calculate & Plot", use_container_width=True)
        if submitted:
            try:
                state = ctrl.calculate_eos_state(eos_name, T_eos, P_eos, P_unit_eos, Tc, Pc, Pc_unit, omega)
                msg = state.get("message", "")
                curve = ctrl.generate_eos_z_curve(eos_name, T_eos, Tc, Pc, Pc_unit, omega, P_min_eos, P_max_eos, P_unit_eos)
                st.session_state["res_t4"] = (msg, curve)
            except Exception as e:
                st.session_state["res_t4"] = (f"Error: {e}", {})
    with col_out:
        if "res_t4" in st.session_state:
            msg, data = st.session_state["res_t4"]
            if msg:
                st.code(msg, language=None)
            if data and "P_values" in data and "Z_values" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["P_values"], data["Z_values"], color="purple", linewidth=2)
                ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal Gas (Z=1)")
                ax.set_xlabel(f"Pressure ({P_unit_eos})")
                ax.set_ylabel("Compressibility Factor Z")
                ax.set_title(f"{eos_name} – Z vs P")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Enthalpy & Entropy
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Enthalpy & Entropy")
            st.markdown("Cp = a + bT + cT² + dT³")
            a = st.number_input("a", value=28.98, step=0.01)
            b = st.number_input("b", value=0.00157, step=0.0001, format="%.5f")
            c = st.number_input("c", value=0.0, step=0.0001, format="%.5f")
            d = st.number_input("d", value=0.0, step=0.000001, format="%.6f")
            T1_K = st.number_input("T1 (K)", value=298.0, step=1.0)
            T2_K = st.number_input("T2 (K)", value=1000.0, step=1.0)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_enthalpy_entropy(a, b, c, d, T1_K, T2_K)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t5"] = (msg, data)
            except Exception as e:
                st.session_state["res_t5"] = (f"Error: {e}", {})
    with col_out:
        if "res_t5" in st.session_state:
            msg, data = st.session_state["res_t5"]
            if msg:
                st.code(msg, language=None)
            if data and "T" in data and "H" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T"], data["H"], color="firebrick", linewidth=2, label="ΔH")
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel("Enthalpy (J/mol)")
                ax.set_title("Enthalpy vs Temperature")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Activity Coeff
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Activity Coefficient VLE")
            model = st.selectbox("Model", ["Margules", "van Laar"])
            A12 = st.number_input("A12", value=1.5, step=0.1)
            A21 = st.number_input("A21", value=0.8, step=0.1)
            Psat1 = st.number_input("Psat1 (mmHg)", value=150.0, step=1.0)
            Psat2 = st.number_input("Psat2 (mmHg)", value=75.0, step=1.0)
            T_K_act = st.number_input("Temperature (K)", value=333.15, step=0.1)
            submitted = st.form_submit_button("Calculate VLE", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_activity_vle(model, A12, A21, Psat1, Psat2, T_K_act)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t6"] = (msg, data)
            except Exception as e:
                st.session_state["res_t6"] = (f"Error: {e}", {})
    with col_out:
        if "res_t6" in st.session_state:
            msg, data = st.session_state["res_t6"]
            if msg:
                st.code(msg, language=None)
            if data and "x1" in data and "y1" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["x1"], data["y1"], color="darkgreen", linewidth=2, label="y1 vs x1")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x")
                ax.set_xlabel("x1 (liquid mole fraction)")
                ax.set_ylabel("y1 (vapor mole fraction)")
                ax.set_title(f"{model} – y-x Diagram")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 7 - Psychrometrics
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t7"):
            st.subheader("Psychrometrics")
            T_db = st.number_input("Dry-bulb Temperature (°C)", value=25.0, step=0.5)
            T_wb = st.number_input("Wet-bulb Temperature (°C)", value=18.0, step=0.5)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_psychrometrics(T_db, T_wb)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t7"] = (msg, data)
            except Exception as e:
                st.session_state["res_t7"] = (f"Error: {e}", {})
    with col_out:
        if "res_t7" in st.session_state:
            msg, data = st.session_state["res_t7"]
            if msg:
                st.code(msg, language=None)
            state = data.get("state", {}) if data else {}
            if state:
                props = {
                    "Humidity ratio\n(g/kg)": state.get("W_g_kg", 0),
                    "Rel. Humidity\n(%)": state.get("RH_pct", 0),
                    "Enthalpy\n(kJ/kg)": state.get("h_kJ_kg", 0),
                    "Spec. Volume\n(m³/kg)": state.get("v_m3_kg", 0),
                }
                labels = list(props.keys())
                values = list(props.values())
                colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8, edgecolor="white")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Value")
                ax.set_title(f"Psychrometric State  (T_db={state.get('T_db_C',0):.1f}°C, T_wb={state.get('T_wb_C',0):.1f}°C)")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 8 - Adiabatic Flame
with tab8:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t8"):
            st.subheader("Adiabatic Flame Temperature")
            dH_comb = st.number_input("ΔH_combustion (kJ/mol)", value=-890.0, step=1.0)
            T_reac = st.number_input("T reactants (°C)", value=25.0, step=1.0)
            Cp_prod = st.number_input("Cp products (J/mol·K)", value=38.0, step=0.5)
            n_prod = st.number_input("n products (mol)", value=3.0, step=0.5)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_adiabatic_flame(dH_comb, T_reac, Cp_prod, n_prod)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t8"] = (msg, data)
            except Exception as e:
                st.session_state["res_t8"] = (f"Error: {e}", {})
    with col_out:
        if "res_t8" in st.session_state:
            msg, data = st.session_state["res_t8"]
            if msg:
                st.code(msg, language=None)
            if data and "T_ad_K" in data:
                T_react_K = T_reac + 273.15
                T_ad_K = data["T_ad_K"]
                fig, ax = plt.subplots(figsize=(8, 4))
                labels = ["T Reactants", "T Adiabatic Flame"]
                values = [T_react_K, T_ad_K]
                colors = ["steelblue", "firebrick"]
                bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="white", height=0.5)
                for bar, val in zip(bars, values):
                    ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f} K  ({val - 273.15:.1f} °C)",
                            va="center", fontsize=10)
                ax.set_xlabel("Temperature (K)")
                ax.set_title("Adiabatic Flame Temperature")
                ax.set_xlim(0, T_ad_K * 1.15)
                ax.grid(True, alpha=0.3, axis="x")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
