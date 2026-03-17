import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.electrochemistry_controller import ElectrochemController

st.set_page_config(layout="wide", page_title="Electrochemistry - ChemEng")
st.title("⚡ Electrochemistry")

ctrl = ElectrochemController()

def _default(key, fn):
    if key not in st.session_state:
        try:
            r = fn(); st.session_state[key] = r if isinstance(r, tuple) else (r.get("message",""), r)
        except Exception: pass

_default("res_t1", lambda: ctrl.run_nernst(1.23,2,1.0,25.0))
_default("res_t2", lambda: ctrl.run_butler_volmer(1e-3,0.5,25.0,0.5))
_default("res_t3", lambda: ctrl.run_faraday(10.0,1.0,63.5,2,0.95))
_default("res_t4", lambda: ctrl.run_fuel_cell(80.0,1.5,1e-4,0.1,0.4,1.4))
_default("res_t5", lambda: ctrl.run_corrosion(1e-2,55.85,2,7.87,0.01))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Nernst Equation", "Butler-Volmer", "Faraday Electrolysis",
    "Fuel Cell", "Corrosion"
])

# Tab 1 - Nernst Equation
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Nernst Equation")
            E0 = st.number_input("Standard cell potential E0 (V)", value=1.23, step=0.01)
            n_nernst = st.number_input("Number of electrons n", value=2, min_value=1, step=1)
            Q_nernst = st.number_input("Reaction quotient Q", value=1.0, step=0.1)
            T_C_nernst = st.number_input("Temperature (°C)", value=25.0, step=1.0)
            submitted = st.form_submit_button("Calculate Nernst", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_nernst(E0, int(n_nernst), Q_nernst, T_C_nernst)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t1"] = (msg, data)
            except Exception as e:
                st.session_state["res_t1"] = (f"Error: {e}", {})
    with col_out:
        if "res_t1" in st.session_state:
            msg, data = st.session_state["res_t1"]
            if msg:
                st.code(msg, language=None)
            if data and "T" in data and "E_cell" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T"], data["E_cell"], color="steelblue", linewidth=2)
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Cell Potential E (V)")
                ax.set_title("Nernst: Cell Potential vs Temperature")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Butler-Volmer
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Butler-Volmer Kinetics")
            i0 = st.number_input("Exchange current density i0 (A/m²)", value=1e-3, step=1e-4, format="%.4e")
            alpha = st.number_input("Transfer coefficient α", value=0.5, step=0.05)
            T_C_bv = st.number_input("Temperature (°C)", value=25.0, step=1.0)
            eta_max = st.number_input("Max overpotential η_max (V)", value=0.5, step=0.05)
            submitted = st.form_submit_button("Calculate Butler-Volmer", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_butler_volmer(i0, alpha, T_C_bv, eta_max)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t2"] = (msg, data)
            except Exception as e:
                st.session_state["res_t2"] = (f"Error: {e}", {})
    with col_out:
        if "res_t2" in st.session_state:
            msg, data = st.session_state["res_t2"]
            if msg:
                st.code(msg, language=None)
            if data and "eta" in data and "i" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                eta_mV = [e * 1000 for e in data["eta"]]
                i_mA = [curr * 1000 for curr in data["i"]]
                ax.plot(eta_mV, i_mA, color="darkorange", linewidth=2)
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Overpotential η (mV)")
                ax.set_ylabel("Current density i (mA/m²)")
                ax.set_title("Butler-Volmer Polarization Curve")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Faraday Electrolysis
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Faraday's Law of Electrolysis")
            current = st.number_input("Current (A)", value=10.0, step=0.5)
            time_h = st.number_input("Time (h)", value=1.0, step=0.1)
            M_molar = st.number_input("Molar mass of deposit (g/mol)", value=63.5, step=0.5)
            n_far = st.number_input("Electrons per mol n", value=2, min_value=1, step=1)
            current_eff = st.number_input("Current efficiency", value=0.95, min_value=0.01, max_value=1.0, step=0.01)
            submitted = st.form_submit_button("Calculate Mass Deposited", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_faraday(current, time_h, M_molar, int(n_far), current_eff)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t3"] = (msg, data)
            except Exception as e:
                st.session_state["res_t3"] = (f"Error: {e}", {})
    with col_out:
        if "res_t3" in st.session_state:
            msg, data = st.session_state["res_t3"]
            if msg:
                st.code(msg, language=None)
            if data and "mass_g" in data and "moles" in data and "Q_C" in data:
                labels_far = ["Charge Q (C)", "Moles (mol)", "Mass deposited (g)"]
                values_far = [data["Q_C"], data["moles"], data["mass_g"]]
                colors_far = ["steelblue", "darkorange", "seagreen"]
                x_pos = np.arange(len(labels_far))
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(x_pos, values_far, color=colors_far, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, values_far):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(values_far) * 0.01,
                            f"{val:.4g}", ha="center", va="bottom", fontsize=9)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels_far)
                ax.set_ylabel("Value")
                ax.set_title("Faraday Electrolysis – Key Quantities")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Fuel Cell
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("PEM Fuel Cell Polarization")
            T_C_fc = st.number_input("Temperature (°C)", value=80.0, step=1.0)
            i_max = st.number_input("Max current density i_max (A/cm²)", value=1.5, step=0.1)
            i0_cathode = st.number_input("Cathode exchange current i0 (A/cm²)", value=1e-4, step=1e-5, format="%.2e")
            R_ohmic = st.number_input("Ohmic resistance R (Ω·cm²)", value=0.1, step=0.01)
            alpha_c = st.number_input("Cathode transfer coefficient α", value=0.4, step=0.05)
            i_limit = st.number_input("Limiting current i_L (A/cm²)", value=1.4, step=0.05)
            submitted = st.form_submit_button("Plot Polarization Curve", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_fuel_cell(T_C_fc, i_max, i0_cathode, R_ohmic, alpha_c, i_limit)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t4"] = (msg, data)
            except Exception as e:
                st.session_state["res_t4"] = (f"Error: {e}", {})
    with col_out:
        if "res_t4" in st.session_state:
            msg, data = st.session_state["res_t4"]
            if msg:
                st.code(msg, language=None)
            if data and "i" in data and "V" in data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                ax1.plot(data["i"], data["V"], color="steelblue", linewidth=2)
                ax1.set_ylabel("Cell Voltage (V)")
                ax1.set_title("Fuel Cell Polarization Curve")
                ax1.set_ylim(bottom=0)
                ax1.grid(True, alpha=0.3)
                ax1.spines[["top", "right"]].set_visible(False)
                if "P" in data:
                    ax2.plot(data["i"], data["P"], color="darkorange", linewidth=2)
                    ax2.set_xlabel("Current density i (A/cm²)")
                    ax2.set_ylabel("Power density (W/cm²)")
                    ax2.set_title("Power Density Curve")
                    ax2.set_ylim(bottom=0)
                    ax2.grid(True, alpha=0.3)
                    ax2.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Corrosion
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Corrosion Rate Analysis")
            i_corr = st.number_input("Corrosion current density i_corr (A/m²)", value=1e-2, step=1e-3, format="%.4e")
            M_molar_corr = st.number_input("Molar mass M (g/mol)", value=55.85, step=0.5)
            n_corr = st.number_input("Electrons per atom n", value=2, min_value=1, step=1)
            rho_corr = st.number_input("Density ρ (g/cm³)", value=7.87, step=0.05)
            area_corr = st.number_input("Exposed area (m²)", value=0.01, step=0.001, format="%.4f")
            submitted = st.form_submit_button("Calculate Corrosion Rate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_corrosion(i_corr, M_molar_corr, int(n_corr), rho_corr, area_corr)
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
            if data and "i_corr_range" in data and "corr_rate_mm_yr" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.loglog(data["i_corr_range"], data["corr_rate_mm_yr"], color="firebrick", linewidth=2)
                ax.axvline(i_corr, color="red", linestyle="--", alpha=0.7, label=f"i_corr = {i_corr:.2e}")
                ax.set_xlabel("Corrosion current density (A/m²)")
                ax.set_ylabel("Corrosion rate (mm/yr)")
                ax.set_title("Corrosion Rate vs Current Density")
                ax.legend()
                ax.grid(True, alpha=0.3, which="both")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
