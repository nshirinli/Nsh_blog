import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.reaction_controller import ReactionController

st.set_page_config(layout="wide", page_title="Reaction Modeling - ChemEng")
st.title("⚗️ Reaction Modeling")

ctrl = ReactionController()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Ideal Reactors", "Arrhenius", "Series/Parallel",
    "Reactor Sizing", "Equilibrium", "Non-Isothermal", "RTD Analysis"
])

# Tab 1 - Ideal Reactors
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Ideal Reactor")
            reactor_type = st.selectbox("Reactor Type", ["Batch", "CSTR", "PFR"])
            n = st.number_input("Reaction order (n)", value=1.0, step=0.5)
            k = st.number_input("Rate constant k", value=0.5, step=0.05)
            ca0 = st.number_input("Initial conc. Ca0 (mol/L)", value=1.0, step=0.1)
            t_or_v = st.number_input("Time (Batch) or Volume (CSTR/PFR)", value=10.0, step=0.5)
            v0 = st.number_input("Volumetric flow v0 (L/s)", value=1.0, step=0.1)
            submitted = st.form_submit_button("Run Simulation", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_ideal_reactor(reactor_type, n, k, ca0, t_or_v, v0)
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
            if data and "t" in data and "ca" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["ca"], color="steelblue", linewidth=2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Concentration Ca (mol/L)")
                ax.set_title(f"{reactor_type} – Concentration Profile")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Arrhenius
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Arrhenius Equation")
            A_arr = st.number_input("Pre-exponential factor A", value=1e8, step=1e7, format="%.2e")
            Ea_J = st.number_input("Activation energy Ea (J/mol)", value=50000.0, step=100.0)
            T_calc = st.number_input("T for k calculation (K)", value=300.0, step=1.0)
            T_min_arr = st.number_input("T min (K)", value=200.0, step=10.0)
            T_max_arr = st.number_input("T max (K)", value=600.0, step=10.0)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_arrhenius(A_arr, Ea_J, T_calc, T_min_arr, T_max_arr)
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
            if data and "T" in data and "k" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.semilogy(data["T"], data["k"], color="darkorange", linewidth=2)
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel("Rate constant k (log scale)")
                ax.set_title("Arrhenius: k vs Temperature")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Series/Parallel
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Series/Parallel Reactions")
            reaction_type = st.selectbox("Reaction type", ["series", "parallel"])
            k1 = st.number_input("k1", value=0.5, step=0.05)
            k2 = st.number_input("k2", value=0.2, step=0.05)
            ca0_sp = st.number_input("Ca0 (mol/L)", value=1.0, step=0.1)
            t_final_sp = st.number_input("Final time (s)", value=10.0, step=0.5)
            cb0 = st.number_input("Cb0 (mol/L)", value=0.0, step=0.1)
            cc0 = st.number_input("Cc0 (mol/L)", value=0.0, step=0.1)
            submitted = st.form_submit_button("Simulate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_series_parallel(reaction_type, k1, k2, ca0_sp, t_final_sp, cb0, cc0)
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
            if data and "t" in data and "ca" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["ca"], label="Ca", linewidth=2)
                if "cb" in data:
                    ax.plot(data["t"], data["cb"], label="Cb", linewidth=2)
                if "cc" in data:
                    ax.plot(data["t"], data["cc"], label="Cc", linewidth=2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Concentration (mol/L)")
                ax.set_title(f"{reaction_type.capitalize()} Reactions")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Reactor Sizing
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Reactor Sizing (Levenspiel)")
            k_sz = st.number_input("Rate constant k", value=0.5, step=0.05)
            n_sz = st.number_input("Reaction order n", value=1.0, step=0.5)
            ca0_sz = st.number_input("Ca0 (mol/L)", value=1.0, step=0.1)
            v0_sz = st.number_input("Volumetric flow v0 (L/s)", value=1.0, step=0.1)
            x_target = st.number_input("Target conversion X", value=0.9, min_value=0.01, max_value=0.999, step=0.01)
            submitted = st.form_submit_button("Size Reactors", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_reactor_sizing(k_sz, n_sz, ca0_sz, v0_sz, x_target)
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
            if data and "x" in data and "V_cstr" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["x"], data["V_cstr"], label="CSTR", color="steelblue", linewidth=2)
                ax.plot(data["x"], data["V_pfr"], label="PFR", color="darkorange", linewidth=2, linestyle="--")
                ax.set_xlabel("Conversion X")
                ax.set_ylabel("Volume (L)")
                ax.set_title("Levenspiel Plot – Reactor Volume vs Conversion")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Equilibrium
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Chemical Equilibrium")
            dH_kJ = st.number_input("ΔH_rxn (kJ/mol)", value=-92.4, step=0.1)
            dG_kJ = st.number_input("ΔG_rxn at T_ref (kJ/mol)", value=-33.3, step=0.1)
            stoich_reac_str = st.text_input("Stoich (reactants)", "1, 3")
            stoich_prod_str = st.text_input("Stoich (products)", "2")
            n_init_str = st.text_input("Initial moles (n_init)", "1, 3, 0")
            P_bar = st.number_input("Pressure (bar)", value=1.0, step=0.1)
            T_ref_K = st.number_input("T_ref (K)", value=298.0, step=1.0)
            T_min_eq = st.number_input("T min (K)", value=200.0, step=10.0)
            T_max_eq = st.number_input("T max (K)", value=800.0, step=10.0)
            submitted = st.form_submit_button("Calculate Equilibrium", use_container_width=True)
        if submitted:
            try:
                stoich_reac = [float(x.strip()) for x in stoich_reac_str.split(",")]
                stoich_prod = [float(x.strip()) for x in stoich_prod_str.split(",")]
                n_init = [float(x.strip()) for x in n_init_str.split(",")]
                result = ctrl.run_equilibrium(dH_kJ, dG_kJ, stoich_reac, stoich_prod, n_init, P_bar, T_ref_K, T_min_eq, T_max_eq)
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
            if data and "T" in data and "Keq" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.semilogy(data["T"], data["Keq"], color="purple", linewidth=2)
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel("Equilibrium constant Keq (log scale)")
                ax.set_title("Keq vs Temperature")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Non-Isothermal
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Non-Isothermal Reactor")
            reactor_ni = st.selectbox("Reactor Type", ["Batch", "CSTR", "PFR"])
            k0_ni = st.number_input("Pre-exp factor k0", value=1e6, step=1e5, format="%.2e")
            Ea_kJ = st.number_input("Ea (kJ/mol)", value=50.0, step=0.5)
            n_ni = st.number_input("Reaction order n", value=1.0, step=0.5)
            ca0_ni = st.number_input("Ca0 (mol/L)", value=1.0, step=0.1)
            T0_C = st.number_input("Initial temperature T0 (°C)", value=25.0, step=1.0)
            v0_ni = st.number_input("Flow v0 (L/s)", value=1.0, step=0.1)
            dH_kJ_ni = st.number_input("ΔH_rxn (kJ/mol)", value=-50.0, step=1.0)
            Cp_J = st.number_input("Cp (J/mol·K)", value=4184.0, step=10.0)
            rho_mol = st.number_input("ρ (mol/L)", value=55.5, step=0.5)
            UA_ni = st.number_input("UA (W/K)", value=0.0, step=0.5)
            Tc_C = st.number_input("Coolant Tc (°C)", value=25.0, step=1.0)
            t_or_V_ni = st.number_input("Final time or volume", value=20.0, step=0.5)
            submitted = st.form_submit_button("Simulate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_nonisothermal(reactor_ni, k0_ni, Ea_kJ, n_ni, ca0_ni, T0_C, v0_ni, dH_kJ_ni, Cp_J, rho_mol, UA_ni, Tc_C, t_or_V_ni)
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
            if data and "t_or_V" in data and "T" in data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                ax1.plot(data["t_or_V"], data["T"], color="firebrick", linewidth=2)
                ax1.set_xlabel("Time / Volume")
                ax1.set_ylabel("Temperature (°C)")
                ax1.set_title("Temperature Profile")
                ax1.grid(True, alpha=0.3)
                ax1.spines[["top", "right"]].set_visible(False)
                if "ca" in data:
                    ax2.plot(data["t_or_V"], data["ca"], color="steelblue", linewidth=2)
                    ax2.set_xlabel("Time / Volume")
                    ax2.set_ylabel("Concentration (mol/L)")
                    ax2.set_title("Concentration Profile")
                    ax2.grid(True, alpha=0.3)
                    ax2.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 7 - RTD
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t7"):
            st.subheader("RTD Analysis")
            rtd_model = st.selectbox("Model", ["CSTR", "PFR", "Tanks-in-Series", "Dispersion"])
            tau_rtd = st.number_input("Mean residence time τ (s)", value=5.0, step=0.5)
            N_or_Pe = st.number_input("N (TIS) or Pe (Dispersion)", value=5.0, step=0.5)
            t_final_rtd = st.number_input("Final time (s)", value=25.0, step=0.5)
            submitted = st.form_submit_button("Calculate RTD", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_rtd(rtd_model, tau_rtd, N_or_Pe, t_final_rtd)
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
            if data and "t" in data and "E" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["E"], color="teal", linewidth=2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("E(t) – RTD function")
                ax.set_title(f"RTD – {rtd_model} Model")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
