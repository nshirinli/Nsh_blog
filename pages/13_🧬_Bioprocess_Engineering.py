import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.bioprocess_controller import BioprocessController

st.set_page_config(layout="wide", page_title="Bioprocess Engineering - ChemEng")
st.title("🧬 Bioprocess Engineering")

ctrl = BioprocessController()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Growth Kinetics", "Batch Bioreactor", "Chemostat",
    "Oxygen Transfer", "Thermal Sterilization"
])

# Tab 1 - Growth Kinetics
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Microbial Growth Kinetics")
            model_gk = st.selectbox("Model", ["monod", "andrews"])
            mu_max = st.number_input("μ_max (1/h)", value=0.9, step=0.05)
            Ks = st.number_input("Ks (g/L)", value=0.2, step=0.01)
            Ki = st.number_input("Ki (g/L, Andrews only)", value=0.5, step=0.05)
            S_max = st.number_input("Max substrate S_max (g/L)", value=10.0, step=0.5)
            submitted = st.form_submit_button("Plot Kinetics", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_growth_kinetics(mu_max, Ks, model_gk, Ki, S_max)
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
            if data and "S" in data and "mu_monod" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["S"], data["mu_monod"], color="steelblue", linewidth=2, label="Monod")
                if "mu_andrews" in data:
                    ax.plot(data["S"], data["mu_andrews"], color="darkorange", linewidth=2, linestyle="--", label="Andrews")
                ax.set_xlabel("Substrate S (g/L)")
                ax.set_ylabel("Specific growth rate μ (1/h)")
                ax.set_title("Microbial Growth Kinetics")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Batch Bioreactor
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Batch Bioreactor Simulation")
            S0 = st.number_input("Initial substrate S0 (g/L)", value=10.0, step=0.5)
            X0 = st.number_input("Initial biomass X0 (g/L)", value=0.1, step=0.01)
            P0 = st.number_input("Initial product P0 (g/L)", value=0.0, step=0.1)
            mu_max_b = st.number_input("μ_max (1/h)", value=0.5, step=0.05)
            Ks_b = st.number_input("Ks (g/L)", value=0.2, step=0.01)
            Yxs = st.number_input("Biomass yield Yxs (g/g)", value=0.5, step=0.05)
            Yps = st.number_input("Product yield Yps (g/g)", value=0.3, step=0.05)
            ms = st.number_input("Maintenance coefficient ms (g/g·h)", value=0.02, step=0.005, format="%.4f")
            t_end = st.number_input("End time (h)", value=20.0, step=1.0)
            submitted = st.form_submit_button("Simulate Batch", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_batch(S0, X0, P0, mu_max_b, Ks_b, Yxs, Yps, ms, t_end)
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
            if data and "t" in data and "S" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["S"], label="Substrate S", color="steelblue", linewidth=2)
                if "X" in data:
                    ax.plot(data["t"], data["X"], label="Biomass X", color="darkorange", linewidth=2)
                if "P" in data:
                    ax.plot(data["t"], data["P"], label="Product P", color="green", linewidth=2)
                ax.set_xlabel("Time (h)")
                ax.set_ylabel("Concentration (g/L)")
                ax.set_title("Batch Bioreactor Simulation")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Chemostat
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Chemostat Steady-State Analysis")
            Sin = st.number_input("Substrate in feed Sin (g/L)", value=10.0, step=0.5)
            mu_max_c = st.number_input("μ_max (1/h)", value=0.9, step=0.05)
            Ks_c = st.number_input("Ks (g/L)", value=0.2, step=0.01)
            Yxs_c = st.number_input("Biomass yield Yxs (g/g)", value=0.5, step=0.05)
            Yps_c = st.number_input("Product yield Yps (g/g)", value=0.3, step=0.05)
            D_op = st.number_input("Operating dilution rate D (1/h)", value=0.3, step=0.05)
            submitted = st.form_submit_button("Analyze Chemostat", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_chemostat(Sin, mu_max_c, Ks_c, Yxs_c, Yps_c, D_op)
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
            if data and "D_range" in data and "X_steady" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["D_range"], data["X_steady"], label="Biomass X*", color="steelblue", linewidth=2)
                if "S_steady" in data:
                    ax.plot(data["D_range"], data["S_steady"], label="Substrate S*", color="darkorange", linewidth=2, linestyle="--")
                ax.axvline(D_op, color="gray", linestyle=":", alpha=0.7, label=f"D = {D_op}")
                ax.set_xlabel("Dilution rate D (1/h)")
                ax.set_ylabel("Steady-state concentration (g/L)")
                ax.set_title("Chemostat – Steady-State vs Dilution Rate")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Oxygen Transfer
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Oxygen Transfer Rate")
            OUR = st.number_input("OUR – Oxygen uptake rate (g/L·h)", value=0.05, step=0.005, format="%.4f")
            C_star = st.number_input("Saturation concentration C* (mg/L)", value=8.0, step=0.1)
            C_L = st.number_input("Dissolved O2 C_L (mg/L)", value=2.0, step=0.1)
            V = st.number_input("Reactor volume V (L)", value=1000.0, step=10.0)
            submitted = st.form_submit_button("Calculate kLa", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_oxygen_transfer(OUR, C_star, C_L, V)
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
            if data and "KLa" in data and "OUR_mmol_L_h" in data:
                kla_val = data["KLa"]
                our_val = data["OUR_mmol_L_h"]
                otr_val = kla_val * (data.get("C_star", C_star) - data.get("C_L", C_L))
                labels_ot = ["OUR\n(mmol/L·h)", "kLa\n(1/h)", "OTR\n(mmol/L·h)"]
                values_ot = [our_val, kla_val, otr_val]
                colors_ot = ["steelblue", "darkorange", "seagreen"]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels_ot, values_ot, color=colors_ot, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, values_ot):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(values_ot) * 0.01,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Value")
                ax.set_title("Oxygen Transfer – OUR, kLa, OTR")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Thermal Sterilization
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Thermal Sterilization")
            T_steril = st.number_input("Sterilization temperature (°C)", value=121.0, step=1.0)
            D_121 = st.number_input("D-value at 121°C (min)", value=2.5, step=0.1)
            z = st.number_input("z-value (°C)", value=10.0, step=0.5)
            N0 = st.number_input("Initial organism count N0", value=1e12, step=1e11, format="%.2e")
            N_target = st.number_input("Target N (probability of failure)", value=0.001, step=0.0001, format="%.4f")
            t_hold = st.number_input("Hold time (min)", value=30.0, step=1.0)
            submitted = st.form_submit_button("Calculate Sterilization", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_sterilization(T_steril, D_121, z, N0, N_target, t_hold)
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
            if data and "D_at_T" in data:
                D_val = data["D_at_T"]
                if D_val and D_val > 0:
                    t_arr = np.linspace(0, max(t_hold * 1.5, data.get("t_required_min", t_hold) * 1.2), 300)
                    N_arr = N0 * np.power(10.0, -t_arr / D_val)
                    N_arr = np.clip(N_arr, 1e-20, None)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.semilogy(t_arr, N_arr, color="steelblue", linewidth=2, label="N(t)")
                    ax.axhline(N_target, color="red", linestyle="--", linewidth=1.5,
                               label=f"Target N = {N_target:.3e}")
                    ax.axvline(t_hold, color="darkorange", linestyle=":", linewidth=1.5,
                               label=f"Hold time = {t_hold:.1f} min")
                    ax.set_xlabel("Time (min)")
                    ax.set_ylabel("Organism count N (log scale)")
                    ax.set_title(f"Thermal Sterilization at {T_steril:.1f}°C")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, which="both")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
