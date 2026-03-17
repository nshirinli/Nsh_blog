import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.separation_controller import SeparationController

st.set_page_config(layout="wide", page_title="Separation - ChemEng")
st.title("🔬 Separation")

ctrl = SeparationController()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "McCabe-Thiele", "Kremser", "Flash Calculation",
    "Extraction", "Adsorption", "Membrane"
])

# Tab 1 - McCabe-Thiele
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("McCabe-Thiele")
            alpha = st.number_input("Relative volatility α", value=2.5, step=0.1)
            R = st.number_input("Reflux ratio R", value=1.5, step=0.1)
            xD = st.number_input("Distillate xD", value=0.95, min_value=0.01, max_value=0.999, step=0.01)
            xB = st.number_input("Bottoms xB", value=0.05, min_value=0.001, max_value=0.99, step=0.01)
            zF = st.number_input("Feed composition zF", value=0.5, min_value=0.001, max_value=0.999, step=0.01)
            q = st.number_input("Feed quality q", value=1.0, step=0.1)
            submitted = st.form_submit_button("Run McCabe-Thiele", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_mccabe_thiele(alpha, R, xD, xB, zF, q)
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
            if data and "x_eq" in data:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(data["x_eq"], data["y_eq"], label="Equilibrium curve", color="steelblue", linewidth=2)
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x")
                if "x_op" in data and "y_op" in data:
                    ax.plot(data["x_op"], data["y_op"], label="Operating line", color="darkorange", linewidth=1.5)
                if "x_steps" in data and "y_steps" in data:
                    ax.step(data["x_steps"], data["y_steps"], where="post", color="green", linewidth=1.5, label=f"Stages: {data.get('stages','?')}")
                ax.set_xlabel("x (liquid mole fraction)")
                ax.set_ylabel("y (vapor mole fraction)")
                ax.set_title("McCabe-Thiele Diagram")
                ax.legend(fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Kremser
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Kremser Equation")
            A_kr = st.number_input("Absorption factor A", value=1.4, step=0.1)
            m_kr = st.number_input("Equilibrium slope m", value=0.5, step=0.05)
            y_in = st.number_input("y_in (inlet gas)", value=0.02, step=0.001, format="%.4f")
            y_out = st.number_input("y_out (outlet gas)", value=0.001, step=0.0001, format="%.4f")
            x_in = st.number_input("x_in (inlet solvent)", value=0.0, step=0.001, format="%.4f")
            submitted = st.form_submit_button("Calculate Kremser", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_kremser(A_kr, m_kr, y_in, y_out, x_in)
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
            if data and "N" in data:
                N_val = data.get("N", 0)
                N_ceil = data.get("N_ceil", 0)
                absorb_eff = data.get("absorb_eff", 0) * 100
                y_in_v = data.get("y_in", y_in)
                y_out_v = data.get("y_out", y_out)
                x_out_v = data.get("x_out", 0)
                import math as _math
                if not _math.isinf(float(N_val if N_val else 0)):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bar_labels = ["N theoretical", "N actual (ceil)", "Absorption\nEff. (%)"]
                    bar_values = [float(N_val), float(N_ceil), absorb_eff]
                    colors = ["steelblue", "darkorange", "seagreen"]
                    bars = ax.bar(bar_labels, bar_values, color=colors, alpha=0.85, edgecolor="white")
                    for bar, val in zip(bars, bar_values):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bar_values) * 0.01,
                                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
                    ax.set_ylabel("Value")
                    ax.set_title("Kremser Equation – Absorption Results")
                    ax.grid(True, alpha=0.3, axis="y")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 3 - Flash Calculation
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Flash Calculation (Rachford-Rice)")
            z_text = st.text_input("Feed mole fractions z (comma-separated)", "0.3, 0.4, 0.3")
            k_text = st.text_input("K-values (comma-separated)", "5.0, 1.0, 0.2")
            submitted = st.form_submit_button("Calculate Flash", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_flash(z_text, k_text, None)
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
            if data and "x" in data and "y" in data:
                n_comp = len(data["x"])
                labels = data.get("labels") or [f"C{i+1}" for i in range(n_comp)]
                x_pos = np.arange(n_comp)
                fig, ax = plt.subplots(figsize=(8, 4))
                bar_width = 0.35
                ax.bar(x_pos - bar_width/2, data["x"], bar_width, label="Liquid x", color="steelblue", alpha=0.8)
                ax.bar(x_pos + bar_width/2, data["y"], bar_width, label="Vapor y", color="darkorange", alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels)
                ax.set_ylabel("Mole fraction")
                ax.set_title(f"Flash Calculation – V/F = {data.get('V_F', '?'):.3f}")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Extraction
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Liquid-Liquid Extraction")
            z_feed = st.number_input("Feed solute fraction z_feed", value=0.3, step=0.01)
            K_D = st.number_input("Distribution coefficient K_D", value=2.0, step=0.1)
            S_over_F = st.number_input("Solvent/Feed ratio S/F", value=1.0, step=0.1)
            n_stages = st.number_input("Number of stages", value=5, min_value=1, max_value=20, step=1)
            mode = st.selectbox("Mode", ["Countercurrent", "Crosscurrent"])
            submitted = st.form_submit_button("Run Extraction", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_extraction(z_feed, K_D, S_over_F, n_stages, mode)
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
            if data and "stage" in data and "x_R" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["stage"], data["x_R"], marker="o", color="steelblue", linewidth=2, label="Raffinate xR")
                if "x_E" in data:
                    ax.plot(data["stage"], data["x_E"], marker="s", color="darkorange", linewidth=2, label="Extract xE")
                ax.set_xlabel("Stage")
                ax.set_ylabel("Solute mole fraction")
                ax.set_title(f"{mode} Extraction")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Adsorption
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Adsorption Isotherm")
            ads_model = st.selectbox("Model", ["Langmuir", "Freundlich"])
            C_max = st.number_input("Max concentration C_max", value=10.0, step=0.5)
            KL = st.number_input("Langmuir KL", value=0.5, step=0.05)
            KF = st.number_input("Freundlich KF", value=1.0, step=0.1)
            n_f = st.number_input("Freundlich n", value=2.0, step=0.1)
            submitted = st.form_submit_button("Calculate Isotherm", use_container_width=True)
        if submitted:
            try:
                if ads_model == "Langmuir":
                    params = {"KL": KL}
                else:
                    params = {"KF": KF, "n": n_f}
                result = ctrl.run_adsorption(ads_model, C_max, params)
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
            if data and "C" in data and "q" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["C"], data["q"], color="teal", linewidth=2)
                ax.set_xlabel("Concentration C (mg/L or mol/L)")
                ax.set_ylabel("Loading q (mg/g)")
                ax.set_title(f"{ads_model} Adsorption Isotherm")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Membrane
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Membrane Separation")
            P_A = st.number_input("Permeability A (Barrer)", value=50.0, step=1.0)
            P_B = st.number_input("Permeability B (Barrer)", value=5.0, step=0.5)
            thickness_um = st.number_input("Membrane thickness (μm)", value=1.0, step=0.1)
            p_feed_bar = st.number_input("Feed pressure (bar)", value=10.0, step=0.5)
            p_perm_bar = st.number_input("Permeate pressure (bar)", value=1.0, step=0.1)
            z_A_feed = st.number_input("Feed mole fraction A", value=0.4, step=0.01)
            submitted = st.form_submit_button("Calculate Membrane", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_membrane(P_A, P_B, thickness_um, p_feed_bar, p_perm_bar, z_A_feed)
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
            if data and "y_A" in data and "theta" in data:
                import numpy as _np2
                theta_arr = _np2.asarray(data["theta"])
                y_A_arr = _np2.asarray(data["y_A"])
                y_B_arr = 1.0 - y_A_arr
                # show permeate composition vs stage cut
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.stackplot(theta_arr, y_A_arr, y_B_arr,
                             labels=["Permeate A", "Permeate B"],
                             colors=["steelblue", "lightcoral"], alpha=0.85)
                ax.axhline(z_A_feed, color="black", linestyle="--", alpha=0.6,
                           label=f"Feed z_A = {z_A_feed:.2f}")
                ax.set_xlabel("Stage cut θ (fraction permeated)")
                ax.set_ylabel("Permeate mole fraction")
                ax.set_title("Membrane Separation – Permeate Composition vs Stage Cut")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
