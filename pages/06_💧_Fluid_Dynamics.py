import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.fluid_dynamics_controller import FluidDynamicsController

st.set_page_config(layout="wide", page_title="Fluid Dynamics - ChemEng")
st.title("💧 Fluid Dynamics")

ctrl = FluidDynamicsController()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Pipe Flow / Moody", "Bernoulli", "Orifice Meter",
    "Pump Sizing", "Isentropic Flow", "Normal Shock"
])

# Tab 1 - Pipe Flow / Moody
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Pipe Flow & Moody Chart")
            rho = st.number_input("Density ρ (kg/m³)", value=1000.0, step=1.0)
            v = st.number_input("Velocity v (m/s)", value=1.0, step=0.1)
            D = st.number_input("Diameter D (m)", value=0.05, step=0.005, format="%.4f")
            mu = st.number_input("Dynamic viscosity μ (Pa·s)", value=0.001, step=0.0001, format="%.5f")
            L = st.number_input("Pipe length L (m)", value=100.0, step=1.0)
            eps = st.number_input("Roughness ε (m)", value=0.0, step=0.00001, format="%.6f",
                                  help="0 for smooth pipe")
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_pipe_flow(rho, v, D, mu, L, eps)
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
            if data and "Re_array" in data and "f_array" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.loglog(data["Re_array"], data["f_array"], color="steelblue", linewidth=2)
                if "Re" in data:
                    ax.axvline(data["Re"], color="red", linestyle="--", alpha=0.7, label=f"Re = {data['Re']:.0f}")
                ax.set_xlabel("Reynolds Number Re")
                ax.set_ylabel("Friction factor f")
                ax.set_title("Moody Chart")
                ax.legend()
                ax.grid(True, alpha=0.3, which="both")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Bernoulli
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Bernoulli Equation")
            P1 = st.number_input("P1 (Pa)", value=200000.0, step=1000.0)
            v1 = st.number_input("v1 (m/s)", value=1.0, step=0.1)
            z1 = st.number_input("z1 (m)", value=5.0, step=0.1)
            P2 = st.number_input("P2 (Pa)", value=100000.0, step=1000.0)
            v2 = st.number_input("v2 (m/s)", value=2.0, step=0.1)
            z2 = st.number_input("z2 (m)", value=0.0, step=0.1)
            rho_b = st.number_input("Density ρ (kg/m³)", value=1000.0, step=1.0)
            submitted = st.form_submit_button("Check Bernoulli", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_bernoulli(P1, v1, z1, P2, v2, z2, rho_b)
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
            if data and "H1" in data and "H2" in data:
                g = 9.81
                # energy components in metres of head
                pts = ["Point 1", "Point 2"]
                P_head = [P1 / (rho_b * g), P2 / (rho_b * g)]
                v_head = [v1**2 / (2 * g), v2**2 / (2 * g)]
                z_head = [z1, z2]
                x = np.arange(2)
                width = 0.5
                fig, ax = plt.subplots(figsize=(8, 4))
                b1 = ax.bar(x, P_head, width, label="Pressure head P/ρg", color="steelblue", alpha=0.85)
                b2 = ax.bar(x, v_head, width, bottom=P_head, label="Velocity head v²/2g", color="darkorange", alpha=0.85)
                b3 = ax.bar(x, z_head, width, bottom=[P_head[i] + v_head[i] for i in range(2)],
                            label="Elevation head z", color="seagreen", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(pts)
                ax.set_ylabel("Head (m)")
                ax.set_title("Bernoulli – Energy Components at Each Point")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Orifice Meter
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Orifice Meter")
            Cd = st.number_input("Discharge coefficient Cd", value=0.61, step=0.01)
            D_orifice = st.number_input("Orifice diameter D (m)", value=0.025, step=0.001, format="%.4f")
            dP = st.number_input("Pressure drop ΔP (Pa)", value=10000.0, step=100.0)
            rho_or = st.number_input("Density ρ (kg/m³)", value=1000.0, step=1.0)
            submitted = st.form_submit_button("Calculate Flow Rate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_orifice(Cd, D_orifice, dP, rho_or)
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
            if data and "Q_m3s" in data:
                # Q vs ΔP sweep
                dP_arr = np.linspace(max(dP * 0.1, 100), dP * 3, 200)
                A_or = 3.14159265 / 4 * D_orifice ** 2
                Q_arr = Cd * A_or * np.sqrt(2 * dP_arr / rho_or)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dP_arr / 1000, Q_arr * 1000, color="teal", linewidth=2)
                ax.axvline(dP / 1000, color="red", linestyle="--", alpha=0.8,
                           label=f"ΔP={dP/1000:.2f} kPa  →  Q={data['Q_m3s']*1000:.4f} L/s")
                ax.set_xlabel("Pressure drop ΔP (kPa)")
                ax.set_ylabel("Flow rate Q (L/s)")
                ax.set_title("Orifice Meter – Q vs ΔP")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Pump Sizing
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Pump Sizing")
            Q_pump = st.number_input("Flow rate Q (m³/s)", value=0.01, step=0.001, format="%.4f")
            rho_pump = st.number_input("Density ρ (kg/m³)", value=1000.0, step=1.0)
            H_pump = st.number_input("Design head H (m)", value=20.0, step=0.5)
            eta_pump = st.number_input("Pump efficiency η", value=0.75, min_value=0.1, max_value=1.0, step=0.05)
            H_shutoff = st.number_input("Shutoff head (m)", value=30.0, step=0.5)
            Q_BEP = st.number_input("BEP flow Q_BEP (m³/s)", value=0.015, step=0.001, format="%.4f")
            H_BEP = st.number_input("BEP head H_BEP (m)", value=18.0, step=0.5)
            H_static = st.number_input("Static head H_static (m)", value=5.0, step=0.5)
            K_friction = st.number_input("Friction coeff K (s²/m⁵)", value=50000.0, step=1000.0)
            submitted = st.form_submit_button("Size Pump", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_pump_sizing(Q_pump, rho_pump, H_pump, eta_pump, H_shutoff, Q_BEP, H_BEP, H_static, K_friction)
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
            if data and "Q_curve" in data and "H_curve" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["Q_curve"], data["H_curve"], label="Pump Curve", color="steelblue", linewidth=2)
                if "H_sys" in data:
                    ax.plot(data["Q_curve"], data["H_sys"], label="System Curve", color="darkorange", linewidth=2, linestyle="--")
                ax.set_xlabel("Flow Rate Q (m³/s)")
                ax.set_ylabel("Head H (m)")
                ax.set_title("Pump & System Curve")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Isentropic
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Isentropic Flow Relations")
            M_iso = st.number_input("Mach number M", value=2.0, step=0.1)
            gamma_iso = st.number_input("Specific heat ratio γ", value=1.4, step=0.05)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_isentropic(M_iso, gamma_iso)
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
            if data and "T0_T_calc" in data and "P0_P_calc" in data:
                ratio_labels = ["T₀/T", "P₀/P", "ρ₀/ρ", "A/A*"]
                ratio_values = [
                    data.get("T0_T_calc", 0),
                    data.get("P0_P_calc", 0),
                    data.get("rho0_rho_calc", data.get("P0_P_calc", 0) / max(data.get("T0_T_calc", 1), 1e-9)),
                    data.get("A_Astar_calc", 0),
                ]
                colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(ratio_labels, ratio_values, color=colors, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, ratio_values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(ratio_values) * 0.01,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
                ax.set_ylabel("Ratio value")
                ax.set_title(f"Isentropic Flow Ratios at M = {M_iso:.2f}  (γ = {gamma_iso:.2f})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Normal Shock
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Normal Shock Relations")
            M1_ns = st.number_input("Upstream Mach M1", value=2.5, step=0.1)
            gamma_ns = st.number_input("Specific heat ratio γ", value=1.4, step=0.05)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_normal_shock(M1_ns, gamma_ns)
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
            if data and "M1_array" in data and "M2_array" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["M1_array"], data["M2_array"], color="purple", linewidth=2)
                ax.set_xlabel("Upstream Mach M1")
                ax.set_ylabel("Downstream Mach M2")
                ax.set_title("Normal Shock: M2 vs M1")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
