import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.safety_controller import SafetyController

st.set_page_config(layout="wide", page_title="Safety & Risk - ChemEng")
st.title("🦺 Safety & Risk")

ctrl = SafetyController()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Toxic Dispersion", "Vapor Cloud Explosion", "Pool Fire",
    "Risk Matrix", "LOPA", "Flammability Limits"
])

# Tab 1 - Toxic Dispersion
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Gaussian Dispersion Model")
            Q_disp = st.number_input("Source strength Q (kg/s)", value=1.0, step=0.1)
            u_disp = st.number_input("Wind speed u (m/s)", value=3.0, step=0.5)
            H_disp = st.number_input("Release height H (m)", value=10.0, step=1.0)
            stability = st.selectbox("Pasquill stability class", ["A", "B", "C", "D", "E", "F"])
            x_max = st.number_input("Max downwind distance x_max (m)", value=2000.0, step=100.0)
            submitted = st.form_submit_button("Calculate Dispersion", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_dispersion(Q_disp, u_disp, H_disp, stability, x_max)
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
            if data and "x" in data and "C_ground" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["x"], data["C_ground"], color="steelblue", linewidth=2, label="Ground-level")
                if "C_center" in data:
                    ax.plot(data["x"], data["C_center"], color="darkorange", linewidth=2, linestyle="--", label="Plume center")
                ax.set_xlabel("Downwind distance x (m)")
                ax.set_ylabel("Concentration (kg/m³)")
                ax.set_title(f"Toxic Dispersion – Stability {stability}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Vapor Cloud Explosion
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Vapor Cloud Explosion – TNT Equivalence")
            m_fuel = st.number_input("Fuel mass (kg)", value=1000.0, step=100.0)
            dHc = st.number_input("Heat of combustion ΔHc (kJ/kg)", value=46000.0, step=500.0)
            alpha = st.number_input("Explosion efficiency α", value=0.03, step=0.005, format="%.4f")
            r_max = st.number_input("Max radius r_max (m)", value=500.0, step=50.0)
            submitted = st.form_submit_button("Calculate Overpressure", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_explosion(m_fuel, dHc, alpha, r_max)
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
            if data and "r" in data and "overpressure" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.semilogy(data["r"], data["overpressure"], color="firebrick", linewidth=2)
                ax.set_xlabel("Radial distance r (m)")
                ax.set_ylabel("Overpressure (kPa, log scale)")
                ax.set_title("VCE Overpressure vs Distance")
                ax.grid(True, alpha=0.3, which="both")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Pool Fire
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Pool Fire Radiation")
            diameter = st.number_input("Pool diameter (m)", value=10.0, step=0.5)
            m_dot = st.number_input("Burning rate m_dot (kg/m²·s)", value=0.05, step=0.005, format="%.4f")
            dHc_pf = st.number_input("Heat of combustion ΔHc (kJ/kg)", value=46000.0, step=500.0)
            eta = st.number_input("Radiative fraction η", value=0.2, step=0.01)
            r_max_pf = st.number_input("Max radius r_max (m)", value=200.0, step=10.0)
            submitted = st.form_submit_button("Calculate Radiation", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_pool_fire(diameter, m_dot, dHc_pf, eta, r_max_pf)
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
            if data and "r" in data and "flux" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["r"], data["flux"], color="darkorange", linewidth=2)
                ax.set_xlabel("Distance from flame edge r (m)")
                ax.set_ylabel("Thermal flux (kW/m²)")
                ax.set_title("Pool Fire – Thermal Radiation")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Risk Matrix
with tab4:
    st.subheader("Risk Matrix (5×5)")
    try:
        risk_data = ctrl.get_risk_matrix_data()
    except Exception:
        risk_data = {}

    likelihood_labels = ["Rare (1)", "Unlikely (2)", "Possible (3)", "Likely (4)", "Almost Certain (5)"]
    consequence_labels = ["Negligible (1)", "Minor (2)", "Moderate (3)", "Major (4)", "Catastrophic (5)"]

    def risk_color(score):
        if score < 4:
            return "#4caf50"   # green
        elif score < 8:
            return "#ffeb3b"   # yellow
        elif score < 15:
            return "#ff9800"   # orange
        else:
            return "#f44336"   # red

    html = "<table style='border-collapse:collapse; width:100%;'>"
    html += "<tr><th style='padding:6px; border:1px solid #ddd;'>L \\ C</th>"
    for cl in consequence_labels:
        html += f"<th style='padding:6px; border:1px solid #ddd; font-size:11px;'>{cl}</th>"
    html += "</tr>"

    for li in range(1, 6):
        html += f"<tr><td style='padding:6px; border:1px solid #ddd; font-weight:bold; font-size:11px;'>{likelihood_labels[li-1]}</td>"
        for ci in range(1, 6):
            score = li * ci
            color = risk_color(score)
            html += f"<td style='padding:8px; border:1px solid #ddd; background:{color}; text-align:center; font-weight:bold;'>{score}</td>"
        html += "</tr>"
    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Legend:** Green < 4 | Yellow 4–7 | Orange 8–14 | Red ≥ 15")

# Tab 5 - LOPA
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Layer of Protection Analysis (LOPA)")
            ie_desc = st.text_input("Initiating event description", "Initiating event")
            ie_freq = st.number_input("Initiating event frequency (per year)", value=1.0, step=0.1)
            ipls_text = st.text_area(
                "IPLs (name, PFD per line)",
                "BPCS, 0.1\nSIS, 0.01",
                help="Format: Layer Name, PFD"
            )
            target_freq = st.number_input("Target mitigated frequency (per year)", value=1e-4, step=1e-5, format="%.2e")
            submitted = st.form_submit_button("Run LOPA", use_container_width=True)
        if submitted:
            try:
                ipls = []
                for line in ipls_text.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        ipls.append({"name": parts[0], "pfd": float(parts[1])})
                result = ctrl.run_lopa(ie_desc, ie_freq, ipls, target_freq)
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
            if data and "ipl_names" in data and "frequencies" in data:
                names = data["ipl_names"]
                freqs = data["frequencies"]
                fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.6)))
                x_pos = np.arange(len(names))
                colors = ["steelblue" if i == 0 else "darkorange" if i < len(names) - 1 else "green"
                          for i in range(len(names))]
                ax.barh(x_pos, freqs, color=colors, alpha=0.8, edgecolor="white")
                if target_freq:
                    ax.axvline(target_freq, color="red", linestyle="--", alpha=0.7, label=f"Target: {target_freq:.1e}")
                ax.set_yticks(x_pos)
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel("Frequency (per year)")
                ax.set_xscale("log")
                ax.set_title("LOPA – Frequency Waterfall")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="x")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Flammability Limits
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Flammability Limits (Le Chatelier)")
            comp_names_fl = st.text_input("Component names (comma-separated)", "CH4, C3H8")
            mole_fracs_fl = st.text_input("Mole fractions (comma-separated)", "0.7, 0.3")
            lfl_vals = st.text_input("LFL values (comma-separated, vol fraction)", "0.05, 0.021")
            ufl_vals = st.text_input("UFL values (comma-separated, vol fraction)", "0.15, 0.095")
            Tb_K = st.number_input("Boiling point Tb (K, single comp or avg)", value=111.7, step=1.0)
            submitted = st.form_submit_button("Calculate Flammability", use_container_width=True)
        if submitted:
            try:
                components = [c.strip() for c in comp_names_fl.split(",")]
                mole_fracs = [float(x.strip()) for x in mole_fracs_fl.split(",")]
                lfl_list = [float(x.strip()) for x in lfl_vals.split(",")]
                ufl_list = [float(x.strip()) for x in ufl_vals.split(",")]
                result = ctrl.run_flammability(components, mole_fracs, lfl_list, ufl_list, Tb_K)
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
            result_r = data.get("result", {}) if data else {}
            if result_r and "components" in result_r and "LFL_vals" in result_r and "UFL_vals" in result_r:
                comps_fl = result_r["components"]
                lfl_r = np.asarray(result_r["LFL_vals"])
                ufl_r = np.asarray(result_r["UFL_vals"])
                lfl_mix = result_r.get("LFLmix", None)
                ufl_mix = result_r.get("UFLmix", None)
                y_pos = np.arange(len(comps_fl))
                fig, ax = plt.subplots(figsize=(8, max(4, len(comps_fl) * 0.8 + 1)))
                ax.barh(y_pos, lfl_r * 100, left=0, height=0.35, color="steelblue",
                        alpha=0.85, label="LFL")
                ax.barh(y_pos + 0.4, ufl_r * 100, left=0, height=0.35, color="firebrick",
                        alpha=0.85, label="UFL")
                if lfl_mix is not None:
                    ax.axvline(lfl_mix * 100, color="steelblue", linestyle="--", linewidth=1.5,
                               label=f"LFL_mix={lfl_mix*100:.2f}%")
                if ufl_mix is not None:
                    ax.axvline(ufl_mix * 100, color="firebrick", linestyle="--", linewidth=1.5,
                               label=f"UFL_mix={ufl_mix*100:.2f}%")
                ax.set_yticks(y_pos + 0.2)
                ax.set_yticklabels(comps_fl)
                ax.set_xlabel("Flammability limit (vol %)")
                ax.set_title("Flammability Limits – LFL and UFL per Component")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="x")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
