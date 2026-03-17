import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.mass_energy_controller import MassEnergyController

st.set_page_config(layout="wide", page_title="Mass & Energy Balances - ChemEng")
st.title("⚖️ Mass & Energy Balances")

ctrl = MassEnergyController()

def _default(key, fn):
    if key not in st.session_state:
        try:
            r = fn(); st.session_state[key] = r if isinstance(r, tuple) else (r.get("message",""), r)
        except Exception: pass

_default("res_t1", lambda: ctrl.run_stream(298.15,101325.0,100.0,["A","B","C"],[0.33,0.34,0.33],[28,32,44]))
_default("res_t2", lambda: ctrl.run_mixer(100.0,350.0,4.18,50.0,300.0,2.1))
_default("res_t3", lambda: ctrl.run_splitter(100.0,[0.4,0.6],["A","B"],[0.7,0.3]))
_default("res_t4", lambda: ctrl.run_material_balance(100.0,[0.4,0.3,0.3],["A","B","C"],0,[-1.0,-1.0,2.0],0.85))
_default("res_t6", lambda: ctrl.run_recycle(100.0,0.95,0.1))

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Stream Properties", "Adiabatic Mixer", "Splitter",
    "Material Balance", "Energy Balance", "Recycle Loop", "Composition Converter"
])

# Tab 1 - Stream Properties
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Stream Properties")
            T_s = st.number_input("Temperature (K)", value=298.15, step=1.0)
            P_s = st.number_input("Pressure (Pa)", value=101325.0, step=100.0)
            F_total = st.number_input("Total molar flow (mol/s)", value=100.0, step=1.0)
            comp_names_s = st.text_input("Component names (comma-separated)", "A, B, C")
            mole_fracs_s = st.text_input("Mole fractions (comma-separated)", "0.33, 0.34, 0.33")
            mol_weights_s = st.text_input("Molecular weights g/mol (comma-separated)", "28, 32, 44")
            submitted = st.form_submit_button("Calculate Stream", use_container_width=True)
        if submitted:
            try:
                components = [c.strip() for c in comp_names_s.split(",")]
                mole_fracs = [float(x.strip()) for x in mole_fracs_s.split(",")]
                mol_weights = [float(x.strip()) for x in mol_weights_s.split(",")]
                result = ctrl.run_stream(T_s, P_s, F_total, components, mole_fracs, mol_weights)
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
            result_r = data.get("result", {}) if data else {}
            if result_r and "mole_fracs" in result_r and "components" in result_r:
                comps = result_r["components"]
                mf = result_r["mole_fracs"]
                x_pos = np.arange(len(comps))
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(comps, mf, color="steelblue", alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, mf):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Mole fraction")
                ax.set_title("Stream Properties – Component Mole Fractions")
                ax.set_ylim(0, max(mf) * 1.15)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Adiabatic Mixer
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Adiabatic Mixer")
            F1 = st.number_input("Stream 1 flow F1 (mol/s)", value=100.0, step=1.0)
            T1_m = st.number_input("Stream 1 temperature T1 (K)", value=350.0, step=1.0)
            Cp1 = st.number_input("Stream 1 Cp (J/mol·K)", value=4.18, step=0.1)
            F2 = st.number_input("Stream 2 flow F2 (mol/s)", value=50.0, step=1.0)
            T2_m = st.number_input("Stream 2 temperature T2 (K)", value=300.0, step=1.0)
            Cp2 = st.number_input("Stream 2 Cp (J/mol·K)", value=2.1, step=0.1)
            submitted = st.form_submit_button("Calculate Mixer", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_mixer(F1, T1_m, Cp1, F2, T2_m, Cp2)
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
            result_r = data.get("result", {}) if data else {}
            if result_r and "T_out_K" in result_r:
                T_out = result_r["T_out_K"]
                labels = ["Stream 1\nT1", "Stream 2\nT2", "Outlet\nT_mix"]
                values = [T1_m, T2_m, T_out]
                colors = ["steelblue", "darkorange", "seagreen"]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                            f"{val:.2f} K\n({val-273.15:.1f}°C)", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Temperature (K)")
                ax.set_title("Adiabatic Mixer – Stream Temperatures")
                ax.set_ylim(0, max(values) * 1.18)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Splitter
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Stream Splitter")
            F_in = st.number_input("Feed flow F_in (mol/s)", value=100.0, step=1.0)
            comp_names_sp = st.text_input("Component names (comma-separated)", "A, B")
            mole_fracs_sp = st.text_input("Mole fractions (comma-separated)", "0.4, 0.6")
            split_fracs_sp = st.text_input("Split fractions (comma-separated, sum=1)", "0.7, 0.3")
            submitted = st.form_submit_button("Calculate Splitter", use_container_width=True)
        if submitted:
            try:
                components = [c.strip() for c in comp_names_sp.split(",")]
                mole_fracs = [float(x.strip()) for x in mole_fracs_sp.split(",")]
                split_fracs = [float(x.strip()) for x in split_fracs_sp.split(",")]
                result = ctrl.run_splitter(F_in, mole_fracs, components, split_fracs)
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
            result_r = data.get("result", {}) if data else {}
            if result_r and "outlets" in result_r:
                outlets = result_r["outlets"]
                try:
                    mf_vals = [float(x.strip()) for x in mole_fracs_sp.split(",")]
                    comp_list = [c.strip() for c in comp_names_sp.split(",")]
                except Exception:
                    mf_vals = []
                    comp_list = []
                if outlets and mf_vals and len(comp_list) == len(mf_vals):
                    outlet_labels = [o.get("label", f"Stream {i+1}") for i, o in enumerate(outlets)]
                    x_pos = np.arange(len(outlet_labels))
                    width = 0.6
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bottoms = np.zeros(len(outlets))
                    colors_sp = ["steelblue", "darkorange", "seagreen", "mediumpurple", "salmon"]
                    for ci, (comp, mf) in enumerate(zip(comp_list, mf_vals)):
                        comp_flows = np.array([o.get("F_mols", 0) * mf for o in outlets])
                        ax.bar(x_pos, comp_flows, width, bottom=bottoms,
                               label=comp, color=colors_sp[ci % len(colors_sp)], alpha=0.85)
                        bottoms += comp_flows
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(outlet_labels)
                    ax.set_ylabel("Component flow (mol/s)")
                    ax.set_title("Splitter – Component Flows per Outlet Stream")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, axis="y")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 4 - Material Balance
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Reactive Material Balance")
            F_feed = st.number_input("Feed flow (mol/s)", value=100.0, step=1.0)
            comp_names_mb = st.text_input("Component names (comma-separated)", "A, B, C")
            z_feed_mb = st.text_input("Feed mole fractions (comma-separated)", "0.4, 0.3, 0.3")
            key_idx = st.number_input("Key component index (0-based)", value=0, min_value=0, step=1)
            stoich_mb = st.text_input("Stoichiometric coefficients (comma-separated, neg=reactant)", "-1, -1, 2")
            conversion = st.number_input("Conversion of key component", value=0.85, min_value=0.0, max_value=1.0, step=0.01)
            submitted = st.form_submit_button("Calculate Balance", use_container_width=True)
        if submitted:
            try:
                components = [c.strip() for c in comp_names_mb.split(",")]
                z_feed = [float(x.strip()) for x in z_feed_mb.split(",")]
                stoich = [float(x.strip()) for x in stoich_mb.split(",")]
                result = ctrl.run_material_balance(F_feed, z_feed, components, int(key_idx), stoich, conversion)
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
            result_r = data.get("result", {}) if data else {}
            if result_r and "z_out" in result_r and "components" in result_r:
                comps_mb = result_r["components"]
                z_out = np.asarray(result_r["z_out"])
                F_out_val = result_r.get("F_out", 1.0)
                F_in_val = result_r.get("F_in", 1.0)
                outlet_flows = z_out * F_out_val
                try:
                    z_in = [float(x.strip()) for x in z_feed_mb.split(",")]
                    inlet_flows = np.asarray(z_in) * F_in_val
                except Exception:
                    inlet_flows = None
                x_pos = np.arange(len(comps_mb))
                width = 0.35
                fig, ax = plt.subplots(figsize=(8, 4))
                if inlet_flows is not None and len(inlet_flows) == len(comps_mb):
                    ax.bar(x_pos - width / 2, inlet_flows, width, label="Inlet", color="steelblue", alpha=0.85)
                ax.bar(x_pos + (width / 2 if inlet_flows is not None else 0), outlet_flows, width,
                       label="Outlet", color="darkorange", alpha=0.85)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(comps_mb)
                ax.set_ylabel("Molar flow (mol/s)")
                ax.set_title("Reactive Material Balance – Inlet vs Outlet Flows")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Energy Balance
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Energy Balance")
            st.markdown("**Sensible Heat Term**")
            n_sensible = st.number_input("n (mol)", value=1.0, step=0.1, key="n_sens")
            Cp_sensible = st.number_input("Cp (J/mol·K)", value=4.18, step=0.1, key="cp_sens")
            T1_sensible = st.number_input("T1 (K)", value=25.0, step=1.0, key="t1_sens")
            T2_sensible = st.number_input("T2 (K)", value=100.0, step=1.0, key="t2_sens")
            st.markdown("**Latent Heat Term**")
            n_latent = st.number_input("n (mol)", value=1.0, step=0.1, key="n_lat")
            dH_latent = st.number_input("ΔH_vap (J/mol)", value=40650.0, step=100.0, key="dH_lat")
            st.markdown("**Reaction Term**")
            n_rxn = st.number_input("n (mol)", value=1.0, step=0.1, key="n_rxn")
            dHrxn = st.number_input("ΔH_rxn (J/mol)", value=-890000.0, step=1000.0, key="dH_rxn")
            submitted = st.form_submit_button("Calculate Energy Balance", use_container_width=True)
        if submitted:
            try:
                sensible_items = [{"n": n_sensible, "Cp": Cp_sensible, "T1": T1_sensible, "T2": T2_sensible}]
                latent_items = [{"n": n_latent, "dH": dH_latent}]
                reaction_items = [{"n": n_rxn, "dHrxn": dHrxn}]
                result = ctrl.run_energy_balance(sensible_items, latent_items, reaction_items)
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
            result_r = data.get("result", {}) if data else {}
            if result_r and "Q_total_J" in result_r:
                Q_sens = n_sensible * Cp_sensible * (T2_sensible - T1_sensible)
                Q_lat = n_latent * dH_latent
                Q_rxn = n_rxn * dHrxn
                labels_eb = ["Sensible Heat", "Latent Heat", "Reaction Heat", "Total Q"]
                values_eb = [Q_sens, Q_lat, Q_rxn, result_r["Q_total_J"]]
                colors_eb = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels_eb, [v / 1000 for v in values_eb],
                              color=colors_eb, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, values_eb):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (max(abs(v) for v in values_eb) / 1000) * 0.01,
                            f"{val/1000:.2f} kJ", ha="center", va="bottom", fontsize=9)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("Energy (kJ)")
                ax.set_title("Energy Balance – Heat Contributions")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Recycle
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Recycle Loop Analysis")
            F_fresh = st.number_input("Fresh feed F_fresh (mol/s)", value=100.0, step=1.0)
            X_sp = st.number_input("Single-pass conversion X_sp", value=0.95, min_value=0.01, max_value=0.999, step=0.01)
            purge_frac = st.number_input("Purge fraction", value=0.1, min_value=0.001, max_value=0.999, step=0.01)
            submitted = st.form_submit_button("Analyze Recycle", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_recycle(F_fresh, X_sp, purge_frac)
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
            if data and "D_values" in data and "F_R_values" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["D_values"], data["F_R_values"], color="steelblue", linewidth=2)
                ax.set_xlabel("Purge fraction D")
                ax.set_ylabel("Recycle flow F_R (mol/s)")
                ax.set_title("Recycle Flow vs Purge Fraction")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 7 - Composition Converter
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t7"):
            st.subheader("Composition Converter")
            mode_conv = st.selectbox("Conversion mode", ["mole_to_mass", "mass_to_mole"])
            comp_names_conv = st.text_input("Component names (comma-separated)", "A, B")
            fracs_conv = st.text_input("Fractions (comma-separated)", "0.4, 0.6")
            mol_weights_conv = st.text_input("Molecular weights (comma-separated)", "28, 44")
            submitted = st.form_submit_button("Convert", use_container_width=True)
        if submitted:
            try:
                components = [c.strip() for c in comp_names_conv.split(",")]
                fracs = [float(x.strip()) for x in fracs_conv.split(",")]
                mol_weights = [float(x.strip()) for x in mol_weights_conv.split(",")]
                result = ctrl.run_composition_convert(mode_conv, fracs, mol_weights, components)
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
            if data and "mole_fracs" in data and "mass_fracs" in data and "components" in data:
                comps_cv = data["components"]
                mf_conv = np.asarray(data["mole_fracs"])
                wf_conv = np.asarray(data["mass_fracs"])
                x_pos = np.arange(len(comps_cv))
                width = 0.35
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(x_pos - width / 2, mf_conv, width, label="Mole fraction", color="steelblue", alpha=0.85)
                ax.bar(x_pos + width / 2, wf_conv, width, label="Mass fraction", color="darkorange", alpha=0.85)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(comps_cv)
                ax.set_ylabel("Fraction")
                ax.set_title("Composition Converter – Mole vs Mass Fractions")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
