import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.heat_transfer_controller import HeatTransferController

st.set_page_config(layout="wide", page_title="Heat Transfer - ChemEng")
st.title("🔥 Heat Transfer")

ctrl = HeatTransferController()

def _default(key, fn):
    if key not in st.session_state:
        try:
            r = fn(); st.session_state[key] = r if isinstance(r, tuple) else (r.get("message",""), r)
        except Exception: pass

_default("res_t1", lambda: ctrl.run_flat_wall(50.0,1.0,100.0,25.0,0.01))
_default("res_t2", lambda: ctrl.run_composite_wall([{"k":50.0,"L":0.01,"name":"Steel"},{"k":0.04,"L":0.1,"name":"Insulation"}],1.0,300.0,25.0))
_default("res_t3", lambda: ctrl.run_cylinder(50.0,1.0,0.05,0.1,200.0,25.0))
_default("res_t4", lambda: ctrl.run_newton_cooling(25.0,0.5,80.0,20.0))
_default("res_t5", lambda: ctrl.run_pipe_convection(10000.0,7.0,0.6,0.05,"Dittus-Boelter",True))
_default("res_t6", lambda: ctrl.run_lmtd(120.0,60.0,20.0,80.0,500.0,10.0,"Counter"))
_default("res_t7", lambda: ctrl.run_ntu(2000.0,3000.0,500.0,10.0,120.0,20.0,"counter"))
_default("res_t8a", lambda: ctrl.run_blackbody(1000.0,1.0))
_default("res_t8b", lambda: ctrl.run_grey_body(800.0,300.0,0.8,1.0,1.0))

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Flat Wall", "Composite Wall", "Cylinder", "Newton Cooling",
    "Pipe Convection", "LMTD Heat Exchanger", "NTU Method", "Radiation"
])

# Tab 1 - Flat Wall
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Flat Wall Conduction")
            k_fw = st.number_input("Thermal conductivity k (W/m·K)", value=50.0, step=1.0)
            A_fw = st.number_input("Area A (m²)", value=1.0, step=0.1)
            T1_fw = st.number_input("T1 (°C)", value=100.0, step=1.0)
            T2_fw = st.number_input("T2 (°C)", value=25.0, step=1.0)
            L_fw = st.number_input("Thickness L (m)", value=0.01, step=0.001, format="%.4f")
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_flat_wall(k_fw, A_fw, T1_fw, T2_fw, L_fw)
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
            # temperature profile across wall computed from in-scope inputs
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([0, L_fw], [T1_fw, T2_fw], color="firebrick", linewidth=2, marker="o", markersize=8)
            ax.fill_between([0, L_fw], [T1_fw, T2_fw], alpha=0.15, color="firebrick")
            ax.set_xlabel("Position x (m)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Flat Wall – Linear Temperature Profile")
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# Tab 2 - Composite Wall
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Composite Wall Conduction")
            layers_text = st.text_area(
                "Layers (k, L, name per line)",
                "50.0, 0.01, Steel\n0.04, 0.1, Insulation",
                help="Format: k (W/m·K), L (m), name"
            )
            A_cw = st.number_input("Area A (m²)", value=1.0, step=0.1)
            T_in_cw = st.number_input("Inner temperature T_in (°C)", value=300.0, step=1.0)
            T_out_cw = st.number_input("Outer temperature T_out (°C)", value=25.0, step=1.0)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                layers = []
                for line in layers_text.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        layers.append({"k": float(parts[0]), "L": float(parts[1]), "name": parts[2]})
                    elif len(parts) == 2:
                        layers.append({"k": float(parts[0]), "L": float(parts[1]), "name": f"Layer{len(layers)+1}"})
                result = ctrl.run_composite_wall(layers, A_cw, T_in_cw, T_out_cw)
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
            if data and "layers" in data:
                layers_data = data["layers"]
                positions = [0.0]
                names = []
                temps = []
                for layer in layers_data:
                    positions.append(positions[-1] + layer.get("L", 0))
                    names.append(layer.get("name", ""))
                    if "T_surface" in layer:
                        temps.append(layer["T_surface"])
                if temps:
                    T_profile = [T_in_cw] + temps
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(positions[:len(T_profile)], T_profile, marker="o", color="firebrick", linewidth=2)
                    for i, name in enumerate(names):
                        ax.axvline(positions[i+1], color="gray", linestyle="--", alpha=0.4)
                    ax.set_xlabel("Position (m)")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title("Composite Wall Temperature Profile")
                    ax.grid(True, alpha=0.3)
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 3 - Cylinder
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Cylindrical Wall Conduction")
            k_cy = st.number_input("k (W/m·K)", value=50.0, step=1.0)
            L_cy = st.number_input("Length L (m)", value=1.0, step=0.1)
            r1_cy = st.number_input("Inner radius r1 (m)", value=0.05, step=0.005, format="%.4f")
            r2_cy = st.number_input("Outer radius r2 (m)", value=0.1, step=0.005, format="%.4f")
            T1_cy = st.number_input("T1 (°C) inner", value=200.0, step=1.0)
            T2_cy = st.number_input("T2 (°C) outer", value=25.0, step=1.0)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_cylinder(k_cy, L_cy, r1_cy, r2_cy, T1_cy, T2_cy)
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
            # T vs r profile for cylinder computed from in-scope inputs
            r_arr = np.linspace(r1_cy, r2_cy, 200)
            # T(r) = T1 + (T2 - T1) * ln(r/r1) / ln(r2/r1)
            import math as _mheat
            ln_ratio = _mheat.log(r2_cy / r1_cy)
            T_arr = T1_cy + (T2_cy - T1_cy) * np.log(r_arr / r1_cy) / ln_ratio
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(r_arr, T_arr, color="firebrick", linewidth=2)
            ax.fill_between(r_arr, T_arr, alpha=0.12, color="firebrick")
            ax.set_xlabel("Radius r (m)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Cylindrical Wall – Temperature Profile T(r)")
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# Tab 4 - Newton Cooling
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Newton's Law of Cooling")
            h_nc = st.number_input("Heat transfer coefficient h (W/m²·K)", value=25.0, step=1.0)
            A_nc = st.number_input("Area A (m²)", value=0.5, step=0.05)
            T_s = st.number_input("Surface temperature T_s (°C)", value=80.0, step=1.0)
            T_f = st.number_input("Fluid temperature T_f (°C)", value=20.0, step=1.0)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_newton_cooling(h_nc, A_nc, T_s, T_f)
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
            if data and "Q_W" in data:
                # Q vs h sweep from h=1 to 5×h_nc
                h_arr = np.linspace(1.0, max(h_nc * 5, 200), 200)
                Q_arr = h_arr * A_nc * abs(T_s - T_f)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(h_arr, Q_arr, color="darkorange", linewidth=2)
                ax.axvline(h_nc, color="steelblue", linestyle="--", alpha=0.8,
                           label=f"h = {h_nc:.1f} W/(m²·K)  →  Q = {data['Q_W']:.2f} W")
                ax.set_xlabel("Heat transfer coefficient h (W/m²·K)")
                ax.set_ylabel("Heat transfer rate Q (W)")
                ax.set_title("Newton Cooling – Q vs h")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Pipe Convection
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Pipe Convection (Nusselt)")
            Re = st.number_input("Reynolds number Re", value=10000.0, step=100.0)
            Pr = st.number_input("Prandtl number Pr", value=7.0, step=0.1)
            k_fluid = st.number_input("Fluid conductivity k (W/m·K)", value=0.6, step=0.01)
            D_pipe = st.number_input("Pipe diameter D (m)", value=0.05, step=0.005, format="%.4f")
            correlation = st.selectbox("Correlation", ["Dittus-Boelter", "Sieder-Tate", "Gnielinski"])
            heating = st.checkbox("Heating (vs cooling)", value=True)
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_pipe_convection(Re, Pr, k_fluid, D_pipe, correlation, heating)
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
            if data and "Re_arr" in data and "Nu_arr" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.loglog(data["Re_arr"], data["Nu_arr"], color="steelblue", linewidth=2, label="Nu vs Re")
                if "Re_calc" in data and "h_calc" in data:
                    # compute Nu at calc point: Nu = h*D/k
                    Nu_calc = data["h_calc"] * D_pipe / k_fluid if k_fluid > 0 else None
                    if Nu_calc:
                        ax.scatter([data["Re_calc"]], [Nu_calc], color="red", s=80, zorder=5,
                                   label=f"Re={data['Re_calc']:.0f}, Nu={Nu_calc:.1f}")
                ax.set_xlabel("Reynolds number Re")
                ax.set_ylabel("Nusselt number Nu")
                ax.set_title(f"Pipe Convection – Nu vs Re  ({correlation})")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, which="both")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - LMTD
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("LMTD Heat Exchanger")
            T_h_in = st.number_input("Hot inlet T_h_in (°C)", value=120.0, step=1.0)
            T_h_out = st.number_input("Hot outlet T_h_out (°C)", value=60.0, step=1.0)
            T_c_in = st.number_input("Cold inlet T_c_in (°C)", value=20.0, step=1.0)
            T_c_out = st.number_input("Cold outlet T_c_out (°C)", value=80.0, step=1.0)
            U_lmtd = st.number_input("Overall heat transfer coeff U (W/m²·K)", value=500.0, step=10.0)
            A_lmtd = st.number_input("Heat transfer area A (m²)", value=10.0, step=0.5)
            flow_lmtd = st.selectbox("Flow configuration", ["Counter", "Parallel"])
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_lmtd(T_h_in, T_h_out, T_c_in, T_c_out, U_lmtd, A_lmtd, flow_lmtd)
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
            # temperature vs position profile computed from in-scope inputs
            pos = np.array([0.0, 1.0])
            if flow_lmtd.lower() == "counter":
                T_hot = np.array([T_h_in, T_h_out])
                T_cold = np.array([T_c_out, T_c_in])
            else:
                T_hot = np.array([T_h_in, T_h_out])
                T_cold = np.array([T_c_in, T_c_out])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(pos, T_hot, color="firebrick", linewidth=2, marker="o", label="Hot stream")
            ax.plot(pos, T_cold, color="steelblue", linewidth=2, marker="o", label="Cold stream")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Inlet", "Outlet"])
            ax.set_ylabel("Temperature (°C)")
            ax.set_title(f"LMTD Heat Exchanger – Temperature Profile  ({flow_lmtd}-flow)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# Tab 7 - NTU
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t7"):
            st.subheader("NTU-Effectiveness Method")
            C_hot = st.number_input("Hot stream capacity C_hot (W/K)", value=2000.0, step=100.0)
            C_cold = st.number_input("Cold stream capacity C_cold (W/K)", value=3000.0, step=100.0)
            U_ntu = st.number_input("U (W/m²·K)", value=500.0, step=10.0)
            A_ntu = st.number_input("Area A (m²)", value=10.0, step=0.5)
            T_h_in_ntu = st.number_input("Hot inlet T_h_in (°C)", value=120.0, step=1.0)
            T_c_in_ntu = st.number_input("Cold inlet T_c_in (°C)", value=20.0, step=1.0)
            hx_type = st.selectbox("HX type", ["counter", "parallel", "shell-tube", "cross"])
            submitted = st.form_submit_button("Calculate", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_ntu(C_hot, C_cold, U_ntu, A_ntu, T_h_in_ntu, T_c_in_ntu, hx_type)
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
            if data and "NTU" in data and "eps" in data:
                # effectiveness vs NTU sweep for the same C_r
                C_r = data.get("C_r", 0.5)
                ntu_arr = np.linspace(0.01, 6.0, 300)
                if hx_type in ("counter",):
                    if abs(C_r - 1.0) < 1e-6:
                        eps_arr = ntu_arr / (1 + ntu_arr)
                    else:
                        eps_arr = (1 - np.exp(-ntu_arr * (1 - C_r))) / (1 - C_r * np.exp(-ntu_arr * (1 - C_r)))
                elif hx_type == "parallel":
                    eps_arr = (1 - np.exp(-ntu_arr * (1 + C_r))) / (1 + C_r)
                else:
                    # approximate for shell-tube / cross
                    eps_arr = 1 - np.exp((ntu_arr ** 0.22 / C_r) * (np.exp(-C_r * ntu_arr ** 0.78) - 1))
                eps_arr = np.clip(eps_arr, 0, 1)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(ntu_arr, eps_arr * 100, color="steelblue", linewidth=2)
                ax.axvline(data["NTU"], color="red", linestyle="--", alpha=0.8,
                           label=f"NTU={data['NTU']:.3f}  →  ε={data['eps']*100:.1f}%")
                ax.set_xlabel("NTU")
                ax.set_ylabel("Effectiveness ε (%)")
                ax.set_title(f"NTU–Effectiveness Curve  ({hx_type},  C_r={C_r:.3f})")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 8 - Radiation
with tab8:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("**Blackbody Radiation**")
        with st.form("form_t8a"):
            T_bb = st.number_input("Temperature T (K)", value=1000.0, step=10.0)
            A_bb = st.number_input("Area A (m²)", value=1.0, step=0.1)
            submitted_bb = st.form_submit_button("Calculate Blackbody", use_container_width=True)
        if submitted_bb:
            try:
                result = ctrl.run_blackbody(T_bb, A_bb)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t8a"] = (msg, data)
            except Exception as e:
                st.session_state["res_t8a"] = (f"Error: {e}", {})

        st.markdown("---")
        st.markdown("**Grey Body Radiation**")
        with st.form("form_t8b"):
            T1_gb = st.number_input("Surface 1 temperature T1 (K)", value=800.0, step=10.0)
            T2_gb = st.number_input("Surface 2 temperature T2 (K)", value=300.0, step=10.0)
            eps_gb = st.number_input("Emissivity ε", value=0.8, min_value=0.0, max_value=1.0, step=0.05)
            A_gb = st.number_input("Area A (m²)", value=1.0, step=0.1, key="A_gb")
            F12_gb = st.number_input("View factor F12", value=1.0, min_value=0.0, max_value=1.0, step=0.05)
            submitted_gb = st.form_submit_button("Calculate Grey Body", use_container_width=True)
        if submitted_gb:
            try:
                result = ctrl.run_grey_body(T1_gb, T2_gb, eps_gb, A_gb, F12_gb)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t8b"] = (msg, data)
            except Exception as e:
                st.session_state["res_t8b"] = (f"Error: {e}", {})

    with col_out:
        if "res_t8a" in st.session_state:
            msg, data = st.session_state["res_t8a"]
            st.markdown("**Blackbody Result:**")
            if msg:
                st.code(msg, language=None)
            if data and "lam_um" in data and "E" in data:
                lam = np.asarray(data["lam_um"])
                E = np.asarray(data["E"])
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(lam, E / 1e6, color="firebrick", linewidth=2)  # W/m³ → W/(m²·μm) scaling
                peak = data.get("wien_lam_um", lam[int(np.argmax(E))])
                ax.axvline(peak, color="gray", linestyle="--", alpha=0.7,
                           label=f"λ_max = {peak:.2f} μm")
                ax.set_xlabel("Wavelength λ (μm)")
                ax.set_ylabel("Spectral emissive power (W/m²·μm)")
                ax.set_title(f"Planck Spectral Distribution  (T = {data.get('T', T_bb):.0f} K)")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        if "res_t8b" in st.session_state:
            msg, data = st.session_state["res_t8b"]
            st.markdown("**Grey Body Result:**")
            if msg:
                st.code(msg, language=None)
