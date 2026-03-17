import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.polymer_controller import PolymerController

st.set_page_config(layout="wide", page_title="Polymer Engineering - ChemEng")
st.title("🧪 Polymer Engineering")

ctrl = PolymerController()

def _default(key, fn):
    if key not in st.session_state:
        try:
            r = fn(); st.session_state[key] = r if isinstance(r, tuple) else (r.get("message",""), r)
        except Exception: pass

_default("res_t1", lambda: ctrl.run_mw_stats([1,2,4,3,1],[10000,20000,30000,40000,50000]))
_default("res_t2", lambda: ctrl.run_flory_huggins(0.5,100))
_default("res_t3", lambda: ctrl.run_mark_houwink(1.1e-4,0.78,1000.0,1000000.0,None))
_default("res_t4", lambda: ctrl.run_glass_transition([{"name":"PS","Tg":100.0,"w":0.6},{"name":"PMMA","Tg":105.0,"w":0.4}]))
_default("res_t5", lambda: ctrl.run_wlf(60.0,0.0,17.44,51.6))
_default("res_t6", lambda: ctrl.run_free_radical(500.0,1e7,1e-5,0.5,0.01,8.7,3600.0))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "MW Statistics", "Flory-Huggins", "Mark-Houwink",
    "Glass Transition", "WLF Equation", "Free-Radical Kinetics"
])

# Tab 1 - MW Statistics
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Molecular Weight Statistics")
            Ni_str = st.text_input("Number of chains Ni (comma-separated)", "1, 2, 4, 3, 1")
            Mi_str = st.text_input("Molecular weights Mi (g/mol, comma-separated)", "10000, 20000, 30000, 40000, 50000")
            submitted = st.form_submit_button("Calculate Statistics", use_container_width=True)
        if submitted:
            try:
                Ni = [float(x.strip()) for x in Ni_str.split(",")]
                Mi = [float(x.strip()) for x in Mi_str.split(",")]
                result = ctrl.run_mw_stats(Ni, Mi)
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
            if data and "Mi" in data and "wi_dist" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                x_pos = np.arange(len(data["Mi"]))
                ax.bar(x_pos, data["wi_dist"], color="steelblue", alpha=0.8, edgecolor="white")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"{m:.0f}" for m in data["Mi"]], rotation=30, ha="right", fontsize=9)
                ax.set_xlabel("Molecular weight Mi (g/mol)")
                ax.set_ylabel("Weight fraction w_i")
                ax.set_title("Molecular Weight Distribution (Weight)")
                Mn = data.get("Mn", "?")
                Mw = data.get("Mw", "?")
                PDI = data.get("PDI", "?")
                ax.set_title(f"MWD – Mn={Mn:.0f}, Mw={Mw:.0f}, PDI={PDI:.2f}" if isinstance(Mn, float) else "MWD")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - Flory-Huggins
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Flory-Huggins Theory")
            chi = st.number_input("Flory-Huggins parameter χ", value=0.5, step=0.05)
            r = st.number_input("Degree of polymerization r", value=100, min_value=1, max_value=10000, step=10)
            submitted = st.form_submit_button("Calculate ΔGmix", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_flory_huggins(chi, int(r))
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
            if data and "phi2" in data and "dGmix" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["phi2"], data["dGmix"], color="darkgreen", linewidth=2)
                if "spinodal" in data and data["spinodal"]:
                    for sp in data["spinodal"]:
                        ax.axvline(sp, color="red", linestyle="--", alpha=0.7, label=f"Spinodal φ={sp:.3f}")
                ax.set_xlabel("Volume fraction φ₂")
                ax.set_ylabel("ΔGmix / nRT")
                ax.set_title(f"Flory-Huggins (χ={chi}, r={r})")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Mark-Houwink
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Mark-Houwink Equation")
            K_mh = st.number_input("K constant", value=1.1e-4, step=1e-5, format="%.2e")
            alpha_mh = st.number_input("α exponent", value=0.78, step=0.01)
            M_min = st.number_input("M min (g/mol)", value=1000.0, step=100.0)
            M_max = st.number_input("M max (g/mol)", value=1000000.0, step=10000.0)
            M_known_input = st.number_input("M_known (0 to skip)", value=0.0, step=1000.0)
            submitted = st.form_submit_button("Calculate Viscosity", use_container_width=True)
        if submitted:
            try:
                M_known = None if M_known_input == 0 else M_known_input
                result = ctrl.run_mark_houwink(K_mh, alpha_mh, M_min, M_max, M_known)
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
            if data and "M" in data and "eta" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.loglog(data["M"], data["eta"], color="purple", linewidth=2)
                if "M_known" in data and "eta_at_M" in data and data["M_known"]:
                    ax.scatter([data["M_known"]], [data["eta_at_M"]], color="red", s=80, zorder=5,
                               label=f"[η](M={data['M_known']:.0f}) = {data['eta_at_M']:.3f}")
                    ax.legend(fontsize=8)
                ax.set_xlabel("Molecular weight M (g/mol)")
                ax.set_ylabel("Intrinsic viscosity [η] (dL/g)")
                ax.set_title("Mark-Houwink Relationship")
                ax.grid(True, alpha=0.3, which="both")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Glass Transition
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Glass Transition Temperature (Tg)")
            st.markdown("Enter blend components (Name, Tg °C, weight fraction):")
            components_text = st.text_area(
                "Components (one per line: Name, Tg, w)",
                "PS, 100, 0.6\nPMMA, 105, 0.4",
                help="Format: Name, Tg (°C), weight fraction"
            )
            submitted = st.form_submit_button("Calculate Tg", use_container_width=True)
        if submitted:
            try:
                components = []
                for line in components_text.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        components.append({"name": parts[0], "Tg": float(parts[1]), "w": float(parts[2])})
                result = ctrl.run_glass_transition(components)
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
            if data and "Tg_C" in data:
                tg_blend = data["Tg_C"]
                try:
                    comp_list_gt = []
                    for line in components_text.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            comp_list_gt.append({"name": parts[0], "Tg": float(parts[1]), "w": float(parts[2])})
                except Exception:
                    comp_list_gt = []
                if comp_list_gt:
                    names_gt = [c["name"] for c in comp_list_gt] + ["Blend Tg"]
                    values_gt = [c["Tg"] for c in comp_list_gt] + [tg_blend]
                    colors_gt = ["steelblue"] * len(comp_list_gt) + ["darkorange"]
                    x_pos = np.arange(len(names_gt))
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(x_pos, values_gt, color=colors_gt, alpha=0.85, edgecolor="white")
                    for bar, val in zip(bars, values_gt):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + max(abs(v) for v in values_gt) * 0.01,
                                f"{val:.1f}°C", ha="center", va="bottom", fontsize=9)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(names_gt)
                    ax.set_ylabel("Tg (°C)")
                    ax.set_title("Glass Transition Temperatures – Components & Blend")
                    ax.grid(True, alpha=0.3, axis="y")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 5 - WLF Equation
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("WLF Equation (Viscoelasticity)")
            T_C = st.number_input("Temperature of interest T (°C)", value=60.0, step=1.0)
            Tg_C = st.number_input("Glass transition temperature Tg (°C)", value=0.0, step=1.0)
            C1 = st.number_input("WLF constant C1", value=17.44, step=0.1)
            C2 = st.number_input("WLF constant C2", value=51.6, step=0.1)
            submitted = st.form_submit_button("Calculate Shift Factor", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_wlf(T_C, Tg_C, C1, C2)
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
            if data and "T" in data and "log_aT" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T"], data["log_aT"], color="teal", linewidth=2)
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("log aT (shift factor)")
                ax.set_title("WLF – Temperature Shift Factor")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Free-Radical Kinetics
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Free-Radical Polymerization Kinetics")
            kp = st.number_input("Propagation rate constant kp (L/mol·s)", value=500.0, step=10.0)
            kt = st.number_input("Termination rate constant kt (L/mol·s)", value=1e7, step=1e6, format="%.2e")
            kd = st.number_input("Initiator decomp. kd (1/s)", value=1e-5, step=1e-6, format="%.2e")
            f = st.number_input("Initiator efficiency f", value=0.5, min_value=0.01, max_value=1.0, step=0.05)
            I0 = st.number_input("Initial initiator [I]0 (mol/L)", value=0.01, step=0.001, format="%.4f")
            M0 = st.number_input("Initial monomer [M]0 (mol/L)", value=8.7, step=0.1)
            t_end = st.number_input("End time (s)", value=3600.0, step=100.0)
            submitted = st.form_submit_button("Simulate Kinetics", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_free_radical(kp, kt, kd, f, I0, M0, t_end)
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
            if data and "t" in data and "M" in data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                ax1.plot(data["t"], data["M"], color="steelblue", linewidth=2)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("[M] (mol/L)")
                ax1.set_title("Monomer Concentration vs Time")
                ax1.grid(True, alpha=0.3)
                ax1.spines[["top", "right"]].set_visible(False)
                if "Rp" in data:
                    ax2.plot(data["t"], data["Rp"], color="darkorange", linewidth=2)
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Rp (mol/L·s)")
                    ax2.set_title("Polymerization Rate vs Time")
                    ax2.grid(True, alpha=0.3)
                    ax2.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
