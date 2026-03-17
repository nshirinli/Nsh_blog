import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.database_controller import DatabaseController

st.set_page_config(layout="wide", page_title="Database - ChemEng")
st.title("🗄️ Chemical Database")

ctrl = DatabaseController()

tab1, tab2, tab3, tab4 = st.tabs([
    "Search & Info", "Vapor Pressure Curve", "Cp Curve", "Compare Properties"
])

# Helper: get all compound names
@st.cache_data
def get_all_compound_names():
    try:
        compounds = ctrl.all_compounds()
        return [c["name"] for c in compounds] if compounds else []
    except Exception:
        return []

all_names = get_all_compound_names()

# Tab 1 - Search & Info
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Search Compounds")
            query = st.text_input("Search query", "water")
            try:
                categories = ["All"] + ctrl.all_categories()
            except Exception:
                categories = ["All"]
            category = st.selectbox("Category", categories)
            submitted = st.form_submit_button("Search", use_container_width=True)
        if submitted:
            try:
                cat_arg = None if category == "All" else category
                results = ctrl.search(query, cat_arg)
                st.session_state["db_results"] = results
                st.session_state["res_t1"] = (f"Found {len(results)} results.", {})
            except Exception as e:
                st.session_state["db_results"] = []
                st.session_state["res_t1"] = (f"Error: {e}", {})

        if "db_results" in st.session_state and st.session_state["db_results"]:
            result_names = [r["name"] for r in st.session_state["db_results"]]
            compound_name = st.selectbox("Select compound for details", result_names)
            if compound_name:
                if st.button("Show Details"):
                    try:
                        detail = ctrl.get_text(compound_name)
                        st.session_state["compound_detail"] = detail
                    except Exception as e:
                        st.session_state["compound_detail"] = f"Error: {e}"

    with col_out:
        if "res_t1" in st.session_state:
            msg, _ = st.session_state["res_t1"]
            if msg:
                st.info(msg)
        if "db_results" in st.session_state and st.session_state["db_results"]:
            st.dataframe(st.session_state["db_results"], use_container_width=True)
        if "compound_detail" in st.session_state:
            st.markdown("**Compound Details:**")
            st.code(st.session_state["compound_detail"], language=None)

# Tab 2 - Vapor Pressure Curve
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Vapor Pressure Curve")
            vp_compound = st.selectbox("Select compound", all_names if all_names else ["water"])
            T_min_vp = st.number_input("T min (°C)", value=-20.0, step=5.0)
            T_max_vp = st.number_input("T max (°C)", value=150.0, step=5.0)
            submitted = st.form_submit_button("Generate Curve", use_container_width=True)
        if submitted:
            try:
                result = ctrl.vapor_pressure_curve(vp_compound, T_min_vp, T_max_vp)
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
            if data and "T" in data and "P" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T"], data["P"], color="steelblue", linewidth=2)
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Vapor Pressure (mmHg or Pa)")
                ax.set_title(f"Vapor Pressure Curve – {vp_compound}")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Cp Curve
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Cp vs Temperature Curve")
            cp_compound = st.selectbox("Select compound", all_names if all_names else ["water"])
            T_min_cp = st.number_input("T min (K)", value=298.0, step=10.0)
            T_max_cp = st.number_input("T max (K)", value=1000.0, step=10.0)
            submitted = st.form_submit_button("Generate Curve", use_container_width=True)
        if submitted:
            try:
                result = ctrl.cp_curve(cp_compound, T_min_cp, T_max_cp)
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
            if data and "T" in data and "Cp" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["T"], data["Cp"], color="darkorange", linewidth=2)
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel("Cp (J/mol·K)")
                ax.set_title(f"Heat Capacity Curve – {cp_compound}")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Compare Properties
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Compare Compound Properties")
            selected_compounds = st.multiselect(
                "Select compounds to compare",
                all_names if all_names else [],
                default=all_names[:3] if len(all_names) >= 3 else all_names
            )
            prop = st.selectbox("Property", ["molecular_weight", "Tc", "Pc", "Vc", "omega", "Tb"])
            submitted = st.form_submit_button("Compare", use_container_width=True)
        if submitted:
            try:
                result = ctrl.compare(selected_compounds, prop)
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
            if data and "names" in data and "values" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                x_pos = np.arange(len(data["names"]))
                ax.bar(x_pos, data["values"], color="steelblue", alpha=0.8, edgecolor="white")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(data["names"], rotation=30, ha="right", fontsize=9)
                ax.set_ylabel(prop)
                ax.set_title(f"Property Comparison: {prop}")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
