import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide", page_title="Process Economics - ChemEng")
st.title("💰 Process Economics")

if "econ_ctrl" not in st.session_state:
    from app.controllers.process_economics_controller import ProcessEconomicsController
    st.session_state.econ_ctrl = ProcessEconomicsController()
ctrl = st.session_state.econ_ctrl

def _default(key, fn):
    if key not in st.session_state:
        try:
            r = fn(); st.session_state[key] = r if isinstance(r, tuple) else (r.get("message",""), r)
        except Exception: pass

_default("res_t2", lambda: ctrl.run_capex_bare_module(600.0,0.15,0.10))
_default("res_t3", lambda: ctrl.run_capex_lang(500000.0,"fluid",0.15))
_default("res_t4", lambda: ctrl.run_opex(1000000.0,200000.0,10,60000.0,0.06,0.01,0.01,0.1))
_default("res_t5", lambda: ctrl.run_cash_flow(2000000.0,15,0.21,"straight-line",0.05))
_default("res_t6", lambda: ctrl.run_profitability(0.10,2000000.0,15,0.21))
_default("res_t7", lambda: ctrl.run_sensitivity(0.10,0.10))

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Equipment Cost", "CAPEX (Bare Module)", "CAPEX (Lang)",
    "OPEX", "Cash Flow", "Profitability", "Sensitivity"
])

# Tab 1 - Equipment Cost
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Equipment Cost Estimation")
            try:
                equip_list = ctrl.get_equipment_list()
            except Exception:
                equip_list = ["Heat Exchanger", "Pump", "Compressor", "Vessel", "Reactor"]
            try:
                mat_list = ctrl.get_material_list()
            except Exception:
                mat_list = ["Carbon Steel", "Stainless Steel 304", "Stainless Steel 316"]
            equip_type = st.selectbox("Equipment type", equip_list)
            size = st.number_input("Size parameter", value=100.0, step=10.0)
            cepci = st.number_input("CEPCI index", value=600.0, step=10.0)
            material = st.selectbox("Material", mat_list)
            pressure_factor = st.number_input("Pressure factor", value=1.0, step=0.05)
            quantity = st.number_input("Quantity", value=1, min_value=1, step=1)
            submitted = st.form_submit_button("Estimate Cost & Add to List", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_equipment_cost(equip_type, size, cepci, material, pressure_factor, int(quantity))
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                st.session_state["res_t1"] = (msg, data)
            except Exception as e:
                st.session_state["res_t1"] = (f"Error: {e}", {})

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Add to Equipment List", use_container_width=True):
                try:
                    ctrl.add_to_equipment_list(equip_type, size, cepci, material, pressure_factor, int(quantity))
                    st.success("Added.")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col_btn2:
            if st.button("Clear Equipment List", use_container_width=True):
                try:
                    ctrl.clear_equipment_list()
                    st.success("Cleared.")
                except Exception as e:
                    st.error(f"Error: {e}")
    with col_out:
        if "res_t1" in st.session_state:
            msg, data = st.session_state["res_t1"]
            if msg:
                st.code(msg, language=None)
            result_r = data.get("result", {}) if data else {}
            if result_r:
                bar_items = {
                    "Purchased\nCost (Cp)": result_r.get("Cp_current_USD", 0),
                    "Bare-Module\nCost (C_BM)": result_r.get("C_BM_USD", 0),
                }
                labels = list(bar_items.keys())
                values = list(bar_items.values())
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, values, color=["steelblue", "darkorange"], alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                            f"${val/1e3:.1f}k", ha="center", va="bottom", fontsize=10)
                ax.set_ylabel("Cost (USD)")
                ax.set_title(f"Equipment Cost – {equip_type}  (qty={int(quantity)})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - CAPEX Bare Module
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("CAPEX – Bare Module Method")
            cepci2 = st.number_input("CEPCI index", value=600.0, step=10.0)
            contingency = st.number_input("Contingency fraction", value=0.15, step=0.01)
            engineering = st.number_input("Engineering fraction", value=0.10, step=0.01)
            submitted = st.form_submit_button("Calculate CAPEX", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_capex_bare_module(cepci2, contingency, engineering)
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
            if data and isinstance(data, dict):
                # Build pie chart from numeric values in data
                pie_keys = [k for k, v in data.items() if isinstance(v, (int, float)) and v > 0 and k != "message"]
                if pie_keys:
                    pie_vals = [data[k] for k in pie_keys]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(pie_vals, labels=pie_keys, autopct="%1.1f%%", startangle=90)
                    ax.set_title("CAPEX Breakdown")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 3 - CAPEX Lang
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("CAPEX – Lang Factor Method")
            total_purchased = st.number_input("Total purchased equipment cost ($)", value=500000.0, step=10000.0)
            plant_type = st.selectbox("Plant type", ["solid", "fluid", "mixed"])
            contingency3 = st.number_input("Contingency fraction", value=0.15, step=0.01)
            submitted = st.form_submit_button("Calculate CAPEX", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_capex_lang(total_purchased, plant_type, contingency3)
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
            result_r3 = data.get("result", {}) if data else {}
            if result_r3:
                bar_items = {
                    "Purchased\nCost": total_purchased,
                    "Fixed Capital\n(FCI)": result_r3.get("FCI", 0),
                    "Working Capital\n(WC)": result_r3.get("WC", 0),
                    "Total Capital\n(TCI)": result_r3.get("TCI", 0),
                }
                labels = list(bar_items.keys())
                values = list(bar_items.values())
                colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                            f"${val/1e6:.2f}M", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Cost (USD)")
                ax.set_title(f"CAPEX Lang Factor  ({plant_type}, F_L={result_r3.get('lang_factor', '?')})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - OPEX
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Operating Expenditure (OPEX)")
            raw_materials = st.number_input("Raw materials cost ($/yr)", value=1000000.0, step=50000.0)
            utilities = st.number_input("Utilities cost ($/yr)", value=200000.0, step=10000.0)
            n_operators = st.number_input("Number of operators", value=10, min_value=1, step=1)
            salary = st.number_input("Operator salary ($/yr)", value=60000.0, step=5000.0)
            maintenance_frac = st.number_input("Maintenance fraction of CAPEX", value=0.06, step=0.01)
            insurance_frac = st.number_input("Insurance fraction of CAPEX", value=0.01, step=0.005)
            tax_frac = st.number_input("Tax fraction of CAPEX", value=0.01, step=0.005)
            cap_charges_frac = st.number_input("Capital charges fraction", value=0.1, step=0.01)
            submitted = st.form_submit_button("Calculate OPEX", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_opex(raw_materials, utilities, int(n_operators), salary,
                                       maintenance_frac, insurance_frac, tax_frac, cap_charges_frac)
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
            if data and "categories" in data:
                categories = data["categories"]
                values = data.get("values", [])
                if categories and values:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(values, labels=categories, autopct="%1.1f%%", startangle=90)
                    ax.set_title("OPEX Breakdown")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 5 - Cash Flow
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Cash Flow Analysis")
            revenue5 = st.number_input("Annual revenue ($/yr)", value=2000000.0, step=100000.0)
            plant_life5 = st.number_input("Plant life (years)", value=15, min_value=1, max_value=50, step=1)
            tax_rate5 = st.number_input("Tax rate", value=0.21, step=0.01)
            depreciation5 = st.selectbox("Depreciation method", ["straight-line", "MACRS"])
            salvage_frac5 = st.number_input("Salvage value fraction", value=0.05, step=0.01)
            submitted = st.form_submit_button("Calculate Cash Flow", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_cash_flow(revenue5, int(plant_life5), tax_rate5, depreciation5, salvage_frac5)
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
            if data and "years" in data and "cumulative_cash_flow" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["years"], data["cumulative_cash_flow"], color="steelblue", linewidth=2, marker="o", markersize=4)
                ax.axhline(0, color="red", linestyle="--", alpha=0.5)
                ax.set_xlabel("Year")
                ax.set_ylabel("Cumulative Cash Flow ($)")
                ax.set_title("Cumulative Cash Flow")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - Profitability
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Profitability Metrics (NPV, IRR, ROI)")
            discount_rate6 = st.number_input("Discount rate", value=0.10, step=0.01)
            revenue6 = st.number_input("Annual revenue ($/yr)", value=2000000.0, step=100000.0)
            plant_life6 = st.number_input("Plant life (years)", value=15, min_value=1, step=1)
            tax_rate6 = st.number_input("Tax rate", value=0.21, step=0.01)
            submitted = st.form_submit_button("Calculate Profitability", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_profitability(discount_rate6, revenue6, int(plant_life6), tax_rate6)
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
            result_r6 = data.get("result", {}) if data else {}
            if result_r6:
                npv = result_r6.get("npv", 0)
                irr = result_r6.get("irr", float("nan"))
                roi = result_r6.get("roi", 0)
                import math as _mpecon
                bar_labels = ["NPV ($M)"]
                bar_values = [npv / 1e6]
                colors = ["steelblue"]
                if not _mpecon.isnan(irr):
                    bar_labels.append("IRR (%)")
                    bar_values.append(irr * 100)
                    colors.append("darkorange")
                bar_labels.append("ROI (%)")
                bar_values.append(roi * 100)
                colors.append("seagreen")
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(bar_labels, bar_values, color=colors, alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, bar_values):
                    offset = max(abs(v) for v in bar_values) * 0.02
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (offset if val >= 0 else -offset * 3),
                            f"{val:.2f}", ha="center", va="bottom", fontsize=10)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("Value")
                ax.set_title(f"Profitability Metrics  (WACC={discount_rate6:.1%})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 7 - Sensitivity
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t7"):
            st.subheader("Sensitivity Analysis (Tornado Chart)")
            discount_rate7 = st.number_input("Base discount rate", value=0.10, step=0.01)
            variation7 = st.number_input("Variation (±fraction)", value=0.10, step=0.01,
                                         help="0.1 = ±10%")
            submitted = st.form_submit_button("Run Sensitivity", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_sensitivity(discount_rate7, variation7)
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
            if data and "params" in data and "npv_low" in data and "npv_high" in data:
                params = data["params"]
                npv_low = data["npv_low"]
                npv_high = data["npv_high"]
                n = len(params)
                y_pos = np.arange(n)
                fig, ax = plt.subplots(figsize=(8, max(4, n * 0.6)))
                for i, (p, lo, hi) in enumerate(zip(params, npv_low, npv_high)):
                    ax.barh(i, hi - lo, left=lo, color="steelblue", alpha=0.8, edgecolor="white")
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(params)
                ax.set_xlabel("NPV ($)")
                ax.set_title("Tornado Chart – NPV Sensitivity")
                ax.grid(True, alpha=0.3, axis="x")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
