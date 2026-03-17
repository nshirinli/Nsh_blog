import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.optimization_controller import OptimizationController

st.set_page_config(layout="wide", page_title="Optimization - ChemEng")
st.title("📈 Optimization")

ctrl = OptimizationController()

tab1, tab2, tab3, tab4 = st.tabs([
    "Linear (LP)", "Nonlinear (NLP)", "Mixed-Integer (MILP)", "Dynamic Control"
])

# Tab 1 - LP
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("Linear Programming")
            lp_sense = st.selectbox("Objective", ["minimize", "maximize"])
            lp_nvars = st.number_input("Number of variables", value=2, min_value=1, max_value=20, step=1)
            lp_obj = st.text_area("Objective function", "x0 + 2*x1",
                                  help="Use x0, x1, ... as variable names")
            lp_cons = st.text_area("Constraints (one per line)", "x0 + x1 <= 4\n2*x0 + x1 <= 6",
                                   help="Use <=, >=, or ==")
            lp_bounds = st.text_area("Variable bounds (one per line: low, high)", "0, 10\n0, 10")
            lp_guess = st.text_input("Initial guess (comma-separated)", "1.0, 1.0")
            submitted = st.form_submit_button("Solve LP", use_container_width=True)
        if submitted:
            try:
                result = ctrl.solve_problem(
                    problem_type="LP",
                    objective_sense=lp_sense,
                    num_vars=int(lp_nvars),
                    objective_text=lp_obj,
                    constraints_text=lp_cons,
                    bounds_text=lp_bounds,
                    integer_vars_text="",
                    initial_guess_text=lp_guess,
                    dynamic_state_vars_text="",
                    dynamic_control_vars_text="",
                    dynamic_horizon_text="",
                    dynamic_intervals=20,
                    dynamic_odes_text="",
                    dynamic_initial_conditions_text="",
                    dynamic_running_cost_text="",
                    dynamic_terminal_cost_text="",
                    dynamic_control_bounds_text="",
                    dynamic_control_guess_text="",
                    dynamic_terminal_constraints_text=""
                )
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
            if data and "variables" in data and data["variables"]:
                var_names = list(data["variables"].keys())
                var_vals = [data["variables"][k] for k in var_names]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(var_names, var_vals, color="steelblue", alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, var_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + abs(max(var_vals, default=1)) * 0.01,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
                ax.set_ylabel("Optimal value")
                ax.set_title(f"LP Optimal Variables  (obj = {data.get('objective_value', 0):.4f})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - NLP
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("Nonlinear Programming")
            nlp_sense = st.selectbox("Objective", ["minimize", "maximize"])
            nlp_nvars = st.number_input("Number of variables", value=2, min_value=1, max_value=20, step=1)
            nlp_obj = st.text_area("Objective function", "x0**2 + x1**2 - 2*x0",
                                   help="Use x0, x1, ... and Python math")
            nlp_cons = st.text_area("Constraints (one per line)", "x0 + x1 <= 4\nx0**2 + x1**2 <= 10",
                                    help="Use <=, >=, or ==")
            nlp_bounds = st.text_area("Variable bounds (one per line: low, high)", "0, 10\n0, 10")
            nlp_guess = st.text_input("Initial guess (comma-separated)", "1.0, 1.0")
            submitted = st.form_submit_button("Solve NLP", use_container_width=True)
        if submitted:
            try:
                result = ctrl.solve_problem(
                    problem_type="NLP",
                    objective_sense=nlp_sense,
                    num_vars=int(nlp_nvars),
                    objective_text=nlp_obj,
                    constraints_text=nlp_cons,
                    bounds_text=nlp_bounds,
                    integer_vars_text="",
                    initial_guess_text=nlp_guess,
                    dynamic_state_vars_text="",
                    dynamic_control_vars_text="",
                    dynamic_horizon_text="",
                    dynamic_intervals=20,
                    dynamic_odes_text="",
                    dynamic_initial_conditions_text="",
                    dynamic_running_cost_text="",
                    dynamic_terminal_cost_text="",
                    dynamic_control_bounds_text="",
                    dynamic_control_guess_text="",
                    dynamic_terminal_constraints_text=""
                )
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
            if data and "variables" in data and data["variables"]:
                var_names = list(data["variables"].keys())
                var_vals = [data["variables"][k] for k in var_names]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(var_names, var_vals, color="darkorange", alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, var_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + abs(max(var_vals, default=1)) * 0.01,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
                ax.set_ylabel("Optimal value")
                ax.set_title(f"NLP Optimal Variables  (obj = {data.get('objective_value', 0):.4f})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - MILP
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Mixed-Integer Linear Programming")
            milp_sense = st.selectbox("Objective", ["minimize", "maximize"])
            milp_nvars = st.number_input("Number of variables", value=2, min_value=1, max_value=20, step=1)
            milp_obj = st.text_area("Objective function", "x0 + 2*x1")
            milp_cons = st.text_area("Constraints (one per line)", "x0 + x1 <= 4\n2*x0 + x1 <= 6")
            milp_bounds = st.text_area("Variable bounds (one per line: low, high)", "0, 10\n0, 10")
            milp_int_vars = st.text_input("Integer variable indices (comma-separated)", "0",
                                          help="e.g. '0' for first variable, '0,1' for first two")
            milp_guess = st.text_input("Initial guess (comma-separated)", "1.0, 1.0")
            submitted = st.form_submit_button("Solve MILP", use_container_width=True)
        if submitted:
            try:
                result = ctrl.solve_problem(
                    problem_type="MILP",
                    objective_sense=milp_sense,
                    num_vars=int(milp_nvars),
                    objective_text=milp_obj,
                    constraints_text=milp_cons,
                    bounds_text=milp_bounds,
                    integer_vars_text=milp_int_vars,
                    initial_guess_text=milp_guess,
                    dynamic_state_vars_text="",
                    dynamic_control_vars_text="",
                    dynamic_horizon_text="",
                    dynamic_intervals=20,
                    dynamic_odes_text="",
                    dynamic_initial_conditions_text="",
                    dynamic_running_cost_text="",
                    dynamic_terminal_cost_text="",
                    dynamic_control_bounds_text="",
                    dynamic_control_guess_text="",
                    dynamic_terminal_constraints_text=""
                )
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
            if data and "variables" in data and data["variables"]:
                var_names = list(data["variables"].keys())
                var_vals = [data["variables"][k] for k in var_names]
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(var_names, var_vals, color="seagreen", alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, var_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + abs(max(var_vals, default=1)) * 0.01,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
                ax.set_ylabel("Optimal value")
                ax.set_title(f"MILP Optimal Variables  (obj = {data.get('objective_value', 0):.4f})")
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Dynamic
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("Dynamic Optimal Control")
            dyn_state_vars = st.text_input("State variables (comma-separated)", "x")
            dyn_control_vars = st.text_input("Control variables (comma-separated)", "u")
            dyn_horizon = st.text_input("Time horizon", "10")
            dyn_intervals = st.number_input("Collocation intervals", value=20, min_value=5, max_value=100, step=5)
            dyn_odes = st.text_area("ODEs (one per state var)", "x[1] * u",
                                    help="e.g. dx/dt = x[1]*u for single state 'x'")
            dyn_ic = st.text_input("Initial conditions (comma-separated)", "0.0")
            dyn_running_cost = st.text_area("Running cost L(x,u)", "x**2 + u**2")
            dyn_terminal_cost = st.text_input("Terminal cost phi(x)", "0")
            dyn_ctrl_bounds = st.text_input("Control bounds (low, high)", "-1.0, 1.0")
            dyn_ctrl_guess = st.text_input("Control initial guess", "0.0")
            dyn_terminal_cons = st.text_input("Terminal constraints (optional)", "")
            submitted = st.form_submit_button("Solve Dynamic", use_container_width=True)
        if submitted:
            try:
                result = ctrl.solve_problem(
                    problem_type="Dynamic",
                    objective_sense="minimize",
                    num_vars=0,
                    objective_text="",
                    constraints_text="",
                    bounds_text="",
                    integer_vars_text="",
                    initial_guess_text="",
                    dynamic_state_vars_text=dyn_state_vars,
                    dynamic_control_vars_text=dyn_control_vars,
                    dynamic_horizon_text=dyn_horizon,
                    dynamic_intervals=int(dyn_intervals),
                    dynamic_odes_text=dyn_odes,
                    dynamic_initial_conditions_text=dyn_ic,
                    dynamic_running_cost_text=dyn_running_cost,
                    dynamic_terminal_cost_text=dyn_terminal_cost,
                    dynamic_control_bounds_text=dyn_ctrl_bounds,
                    dynamic_control_guess_text=dyn_ctrl_guess,
                    dynamic_terminal_constraints_text=dyn_terminal_cons
                )
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
            if data and "state_profiles" in data and "time_points" in data:
                t_pts = data["time_points"]
                state_profiles = data["state_profiles"]
                control_profiles = data.get("control_profiles", {})
                n_plots = 1 + (1 if control_profiles else 0)
                fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
                if n_plots == 1:
                    axes = [axes]
                for name, vals in state_profiles.items():
                    axes[0].plot(t_pts[:len(vals)], vals, linewidth=2, label=name)
                axes[0].set_xlabel("Time")
                axes[0].set_ylabel("State")
                axes[0].set_title("Dynamic Optimal Control – State Trajectories")
                axes[0].legend(fontsize=9)
                axes[0].grid(True, alpha=0.3)
                axes[0].spines[["top", "right"]].set_visible(False)
                if control_profiles and n_plots > 1:
                    t_ctrl = data.get("time_points_control", t_pts)
                    for name, vals in control_profiles.items():
                        axes[1].step(t_ctrl[:len(vals)], vals, where="post", linewidth=2, label=name)
                    axes[1].set_xlabel("Time")
                    axes[1].set_ylabel("Control")
                    axes[1].set_title("Control Trajectories")
                    axes[1].legend(fontsize=9)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
