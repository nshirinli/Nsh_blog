import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from app.controllers.process_control_controller import ProcessControlController

st.set_page_config(layout="wide", page_title="Process Control - ChemEng")
st.title("📊 Process Control")

ctrl = ProcessControlController()

def _default(key, fn):
    if key not in st.session_state:
        try:
            r = fn(); st.session_state[key] = r if isinstance(r, tuple) else (r.get("message",""), r)
        except Exception: pass

_default("res_t1", lambda: ctrl.run_fopdt_step(2.0,5.0,1.0,1.0,None))
_default("res_t2", lambda: ctrl.run_fopdt_ramp(2.0,5.0,1.0,0.5,None))
_default("res_t3", lambda: ctrl.run_second_order(1.0,2.0,0.5,1.0,None))
_default("res_t4", lambda: ctrl.run_pid_simulation(2.0,5.0,1.0,2.0,5.0,0.0,1.0,0.0,None,None))
_default("res_t5", lambda: ctrl.run_tuning_open_loop(2.0,5.0,1.0))
_default("res_t6", lambda: ctrl.run_tuning_closed_loop(5.0,8.0))
_default("res_t7", lambda: ctrl.run_bode_process(2.0,5.0,1.0))

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "FOPDT Step", "FOPDT Ramp", "Second Order",
    "PID Simulation", "Open-Loop Tuning", "Closed-Loop Tuning", "Bode Plot"
])

# Tab 1 - FOPDT Step
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t1"):
            st.subheader("FOPDT Step Response")
            Kp1 = st.number_input("Process gain Kp", value=2.0, step=0.1)
            tau1 = st.number_input("Time constant τ (s)", value=5.0, step=0.5)
            theta1 = st.number_input("Dead time θ (s)", value=1.0, step=0.1)
            delta_u1 = st.number_input("Step size Δu", value=1.0, step=0.1)
            t_final1 = st.number_input("Final time (0 = auto)", value=0.0, step=1.0)
            submitted = st.form_submit_button("Simulate", use_container_width=True)
        if submitted:
            try:
                t_final_val = None if t_final1 == 0 else t_final1
                result = ctrl.run_fopdt_step(Kp1, tau1, theta1, delta_u1, t_final_val)
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
            if data and "t" in data and "y" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["y"], color="steelblue", linewidth=2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Process output y(t)")
                ax.set_title("FOPDT Step Response")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 2 - FOPDT Ramp
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t2"):
            st.subheader("FOPDT Ramp Response")
            Kp2 = st.number_input("Process gain Kp", value=2.0, step=0.1)
            tau2 = st.number_input("Time constant τ (s)", value=5.0, step=0.5)
            theta2 = st.number_input("Dead time θ (s)", value=1.0, step=0.1)
            ramp_rate = st.number_input("Ramp rate", value=0.5, step=0.05)
            t_final2 = st.number_input("Final time (0 = auto)", value=0.0, step=1.0)
            submitted = st.form_submit_button("Simulate", use_container_width=True)
        if submitted:
            try:
                t_final_val2 = None if t_final2 == 0 else t_final2
                result = ctrl.run_fopdt_ramp(Kp2, tau2, theta2, ramp_rate, t_final_val2)
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
            if data and "t" in data and "y" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["y"], color="darkorange", linewidth=2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Process output y(t)")
                ax.set_title("FOPDT Ramp Response")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 3 - Second Order
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t3"):
            st.subheader("Second Order Step Response")
            Kp3 = st.number_input("Process gain Kp", value=1.0, step=0.1)
            tau_n = st.number_input("Natural period τn (s)", value=2.0, step=0.1)
            zeta = st.number_input("Damping ratio ζ", value=0.5, step=0.05)
            delta_u3 = st.number_input("Step size Δu", value=1.0, step=0.1)
            t_final3 = st.number_input("Final time (0 = auto)", value=0.0, step=1.0)
            submitted = st.form_submit_button("Simulate", use_container_width=True)
        if submitted:
            try:
                t_final_val3 = None if t_final3 == 0 else t_final3
                result = ctrl.run_second_order(Kp3, tau_n, zeta, delta_u3, t_final_val3)
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
            if data and "t" in data and "y" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data["t"], data["y"], color="purple", linewidth=2)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Process output y(t)")
                ax.set_title(f"2nd Order Response (ζ={zeta})")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - PID Simulation
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t4"):
            st.subheader("PID Control Simulation")
            Kp4 = st.number_input("Process gain Kp", value=2.0, step=0.1)
            tau4 = st.number_input("Process τ (s)", value=5.0, step=0.5)
            theta4 = st.number_input("Dead time θ (s)", value=1.0, step=0.1)
            Kc4 = st.number_input("Controller gain Kc", value=2.0, step=0.1)
            Ti4 = st.number_input("Integral time Ti (s)", value=5.0, step=0.5)
            Td4 = st.number_input("Derivative time Td (s)", value=0.0, step=0.5)
            setpoint4 = st.number_input("Setpoint", value=1.0, step=0.1)
            disturbance4 = st.number_input("Disturbance magnitude", value=0.0, step=0.1)
            dist_time4 = st.number_input("Disturbance time (0 = none)", value=0.0, step=1.0)
            t_final4 = st.number_input("Final time (0 = auto)", value=0.0, step=1.0)
            submitted = st.form_submit_button("Simulate PID", use_container_width=True)
        if submitted:
            try:
                t_final_val4 = None if t_final4 == 0 else t_final4
                dist_time_val = None if dist_time4 == 0 else dist_time4
                result = ctrl.run_pid_simulation(Kp4, tau4, theta4, Kc4, Ti4, Td4, setpoint4, disturbance4, dist_time_val, t_final_val4)
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
            if data and "t" in data and "y" in data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                ax1.plot(data["t"], data["y"], color="steelblue", linewidth=2, label="Output y")
                ax1.axhline(setpoint4, color="red", linestyle="--", alpha=0.6, label="Setpoint")
                ax1.set_ylabel("Output y(t)")
                ax1.set_title("PID Closed-Loop Response")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.spines[["top", "right"]].set_visible(False)
                if "u" in data:
                    ax2.plot(data["t"], data["u"], color="darkorange", linewidth=2, label="Control u")
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Control signal u(t)")
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.3)
                    ax2.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Open-Loop Tuning
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t5"):
            st.subheader("Open-Loop Tuning (IMC, ZN, ITAE)")
            Kp5 = st.number_input("Process gain Kp", value=2.0, step=0.1)
            tau5 = st.number_input("Process τ (s)", value=5.0, step=0.5)
            theta5 = st.number_input("Dead time θ (s)", value=1.0, step=0.1)
            submitted = st.form_submit_button("Get Tuning Rules", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_tuning_open_loop(Kp5, tau5, theta5)
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
            if data:
                # Collect Kc, Ti, Td for each tuning method that has them
                method_names, Kc_vals, Ti_vals, Td_vals = [], [], [], []
                for key in ["P", "PI", "PID", "ITAE_PI", "ITAE_PID", "IMC_PI"]:
                    if key in data and isinstance(data[key], dict):
                        r = data[key]
                        if "Kc" in r:
                            method_names.append(key)
                            Kc_vals.append(float(r["Kc"]))
                            Ti_vals.append(float(r.get("Ti", 0)))
                            Td_vals.append(float(r.get("Td", 0)))
                if method_names:
                    x = np.arange(len(method_names))
                    width = 0.25
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(x - width, Kc_vals, width, label="Kc", color="steelblue", alpha=0.85)
                    ax.bar(x,         Ti_vals, width, label="Ti (s)", color="darkorange", alpha=0.85)
                    ax.bar(x + width, Td_vals, width, label="Td (s)", color="seagreen", alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(method_names, rotation=25, ha="right", fontsize=9)
                    ax.set_ylabel("Parameter value")
                    ax.set_title("Open-Loop Tuning – Controller Parameters by Method")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, axis="y")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 6 - Closed-Loop Tuning
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t6"):
            st.subheader("Ziegler-Nichols Closed-Loop Tuning")
            Kcu = st.number_input("Ultimate gain Kcu", value=5.0, step=0.1)
            Pu = st.number_input("Ultimate period Pu (s)", value=8.0, step=0.5)
            submitted = st.form_submit_button("Get Tuning Rules", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_tuning_closed_loop(Kcu, Pu)
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
            if data:
                method_names, Kc_vals, Ti_vals, Td_vals = [], [], [], []
                for key in ["P", "PI", "PID"]:
                    if key in data and isinstance(data[key], dict):
                        r = data[key]
                        if "Kc" in r:
                            method_names.append(key)
                            Kc_vals.append(float(r["Kc"]))
                            Ti_vals.append(float(r.get("Ti", 0)))
                            Td_vals.append(float(r.get("Td", 0)))
                if method_names:
                    x = np.arange(len(method_names))
                    width = 0.25
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(x - width, Kc_vals, width, label="Kc", color="steelblue", alpha=0.85)
                    ax.bar(x,         Ti_vals, width, label="Ti (s)", color="darkorange", alpha=0.85)
                    ax.bar(x + width, Td_vals, width, label="Td (s)", color="seagreen", alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(method_names, fontsize=10)
                    ax.set_ylabel("Parameter value")
                    ax.set_title(f"Ziegler-Nichols Closed-Loop Tuning  (Kcu={Kcu}, Pu={Pu} s)")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3, axis="y")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# Tab 7 - Bode Plot
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.form("form_t7"):
            st.subheader("Bode Plot")
            Kp7 = st.number_input("Process gain Kp", value=2.0, step=0.1)
            tau7 = st.number_input("Process τ (s)", value=5.0, step=0.5)
            theta7 = st.number_input("Dead time θ (s)", value=1.0, step=0.1)
            show_cl = st.checkbox("Include closed-loop (PID)", value=False)
            Kc7 = st.number_input("Kc (for CL)", value=2.0, step=0.1)
            Ti7 = st.number_input("Ti (for CL)", value=5.0, step=0.5)
            Td7 = st.number_input("Td (for CL)", value=0.0, step=0.5)
            submitted = st.form_submit_button("Generate Bode", use_container_width=True)
        if submitted:
            try:
                result = ctrl.run_bode_process(Kp7, tau7, theta7)
                if isinstance(result, tuple):
                    msg, data = result
                else:
                    data = result
                    msg = data.get("message", "")
                cl_data = {}
                if show_cl:
                    cl_result = ctrl.run_bode_loop(Kp7, tau7, theta7, Kc7, Ti7, Td7)
                    if isinstance(cl_result, tuple):
                        _, cl_data = cl_result
                    else:
                        cl_data = cl_result
                st.session_state["res_t7"] = (msg, data, cl_data)
            except Exception as e:
                st.session_state["res_t7"] = (f"Error: {e}", {}, {})
    with col_out:
        if "res_t7" in st.session_state:
            entry = st.session_state["res_t7"]
            msg = entry[0]
            data = entry[1]
            cl_data = entry[2] if len(entry) > 2 else {}
            if msg:
                st.code(msg, language=None)
            if data and "omega" in data and "magnitude_dB" in data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                ax1.semilogx(data["omega"], data["magnitude_dB"], color="steelblue", linewidth=2, label="OL Process")
                if cl_data and "magnitude_dB" in cl_data:
                    ax1.semilogx(cl_data["omega"], cl_data["magnitude_dB"], color="darkorange", linewidth=2, linestyle="--", label="CL")
                ax1.axhline(-3, color="gray", linestyle=":", alpha=0.5)
                ax1.set_ylabel("Magnitude (dB)")
                ax1.set_title("Bode Plot")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3, which="both")
                ax1.spines[["top", "right"]].set_visible(False)

                ax2.semilogx(data["omega"], data["phase_deg"], color="steelblue", linewidth=2, label="OL Process")
                if cl_data and "phase_deg" in cl_data:
                    ax2.semilogx(cl_data["omega"], cl_data["phase_deg"], color="darkorange", linewidth=2, linestyle="--", label="CL")
                ax2.axhline(-180, color="gray", linestyle=":", alpha=0.5)
                ax2.set_xlabel("Frequency ω (rad/s)")
                ax2.set_ylabel("Phase (deg)")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3, which="both")
                ax2.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
