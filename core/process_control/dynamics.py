"""Process Control dynamics: FOPDT, PID simulation, Bode plots, tuning rules."""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# 1. First Order Plus Dead Time (FOPDT) model
#    G(s) = Kp * exp(-theta*s) / (tau*s + 1)
# ---------------------------------------------------------------------------

def fopdt_step_response(
    Kp: float,
    tau: float,
    theta: float,
    delta_u: float = 1.0,
    t_final: float | None = None,
    n_pts: int = 500,
) -> dict:
    """Step response of FOPDT model.

    y(t) = Kp * delta_u * (1 - exp(-(t - theta)/tau))  for t > theta
    Returns t, y, dy/dt arrays.
    """
    if t_final is None:
        t_final = max(theta + 5 * tau, 10.0)
    t_arr = np.linspace(0.0, t_final, n_pts)
    y_arr = np.where(
        t_arr <= theta,
        0.0,
        Kp * delta_u * (1.0 - np.exp(-(t_arr - theta) / tau))
    )
    # Derivative
    dydt = np.where(
        t_arr <= theta,
        0.0,
        Kp * delta_u / tau * np.exp(-(t_arr - theta) / tau)
    )
    return {
        "t": t_arr, "y": y_arr, "dydt": dydt,
        "Kp": Kp, "tau": tau, "theta": theta, "delta_u": delta_u,
        "y_ss": Kp * delta_u,
        "t63": theta + tau,   # time to reach 63.2% of steady-state
        "t28": theta + tau * math.log(1 / (1 - 0.283)),   # 28.3%
    }


def fopdt_ramp_response(
    Kp: float, tau: float, theta: float,
    ramp_rate: float = 1.0, t_final: float | None = None, n_pts: int = 500,
) -> dict:
    """Ramp (u = ramp_rate * t) response of FOPDT."""
    if t_final is None:
        t_final = max(theta + 5 * tau, 20.0)
    t_arr = np.linspace(0.0, t_final, n_pts)
    # y(t) = Kp * ramp_rate * ((t - theta) - tau*(1 - exp(-(t-theta)/tau))) for t>theta
    y_arr = np.where(
        t_arr <= theta,
        0.0,
        Kp * ramp_rate * (
            (t_arr - theta) - tau * (1.0 - np.exp(-(t_arr - theta) / tau))
        )
    )
    return {"t": t_arr, "y": y_arr, "Kp": Kp, "tau": tau, "theta": theta}


def second_order_response(
    Kp: float, tau_n: float, zeta: float,
    delta_u: float = 1.0, t_final: float | None = None, n_pts: int = 500,
) -> dict:
    """Step response of second-order system G(s) = Kp / (tau_n^2 s^2 + 2*zeta*tau_n*s + 1).

    zeta < 1 → underdamped, zeta = 1 → critically damped, zeta > 1 → overdamped.
    """
    if t_final is None:
        t_final = 10 * tau_n
    t_arr = np.linspace(0.0, t_final, n_pts)
    y_ss = Kp * delta_u

    if abs(zeta - 1.0) < 1e-4:   # critically damped
        y_arr = y_ss * (1.0 - (1.0 + t_arr / tau_n) * np.exp(-t_arr / tau_n))
    elif zeta < 1.0:              # underdamped
        omega_d = math.sqrt(1.0 - zeta**2) / tau_n
        y_arr = y_ss * (1.0 - np.exp(-zeta * t_arr / tau_n) * (
            np.cos(omega_d * t_arr)
            + zeta / math.sqrt(1.0 - zeta**2) * np.sin(omega_d * t_arr)
        ))
    else:                          # overdamped
        p1 = (-zeta + math.sqrt(zeta**2 - 1)) / tau_n
        p2 = (-zeta - math.sqrt(zeta**2 - 1)) / tau_n
        y_arr = y_ss * (1.0 + (p2 * np.exp(p1 * t_arr) - p1 * np.exp(p2 * t_arr)) / (p1 - p2))

    # Overshoot (underdamped only)
    overshoot = 0.0
    if zeta < 1.0:
        overshoot = math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2)) * 100  # %

    # Rise time (10% to 90%)
    mask_10 = y_arr >= 0.1 * y_ss
    mask_90 = y_arr >= 0.9 * y_ss
    t_rise = (t_arr[mask_90][0] - t_arr[mask_10][0]) if mask_90.any() and mask_10.any() else float("nan")

    return {
        "t": t_arr, "y": y_arr, "y_ss": y_ss,
        "overshoot_pct": overshoot, "t_rise": t_rise,
        "Kp": Kp, "tau_n": tau_n, "zeta": zeta,
    }


# ---------------------------------------------------------------------------
# 2. PID Controller Simulation
#    u(t) = Kc * [e(t) + (1/Ti)*∫e dt + Td * de/dt]
# ---------------------------------------------------------------------------

def pid_simulation(
    Kp: float,
    tau: float,
    theta: float,
    Kc: float,
    Ti: float,
    Td: float = 0.0,
    setpoint: float = 1.0,
    disturbance: float = 0.0,
    dist_time: float | None = None,
    t_final: float | None = None,
    n_pts: int = 600,
) -> dict:
    """Simulate PID control of a FOPDT process using ODE integration.

    State: [y, integral_error, y_delayed]
    Approximates dead time with Padé approximation (1st order).
    Returns t, y (output), u (manipulated variable), e (error) arrays.
    """
    if t_final is None:
        t_final = max(theta + 10 * tau, 30.0)
    if dist_time is None:
        dist_time = t_final * 0.6

    # 1st-order Padé approximation of delay: e^(-θs) ≈ (1 - θs/2)/(1 + θs/2)
    # => add state x_d: dx_d/dt = (2/θ)*(u - x_d) - (2/θ)*u
    #    delay_out = -u + (2/θ)*x_d  ... simplified

    dt = t_final / n_pts
    t_arr = np.linspace(0.0, t_final, n_pts)

    y = 0.0         # process output
    integral = 0.0
    e_prev = 0.0
    u_arr = np.zeros(n_pts)
    y_arr = np.zeros(n_pts)
    e_arr = np.zeros(n_pts)

    # Simple discrete Euler simulation with dead-time buffer
    delay_steps = max(1, int(theta / dt))
    u_history = [0.0] * (delay_steps + 2)

    for i, t in enumerate(t_arr):
        sp = setpoint
        e = sp - y
        integral += e * dt
        de = (e - e_prev) / dt if i > 0 else 0.0
        e_prev = e

        # PID output
        u = Kc * (e + (1.0 / Ti if Ti > 1e-12 else 0.0) * integral + Td * de)
        u = max(-10.0, min(10.0, u))  # anti-windup clamp

        # Disturbance
        d = disturbance if t >= dist_time else 0.0

        # Delayed u
        u_history.append(u)
        u_delayed = u_history[-delay_steps - 1]

        # FOPDT discrete: dy = dt/tau * (Kp*(u_delayed + d) - y)
        y += dt / tau * (Kp * (u_delayed + d) - y)

        y_arr[i] = y
        u_arr[i] = u
        e_arr[i] = e

    # Performance metrics
    iae = float(np.trapz(np.abs(e_arr), t_arr))
    ise = float(np.trapz(e_arr**2, t_arr))
    overshoot = float(max(0.0, (np.max(y_arr) - setpoint) / setpoint * 100))
    settle_mask = np.abs(y_arr - setpoint) < 0.02 * abs(setpoint)
    settle_t = float(t_arr[np.where(settle_mask)[0][-1]]) if settle_mask.any() else float("nan")

    return {
        "t": t_arr, "y": y_arr, "u": u_arr, "e": e_arr,
        "setpoint": setpoint,
        "IAE": iae, "ISE": ise,
        "overshoot_pct": overshoot,
        "settling_time": settle_t,
        "Kc": Kc, "Ti": Ti, "Td": Td,
    }


# ---------------------------------------------------------------------------
# 3. Controller Tuning Rules
# ---------------------------------------------------------------------------

def ziegler_nichols_open_loop(Kp_proc: float, tau: float, theta: float) -> dict:
    """Ziegler-Nichols open-loop (process reaction curve) tuning.
    Based on FOPDT model parameters.
    """
    # R = slope at inflection = Kp_proc / tau (for FOPDT)
    R = Kp_proc / tau
    L = theta

    results = {}
    if R * L > 1e-15:
        results["P"]   = {"Kc": 1.0 / (R * L)}
        results["PI"]  = {"Kc": 0.9 / (R * L), "Ti": 3.33 * L}
        results["PID"] = {"Kc": 1.2 / (R * L), "Ti": 2.0 * L, "Td": 0.5 * L}

    # ITAE tuning (Kaya 1999)
    results["ITAE_PI"]  = {
        "Kc": (0.586 / Kp_proc) * (theta / tau)**(-0.916),
        "Ti": tau / (1.03 - 0.165 * (theta / tau)),
    }
    results["ITAE_PID"] = {
        "Kc": (0.965 / Kp_proc) * (theta / tau)**(-0.85),
        "Ti": tau / (0.796 - 0.147 * (theta / tau)),
        "Td": 0.308 * tau * (theta / tau)**0.929,
    }

    # IMC-based PI (lambda = tau for moderate response)
    lam = tau  # closed-loop time constant = tau (can be user-specified)
    Kc_imc = tau / (Kp_proc * (lam + theta))
    results["IMC_PI"] = {"Kc": Kc_imc, "Ti": tau, "lambda": lam}

    return results


def ziegler_nichols_closed_loop(Kcu: float, Pu: float) -> dict:
    """Ziegler-Nichols closed-loop (ultimate gain) tuning."""
    return {
        "P":   {"Kc": 0.5 * Kcu},
        "PI":  {"Kc": 0.45 * Kcu, "Ti": Pu / 1.2},
        "PID": {"Kc": 0.6 * Kcu,  "Ti": Pu / 2.0, "Td": Pu / 8.0},
    }


# ---------------------------------------------------------------------------
# 4. Bode Plot / Frequency Response for FOPDT
# ---------------------------------------------------------------------------

def bode_fopdt(
    Kp: float, tau: float, theta: float,
    omega_min: float = 1e-2, omega_max: float = 1e2, n_pts: int = 400,
) -> dict:
    """Bode plot data for FOPDT open-loop G(jω) * C(jω) = Kp*exp(-jωθ)/(1+jωτ).
    (Without controller — process transfer function only.)
    Returns omega, magnitude_dB, phase_deg, GM, PM, crossover frequencies.
    """
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_pts)
    magnitude = Kp / np.sqrt(1.0 + (omega * tau)**2)
    phase_rad = -np.arctan(omega * tau) - omega * theta  # dead time contribution
    phase_deg = np.degrees(phase_rad)
    magnitude_dB = 20 * np.log10(magnitude)

    # Gain crossover (|G| = 1 → 0 dB)
    gc_idx = np.argmin(np.abs(magnitude - 1.0))
    omega_gc = float(omega[gc_idx])
    PM = float(180.0 + phase_deg[gc_idx])  # phase margin

    # Phase crossover (phase = -180°)
    sign_cross = np.sign(phase_deg + 180.0)
    pc_idx = np.where(np.diff(sign_cross) != 0)[0]
    if len(pc_idx) > 0:
        omega_pc = float(omega[pc_idx[0]])
        GM_dB = float(-magnitude_dB[pc_idx[0]])
    else:
        omega_pc = float("nan")
        GM_dB = float("inf")

    return {
        "omega": omega,
        "magnitude_dB": magnitude_dB,
        "phase_deg": phase_deg,
        "GM_dB": GM_dB,
        "PM_deg": PM,
        "omega_gc": omega_gc,
        "omega_pc": omega_pc,
    }


def bode_pid_loop(
    Kp: float, tau: float, theta: float,
    Kc: float, Ti: float, Td: float = 0.0,
    omega_min: float = 1e-2, omega_max: float = 1e2, n_pts: int = 400,
) -> dict:
    """Bode plot of closed-loop open-loop transfer function L(jω) = C(jω)*G(jω)."""
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_pts)
    G_mag = Kp / np.sqrt(1.0 + (omega * tau)**2)
    G_phase = -np.arctan(omega * tau) - omega * theta

    # PID in parallel form: C = Kc*(1 + 1/(Ti*s) + Td*s)
    # |C(jω)| = Kc * sqrt(1 + (1/(Ti*ω) - Td*ω)^2 ... wait actually:
    # C(jω) = Kc*(1 + 1/(jω*Ti) + jω*Td) = Kc*(1 + Td*jω - j/(Ti*ω))
    # Real part = Kc*(1), Imag part = Kc*(Td*ω - 1/(Ti*ω))
    C_real = Kc * np.ones_like(omega)
    C_imag = Kc * (Td * omega - (1.0 / (Ti * omega) if Ti > 1e-12 else 0.0))
    C_mag = np.sqrt(C_real**2 + C_imag**2)
    C_phase = np.arctan2(C_imag, C_real)

    L_mag = G_mag * C_mag
    L_phase_deg = np.degrees(G_phase + C_phase)
    L_mag_dB = 20 * np.log10(np.maximum(L_mag, 1e-15))

    # Gain margin
    sign_cross = np.sign(L_phase_deg + 180.0)
    pc_idx = np.where(np.diff(sign_cross) != 0)[0]
    if len(pc_idx) > 0:
        GM_dB = float(-L_mag_dB[pc_idx[0]])
        omega_pc = float(omega[pc_idx[0]])
    else:
        GM_dB = float("inf")
        omega_pc = float("nan")

    # Phase margin
    gc_idx = np.argmin(np.abs(L_mag - 1.0))
    PM = float(180.0 + L_phase_deg[gc_idx])
    omega_gc = float(omega[gc_idx])

    return {
        "omega": omega,
        "L_mag_dB": L_mag_dB,
        "L_phase_deg": L_phase_deg,
        "GM_dB": GM_dB, "PM_deg": PM,
        "omega_gc": omega_gc, "omega_pc": omega_pc,
    }


# ---------------------------------------------------------------------------
# 5. Process Reaction Curve fitting (Smith method)
# ---------------------------------------------------------------------------

def fit_fopdt_from_prm(t_arr: np.ndarray, y_arr: np.ndarray, delta_u: float) -> dict:
    """Estimate Kp, tau, theta from step-test data using the 28.3%/63.2% method."""
    y_ss = float(y_arr[-1])
    Kp_est = y_ss / delta_u if abs(delta_u) > 1e-12 else 1.0

    y28 = 0.283 * y_ss
    y63 = 0.632 * y_ss

    t28 = float(np.interp(y28, y_arr, t_arr)) if y28 <= y_ss else float("nan")
    t63 = float(np.interp(y63, y_arr, t_arr)) if y63 <= y_ss else float("nan")

    if not math.isnan(t28) and not math.isnan(t63):
        tau_est = 1.5 * (t63 - t28)
        theta_est = t63 - tau_est
    else:
        tau_est = float("nan")
        theta_est = float("nan")

    return {
        "Kp": Kp_est, "tau": tau_est, "theta": max(0.0, theta_est),
        "t28": t28, "t63": t63, "y_ss": y_ss,
    }
