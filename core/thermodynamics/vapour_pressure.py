from __future__ import annotations

import math
import numpy as np

from core.thermodynamics.ideal_gas import R


def antoine_pressure(A: float, B: float, C: float, T_celsius: float) -> float:
    """
    Antoine equation:
    log10(P_mmHg) = A - B / (T_C + C)
    """
    denominator = T_celsius + C
    if abs(denominator) < 1e-12:
        raise ValueError("Invalid Antoine denominator: T + C is too close to zero.")
    return 10 ** (A - B / denominator)


def antoine_temperature(A: float, B: float, C: float, P_mmhg: float) -> float:
    """
    Inverse Antoine equation, returns temperature in Celsius.
    """
    if P_mmhg <= 0:
        raise ValueError("Pressure must be positive.")
    log_p = math.log10(P_mmhg)
    denominator = A - log_p
    if abs(denominator) < 1e-12:
        raise ValueError("Invalid Antoine inversion: A - log10(P) is too close to zero.")
    return B / denominator - C


def antoine_curve(
    A: float,
    B: float,
    C: float,
    T_min_celsius: float,
    T_max_celsius: float,
    num_points: int = 200,
):
    if T_max_celsius <= T_min_celsius:
        raise ValueError("Maximum temperature must be greater than minimum temperature.")

    temperatures_c = np.linspace(T_min_celsius, T_max_celsius, num_points)
    pressures_mmhg = np.array([antoine_pressure(A, B, C, t) for t in temperatures_c])
    return temperatures_c, pressures_mmhg


def estimate_heat_of_vaporization(
    T1_kelvin: float,
    P1_pascal: float,
    T2_kelvin: float,
    P2_pascal: float,
) -> float:
    """
    Clausius–Clapeyron estimate using two points:
    ln(P2/P1) = -ΔHvap/R * (1/T2 - 1/T1)
    """
    if T1_kelvin <= 0 or T2_kelvin <= 0:
        raise ValueError("Temperatures must be positive in Kelvin.")
    if P1_pascal <= 0 or P2_pascal <= 0:
        raise ValueError("Pressures must be positive.")

    denominator = (1.0 / T2_kelvin) - (1.0 / T1_kelvin)
    if abs(denominator) < 1e-15:
        raise ValueError("Temperatures are too close to estimate ΔHvap.")

    delta_h = -R * math.log(P2_pascal / P1_pascal) / denominator
    return delta_h


def clausius_clapeyron_line(
    T1_kelvin: float,
    P1_pascal: float,
    T2_kelvin: float,
    P2_pascal: float,
    num_points: int = 100,
):
    if T1_kelvin <= 0 or T2_kelvin <= 0:
        raise ValueError("Temperatures must be positive in Kelvin.")
    if P1_pascal <= 0 or P2_pascal <= 0:
        raise ValueError("Pressures must be positive.")

    inv_t_1 = 1.0 / T1_kelvin
    inv_t_2 = 1.0 / T2_kelvin

    x_values = np.linspace(min(inv_t_1, inv_t_2), max(inv_t_1, inv_t_2), num_points)

    slope = (math.log(P2_pascal) - math.log(P1_pascal)) / (inv_t_2 - inv_t_1)
    intercept = math.log(P1_pascal) - slope * inv_t_1
    y_values = slope * x_values + intercept

    return x_values, y_values