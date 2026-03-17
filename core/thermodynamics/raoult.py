from __future__ import annotations

import numpy as np

from core.thermodynamics.vapour_pressure import antoine_pressure


def _validate_fraction(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1.")


def _bisection(function, low: float, high: float, tol: float = 1e-6, max_iter: int = 200) -> float:
    f_low = function(low)
    f_high = function(high)

    if abs(f_low) < tol:
        return low
    if abs(f_high) < tol:
        return high

    if f_low * f_high > 0:
        raise ValueError(
            "Root could not be bracketed. Expand the temperature range or check the inputs."
        )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = function(mid)

        if abs(f_mid) < tol or abs(high - low) < tol:
            return mid

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def bubble_pressure_binary(
    x1: float,
    T_celsius: float,
    constants_1: tuple[float, float, float],
    constants_2: tuple[float, float, float],
):
    _validate_fraction(x1, "x1")
    x2 = 1.0 - x1

    A1, B1, C1 = constants_1
    A2, B2, C2 = constants_2

    p_sat_1 = antoine_pressure(A1, B1, C1, T_celsius)
    p_sat_2 = antoine_pressure(A2, B2, C2, T_celsius)

    bubble_pressure = x1 * p_sat_1 + x2 * p_sat_2
    y1 = x1 * p_sat_1 / bubble_pressure if bubble_pressure > 0 else 0.0
    y2 = 1.0 - y1

    return {
        "P_bubble_mmhg": bubble_pressure,
        "y1": y1,
        "y2": y2,
        "P_sat_1_mmhg": p_sat_1,
        "P_sat_2_mmhg": p_sat_2,
    }


def dew_pressure_binary(
    y1: float,
    T_celsius: float,
    constants_1: tuple[float, float, float],
    constants_2: tuple[float, float, float],
):
    _validate_fraction(y1, "y1")
    y2 = 1.0 - y1

    A1, B1, C1 = constants_1
    A2, B2, C2 = constants_2

    p_sat_1 = antoine_pressure(A1, B1, C1, T_celsius)
    p_sat_2 = antoine_pressure(A2, B2, C2, T_celsius)

    denominator = y1 / p_sat_1 + y2 / p_sat_2
    if denominator <= 0:
        raise ValueError("Invalid dew-pressure denominator.")

    dew_pressure = 1.0 / denominator
    x1 = y1 * dew_pressure / p_sat_1
    x2 = 1.0 - x1

    return {
        "P_dew_mmhg": dew_pressure,
        "x1": x1,
        "x2": x2,
        "P_sat_1_mmhg": p_sat_1,
        "P_sat_2_mmhg": p_sat_2,
    }


def bubble_temperature_binary(
    x1: float,
    P_mmhg: float,
    constants_1: tuple[float, float, float],
    constants_2: tuple[float, float, float],
    t_low: float = -100.0,
    t_high: float = 250.0,
):
    if P_mmhg <= 0:
        raise ValueError("Pressure must be positive.")

    def objective(T_celsius: float) -> float:
        result = bubble_pressure_binary(x1, T_celsius, constants_1, constants_2)
        return result["P_bubble_mmhg"] - P_mmhg

    T_bubble = _bisection(objective, t_low, t_high)
    result = bubble_pressure_binary(x1, T_bubble, constants_1, constants_2)

    return {
        "T_bubble_celsius": T_bubble,
        "y1": result["y1"],
        "y2": result["y2"],
        "P_sat_1_mmhg": result["P_sat_1_mmhg"],
        "P_sat_2_mmhg": result["P_sat_2_mmhg"],
    }


def dew_temperature_binary(
    y1: float,
    P_mmhg: float,
    constants_1: tuple[float, float, float],
    constants_2: tuple[float, float, float],
    t_low: float = -100.0,
    t_high: float = 250.0,
):
    if P_mmhg <= 0:
        raise ValueError("Pressure must be positive.")

    def objective(T_celsius: float) -> float:
        result = dew_pressure_binary(y1, T_celsius, constants_1, constants_2)
        return result["P_dew_mmhg"] - P_mmhg

    T_dew = _bisection(objective, t_low, t_high)
    result = dew_pressure_binary(y1, T_dew, constants_1, constants_2)

    return {
        "T_dew_celsius": T_dew,
        "x1": result["x1"],
        "x2": result["x2"],
        "P_sat_1_mmhg": result["P_sat_1_mmhg"],
        "P_sat_2_mmhg": result["P_sat_2_mmhg"],
    }


def txy_curve_binary(
    P_mmhg: float,
    constants_1: tuple[float, float, float],
    constants_2: tuple[float, float, float],
    num_points: int = 41,
    t_low: float = -100.0,
    t_high: float = 250.0,
):
    if P_mmhg <= 0:
        raise ValueError("Pressure must be positive.")

    x_values = np.linspace(0.0, 1.0, num_points)
    bubble_temperatures = []
    dew_y_values = []

    for x1 in x_values:
        result = bubble_temperature_binary(
            x1=x1,
            P_mmhg=P_mmhg,
            constants_1=constants_1,
            constants_2=constants_2,
            t_low=t_low,
            t_high=t_high,
        )
        bubble_temperatures.append(result["T_bubble_celsius"])
        dew_y_values.append(result["y1"])

    return {
        "x1_bubble": x_values,
        "T_bubble_celsius": np.array(bubble_temperatures),
        "y1_dew": np.array(dew_y_values),
    }