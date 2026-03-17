from __future__ import annotations

import numpy as np

R = 8.314462618  # J/(mol·K)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def ideal_gas_residual(T: float, P: float, V: float, n: float) -> float:
    _require_positive(T, "Temperature")
    _require_positive(P, "Pressure")
    _require_positive(V, "Volume")
    _require_positive(n, "Moles")
    return P * V - n * R * T


def compressibility_factor(T: float, P: float, V: float, n: float) -> float:
    _require_positive(T, "Temperature")
    _require_positive(P, "Pressure")
    _require_positive(V, "Volume")
    _require_positive(n, "Moles")
    return (P * V) / (n * R * T)


def solve_ideal_gas(
    T: float | None = None,
    P: float | None = None,
    V: float | None = None,
    n: float | None = None,
) -> dict:
    values = {"T": T, "P": P, "V": V, "n": n}
    missing = [key for key, value in values.items() if value is None]

    if len(missing) != 1:
        raise ValueError("Exactly one of T, P, V, or n must be left blank.")

    for key, value in values.items():
        if value is not None:
            _require_positive(value, key)

    solved_for = missing[0]

    if solved_for == "T":
        T = (P * V) / (n * R)
    elif solved_for == "P":
        P = (n * R * T) / V
    elif solved_for == "V":
        V = (n * R * T) / P
    elif solved_for == "n":
        n = (P * V) / (R * T)

    return {
        "T": float(T),
        "P": float(P),
        "V": float(V),
        "n": float(n),
        "solved_for": solved_for,
    }


def pressure_temperature_curve(
    V: float,
    n: float,
    T_min: float = 200.0,
    T_max: float = 800.0,
    num_points: int = 200,
):
    _require_positive(V, "Volume")
    _require_positive(n, "Moles")
    _require_positive(T_min, "Minimum temperature")
    _require_positive(T_max, "Maximum temperature")

    if T_max <= T_min:
        raise ValueError("Maximum temperature must be greater than minimum temperature.")

    temperatures = np.linspace(T_min, T_max, num_points)
    pressures = n * R * temperatures / V
    return temperatures, pressures