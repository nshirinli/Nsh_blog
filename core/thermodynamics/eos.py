from __future__ import annotations

import math
import numpy as np

from core.thermodynamics.ideal_gas import R


def _unique_sorted_real_roots(coefficients, tol: float = 1e-9) -> list[float]:
    roots = np.roots(coefficients)
    real_roots = sorted(float(root.real) for root in roots if abs(root.imag) < tol)

    unique_roots: list[float] = []
    for root in real_roots:
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-7:
            unique_roots.append(root)

    return unique_roots


def _select_phase_roots(roots: list[float]) -> tuple[float, float]:
    positive_roots = [root for root in roots if root > 0]

    if not positive_roots:
        raise ValueError("No positive real compressibility roots were found.")

    z_liquid = min(positive_roots)
    z_vapor = max(positive_roots)
    return z_liquid, z_vapor


def van_der_waals_parameters(Tc: float, Pc: float) -> tuple[float, float]:
    if Tc <= 0 or Pc <= 0:
        raise ValueError("Critical temperature and pressure must be positive.")

    a = 27.0 * (R ** 2) * (Tc ** 2) / (64.0 * Pc)
    b = R * Tc / (8.0 * Pc)
    return a, b


def solve_van_der_waals_state(T: float, P: float, Tc: float, Pc: float) -> dict:
    if T <= 0 or P <= 0:
        raise ValueError("Temperature and pressure must be positive.")

    a, b = van_der_waals_parameters(Tc, Pc)
    A = a * P / (R ** 2 * T ** 2)
    B = b * P / (R * T)

    coefficients = [
        1.0,
        -(1.0 + B),
        A,
        -A * B,
    ]

    roots = _unique_sorted_real_roots(coefficients)
    z_liquid, z_vapor = _select_phase_roots(roots)

    return {
        "model": "van der Waals",
        "A": A,
        "B": B,
        "a": a,
        "b": b,
        "roots": roots,
        "z_liquid": z_liquid,
        "z_vapor": z_vapor,
        "v_liquid_m3_per_mol": z_liquid * R * T / P,
        "v_vapor_m3_per_mol": z_vapor * R * T / P,
    }


def peng_robinson_parameters(T: float, Tc: float, Pc: float, omega: float) -> dict:
    if T <= 0 or Tc <= 0 or Pc <= 0:
        raise ValueError("T, Tc, and Pc must be positive.")

    Tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * (omega ** 2)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(Tr))) ** 2

    a_c = 0.45724 * (R ** 2) * (Tc ** 2) / Pc
    a = a_c * alpha
    b = 0.07780 * R * Tc / Pc

    A = a * P_dummy(1.0)  # placeholder pattern not used directly
    return {
        "Tr": Tr,
        "kappa": kappa,
        "alpha": alpha,
        "a_c": a_c,
        "a": a,
        "b": b,
    }


def P_dummy(value: float) -> float:
    return value


def solve_peng_robinson_state(T: float, P: float, Tc: float, Pc: float, omega: float) -> dict:
    if T <= 0 or P <= 0:
        raise ValueError("Temperature and pressure must be positive.")

    params = peng_robinson_parameters(T, Tc, Pc, omega)
    a = params["a"]
    b = params["b"]

    A = a * P / (R ** 2 * T ** 2)
    B = b * P / (R * T)

    coefficients = [
        1.0,
        -(1.0 - B),
        A - 3.0 * B ** 2 - 2.0 * B,
        -(A * B - B ** 2 - B ** 3),
    ]

    roots = _unique_sorted_real_roots(coefficients)
    z_liquid, z_vapor = _select_phase_roots(roots)

    return {
        "model": "Peng–Robinson",
        "A": A,
        "B": B,
        "a": a,
        "b": b,
        "alpha": params["alpha"],
        "kappa": params["kappa"],
        "roots": roots,
        "z_liquid": z_liquid,
        "z_vapor": z_vapor,
        "v_liquid_m3_per_mol": z_liquid * R * T / P,
        "v_vapor_m3_per_mol": z_vapor * R * T / P,
    }


def peng_robinson_fugacity_coefficient(
    T: float,
    P: float,
    Tc: float,
    Pc: float,
    omega: float,
    Z: float,
) -> dict:
    state = solve_peng_robinson_state(T, P, Tc, Pc, omega)
    A = state["A"]
    B = state["B"]

    if Z <= B:
        raise ValueError("Selected Z root is not physically valid for fugacity calculation.")

    sqrt2 = math.sqrt(2.0)
    numerator = Z + (1.0 + sqrt2) * B
    denominator = Z + (1.0 - sqrt2) * B

    if denominator <= 0 or numerator <= 0 or (Z - B) <= 0:
        raise ValueError("Invalid logarithm argument in fugacity calculation.")

    ln_phi = (
        Z
        - 1.0
        - math.log(Z - B)
        - (A / (2.0 * sqrt2 * B)) * math.log(numerator / denominator)
    )
    phi = math.exp(ln_phi)

    return {
        "phi": phi,
        "ln_phi": ln_phi,
    }


def solve_eos_state(
    eos_name: str,
    T: float,
    P: float,
    Tc: float,
    Pc: float,
    omega: float = 0.0,
) -> dict:
    eos_key = eos_name.strip().lower()

    if eos_key in {"vdw", "van der waals", "van_der_waals"}:
        return solve_van_der_waals_state(T=T, P=P, Tc=Tc, Pc=Pc)

    if eos_key in {"pr", "peng-robinson", "peng robinson", "peng_robinson"}:
        state = solve_peng_robinson_state(T=T, P=P, Tc=Tc, Pc=Pc, omega=omega)

        try:
            vapor_phi = peng_robinson_fugacity_coefficient(
                T=T,
                P=P,
                Tc=Tc,
                Pc=Pc,
                omega=omega,
                Z=state["z_vapor"],
            )
            liquid_phi = peng_robinson_fugacity_coefficient(
                T=T,
                P=P,
                Tc=Tc,
                Pc=Pc,
                omega=omega,
                Z=state["z_liquid"],
            )
            state["phi_vapor"] = vapor_phi["phi"]
            state["ln_phi_vapor"] = vapor_phi["ln_phi"]
            state["phi_liquid"] = liquid_phi["phi"]
            state["ln_phi_liquid"] = liquid_phi["ln_phi"]
        except ValueError:
            state["phi_vapor"] = None
            state["ln_phi_vapor"] = None
            state["phi_liquid"] = None
            state["ln_phi_liquid"] = None

        return state

    raise ValueError("Unsupported EOS. Use 'van der Waals' or 'Peng-Robinson'.")


def eos_z_curve(
    eos_name: str,
    T: float,
    Tc: float,
    Pc: float,
    omega: float,
    P_min: float,
    P_max: float,
    num_points: int = 120,
):
    if P_min <= 0 or P_max <= 0:
        raise ValueError("Pressure range must be positive.")
    if P_max <= P_min:
        raise ValueError("Maximum pressure must be greater than minimum pressure.")

    pressures = np.linspace(P_min, P_max, num_points)
    z_vapor_values = []
    z_liquid_values = []

    for pressure in pressures:
        state = solve_eos_state(
            eos_name=eos_name,
            T=T,
            P=pressure,
            Tc=Tc,
            Pc=Pc,
            omega=omega,
        )
        z_vapor_values.append(state["z_vapor"])
        z_liquid_values.append(state["z_liquid"])

    return pressures, np.array(z_vapor_values), np.array(z_liquid_values)