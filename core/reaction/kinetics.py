import math


def first_order_rate(k: float, ca: float) -> float:
    return -k * ca


def nth_order_rate(k: float, ca: float, n: float) -> float:
    """dCA/dt for nth-order irreversible A -> products."""
    return -k * (max(ca, 0.0) ** n)


def arrhenius_k(A: float, Ea_J_per_mol: float, T_K: float) -> float:
    """k(T) = A * exp(-Ea / (R*T)).  R = 8.314 J/mol/K, T in Kelvin."""
    R = 8.314
    return A * math.exp(-Ea_J_per_mol / (R * T_K))
