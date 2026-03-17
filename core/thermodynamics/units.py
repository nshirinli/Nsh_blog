from __future__ import annotations

PRESSURE_TO_PA = {
    "Pa": 1.0,
    "kPa": 1.0e3,
    "bar": 1.0e5,
    "atm": 101325.0,
    "mmHg": 133.32236842105263,
}

VOLUME_TO_M3 = {
    "m³": 1.0,
    "m^3": 1.0,
    "L": 1.0e-3,
    "mL": 1.0e-6,
}


def temperature_to_kelvin(value: float, unit: str) -> float:
    if unit == "K":
        return value
    if unit in {"°C", "C"}:
        return value + 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def temperature_from_kelvin(value: float, unit: str) -> float:
    if unit == "K":
        return value
    if unit in {"°C", "C"}:
        return value - 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def temperature_to_celsius(value: float, unit: str) -> float:
    if unit in {"°C", "C"}:
        return value
    if unit == "K":
        return value - 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def temperature_from_celsius(value: float, unit: str) -> float:
    if unit in {"°C", "C"}:
        return value
    if unit == "K":
        return value + 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")


def pressure_to_pascal(value: float, unit: str) -> float:
    if unit not in PRESSURE_TO_PA:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return value * PRESSURE_TO_PA[unit]


def pressure_from_pascal(value: float, unit: str) -> float:
    if unit not in PRESSURE_TO_PA:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return value / PRESSURE_TO_PA[unit]


def pressure_to_mmhg(value: float, unit: str) -> float:
    return pressure_from_pascal(pressure_to_pascal(value, unit), "mmHg")


def pressure_from_mmhg(value: float, unit: str) -> float:
    return pressure_from_pascal(pressure_to_pascal(value, "mmHg"), unit)


def volume_to_m3(value: float, unit: str) -> float:
    if unit not in VOLUME_TO_M3:
        raise ValueError(f"Unsupported volume unit: {unit}")
    return value * VOLUME_TO_M3[unit]


def volume_from_m3(value: float, unit: str) -> float:
    if unit not in VOLUME_TO_M3:
        raise ValueError(f"Unsupported volume unit: {unit}")
    return value / VOLUME_TO_M3[unit]