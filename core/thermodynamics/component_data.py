from __future__ import annotations

COMPONENTS = {
    "Water": {
        "name": "Water",
        "formula": "H2O",
        "mw_g_per_mol": 18.01528,
        "normal_boiling_point_C": 100.0,
        "antoine": {
            "A": 8.07131,
            "B": 1730.63,
            "C": 233.426,
            "Tmin_C": 1.0,
            "Tmax_C": 100.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 647.096,
            "Pc_bar": 220.64,
            "omega": 0.344,
            "Zc": 0.229,
        },
    },
    "Methanol": {
        "name": "Methanol",
        "formula": "CH3OH",
        "mw_g_per_mol": 32.04186,
        "normal_boiling_point_C": 64.7,
        "antoine": {
            "A": 8.08097,
            "B": 1582.271,
            "C": 239.726,
            "Tmin_C": 15.0,
            "Tmax_C": 100.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 512.6,
            "Pc_bar": 80.97,
            "omega": 0.559,
            "Zc": 0.224,
        },
    },
    "Ethanol": {
        "name": "Ethanol",
        "formula": "C2H5OH",
        "mw_g_per_mol": 46.06844,
        "normal_boiling_point_C": 78.37,
        "antoine": {
            "A": 8.20417,
            "B": 1642.89,
            "C": 230.300,
            "Tmin_C": 0.0,
            "Tmax_C": 78.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 514.0,
            "Pc_bar": 61.37,
            "omega": 0.644,
            "Zc": 0.241,
        },
    },
    "Acetone": {
        "name": "Acetone",
        "formula": "C3H6O",
        "mw_g_per_mol": 58.07914,
        "normal_boiling_point_C": 56.05,
        "antoine": {
            "A": 7.02447,
            "B": 1161.0,
            "C": 224.0,
            "Tmin_C": -10.0,
            "Tmax_C": 80.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 508.1,
            "Pc_bar": 47.0,
            "omega": 0.304,
            "Zc": 0.233,
        },
    },
    "Benzene": {
        "name": "Benzene",
        "formula": "C6H6",
        "mw_g_per_mol": 78.11184,
        "normal_boiling_point_C": 80.1,
        "antoine": {
            "A": 6.90565,
            "B": 1211.033,
            "C": 220.79,
            "Tmin_C": 7.0,
            "Tmax_C": 104.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 562.02,
            "Pc_bar": 48.94,
            "omega": 0.212,
            "Zc": 0.268,
        },
    },
    "Toluene": {
        "name": "Toluene",
        "formula": "C7H8",
        "mw_g_per_mol": 92.13842,
        "normal_boiling_point_C": 110.6,
        "antoine": {
            "A": 6.95464,
            "B": 1344.8,
            "C": 219.48,
            "Tmin_C": 10.0,
            "Tmax_C": 190.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 591.75,
            "Pc_bar": 41.06,
            "omega": 0.263,
            "Zc": 0.264,
        },
    },
    "n-Hexane": {
        "name": "n-Hexane",
        "formula": "C6H14",
        "mw_g_per_mol": 86.17536,
        "normal_boiling_point_C": 68.7,
        "antoine": {
            "A": 6.8763,
            "B": 1171.53,
            "C": 224.366,
            "Tmin_C": -20.0,
            "Tmax_C": 100.0,
            "pressure_unit": "mmHg",
        },
        "critical": {
            "Tc_K": 507.6,
            "Pc_bar": 30.25,
            "omega": 0.301,
            "Zc": 0.266,
        },
    },
}


def list_component_names() -> list[str]:
    return sorted(COMPONENTS.keys())


def get_component(name: str) -> dict:
    if name in COMPONENTS:
        return COMPONENTS[name]

    lowered = name.strip().lower()
    for key, value in COMPONENTS.items():
        if key.lower() == lowered or value["name"].lower() == lowered:
            return value

    raise ValueError(f"Component '{name}' was not found in the database.")