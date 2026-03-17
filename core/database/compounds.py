"""Chemical compound database with physical, thermodynamic and transport properties.

Data sources: Perry's Chemical Engineers' Handbook, NIST WebBook, Poling et al.
Antoine equation:  log10(P / mmHg) = A - B / (T + C),   T in °C
Cp polynomial:     Cp [J/(mol·K)] = a + b*T + c*T^2 + d*T^3   (T in K)
"""
from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Raw data dictionary
# ---------------------------------------------------------------------------
# Keys for each compound:
#   name, formula, cas, category,
#   MW (g/mol), Tb (K), Tm (K),
#   Tc (K), Pc (bar), Vc (cm³/mol), omega, Zc,
#   antoine (A, B, C),
#   cp (a, b, c, d),
#   dHf (kJ/mol, 298 K), dGf (kJ/mol, 298 K),
#   mu25 (viscosity mPa·s at 25°C, liquid),
#   rho25 (density kg/m³ at 25°C, liquid),
#   kth (thermal conductivity W/(m·K) at 25°C, liquid)

_COMPOUNDS: list[dict[str, Any]] = [
    # ── Hydrocarbons ─────────────────────────────────────────────────────────
    {
        "name": "Methane", "formula": "CH4", "cas": "74-82-8",
        "category": "Hydrocarbon (gas)",
        "MW": 16.043, "Tb": 111.7, "Tm": 90.7,
        "Tc": 190.6, "Pc": 46.1, "Vc": 99.0, "omega": 0.011, "Zc": 0.286,
        "antoine": (6.61184, 389.93, 266.00),
        "cp": (19.89, 5.024e-2, 1.269e-5, -11.01e-9),
        "dHf": -74.8, "dGf": -50.7,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Ethane", "formula": "C2H6", "cas": "74-84-0",
        "category": "Hydrocarbon (gas)",
        "MW": 30.069, "Tb": 184.6, "Tm": 90.4,
        "Tc": 305.3, "Pc": 48.7, "Vc": 148.0, "omega": 0.099, "Zc": 0.279,
        "antoine": (6.80896, 656.40, 256.00),
        "cp": (5.409, 1.781e-1, -6.938e-5, 8.713e-9),
        "dHf": -83.8, "dGf": -31.9,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Propane", "formula": "C3H8", "cas": "74-98-6",
        "category": "Hydrocarbon (gas)",
        "MW": 44.096, "Tb": 231.1, "Tm": 85.5,
        "Tc": 369.8, "Pc": 42.5, "Vc": 200.0, "omega": 0.153, "Zc": 0.276,
        "antoine": (6.80338, 804.00, 247.04),
        "cp": (-4.224, 3.063e-1, -1.586e-4, 3.215e-8),
        "dHf": -103.8, "dGf": -23.4,
        "mu25": None, "rho25": 493.0, "kth": None,
    },
    {
        "name": "n-Butane", "formula": "C4H10", "cas": "106-97-8",
        "category": "Hydrocarbon (gas)",
        "MW": 58.122, "Tb": 272.7, "Tm": 134.9,
        "Tc": 425.1, "Pc": 37.96, "Vc": 255.0, "omega": 0.200, "Zc": 0.274,
        "antoine": (6.80896, 935.86, 238.73),
        "cp": (9.487, 3.313e-1, -1.108e-4, -2.822e-9),
        "dHf": -126.2, "dGf": -17.0,
        "mu25": None, "rho25": 579.0, "kth": None,
    },
    {
        "name": "n-Pentane", "formula": "C5H12", "cas": "109-66-0",
        "category": "Hydrocarbon (liq)",
        "MW": 72.149, "Tb": 309.2, "Tm": 143.4,
        "Tc": 469.7, "Pc": 33.7, "Vc": 311.0, "omega": 0.251, "Zc": 0.268,
        "antoine": (6.85221, 1064.63, 232.00),
        "cp": ((-3.626), 4.873e-1, (-2.580e-4), 5.305e-8),
        "dHf": -146.4, "dGf": -8.65,
        "mu25": 0.224, "rho25": 621.0, "kth": 0.113,
    },
    {
        "name": "n-Hexane", "formula": "C6H14", "cas": "110-54-3",
        "category": "Hydrocarbon (liq)",
        "MW": 86.175, "Tb": 341.9, "Tm": 177.8,
        "Tc": 507.6, "Pc": 30.25, "Vc": 368.0, "omega": 0.301, "Zc": 0.264,
        "antoine": (6.87601, 1171.17, 224.41),
        "cp": (-4.413, 5.820e-1, (-3.119e-4), 6.494e-8),
        "dHf": -167.1, "dGf": -0.05,
        "mu25": 0.307, "rho25": 655.0, "kth": 0.120,
    },
    {
        "name": "n-Octane", "formula": "C8H18", "cas": "111-65-9",
        "category": "Hydrocarbon (liq)",
        "MW": 114.229, "Tb": 398.8, "Tm": 216.4,
        "Tc": 568.7, "Pc": 24.86, "Vc": 486.0, "omega": 0.399, "Zc": 0.259,
        "antoine": (6.91868, 1351.99, 209.155),
        "cp": (-6.096, 7.712e-1, (-4.195e-4), 8.855e-8),
        "dHf": -208.5, "dGf": 16.4,
        "mu25": 0.537, "rho25": 698.6, "kth": 0.128,
    },
    {
        "name": "Benzene", "formula": "C6H6", "cas": "71-43-2",
        "category": "Aromatic",
        "MW": 78.112, "Tb": 353.2, "Tm": 278.7,
        "Tc": 562.2, "Pc": 48.9, "Vc": 259.0, "omega": 0.212, "Zc": 0.271,
        "antoine": (6.90565, 1211.033, 220.790),
        "cp": (-33.92, 4.739e-1, (-3.017e-4), 7.130e-8),
        "dHf": 82.9, "dGf": 129.7,
        "mu25": 0.604, "rho25": 876.5, "kth": 0.148,
    },
    {
        "name": "Toluene", "formula": "C7H8", "cas": "108-88-3",
        "category": "Aromatic",
        "MW": 92.139, "Tb": 383.8, "Tm": 178.2,
        "Tc": 591.8, "Pc": 41.08, "Vc": 316.0, "omega": 0.263, "Zc": 0.264,
        "antoine": (6.95334, 1343.943, 219.377),
        "cp": (-24.35, 5.125e-1, (-2.765e-4), 4.911e-8),
        "dHf": 50.0, "dGf": 122.3,
        "mu25": 0.560, "rho25": 867.0, "kth": 0.138,
    },
    {
        "name": "Ethylbenzene", "formula": "C8H10", "cas": "100-41-4",
        "category": "Aromatic",
        "MW": 106.165, "Tb": 409.4, "Tm": 178.2,
        "Tc": 617.2, "Pc": 36.06, "Vc": 374.0, "omega": 0.304, "Zc": 0.263,
        "antoine": (6.95719, 1424.255, 213.206),
        "cp": (-43.10, 6.074e-1, (-3.374e-4), 7.528e-8),
        "dHf": 29.9, "dGf": 130.7,
        "mu25": 0.631, "rho25": 867.0, "kth": 0.130,
    },
    {
        "name": "Cyclohexane", "formula": "C6H12", "cas": "110-82-7",
        "category": "Cycloalkane",
        "MW": 84.159, "Tb": 353.9, "Tm": 279.7,
        "Tc": 553.6, "Pc": 40.73, "Vc": 308.0, "omega": 0.212, "Zc": 0.273,
        "antoine": (6.84498, 1203.526, 222.863),
        "cp": (-52.53, 6.309e-1, (-3.694e-4), 8.445e-8),
        "dHf": -156.4, "dGf": 26.8,
        "mu25": 0.894, "rho25": 779.0, "kth": 0.123,
    },
    # ── Alcohols ──────────────────────────────────────────────────────────────
    {
        "name": "Methanol", "formula": "CH4O", "cas": "67-56-1",
        "category": "Alcohol",
        "MW": 32.042, "Tb": 337.8, "Tm": 175.5,
        "Tc": 512.6, "Pc": 80.97, "Vc": 118.0, "omega": 0.564, "Zc": 0.224,
        "antoine": (7.87863, 1473.11, 230.00),
        "cp": (21.15, 7.092e-2, 2.587e-5, -2.852e-8),
        "dHf": -201.0, "dGf": -162.6,
        "mu25": 0.551, "rho25": 791.0, "kth": 0.202,
    },
    {
        "name": "Ethanol", "formula": "C2H6O", "cas": "64-17-5",
        "category": "Alcohol",
        "MW": 46.068, "Tb": 351.4, "Tm": 159.1,
        "Tc": 513.9, "Pc": 61.48, "Vc": 167.0, "omega": 0.645, "Zc": 0.240,
        "antoine": (8.11220, 1592.864, 226.184),
        "cp": (9.014, 2.141e-1, -8.390e-5, 1.373e-9),
        "dHf": -235.1, "dGf": -168.6,
        "mu25": 1.074, "rho25": 789.0, "kth": 0.167,
    },
    {
        "name": "1-Propanol", "formula": "C3H8O", "cas": "71-23-8",
        "category": "Alcohol",
        "MW": 60.095, "Tb": 370.3, "Tm": 147.0,
        "Tc": 536.8, "Pc": 51.75, "Vc": 219.0, "omega": 0.629, "Zc": 0.253,
        "antoine": (7.74416, 1437.686, 198.463),
        "cp": (2.470, 3.325e-1, -1.855e-4, 4.296e-8),
        "dHf": -255.1, "dGf": -162.5,
        "mu25": 1.945, "rho25": 803.0, "kth": 0.161,
    },
    {
        "name": "Isopropanol", "formula": "C3H8O", "cas": "67-63-0",
        "category": "Alcohol",
        "MW": 60.095, "Tb": 355.4, "Tm": 185.3,
        "Tc": 508.3, "Pc": 47.62, "Vc": 220.0, "omega": 0.665, "Zc": 0.248,
        "antoine": (8.11778, 1580.92, 219.61),
        "cp": (32.43, 1.885e-1, 6.406e-5, -9.261e-8),
        "dHf": -272.7, "dGf": -173.4,
        "mu25": 2.040, "rho25": 786.0, "kth": 0.145,
    },
    {
        "name": "Glycerol", "formula": "C3H8O3", "cas": "56-81-5",
        "category": "Alcohol",
        "MW": 92.094, "Tb": 563.0, "Tm": 291.4,
        "Tc": 726.0, "Pc": 66.9, "Vc": 255.0, "omega": 1.544, "Zc": 0.282,
        "antoine": (6.16, 2000.0, 100.0),   # approximate
        "cp": (54.35, 4.016e-1, -1.874e-4, 3.282e-8),
        "dHf": -669.6, "dGf": -477.1,
        "mu25": 945.0, "rho25": 1261.0, "kth": 0.285,
    },
    # ── Ketones & Aldehydes ───────────────────────────────────────────────────
    {
        "name": "Acetone", "formula": "C3H6O", "cas": "67-64-1",
        "category": "Ketone",
        "MW": 58.079, "Tb": 329.2, "Tm": 178.5,
        "Tc": 508.1, "Pc": 47.01, "Vc": 209.0, "omega": 0.307, "Zc": 0.232,
        "antoine": (7.02447, 1161.0, 224.0),
        "cp": (6.301, 2.606e-1, -1.253e-4, 2.038e-8),
        "dHf": -248.4, "dGf": -155.4,
        "mu25": 0.306, "rho25": 791.0, "kth": 0.161,
    },
    {
        "name": "Acetaldehyde", "formula": "C2H4O", "cas": "75-07-0",
        "category": "Aldehyde",
        "MW": 44.052, "Tb": 293.2, "Tm": 150.2,
        "Tc": 461.0, "Pc": 55.5, "Vc": 154.0, "omega": 0.303, "Zc": 0.220,
        "antoine": (7.05409, 1070.60, 236.00),
        "cp": (7.72, 1.924e-1, -9.915e-5, 2.098e-8),
        "dHf": -165.6, "dGf": -133.0,
        "mu25": 0.222, "rho25": 788.0, "kth": 0.170,
    },
    # ── Acids ─────────────────────────────────────────────────────────────────
    {
        "name": "Acetic Acid", "formula": "C2H4O2", "cas": "64-19-7",
        "category": "Carboxylic Acid",
        "MW": 60.052, "Tb": 391.1, "Tm": 289.8,
        "Tc": 592.7, "Pc": 57.86, "Vc": 171.0, "omega": 0.467, "Zc": 0.200,
        "antoine": (7.38782, 1533.313, 222.309),
        "cp": (4.836, 2.551e-1, -1.321e-4, 2.567e-8),
        "dHf": -432.8, "dGf": -374.0,
        "mu25": 1.056, "rho25": 1049.0, "kth": 0.158,
    },
    {
        "name": "Formic Acid", "formula": "CH2O2", "cas": "64-18-6",
        "category": "Carboxylic Acid",
        "MW": 46.026, "Tb": 373.7, "Tm": 281.5,
        "Tc": 588.0, "Pc": 57.34, "Vc": 115.9, "omega": 0.473, "Zc": 0.135,
        "antoine": (7.58100, 1699.0, 232.0),
        "cp": (29.24, 5.028e-2, 4.164e-5, -2.091e-8),
        "dHf": -362.6, "dGf": -335.7,
        "mu25": 1.607, "rho25": 1220.0, "kth": 0.267,
    },
    # ── Ethers & Esters ───────────────────────────────────────────────────────
    {
        "name": "Diethyl Ether", "formula": "C4H10O", "cas": "60-29-7",
        "category": "Ether",
        "MW": 74.122, "Tb": 307.6, "Tm": 156.9,
        "Tc": 466.7, "Pc": 36.4, "Vc": 280.0, "omega": 0.281, "Zc": 0.262,
        "antoine": (6.78990, 994.195, 220.0),
        "cp": (0.726, 4.493e-1, (-2.254e-4), 4.196e-8),
        "dHf": -252.1, "dGf": -122.2,
        "mu25": 0.233, "rho25": 713.0, "kth": 0.138,
    },
    {
        "name": "Ethyl Acetate", "formula": "C4H8O2", "cas": "141-78-6",
        "category": "Ester",
        "MW": 88.105, "Tb": 350.2, "Tm": 189.6,
        "Tc": 523.3, "Pc": 38.3, "Vc": 286.0, "omega": 0.363, "Zc": 0.252,
        "antoine": (7.09808, 1238.71, 217.0),
        "cp": (-15.82, 5.012e-1, (-2.783e-4), 6.061e-8),
        "dHf": -463.3, "dGf": -318.1,
        "mu25": 0.441, "rho25": 900.0, "kth": 0.152,
    },
    # ── Nitrogen compounds ────────────────────────────────────────────────────
    {
        "name": "Ammonia", "formula": "NH3", "cas": "7664-41-7",
        "category": "Inorganic",
        "MW": 17.031, "Tb": 239.7, "Tm": 195.4,
        "Tc": 405.7, "Pc": 112.8, "Vc": 72.5, "omega": 0.253, "Zc": 0.242,
        "antoine": (7.36050, 926.132, 240.17),
        "cp": (19.99, 4.992e-2, -1.488e-5, 1.789e-9),
        "dHf": -46.1, "dGf": -16.5,
        "mu25": None, "rho25": 610.0, "kth": None,
    },
    {
        "name": "Aniline", "formula": "C6H7N", "cas": "62-53-3",
        "category": "Amine",
        "MW": 93.128, "Tb": 457.6, "Tm": 267.1,
        "Tc": 699.0, "Pc": 53.09, "Vc": 274.0, "omega": 0.382, "Zc": 0.250,
        "antoine": (7.32010, 1731.515, 206.049),
        "cp": (-11.01, 4.861e-1, (-2.920e-4), 6.978e-8),
        "dHf": 87.5, "dGf": 166.7,
        "mu25": 3.71, "rho25": 1022.0, "kth": 0.173,
    },
    # ── Water & Inorganics ────────────────────────────────────────────────────
    {
        "name": "Water", "formula": "H2O", "cas": "7732-18-5",
        "category": "Inorganic",
        "MW": 18.015, "Tb": 373.15, "Tm": 273.15,
        "Tc": 647.1, "Pc": 220.64, "Vc": 56.0, "omega": 0.345, "Zc": 0.229,
        "antoine": (8.07131, 1730.63, 233.426),
        "cp": (32.24, 1.924e-3, 1.055e-5, -3.596e-9),
        "dHf": -241.8, "dGf": -228.6,
        "mu25": 0.890, "rho25": 997.0, "kth": 0.607,
    },
    {
        "name": "Hydrogen", "formula": "H2", "cas": "1333-74-0",
        "category": "Inorganic (gas)",
        "MW": 2.016, "Tb": 20.3, "Tm": 14.0,
        "Tc": 33.2, "Pc": 13.13, "Vc": 64.3, "omega": -0.216, "Zc": 0.306,
        "antoine": (5.82060, 71.41, 280.42),
        "cp": (29.11, -1.916e-3, 4.003e-6, -8.704e-10),
        "dHf": 0.0, "dGf": 0.0,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Nitrogen", "formula": "N2", "cas": "7727-37-9",
        "category": "Inorganic (gas)",
        "MW": 28.014, "Tb": 77.4, "Tm": 63.2,
        "Tc": 126.2, "Pc": 34.0, "Vc": 89.5, "omega": 0.040, "Zc": 0.289,
        "antoine": (6.49457, 255.68, 266.55),
        "cp": (29.11, -1.916e-3, 4.003e-6, -8.704e-10),
        "dHf": 0.0, "dGf": 0.0,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Oxygen", "formula": "O2", "cas": "7782-44-7",
        "category": "Inorganic (gas)",
        "MW": 32.000, "Tb": 90.2, "Tm": 54.4,
        "Tc": 154.6, "Pc": 50.43, "Vc": 73.4, "omega": 0.022, "Zc": 0.288,
        "antoine": (6.69144, 319.01, 266.70),
        "cp": (29.96, 4.184e-3, -1.674e-6, 3.640e-10),
        "dHf": 0.0, "dGf": 0.0,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Carbon Dioxide", "formula": "CO2", "cas": "124-38-9",
        "category": "Inorganic (gas)",
        "MW": 44.010, "Tb": 194.7, "Tm": 216.6,
        "Tc": 304.2, "Pc": 73.83, "Vc": 94.0, "omega": 0.225, "Zc": 0.274,
        "antoine": (6.81228, 1301.679, 3.494),
        "cp": (19.80, 7.344e-2, -5.602e-5, 1.715e-8),
        "dHf": -393.5, "dGf": -394.4,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Carbon Monoxide", "formula": "CO", "cas": "630-08-0",
        "category": "Inorganic (gas)",
        "MW": 28.010, "Tb": 81.7, "Tm": 68.1,
        "Tc": 132.9, "Pc": 34.99, "Vc": 93.1, "omega": 0.048, "Zc": 0.295,
        "antoine": (6.69144, 319.01, 266.70),
        "cp": (29.11, -1.916e-3, 4.003e-6, -8.704e-10),
        "dHf": -110.5, "dGf": -137.2,
        "mu25": None, "rho25": None, "kth": None,
    },
    {
        "name": "Sulfur Dioxide", "formula": "SO2", "cas": "7446-09-5",
        "category": "Inorganic (gas)",
        "MW": 64.065, "Tb": 263.1, "Tm": 197.7,
        "Tc": 430.8, "Pc": 78.84, "Vc": 122.0, "omega": 0.245, "Zc": 0.269,
        "antoine": (7.28220, 999.90, 237.19),
        "cp": (25.78, 5.795e-2, -3.812e-5, 8.612e-9),
        "dHf": -296.8, "dGf": -300.2,
        "mu25": None, "rho25": 1460.0, "kth": None,
    },
    {
        "name": "Hydrogen Chloride", "formula": "HCl", "cas": "7647-01-0",
        "category": "Inorganic (gas)",
        "MW": 36.461, "Tb": 188.1, "Tm": 158.9,
        "Tc": 324.7, "Pc": 82.63, "Vc": 81.0, "omega": 0.132, "Zc": 0.249,
        "antoine": (7.14480, 744.485, 258.45),
        "cp": (30.17, (-7.201e-3), 1.246e-5, (-3.898e-9)),
        "dHf": -92.3, "dGf": -95.3,
        "mu25": None, "rho25": None, "kth": None,
    },
    # ── Chlorinated ───────────────────────────────────────────────────────────
    {
        "name": "Chloroform", "formula": "CHCl3", "cas": "67-66-3",
        "category": "Halogenated",
        "MW": 119.378, "Tb": 334.3, "Tm": 209.6,
        "Tc": 536.4, "Pc": 54.72, "Vc": 239.0, "omega": 0.218, "Zc": 0.293,
        "antoine": (6.90328, 1163.03, 227.4),
        "cp": (24.00, 1.892e-1, -1.841e-4, 6.657e-8),
        "dHf": -103.1, "dGf": -70.3,
        "mu25": 0.537, "rho25": 1492.0, "kth": 0.117,
    },
    {
        "name": "Dichloromethane", "formula": "CH2Cl2", "cas": "75-09-2",
        "category": "Halogenated",
        "MW": 84.932, "Tb": 312.9, "Tm": 178.0,
        "Tc": 510.0, "Pc": 60.8, "Vc": 185.0, "omega": 0.199, "Zc": 0.265,
        "antoine": (7.09070, 1138.91, 231.49),
        "cp": (20.89, 1.621e-1, -1.488e-4, 5.210e-8),
        "dHf": -95.3, "dGf": -68.9,
        "mu25": 0.425, "rho25": 1325.0, "kth": 0.140,
    },
    # ── Polymers / monomers ───────────────────────────────────────────────────
    {
        "name": "Styrene", "formula": "C8H8", "cas": "100-42-5",
        "category": "Monomer",
        "MW": 104.149, "Tb": 418.3, "Tm": 242.5,
        "Tc": 636.0, "Pc": 38.4, "Vc": 352.0, "omega": 0.296, "Zc": 0.256,
        "antoine": (6.92409, 1445.58, 209.44),
        "cp": (-28.25, 5.610e-1, (-3.245e-4), 7.192e-8),
        "dHf": 103.8, "dGf": 202.5,
        "mu25": 0.765, "rho25": 906.0, "kth": 0.143,
    },
    {
        "name": "Vinyl Chloride", "formula": "C2H3Cl", "cas": "75-01-4",
        "category": "Monomer",
        "MW": 62.499, "Tb": 259.8, "Tm": 119.4,
        "Tc": 432.0, "Pc": 56.0, "Vc": 169.0, "omega": 0.100, "Zc": 0.265,
        "antoine": (6.9, 1000.0, 230.0),
        "cp": (5.10, 1.966e-1, (-1.259e-4), 3.161e-8),
        "dHf": -94.1, "dGf": -72.9,
        "mu25": None, "rho25": 910.0, "kth": None,
    },
    # ── Refrigerants ─────────────────────────────────────────────────────────
    {
        "name": "R-134a (HFC-134a)", "formula": "C2H2F4", "cas": "811-97-2",
        "category": "Refrigerant",
        "MW": 102.03, "Tb": 247.1, "Tm": 172.0,
        "Tc": 374.2, "Pc": 40.6, "Vc": 198.0, "omega": 0.327, "Zc": 0.260,
        "antoine": (7.11, 1000.0, 230.0),
        "cp": (19.47, 1.979e-1, -1.280e-4, 3.120e-8),
        "dHf": -901.0, "dGf": -839.0,
        "mu25": None, "rho25": 1206.0, "kth": None,
    },
]

# ---------------------------------------------------------------------------
# Index for fast lookup
# ---------------------------------------------------------------------------
_BY_NAME: dict[str, dict] = {c["name"].lower(): c for c in _COMPOUNDS}
_BY_FORMULA: dict[str, list[dict]] = {}
_BY_CATEGORY: dict[str, list[dict]] = {}

for _c in _COMPOUNDS:
    _BY_FORMULA.setdefault(_c["formula"].lower(), []).append(_c)
    _BY_CATEGORY.setdefault(_c["category"], []).append(_c)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_compounds() -> list[dict]:
    """Return all compounds."""
    return list(_COMPOUNDS)


def list_categories() -> list[str]:
    """Return all unique categories (sorted)."""
    return sorted(_BY_CATEGORY.keys())


def search(query: str, category: str | None = None) -> list[dict]:
    """Search by name or formula substring (case-insensitive)."""
    q = query.strip().lower()
    results = []
    for c in _COMPOUNDS:
        if category and c["category"] != category:
            continue
        if q in c["name"].lower() or q in c["formula"].lower() or q in c.get("cas", ""):
            results.append(c)
    return results


def get_compound(name: str) -> dict | None:
    """Return compound dict by exact name (case-insensitive)."""
    return _BY_NAME.get(name.strip().lower())


def get_properties_text(name: str) -> str:
    """Format a compound's properties as a readable text report."""
    c = get_compound(name)
    if c is None:
        return f"Compound '{name}' not found in database."

    def _fmt(v, unit="", decimals=4):
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f} {unit}".strip()

    A, B, C = c["antoine"]
    a, b, cc, d = c["cp"]

    lines = [
        f"{'='*55}",
        f"  {c['name']}   ({c['formula']})   CAS: {c.get('cas', 'N/A')}",
        f"  Category: {c['category']}",
        f"{'='*55}",
        "",
        "Molecular",
        f"  Molecular weight MW     = {c['MW']:.4f} g/mol",
        "",
        "Phase-change temperatures",
        f"  Normal boiling point Tb = {c['Tb']:.2f} K  ({c['Tb']-273.15:.2f} °C)",
        f"  Melting point       Tm  = {c['Tm']:.2f} K  ({c['Tm']-273.15:.2f} °C)",
        "",
        "Critical properties",
        f"  Tc    = {c['Tc']:.2f} K  ({c['Tc']-273.15:.2f} °C)",
        f"  Pc    = {c['Pc']:.4f} bar  ({c['Pc']*100:.2f} kPa)",
        f"  Vc    = {c['Vc']:.2f} cm³/mol",
        f"  ω     = {c['omega']:.4f}  (acentric factor)",
        f"  Zc    = {c['Zc']:.4f}",
        "",
        "Antoine equation  [log₁₀(P/mmHg) = A − B/(T+C), T in °C]",
        f"  A = {A:.5f},  B = {B:.3f},  C = {C:.3f}",
        f"  Valid approx. near Tb ({c['Tb']-273.15:.1f} °C)",
        "",
        "Cp polynomial  [Cp = a + bT + cT² + dT³,  J/(mol·K)]",
        f"  a = {a:.4g},  b = {b:.4g},  c = {cc:.4g},  d = {d:.4g}",
        "",
        "Standard enthalpies at 298 K",
        f"  ΔHf° = {_fmt(c['dHf'], 'kJ/mol')}",
        f"  ΔGf° = {_fmt(c['dGf'], 'kJ/mol')}",
        "",
        "Transport & physical (liquid at ~25°C)",
        f"  Viscosity       μ   = {_fmt(c['mu25'], 'mPa·s')}",
        f"  Density         ρ   = {_fmt(c['rho25'], 'kg/m³')}",
        f"  Therm. conduct. k   = {_fmt(c['kth'], 'W/(m·K)')}",
    ]
    return "\n".join(lines)


def compute_vapor_pressure(name: str, T_C: float) -> float | None:
    """Compute vapor pressure (mmHg) at T_C (°C) using Antoine equation."""
    c = get_compound(name)
    if c is None:
        return None
    A, B, C = c["antoine"]
    log_p = A - B / (T_C + C)
    return 10.0 ** log_p


def compute_cp(name: str, T_K: float) -> float | None:
    """Compute ideal-gas Cp [J/(mol·K)] at T_K using polynomial."""
    c = get_compound(name)
    if c is None:
        return None
    a, b, cc, d = c["cp"]
    return a + b * T_K + cc * T_K**2 + d * T_K**3


def compare_property(compounds: list[str], prop: str) -> dict:
    """Return a dict {name: value} for the given property key across compounds."""
    result = {}
    for name in compounds:
        c = get_compound(name)
        if c is not None:
            result[name] = c.get(prop)
    return result
