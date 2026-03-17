"""Process Economics engine for ChemEng Platform.

Covers:
  * Equipment purchased-cost correlations  (Turton et al., 2012)
  * CAPEX  — bare-module method & Lang-factor shortcut
  * OPEX   — standard chemical-plant cost breakdown
  * Cash-flow analysis  (straight-line or MACRS-5 depreciation)
  * Profitability indices  (NPV, IRR, payback period, ROI)
  * Sensitivity / tornado analysis

All monetary values are in USD.
CEPCI reference: 397 (Turton 2001 base year).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Equipment cost data  —  Turton et al. (2012), Table A.1
# log10(Cp0) = K1 + K2·log10(A) + K3·(log10(A))²
# Cp0 in USD at CEPCI = 397  (year 2001)
# ---------------------------------------------------------------------------
_EQUIPMENT: dict[str, dict] = {
    "Vessel — Vertical": {
        "param": "Volume (m³)", "A_min": 0.30, "A_max": 520.0,
        "K": (3.4974, 0.4485, 0.1074), "F_BM": 4.16,
    },
    "Vessel — Horizontal": {
        "param": "Volume (m³)", "A_min": 0.10, "A_max": 628.0,
        "K": (3.5565, 0.3776, 0.0905), "F_BM": 3.05,
    },
    "Heat Exchanger — Fixed Tube Sheet": {
        "param": "Area (m²)", "A_min": 10.0, "A_max": 1000.0,
        "K": (4.3247, -0.3030, 0.1634), "F_BM": 3.17,
    },
    "Heat Exchanger — Floating Head": {
        "param": "Area (m²)", "A_min": 10.0, "A_max": 1000.0,
        "K": (4.8306, -0.8509, 0.3187), "F_BM": 3.17,
    },
    "Heat Exchanger — U-Tube": {
        "param": "Area (m²)", "A_min": 10.0, "A_max": 1000.0,
        "K": (4.4646, -0.5188, 0.1705), "F_BM": 3.17,
    },
    "Pump — Centrifugal": {
        "param": "Power (kW)", "A_min": 1.0, "A_max": 300.0,
        "K": (3.3892, 0.0536, 0.1538), "F_BM": 3.30,
    },
    "Compressor — Centrifugal": {
        "param": "Power (kW)", "A_min": 75.0, "A_max": 30_000.0,
        "K": (2.2897, 1.3604, -0.1027), "F_BM": 6.29,
    },
    "Fired Heater": {
        "param": "Duty (kW)", "A_min": 1_000.0, "A_max": 100_000.0,
        "K": (7.3488, -1.1666, 0.2028), "F_BM": 2.19,
    },
    "Distillation Column (shell)": {
        "param": "Volume (m³)", "A_min": 0.30, "A_max": 520.0,
        "K": (3.4974, 0.4485, 0.1074), "F_BM": 4.16,
    },
    "Storage Tank — Fixed Roof": {
        "param": "Volume (m³)", "A_min": 1.0, "A_max": 1_000.0,
        "K": (4.8509, -0.3973, 0.1445), "F_BM": 1.50,
    },
}

_CEPCI_2001 = 397.0
_CEPCI_2024 = 830.0   # approximate

_MATERIAL_FACTORS: dict[str, float] = {
    "Carbon Steel": 1.00,
    "Stainless Steel 304": 1.70,
    "Stainless Steel 316": 2.10,
    "Hastelloy C": 5.40,
    "Monel": 3.60,
    "Titanium": 7.70,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def equipment_list() -> list[str]:
    return list(_EQUIPMENT.keys())


def material_list() -> list[str]:
    return list(_MATERIAL_FACTORS.keys())


# ---------------------------------------------------------------------------
# Tab 1 — Equipment cost
# ---------------------------------------------------------------------------

def equipment_cost(
    equip_type: str,
    size: float,
    cepci: float = _CEPCI_2024,
    material: str = "Carbon Steel",
    pressure_factor: float = 1.0,
    quantity: int = 1,
) -> dict:
    """Purchased & bare-module cost for a single equipment item.

    Returns
    -------
    dict with keys: Cp0, Cp_current, C_BM, F_BM, Fm, Fp, param_units
    """
    data = _EQUIPMENT.get(equip_type)
    if data is None:
        raise ValueError(f"Unknown equipment type: {equip_type!r}")

    K1, K2, K3 = data["K"]
    logA = np.log10(max(size, 1e-9))
    Cp0 = 10 ** (K1 + K2 * logA + K3 * logA ** 2)
    Cp_current = Cp0 * (cepci / _CEPCI_2001)

    Fm = _MATERIAL_FACTORS.get(material, 1.0)
    F_BM = data["F_BM"]
    C_BM = Cp_current * F_BM * Fm * pressure_factor

    return {
        "equip_type": equip_type,
        "size": size,
        "quantity": quantity,
        "Cp0_USD": Cp0 * quantity,
        "Cp_current_USD": Cp_current * quantity,
        "C_BM_USD": C_BM * quantity,
        "F_BM": F_BM,
        "Fm": Fm,
        "Fp": pressure_factor,
        "param_units": data["param"],
        "cepci_ratio": cepci / _CEPCI_2001,
    }


# ---------------------------------------------------------------------------
# Tab 2 — CAPEX
# ---------------------------------------------------------------------------

def capex_bare_module(
    items: list[dict],
    cepci: float = _CEPCI_2024,
    contingency: float = 0.15,
    engineering: float = 0.30,
) -> dict:
    """Total capital from equipment list using the bare-module method.

    Each item dict: {"type", "size", "material", "pressure_factor", "quantity"}
    """
    rows = []
    total_Cp = total_CBM = 0.0
    for item in items:
        r = equipment_cost(
            item["type"], item["size"], cepci,
            item.get("material", "Carbon Steel"),
            item.get("pressure_factor", 1.0),
            item.get("quantity", 1),
        )
        total_Cp += r["Cp_current_USD"]
        total_CBM += r["C_BM_USD"]
        rows.append(r)

    C_TM = 1.18 * total_CBM             # total module cost
    C_GR = C_TM + 0.50 * total_CBM     # grassroots capital

    FCI = C_GR * (1 + contingency + engineering)
    WC = 0.15 * FCI

    return {
        "items": rows,
        "total_Cp": total_Cp,
        "total_CBM": total_CBM,
        "C_TM": C_TM,
        "C_GR": C_GR,
        "FCI": FCI,
        "WC": WC,
        "TCI": FCI + WC,
        "contingency": contingency,
        "engineering": engineering,
    }


def capex_lang(
    total_purchased: float,
    plant_type: str = "fluid",
    contingency: float = 0.15,
    working_capital_frac: float = 0.15,
) -> dict:
    """Quick CAPEX estimate via the Lang-factor method."""
    factors = {"fluid": 4.74, "fluid-solid": 3.63, "solid": 3.10}
    F_L = factors.get(plant_type, 4.74)
    FCI = total_purchased * F_L * (1 + contingency)
    WC = working_capital_frac * FCI
    return {
        "lang_factor": F_L,
        "plant_type": plant_type,
        "purchased_equipment": total_purchased,
        "FCI": FCI,
        "WC": WC,
        "TCI": FCI + WC,
        "contingency": contingency,
    }


# ---------------------------------------------------------------------------
# Tab 3 — OPEX
# ---------------------------------------------------------------------------

def annual_opex(
    FCI: float,
    raw_materials: float = 0.0,
    utilities: float = 0.0,
    n_operators: int = 10,
    operator_salary: float = 70_000.0,
    maintenance_frac: float = 0.06,
    overhead_frac: float = 0.60,
    insurance_frac: float = 0.01,
    local_taxes_frac: float = 0.02,
    capital_charges_frac: float = 0.08,
) -> dict:
    """Annual operating cost breakdown for a chemical plant."""
    labor = n_operators * operator_salary * 1.30   # 30 % benefits
    maintenance = maintenance_frac * FCI
    supplies = 0.009 * FCI
    lab = 0.15 * labor
    overhead = overhead_frac * (labor + maintenance + supplies)
    admin = 0.20 * overhead
    insurance = insurance_frac * FCI
    taxes = local_taxes_frac * FCI
    cap_charges = capital_charges_frac * FCI

    variable = raw_materials + utilities
    fixed = labor + maintenance + supplies + lab + overhead + admin + insurance + taxes + cap_charges
    total = variable + fixed

    return {
        "raw_materials": raw_materials,
        "utilities": utilities,
        "labor": labor,
        "maintenance": maintenance,
        "supplies": supplies,
        "lab_costs": lab,
        "plant_overhead": overhead,
        "admin": admin,
        "insurance": insurance,
        "local_taxes": taxes,
        "capital_charges": cap_charges,
        "variable_opex": variable,
        "fixed_opex": fixed,
        "total_opex": total,
    }


# ---------------------------------------------------------------------------
# Tab 4 — Cash flow
# ---------------------------------------------------------------------------

def cash_flow_analysis(
    FCI: float,
    WC: float,
    annual_revenue: float,
    annual_opex: float,
    plant_life: int = 20,
    tax_rate: float = 0.21,
    depreciation_method: str = "straight-line",
    salvage_frac: float = 0.0,
) -> dict:
    """Multi-year cash flow table.

    Year 0 : -(FCI + WC)
    Years 1-N : (revenue - opex - tax) + depreciation
    Year N adds: WC recovery + salvage value
    """
    salvage = salvage_frac * FCI
    depreciable = FCI - salvage

    if depreciation_method == "MACRS-5":
        rates = np.array([0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576])
        depreciation = np.zeros(plant_life)
        for i, r in enumerate(rates):
            if i < plant_life:
                depreciation[i] = depreciable * r
    else:  # straight-line
        depreciation = np.full(plant_life, depreciable / plant_life)

    years = np.arange(1, plant_life + 1)
    ebitda = np.full(plant_life, annual_revenue - annual_opex)
    ebit = ebitda - depreciation
    tax = np.maximum(ebit * tax_rate, 0.0)
    net_income = ebit - tax
    cash_flow = net_income + depreciation

    # Working capital + salvage in last year
    cash_flow_adj = cash_flow.copy()
    cash_flow_adj[-1] += WC + salvage

    cumulative = np.cumsum(cash_flow_adj)

    return {
        "years": years,
        "depreciation": depreciation,
        "ebitda": ebitda,
        "ebit": ebit,
        "tax": tax,
        "net_income": net_income,
        "cash_flow": cash_flow,
        "cash_flow_adj": cash_flow_adj,
        "cumulative": cumulative,
        "initial_investment": FCI + WC,
    }


# ---------------------------------------------------------------------------
# Tab 5 — Profitability
# ---------------------------------------------------------------------------

def profitability(
    initial_investment: float,
    cash_flows: np.ndarray,
    discount_rate: float = 0.10,
) -> dict:
    """NPV, IRR, discounted payback, and ROI."""
    years = np.arange(1, len(cash_flows) + 1)
    pv = cash_flows / (1.0 + discount_rate) ** years
    npv = float(pv.sum() - initial_investment)

    # IRR
    irr = float("nan")
    def _npv(r):
        return (cash_flows / (1.0 + r) ** years).sum() - initial_investment
    try:
        if _npv(-0.999) * _npv(10.0) < 0:
            irr = float(brentq(_npv, -0.999, 10.0, maxiter=300))
    except Exception:
        pass

    # Simple (undiscounted) payback
    cum = np.cumsum(cash_flows)
    payback = float("nan")
    for i, c in enumerate(cum):
        if c >= initial_investment:
            prev = cum[i - 1] if i > 0 else 0.0
            payback = float(i + (initial_investment - prev) / (c - prev))
            break

    # Discounted payback
    cum_pv = np.cumsum(pv)
    dpb = float("nan")
    for i, c in enumerate(cum_pv):
        if c >= initial_investment:
            prev = cum_pv[i - 1] if i > 0 else 0.0
            dpb = float(i + (initial_investment - prev) / (c - prev))
            break

    roi = float(cash_flows.mean() / initial_investment)

    return {
        "npv": npv,
        "irr": irr,
        "payback": payback,
        "discounted_payback": dpb,
        "roi": roi,
        "pv": pv,
        "cumulative_pv": cum_pv - initial_investment,
        "years": years,
        "discount_rate": discount_rate,
    }


# ---------------------------------------------------------------------------
# Tab 6 — Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    base_FCI: float,
    base_WC: float,
    base_revenue: float,
    base_opex: float,
    plant_life: int = 20,
    tax_rate: float = 0.21,
    discount_rate: float = 0.10,
    variation: float = 0.20,  # ±20 %
) -> dict:
    """Tornado chart data: vary each key parameter ±variation and record NPV swing."""
    params = {
        "Revenue": base_revenue,
        "OPEX": base_opex,
        "FCI": base_FCI,
        "Discount Rate": discount_rate,
        "Tax Rate": tax_rate,
    }

    base_cf = cash_flow_analysis(base_FCI, base_WC, base_revenue, base_opex, plant_life, tax_rate)
    base_prof = profitability(base_cf["initial_investment"], base_cf["cash_flow_adj"], discount_rate)
    base_npv = base_prof["npv"]

    results = {}
    for param, base_val in params.items():
        lo_val = base_val * (1 - variation)
        hi_val = base_val * (1 + variation)
        npvs = []
        for val in (lo_val, hi_val):
            kw = dict(FCI=base_FCI, WC=base_WC, revenue=base_revenue,
                      opex=base_opex, dr=discount_rate, tr=tax_rate)
            if param == "Revenue":
                kw["revenue"] = val
            elif param == "OPEX":
                kw["opex"] = val
            elif param == "FCI":
                kw["FCI"] = val
                kw["WC"] = base_WC * (val / base_FCI)
            elif param == "Discount Rate":
                kw["dr"] = val
            elif param == "Tax Rate":
                kw["tr"] = min(val, 0.99)

            cf = cash_flow_analysis(kw["FCI"], kw["WC"], kw["revenue"], kw["opex"],
                                    plant_life, kw["tr"])
            pr = profitability(cf["initial_investment"], cf["cash_flow_adj"], kw["dr"])
            npvs.append(pr["npv"])

        results[param] = {
            "low_npv": npvs[0],
            "high_npv": npvs[1],
            "swing": abs(npvs[1] - npvs[0]),
        }

    return {
        "base_npv": base_npv,
        "variation": variation,
        "results": results,
    }
