"""Process Economics controller — formats economics results for the UI."""
from __future__ import annotations

import numpy as np

from core.process_economics.economics import (
    equipment_list,
    material_list,
    equipment_cost,
    capex_bare_module,
    capex_lang,
    annual_opex,
    cash_flow_analysis,
    profitability,
    sensitivity_analysis,
)


def _M(v: float) -> str:
    """Format a dollar value as $X.XX M or $X,XXX k."""
    if abs(v) >= 1e6:
        return f"${v/1e6:.3f} M"
    elif abs(v) >= 1e3:
        return f"${v/1e3:.1f} k"
    else:
        return f"${v:.0f}"


class ProcessEconomicsController:
    """Stateful controller.  Stores CAPEX equipment list and cash-flow results
    so that later tabs can build on earlier inputs."""

    def __init__(self) -> None:
        self._equip_items: list[dict] = []   # accumulated equipment list
        self._FCI: float = 0.0
        self._WC: float = 0.0
        self._opex: float = 0.0
        self._revenue: float = 0.0
        self._cf_result: dict = {}
        self._plant_life: int = 20
        self._tax_rate: float = 0.21
        self._discount_rate: float = 0.10

    # ------------------------------------------------------------------
    # Helpers (queried by the page to populate dropdowns)
    # ------------------------------------------------------------------

    def get_equipment_list(self) -> list[str]:
        return equipment_list()

    def get_material_list(self) -> list[str]:
        return material_list()

    # ------------------------------------------------------------------
    # Tab 1 — Equipment Cost
    # ------------------------------------------------------------------

    def run_equipment_cost(
        self,
        equip_type: str,
        size: float,
        cepci: float,
        material: str,
        pressure_factor: float,
        quantity: int,
    ) -> tuple[str, dict]:
        try:
            r = equipment_cost(equip_type, size, cepci, material, pressure_factor, quantity)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            f"Equipment Cost Estimate",
            "═" * 48,
            f"Equipment type : {equip_type}",
            f"Size parameter : {size:.4g}  ({r['param_units']})",
            f"Quantity       : {quantity}",
            f"Material       : {material}",
            f"Pressure factor: {pressure_factor:.3f}",
            f"CEPCI          : {cepci:.0f}  (ref 397, year 2001)",
            "",
            "Cost breakdown:",
            f"  Base purchased cost  (CEPCI 397) : {_M(r['Cp0_USD'] / (cepci / 397.0))}",
            f"  Purchased cost (current CEPCI)   : {_M(r['Cp_current_USD'])}",
            f"  Bare-module factor  F_BM         : {r['F_BM']:.2f}",
            f"  Material factor     Fm           : {r['Fm']:.2f}",
            f"  Pressure factor     Fp           : {r['Fp']:.3f}",
            "",
            f"  Bare-Module Cost  C_BM           : {_M(r['C_BM_USD'])}",
        ]
        return "\n".join(lines), {"type": "equipment_cost", "result": r}

    def add_to_equipment_list(
        self,
        equip_type: str,
        size: float,
        material: str,
        pressure_factor: float,
        quantity: int,
    ) -> str:
        item = {
            "type": equip_type,
            "size": size,
            "material": material,
            "pressure_factor": pressure_factor,
            "quantity": quantity,
        }
        self._equip_items.append(item)
        return (f"Added: {quantity}× {equip_type}  (size={size:.4g})  "
                f"→ {len(self._equip_items)} item(s) in list.")

    def clear_equipment_list(self) -> str:
        self._equip_items.clear()
        return "Equipment list cleared."

    # ------------------------------------------------------------------
    # Tab 2 — CAPEX
    # ------------------------------------------------------------------

    def run_capex_bare_module(
        self,
        cepci: float,
        contingency: float,
        engineering: float,
    ) -> tuple[str, dict]:
        if not self._equip_items:
            return "No equipment in list. Add items in Tab 1 first.", {}

        try:
            r = capex_bare_module(self._equip_items, cepci, contingency, engineering)
        except Exception as exc:
            return f"Error: {exc}", {}

        self._FCI = r["FCI"]
        self._WC = r["WC"]

        lines = [
            "CAPEX — Bare-Module Method",
            "═" * 48,
        ]
        lines.append(f"{'Equipment':<36}  {'Cp':>12}  {'C_BM':>12}")
        lines.append("─" * 64)
        for item in r["items"]:
            lines.append(
                f"  {item['equip_type'][:34]:<34}  {_M(item['Cp_current_USD']):>12}  {_M(item['C_BM_USD']):>12}"
            )
        lines += [
            "─" * 64,
            f"  {'Total purchased equipment cost':<34}  {_M(r['total_Cp']):>12}",
            f"  {'Total bare-module cost (ΣC_BM)':<34}  {_M(r['total_CBM']):>12}",
            "",
            f"Total module cost   C_TM = 1.18 × ΣC_BM   : {_M(r['C_TM'])}",
            f"Grassroots capital  C_GR = C_TM + 0.50·C_BM: {_M(r['C_GR'])}",
            "",
            f"Contingency   ({contingency:.0%})                  : {_M(r['C_GR'] * contingency)}",
            f"Engineering   ({engineering:.0%})                  : {_M(r['C_GR'] * engineering)}",
            "",
            f"Fixed Capital Investment  (FCI)             : {_M(r['FCI'])}",
            f"Working Capital Estimate  (15% of FCI)      : {_M(r['WC'])}",
            f"Total Capital Investment  (TCI)             : {_M(r['TCI'])}",
        ]
        return "\n".join(lines), {"type": "capex", "result": r}

    def run_capex_lang(
        self,
        total_purchased: float,
        plant_type: str,
        contingency: float,
    ) -> tuple[str, dict]:
        try:
            r = capex_lang(total_purchased, plant_type, contingency)
        except Exception as exc:
            return f"Error: {exc}", {}

        self._FCI = r["FCI"]
        self._WC = r["WC"]

        lines = [
            "CAPEX — Lang Factor Method",
            "═" * 48,
            f"Plant type             : {plant_type}",
            f"Lang factor  F_L       : {r['lang_factor']:.2f}",
            f"Contingency            : {contingency:.0%}",
            "",
            f"Total purchased cost   : {_M(total_purchased)}",
            f"Fixed Capital (FCI)    : {_M(r['FCI'])}",
            f"Working Capital (WC)   : {_M(r['WC'])}",
            f"Total Capital (TCI)    : {_M(r['TCI'])}",
        ]
        return "\n".join(lines), {"type": "capex_lang", "result": r}

    # ------------------------------------------------------------------
    # Tab 3 — OPEX
    # ------------------------------------------------------------------

    def run_opex(
        self,
        raw_materials: float,
        utilities: float,
        n_operators: int,
        salary: float,
        maintenance_frac: float,
        insurance_frac: float,
        tax_frac: float,
        cap_charges_frac: float,
    ) -> tuple[str, dict]:
        if self._FCI <= 0:
            return "FCI is zero. Run CAPEX estimation (Tab 2) first.", {}

        try:
            r = annual_opex(
                self._FCI,
                raw_materials=raw_materials,
                utilities=utilities,
                n_operators=n_operators,
                operator_salary=salary,
                maintenance_frac=maintenance_frac,
                insurance_frac=insurance_frac,
                local_taxes_frac=tax_frac,
                capital_charges_frac=cap_charges_frac,
            )
        except Exception as exc:
            return f"Error: {exc}", {}

        self._opex = r["total_opex"]

        lines = [
            "Annual OPEX Breakdown",
            "═" * 48,
            "Variable costs:",
            f"  Raw materials         : {_M(r['raw_materials'])}",
            f"  Utilities             : {_M(r['utilities'])}",
            f"  Variable subtotal     : {_M(r['variable_opex'])}",
            "",
            "Fixed costs:",
            f"  Labor (incl. benefits): {_M(r['labor'])}",
            f"  Maintenance           : {_M(r['maintenance'])}  ({maintenance_frac:.0%} FCI)",
            f"  Operating supplies    : {_M(r['supplies'])}",
            f"  Lab & QC costs        : {_M(r['lab_costs'])}",
            f"  Plant overhead        : {_M(r['plant_overhead'])}",
            f"  Administrative        : {_M(r['admin'])}",
            f"  Insurance             : {_M(r['insurance'])}",
            f"  Local taxes           : {_M(r['local_taxes'])}",
            f"  Capital charges       : {_M(r['capital_charges'])}",
            f"  Fixed subtotal        : {_M(r['fixed_opex'])}",
            "",
            f"Total Annual OPEX       : {_M(r['total_opex'])}",
        ]
        return "\n".join(lines), {"type": "opex", "result": r}

    # ------------------------------------------------------------------
    # Tab 4 — Cash Flow
    # ------------------------------------------------------------------

    def run_cash_flow(
        self,
        revenue: float,
        plant_life: int,
        tax_rate: float,
        depreciation: str,
        salvage_frac: float,
    ) -> tuple[str, dict]:
        if self._FCI <= 0:
            return "FCI is zero. Run CAPEX estimation (Tab 2) first.", {}

        self._revenue = revenue
        self._plant_life = plant_life
        self._tax_rate = tax_rate

        try:
            r = cash_flow_analysis(
                self._FCI, self._WC, revenue, self._opex,
                plant_life, tax_rate, depreciation, salvage_frac,
            )
        except Exception as exc:
            return f"Error: {exc}", {}

        self._cf_result = r
        years = r["years"]

        lines = [
            "Cash Flow Analysis",
            "═" * 80,
            f"{'Year':>4}  {'Deprec.':>12}  {'EBITDA':>12}  {'EBIT':>12}  "
            f"{'Tax':>10}  {'Net Income':>12}  {'Cash Flow':>12}",
            "─" * 80,
        ]
        for i in range(len(years)):
            cf_adj = r["cash_flow_adj"][i]
            marker = "*" if i == len(years) - 1 else " "
            lines.append(
                f"{int(years[i]):>4}  {_M(r['depreciation'][i]):>12}  "
                f"{_M(r['ebitda'][i]):>12}  {_M(r['ebit'][i]):>12}  "
                f"{_M(r['tax'][i]):>10}  {_M(r['net_income'][i]):>12}  "
                f"{_M(cf_adj):>12}{marker}"
            )
        lines += [
            "─" * 80,
            "* includes working-capital recovery",
            "",
            f"Year-0 investment   : {_M(r['initial_investment'])}",
            f"FCI                 : {_M(self._FCI)}",
            f"Working capital     : {_M(self._WC)}",
            f"Revenue / year      : {_M(revenue)}",
            f"OPEX / year         : {_M(self._opex)}",
            f"Tax rate            : {tax_rate:.0%}",
            f"Depreciation method : {depreciation}",
        ]
        return "\n".join(lines), {"type": "cash_flow", "result": r}

    # ------------------------------------------------------------------
    # Tab 5 — Profitability
    # ------------------------------------------------------------------

    def run_profitability(
        self,
        discount_rate: float,
        revenue: float | None = None,
        plant_life: int | None = None,
        tax_rate: float | None = None,
    ) -> tuple[str, dict]:
        """Re-use stored CF result (or recompute if revenue changed)."""
        if self._FCI <= 0:
            return "FCI is zero. Run CAPEX estimation (Tab 2) first.", {}

        if revenue is not None:
            self._revenue = revenue
        if plant_life is not None:
            self._plant_life = plant_life
        if tax_rate is not None:
            self._tax_rate = tax_rate

        self._discount_rate = discount_rate

        try:
            cf = cash_flow_analysis(
                self._FCI, self._WC,
                self._revenue, self._opex,
                self._plant_life, self._tax_rate,
            )
            r = profitability(cf["initial_investment"], cf["cash_flow_adj"], discount_rate)
        except Exception as exc:
            return f"Error: {exc}", {}

        irr_str = f"{r['irr']*100:.2f}%" if not np.isnan(r["irr"]) else "N/A"
        pb_str = f"{r['payback']:.1f} yr" if not np.isnan(r["payback"]) else "N/A"
        dpb_str = f"{r['discounted_payback']:.1f} yr" if not np.isnan(r["discounted_payback"]) else "N/A"

        lines = [
            "Profitability Summary",
            "═" * 48,
            f"Discount rate (WACC)    : {discount_rate:.1%}",
            "",
            f"Net Present Value (NPV) : {_M(r['npv'])}",
            f"Internal Rate of Return : {irr_str}",
            f"Payback period          : {pb_str}",
            f"Discounted payback      : {dpb_str}",
            f"ROI  (avg CF / invest.) : {r['roi']:.1%}",
            "",
            "Decision rules:",
            f"  NPV > 0 ?  {'✓ YES — project adds value' if r['npv'] > 0 else '✗ NO — project destroys value'}",
            f"  IRR > WACC?",
        ]
        if not np.isnan(r["irr"]):
            verdict = "✓ YES" if r["irr"] > discount_rate else "✗ NO"
            lines.append(f"    IRR {r['irr']*100:.2f}% vs WACC {discount_rate:.1%} → {verdict}")

        return "\n".join(lines), {
            "type": "profitability",
            "result": r,
            "cf": cf,
        }

    # ------------------------------------------------------------------
    # Tab 6 — Sensitivity
    # ------------------------------------------------------------------

    def run_sensitivity(
        self,
        discount_rate: float,
        variation: float,
    ) -> tuple[str, dict]:
        if self._FCI <= 0:
            return "FCI is zero. Run CAPEX estimation (Tab 2) first.", {}

        try:
            r = sensitivity_analysis(
                self._FCI, self._WC, self._revenue, self._opex,
                self._plant_life, self._tax_rate,
                discount_rate, variation,
            )
        except Exception as exc:
            return f"Error: {exc}", {}

        base_npv = r["base_npv"]
        lines = [
            "Sensitivity Analysis  (Tornado Chart)",
            "═" * 60,
            f"Base-case NPV : {_M(base_npv)}",
            f"Variation     : ±{variation:.0%}",
            "",
            f"{'Parameter':<20}  {'Low NPV':>14}  {'High NPV':>14}  {'Swing':>12}",
            "─" * 64,
        ]
        sorted_items = sorted(
            r["results"].items(), key=lambda x: x[1]["swing"], reverse=True
        )
        for param, vals in sorted_items:
            lines.append(
                f"  {param:<18}  {_M(vals['low_npv']):>14}  {_M(vals['high_npv']):>14}  "
                f"{_M(vals['swing']):>12}"
            )

        return "\n".join(lines), {
            "type": "sensitivity",
            "result": r,
        }
