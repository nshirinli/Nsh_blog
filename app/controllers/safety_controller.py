"""Safety & Risk Analysis controller — formats results for the UI."""
from __future__ import annotations

import numpy as np

from core.safety.risk import (
    gaussian_plume,
    vapor_cloud_explosion,
    pool_fire,
    risk_level,
    risk_matrix_data,
    lopa,
    flammability_limits_mixture,
    flash_point_estimate,
    minimum_oxygen_concentration,
)


class SafetyController:

    # ------------------------------------------------------------------
    # Tab 1 — Gaussian Plume Dispersion
    # ------------------------------------------------------------------

    def run_dispersion(
        self,
        Q: float,
        u: float,
        H: float,
        stability: str,
        x_max: float,
    ) -> tuple[str, dict]:
        try:
            r = gaussian_plume(Q, u, H, stability, x_max)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Gaussian Plume Dispersion",
            "═" * 52,
            f"Source strength  Q  : {Q:.4g} g/s",
            f"Wind speed       u  : {u:.4g} m/s",
            f"Release height   H  : {H:.4g} m",
            f"Stability class     : {stability}",
            "",
            f"Peak centreline concentration : {r['C_peak_mg_m3']:.4g} mg/m³",
            f"Location of peak              : {r['x_peak_m']:.4g} m downwind",
            "",
            "Concentration at selected downwind distances:",
            f"  {'x (m)':>8}  {'C (mg/m³)':>12}  {'σy (m)':>8}  {'σz (m)':>8}",
            "─" * 44,
        ]
        xs = r["x"]
        Cs = r["C_mg_m3"]
        sy = r["sigma_y"]
        sz = r["sigma_z"]
        checkpoints = [50, 100, 200, 500, 1000, 2000, 5000]
        for xc in checkpoints:
            idx = np.searchsorted(xs, xc)
            if idx < len(xs):
                lines.append(
                    f"  {xs[idx]:>8.0f}  {Cs[idx]:>12.4g}  "
                    f"{sy[idx]:>8.2f}  {sz[idx]:>8.2f}"
                )
        return "\n".join(lines), {"type": "dispersion", "result": r}

    # ------------------------------------------------------------------
    # Tab 2 — Vapor Cloud Explosion
    # ------------------------------------------------------------------

    def run_explosion(
        self,
        m_fuel: float,
        dHc: float,
        alpha: float,
        r_max: float,
    ) -> tuple[str, dict]:
        try:
            r = vapor_cloud_explosion(m_fuel, dHc, alpha, r_max)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Vapor Cloud Explosion  (TNT Equivalency Method)",
            "═" * 52,
            f"Flammable mass    : {m_fuel:.4g} kg",
            f"Heat of combustion: {dHc:.4g} MJ/kg",
            f"Yield factor α    : {alpha:.3f}  ({alpha*100:.1f} %)",
            "",
            f"TNT equivalent mass : {r['W_tnt_kg']:.4f} kg",
            "",
            "Overpressure at selected distances:",
            f"  {'r (m)':>8}  {'Ps (kPa)':>10}  {'Effect':>35}",
            "─" * 60,
        ]
        rs = r["r"]
        Ps = r["Ps_kPa"]
        for dist in [10, 25, 50, 100, 200, 500]:
            idx = np.searchsorted(rs, dist)
            if idx < len(rs):
                ps_val = Ps[idx]
                if ps_val >= 70:
                    effect = "Lung damage likely"
                elif ps_val >= 17:
                    effect = "Serious structural damage"
                elif ps_val >= 3.0:
                    effect = "Partial wall collapse"
                elif ps_val >= 0.5:
                    effect = "Window breakage"
                else:
                    effect = "Negligible"
                lines.append(f"  {dist:>8.0f}  {ps_val:>10.3g}  {effect:>35}")

        lines += ["", "Damage threshold distances:"]
        for label, dist in r["damage_distances"].items():
            lines.append(f"  {label:<40}: {dist:>8.1f} m")
        return "\n".join(lines), {"type": "explosion", "result": r}

    # ------------------------------------------------------------------
    # Tab 3 — Pool Fire
    # ------------------------------------------------------------------

    def run_pool_fire(
        self,
        diameter: float,
        m_dot: float,
        dHc: float,
        eta: float,
        r_max: float,
    ) -> tuple[str, dict]:
        try:
            r = pool_fire(diameter, m_dot, dHc, eta=eta, r_max=r_max)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Pool Fire Analysis",
            "═" * 52,
            f"Pool diameter      : {diameter:.4g} m  (area={np.pi*diameter**2/4:.3g} m²)",
            f"Burning rate       : {m_dot:.4g} kg/(m²·s)",
            f"Heat of combustion : {dHc:.4g} MJ/kg",
            f"Radiative fraction : {eta:.2f}",
            "",
            f"Flame height       : {r['flame_height_m']:.3f} m  (L/D = {r['L_D']:.2f})",
            f"Total HRR          : {r['Q_comb_MW']:.4g} MW",
            f"Radiative power    : {r['Q_rad_MW']:.4g} MW",
            "",
            "Incident heat flux at distance from pool edge:",
            f"  {'r (m)':>8}  {'q (kW/m²)':>12}  {'Hazard level':>28}",
            "─" * 56,
        ]
        rs = r["r"]
        qs = r["q_W_m2"]
        for dist in [5, 10, 20, 30, 50, 75, 100]:
            idx = np.searchsorted(rs, dist)
            if idx < len(rs):
                q_kW = qs[idx] / 1000.0
                if q_kW >= 35:
                    hazard = "Steel structural failure"
                elif q_kW >= 12.5:
                    hazard = "Piloted ignition"
                elif q_kW >= 4.0:
                    hazard = "1% fatality (10 s)"
                elif q_kW >= 1.6:
                    hazard = "Pain threshold"
                else:
                    hazard = "No harm"
                lines.append(f"  {dist:>8.0f}  {q_kW:>12.4g}  {hazard:>28}")

        lines += ["", "Hazard distances:"]
        for label, dist in r["harm_distances"].items():
            lines.append(f"  {label:<45}: {dist:>6.1f} m")
        return "\n".join(lines), {"type": "pool_fire", "result": r}

    # ------------------------------------------------------------------
    # Tab 4 — Risk Matrix
    # ------------------------------------------------------------------

    def run_risk_assessment(
        self,
        hazards: list[dict],  # [{"description", "severity", "likelihood"}, ...]
    ) -> tuple[str, dict]:
        results = []
        for h in hazards:
            r = risk_level(h["severity"], h["likelihood"])
            results.append({**h, **r})

        matrix = risk_matrix_data()
        lines = [
            "Risk Assessment",
            "═" * 60,
            f"{'#':<3}  {'Description':<30}  {'S':>2}  {'L':>2}  {'Risk Level':>12}",
            "─" * 56,
        ]
        for i, item in enumerate(results, 1):
            lines.append(
                f"  {i:<3}  {item.get('description','')[:28]:<30}  "
                f"{item['severity']:>2}  {item['likelihood']:>2}  "
                f"{item['name']:>12}"
            )
        lines += [
            "",
            "Risk Matrix Key:",
            "  L = Low  |  M = Medium  |  H = High  |  C = Critical",
            "  Severity (rows): 1=Negligible → 5=Catastrophic",
            "  Likelihood (cols): 1=Very Unlikely → 5=Very Likely",
        ]
        return "\n".join(lines), {
            "type": "risk_matrix",
            "hazards": results,
            "matrix": matrix,
        }

    def get_risk_matrix_data(self) -> dict:
        return risk_matrix_data()

    # ------------------------------------------------------------------
    # Tab 5 — LOPA
    # ------------------------------------------------------------------

    def run_lopa(
        self,
        ie_description: str,
        ie_freq: float,
        ipls: list[dict],
        target_freq: float,
    ) -> tuple[str, dict]:
        try:
            r = lopa(ie_description, ie_freq, ipls, target_freq)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Layer of Protection Analysis  (LOPA)",
            "═" * 60,
            f"Initiating Event : {ie_description}",
            f"IE Frequency     : {ie_freq:.2e} /yr",
            "",
            "Independent Protection Layers:",
            f"  {'#':<3}  {'IPL Description':<35}  {'PFD':>10}",
            "─" * 54,
        ]
        pfd_running = ie_freq
        for i, ipl in enumerate(r["ipls"], 1):
            lines.append(
                f"  {i:<3}  {ipl.get('description','')[:33]:<35}  "
                f"{ipl.get('pfd', 1.0):>10.2e}"
            )
            pfd_running *= ipl.get("pfd", 1.0)

        verdict = "MEETS TARGET" if r["meets_target"] else "DOES NOT MEET TARGET"
        lines += [
            "─" * 54,
            f"Total PFD product         : {r['pfd_product']:.3e}",
            f"Mitigated event frequency : {r['mitigated_freq']:.3e} /yr",
            f"Target frequency          : {r['target_freq']:.3e} /yr",
            f"Risk reduction factor     : {r['risk_reduction_factor']:.2e}",
            "",
            f"  ➜  {verdict}",
        ]
        if not r["meets_target"]:
            lines.append(
                f"  Additional RRF needed     : {r['additional_rrf_needed']:.2e}"
            )
        return "\n".join(lines), {"type": "lopa", "result": r}

    # ------------------------------------------------------------------
    # Tab 6 — Flammability
    # ------------------------------------------------------------------

    def run_flammability(
        self,
        components: list[str],
        mole_fracs: list[float],
        lfl_vals: list[float],
        ufl_vals: list[float],
        Tb_K: float | None,
        lfl_single: float | None,
        stoich_O2: float | None,
    ) -> tuple[str, dict]:
        try:
            r = flammability_limits_mixture(components, mole_fracs, lfl_vals, ufl_vals)
        except Exception as exc:
            return f"Error: {exc}", {}

        lines = [
            "Flammability Analysis",
            "═" * 52,
            "",
            "Mixture Flammability  (Le Chatelier's Rule):",
            f"  {'Component':<18}  {'Mole frac':>10}  {'LFL (%)':>8}  {'UFL (%)':>8}",
            "─" * 50,
        ]
        for i, comp in enumerate(r["components"]):
            lines.append(
                f"  {comp:<18}  {r['mole_fracs'][i]:>10.4f}  "
                f"{r['LFL_vals'][i]:>8.2f}  {r['UFL_vals'][i]:>8.2f}"
            )
        lines += [
            "─" * 50,
            f"  Mixture LFL           : {r['LFLmix']:.3f} vol%",
            f"  Mixture UFL           : {r['UFLmix']:.3f} vol%",
            f"  Flammable range       : {r['flammable_range_pct']:.3f} vol%",
            f"  Stoichiometric conc.  : {r['Cst_pct']:.3f} vol%  (est.)",
        ]

        flash_r: dict = {}
        if Tb_K is not None and lfl_single is not None:
            flash_r = flash_point_estimate(Tb_K, lfl_single)
            lines += [
                "",
                "Flash Point Estimate:",
                f"  Boiling point     : {Tb_K:.2f} K  ({Tb_K-273.15:.1f} °C)",
                f"  Flash point       : {flash_r['Tflash_K']:.2f} K  "
                f"({flash_r['Tflash_C']:.1f} °C)",
                f"  {flash_r['note']}",
            ]

        if stoich_O2 is not None and lfl_single is not None:
            moc = minimum_oxygen_concentration(lfl_single, stoich_O2)
            lines += [
                "",
                "Inerting / MOC:",
                f"  Minimum Oxygen Concentration : {moc:.2f} vol% O₂",
                f"  Inert below                  : {moc:.2f} vol% O₂ to prevent ignition",
            ]

        return "\n".join(lines), {
            "type": "flammability",
            "result": r,
            "flash": flash_r,
        }
