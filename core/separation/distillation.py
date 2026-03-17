import math

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Shared VLE / operating-line helpers
# ---------------------------------------------------------------------------

def _vle_y(alpha: float, x: float) -> float:
    return alpha * x / (1.0 + (alpha - 1.0) * x)


def _vle_x(alpha: float, y: float) -> float:
    """Inverse of VLE: x = y / (alpha - (alpha-1)*y)."""
    denom = alpha - (alpha - 1.0) * y
    return y / denom if denom > 1e-15 else 0.0


def _rect_y(R: float, xD: float, x: float) -> float:
    return (R / (R + 1.0)) * x + xD / (R + 1.0)


def _strip_y(x_int: float, y_int: float, xB: float, x: float) -> float:
    if abs(x_int - xB) < 1e-12:
        return float(xB)
    slope = (y_int - xB) / (x_int - xB)
    return slope * (x - xB) + xB


def _q_intersection(R: float, xD: float, q: float, zF: float):
    """Intersection of q-line and rectifying operating line."""
    if abs(q - 1.0) < 1e-10:          # saturated-liquid feed → vertical q-line
        x_int = float(zF)
    else:
        denom = R + q
        x_int = (zF * (R + 1.0) + xD * (q - 1.0)) / denom if abs(denom) > 1e-15 else float(zF)
    y_int = _rect_y(R, xD, x_int)
    return float(x_int), float(y_int)


def _min_reflux(alpha: float, zF: float, xD: float, q: float) -> float:
    """Underwood minimum reflux for constant-relative-volatility binary system."""
    def f(theta):
        return alpha * zF / (alpha - theta) + (1.0 - zF) / (1.0 - theta) - (1.0 - q)
    try:
        theta = brentq(f, 1.0 + 1e-9, alpha - 1e-9)
        r_min_p1 = alpha * xD / (alpha - theta) + (1.0 - xD) / (1.0 - theta)
        return max(0.0, r_min_p1 - 1.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Tab 1 — McCabe-Thiele (binary distillation)
# ---------------------------------------------------------------------------

def solve_mccabe_thiele(
    alpha: float, R: float, xD: float, xB: float, zF: float, q: float
) -> dict:
    """
    Step off theoretical stages on a McCabe-Thiele diagram.

    Stepping strategy:
      * Start at (xD, xD) on the y = x diagonal.
      * Horizontal to equilibrium curve  (constant y, find x).
      * Vertical   to operating line     (constant x, find y).
      * Switch from rectifying to stripping line when x drops below x_int.

    Returns a dict with stage steps, counts, feed tray, and R_min.
    """
    x_int, y_int = _q_intersection(R, xD, q, zF)
    R_min = _min_reflux(alpha, zF, xD, q)

    steps = []          # list of (x0, y0, x1, y1) line segments
    n_stages = 0
    feed_stage = None

    x_cur, y_cur = xD, xD   # start at (xD, xD)

    for _ in range(150):
        # Horizontal to equilibrium curve
        x_eq = _vle_x(alpha, y_cur)
        steps.append((x_cur, y_cur, x_eq, y_cur))
        n_stages += 1

        if x_eq <= xB + 1e-8:
            break

        # First crossing below intersection -> feed tray
        if x_eq < x_int and feed_stage is None:
            feed_stage = n_stages

        # Vertical to operating line
        if x_eq < x_int:
            y_new = _strip_y(x_int, y_int, xB, x_eq)
        else:
            y_new = _rect_y(R, xD, x_eq)

        steps.append((x_eq, y_cur, x_eq, y_new))
        x_cur, y_cur = x_eq, y_new

        if y_cur <= xB + 1e-8:
            break

    if feed_stage is None:
        feed_stage = n_stages

    return {
        "n_stages": n_stages,
        "feed_stage": feed_stage,
        "steps": steps,
        "x_int": x_int,
        "y_int": y_int,
        "R_min": R_min,
        "R": R,
        "alpha": alpha,
        "xD": xD,
        "xB": xB,
        "zF": zF,
        "q": q,
    }


# ---------------------------------------------------------------------------
# Tab 2 — Kremser equation (absorption)
# ---------------------------------------------------------------------------

def solve_kremser(
    A: float, m: float, y_in: float, y_out: float, x_in: float
) -> dict:
    """
    Kremser analytical stage count for a dilute absorber.

    A     = L / (m*G)  absorption factor  (A > 1 for feasible absorption)
    m     = slope of linear equilibrium line  y* = m*x
    y_in  = inlet  gas mole fraction  (bottom of column, high solute)
    y_out = outlet gas mole fraction  (top,  desired clean gas)
    x_in  = inlet  liquid mole fraction (top, lean solvent)

    Returns N (Kremser), x_out (rich solvent), and stage steps for plotting.
    """
    L_over_G = A * m   # operating-line slope
    # Material balance: L*(x_out - x_in) = G*(y_in - y_out)
    #   => x_out = x_in + (y_in - y_out) / (L/G)
    x_out = float(x_in) + (y_in - y_out) / L_over_G

    y_eq_in = m * x_in   # equilibrium with lean solvent (y* at x_in)

    # Kremser: N = log[(y_in - m*x_in)/(y_out - m*x_in) * (1 - 1/A) + 1/A] / log(A)
    if abs(A - 1.0) < 1e-6:
        denom = y_out - y_eq_in
        N = (y_in - y_out) / denom if abs(denom) > 1e-15 else float("inf")
    else:
        denom = y_out - y_eq_in
        if abs(denom) < 1e-15:
            N = float("inf")
        else:
            arg = (y_in - y_eq_in) / denom * (1.0 - 1.0 / A) + 1.0 / A
            N = math.log(arg) / math.log(A) if arg > 0 else float("inf")

    absorb_eff = (y_in - y_out) / (y_in - y_eq_in) if abs(y_in - y_eq_in) > 1e-15 else 0.0

    # Stage steps for x-y diagram (stepping top -> bottom of absorber)
    # Operating line: y = L_over_G * (x - x_in) + y_out
    # Equilibrium:    y* = m*x  =>  x* = y/m
    # Each step: horizontal right to eq line, then vertical up to operating line.
    stage_steps = []
    x_cur = float(x_in)
    y_cur = float(y_out)

    n_plot = int(min(N if N != float("inf") else 25, 35)) + 3

    for _ in range(n_plot):
        if x_cur >= x_out - 1e-8 or y_cur >= y_in - 1e-8:
            break
        x_eq = y_cur / m
        stage_steps.append((x_cur, y_cur, x_eq, y_cur))       # horizontal ->
        y_new = L_over_G * (x_eq - x_in) + y_out
        stage_steps.append((x_eq, y_cur, x_eq, y_new))        # vertical   ^
        x_cur = x_eq
        y_cur = y_new

    return {
        "N": N,
        "N_ceil": math.ceil(N) if N != float("inf") else float("inf"),
        "x_out": x_out,
        "absorb_eff": absorb_eff,
        "A": A,
        "m": m,
        "y_in": y_in,
        "y_out": y_out,
        "x_in": x_in,
        "L_over_G": L_over_G,
        "stage_steps": stage_steps,
    }


# ---------------------------------------------------------------------------
# Tab 3 — Rachford-Rice flash
# ---------------------------------------------------------------------------

def solve_flash(z: list, K: list) -> dict:
    """
    Isothermal flash via Rachford-Rice equation.

    z — feed mole fractions  (must sum to ~1)
    K — equilibrium K-values  (y_i = K_i * x_i)

    Solves: sum_i [ z_i*(K_i-1) / (1 + beta*(K_i-1)) ] = 0   for beta = V/F

    Returns beta, liquid x_i, vapor y_i, and component data.
    """
    z_arr = np.array(z, dtype=float)
    K_arr = np.array(K, dtype=float)

    def rr(beta):
        return float(np.sum(z_arr * (K_arr - 1.0) / (1.0 + beta * (K_arr - 1.0))))

    if np.all(K_arr >= 1.0 - 1e-10):
        beta = 1.0 - 1e-10
    elif np.all(K_arr <= 1.0 + 1e-10):
        beta = 1e-10
    else:
        K_min = float(np.min(K_arr))
        hi = min(1.0 - 1e-8, 1.0 / (1.0 - K_min) - 1e-8) if K_min < 1.0 else 1.0 - 1e-8
        lo = 1e-8
        try:
            beta = brentq(rr, lo, hi, xtol=1e-12)
        except Exception:
            try:
                beta = brentq(rr, 1e-10, 1.0 - 1e-10, xtol=1e-12)
            except Exception:
                beta = 0.5

    x_arr = z_arr / (1.0 + beta * (K_arr - 1.0))
    y_arr = K_arr * x_arr

    return {
        "beta": float(beta),
        "x": x_arr.tolist(),
        "y": y_arr.tolist(),
        "z": z_arr.tolist(),
        "K": K_arr.tolist(),
        "n_comp": len(z),
    }


# ---------------------------------------------------------------------------
# Tab 4 — Liquid-Liquid Extraction
# ---------------------------------------------------------------------------

def solve_extraction(
    z_feed: float,
    K_D: float,
    solvent_ratio: float,
    n_stages: int,
    mode: str = "crosscurrent",
) -> dict:
    """Single-solute liquid-liquid extraction.

    z_feed      — feed solute mass fraction in raffinate phase
    K_D         — distribution coefficient  y* = K_D * x  (extract/raffinate)
    solvent_ratio — S/F (solvent to feed ratio, on a solute-free basis)
    n_stages    — number of equilibrium stages
    mode        — 'crosscurrent' or 'countercurrent'

    Returns extraction efficiency, raffinate/extract profiles, and stage data.
    """
    x = float(z_feed)   # raffinate composition (mass fraction or mole fraction)

    x_stages = [x]
    y_stages = []
    E_cumulative = []

    if mode == "crosscurrent":
        # Each stage uses fresh solvent: y = K_D*x  =>  x_out = x_in / (1 + K_D*S/F)
        factor = 1.0 / (1.0 + K_D * solvent_ratio)
        for _ in range(n_stages):
            y = K_D * x
            y_stages.append(y)
            x = x * factor
            x_stages.append(x)
            eta = (z_feed - x) / z_feed if z_feed > 0 else 0.0
            E_cumulative.append(eta)
    else:
        # Countercurrent: analytical solution using absorption factor A = S*K_D/F
        # Kremser-type: x_N / x_0 = (1 - 1/A^(N+1)) / (1 - 1/A^N) for y_in_extract=0
        A = solvent_ratio * K_D   # extraction factor
        if abs(A - 1.0) < 1e-6:
            x_N = z_feed / (n_stages + 1.0)
        else:
            x_N = z_feed * (A - 1.0) / (A ** (n_stages + 1) - 1.0)
        # Build stage profile by back-calculation
        x_stages = []
        xi = z_feed
        for stage in range(1, n_stages + 1):
            x_eq = x_N * (A ** stage - 1) / (A - 1) if abs(A - 1) > 1e-6 else x_N * stage
            x_stages.append(max(xi, 0.0))
            yi = K_D * xi
            y_stages.append(yi)
            eta = (z_feed - xi) / z_feed if z_feed > 0 else 0.0
            E_cumulative.append(eta)
            xi = x_N * (A ** stage - 1) / (A - 1) if abs(A - 1) > 1e-6 else x_N * stage
        x_stages.append(max(x_N, 0.0))
        x = x_N

    E_total = (z_feed - x_stages[-1]) / z_feed if z_feed > 0 else 0.0
    return {
        "x_stages": x_stages,
        "y_stages": y_stages,
        "E_stages": E_cumulative,
        "E_total": E_total,
        "n_stages": n_stages,
        "K_D": K_D,
        "S_over_F": solvent_ratio,
        "mode": mode,
        "z_feed": z_feed,
    }


# ---------------------------------------------------------------------------
# Tab 5 — Adsorption Isotherms
# ---------------------------------------------------------------------------

def solve_adsorption_isotherm(
    model: str,
    C_max: float,
    params: dict,
    n_points: int = 300,
) -> dict:
    """Compute adsorption isotherm q vs C.

    Models:
      'Langmuir'   : q = q_max * K_L * C / (1 + K_L * C)
      'Freundlich' : q = K_F * C^(1/n_F)
      'BET'        : q = q_m * C * K_BET / ((Cs-C)*(1 + (K_BET-1)*C/Cs))
      'Temkin'     : q = (RT/b) * ln(A_T * C)   [= (RT/b) * (ln(A_T) + ln(C))]
    """
    C_arr = np.linspace(1e-8, C_max, n_points)

    if model == "Langmuir":
        q_max = params["q_max"]
        K_L = params["K_L"]
        q_arr = q_max * K_L * C_arr / (1.0 + K_L * C_arr)
        q_half = q_max / 2.0
        RL = 1.0 / (1.0 + K_L * C_max)   # separation factor

    elif model == "Freundlich":
        K_F = params["K_F"]
        n_F = params["n_F"]
        q_arr = K_F * (C_arr ** (1.0 / n_F))
        q_half = None
        RL = None

    elif model == "BET":
        q_m = params["q_m"]
        K_BET = params["K_BET"]
        Cs = params["Cs"]
        C_arr = np.linspace(1e-8, min(C_max, Cs * 0.9999), n_points)
        q_arr = q_m * K_BET * C_arr / ((Cs - C_arr) * (1.0 + (K_BET - 1.0) * C_arr / Cs))
        q_half = None
        RL = None

    elif model == "Temkin":
        b = params["b"]
        A_T = params["A_T"]
        R = 8.314
        T = params.get("T_K", 298.15)
        C_arr = np.linspace(1.0 / A_T + 1e-8, C_max, n_points)
        q_arr = (R * T / b) * np.log(A_T * C_arr)
        q_half = None
        RL = None

    else:
        raise ValueError(f"Unknown isotherm model: {model}")

    return {
        "C": C_arr,
        "q": q_arr,
        "model": model,
        "params": params,
        "q_half": q_half if "q_half" in dir() else None,
        "RL": RL if "RL" in dir() else None,
    }


# ---------------------------------------------------------------------------
# Tab 6 — Membrane Separation (gas permeation)
# ---------------------------------------------------------------------------

def solve_membrane_separation(
    P_A: float,
    P_B: float,
    thickness_um: float,
    p_feed_bar: float,
    p_perm_bar: float,
    z_A_feed: float,
    stage_cut_arr: np.ndarray | None = None,
) -> dict:
    """Binary gas membrane separation (solution-diffusion model).

    P_A, P_B : permeabilities [Barrer]  1 Barrer = 1e-10 cm³(STP)·cm / (cm²·s·cmHg)
    thickness : membrane thickness [μm]
    p_feed    : feed-side pressure [bar]
    p_perm    : permeate-side pressure [bar]
    z_A_feed  : feed mole fraction of more-permeable component A
    stage_cut : theta = V_permeate / V_feed (0 to 1)

    Returns permeate composition y_A, selectivity, and flux data vs stage cut.
    """
    BARRER_TO_SI = 3.348e-16   # m³(STP)·m / (m²·s·Pa)
    Pa_per_bar = 1e5
    l_m = thickness_um * 1e-6  # μm -> m

    P_A_SI = P_A * BARRER_TO_SI
    P_B_SI = P_B * BARRER_TO_SI

    selectivity = P_A / P_B if P_B > 0 else float("inf")
    alpha_m = selectivity

    p_f = p_feed_bar * Pa_per_bar
    p_p = p_perm_bar * Pa_per_bar

    if stage_cut_arr is None:
        stage_cut_arr = np.linspace(0.001, 0.999, 200)

    # For a crossflow membrane: permeate y_A vs stage cut theta
    # Perfect mixing on both sides (simplified):
    # y_A / (1-y_A) = alpha_m * (z_A * p_f - y_A * p_p) /
    #                            ((1-z_A)*p_f - (1-y_A)*p_p)
    # Solve numerically for y_A given stage cut (mass balance + eq above)
    z_A = z_A_feed
    y_A_arr = np.zeros(len(stage_cut_arr))

    for i, theta in enumerate(stage_cut_arr):
        # Mass balance: theta*y_A + (1-theta)*x_A = z_A
        # Membrane eq (feed x_A, permeate y_A, perfect-mixing approximation):
        # y_A = alpha_m * x_A * p_f / (p_p + x_A*(p_f - p_p) + alpha_m*x_A*p_f - ...)
        # Simplified: use limiting case (low theta, x_A ≈ z_A)
        x_A = (z_A - theta * 1.0) / (1.0 - theta) if theta < 0.999 else z_A
        x_A = max(1e-8, min(1.0 - 1e-8, x_A))

        # Iteratively solve for y_A consistent with x_A and mass balance
        y_A = z_A  # initial guess
        for _ in range(50):
            # Membrane equation (simplified crossflow)
            denom = (p_f * x_A + p_p * (1.0 - x_A))
            if denom < 1e-20:
                y_A = 0.0
                break
            y_A_new = alpha_m * x_A * p_f / (
                p_f * x_A * (alpha_m - 1.0) + p_p + (alpha_m - 1.0) * p_p * x_A + p_f
            )
            y_A_new = max(0.0, min(1.0, y_A_new))
            x_A_new = (z_A - theta * y_A_new) / (1.0 - theta) if theta < 0.9999 else z_A
            x_A_new = max(1e-8, min(1.0 - 1e-8, x_A_new))
            if abs(y_A_new - y_A) < 1e-8:
                y_A = y_A_new
                break
            y_A, x_A = y_A_new, x_A_new
        y_A_arr[i] = y_A

    # Flux at zero stage cut
    J_A = P_A_SI / l_m * (z_A * p_f - y_A_arr[0] * p_p)
    J_B_total = P_B_SI / l_m * ((1 - z_A) * p_f - (1 - y_A_arr[0]) * p_p)
    J_total = J_A + J_B_total

    return {
        "stage_cut": stage_cut_arr,
        "y_A": y_A_arr,
        "selectivity": selectivity,
        "J_A_mol_m2_s": float(J_A / 22.4e-3),   # approx, STP
        "J_total": float(J_total / 22.4e-3),
        "P_A_barrer": P_A,
        "P_B_barrer": P_B,
        "thickness_um": thickness_um,
        "alpha_m": alpha_m,
        "z_A_feed": z_A,
        "p_feed_bar": p_feed_bar,
        "p_perm_bar": p_perm_bar,
    }
