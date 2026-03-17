import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, minimize
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
TOL = 1e-6


def solve_optimization_problem(
    problem_type: str,
    objective_sense: str,
    num_vars: int,
    objective_text: str,
    constraints_text: str,
    bounds_text: str,
    integer_vars_text: str,
    initial_guess_text: str,
    dynamic_state_vars_text: str,
    dynamic_control_vars_text: str,
    dynamic_horizon_text: str,
    dynamic_intervals: int,
    dynamic_odes_text: str,
    dynamic_initial_conditions_text: str,
    dynamic_running_cost_text: str,
    dynamic_terminal_cost_text: str,
    dynamic_control_bounds_text: str,
    dynamic_control_guess_text: str,
    dynamic_terminal_constraints_text: str,
) -> dict:
    if problem_type == "Dynamic":
        return _solve_dynamic_problem(
            objective_sense=objective_sense,
            dynamic_state_vars_text=dynamic_state_vars_text,
            dynamic_control_vars_text=dynamic_control_vars_text,
            dynamic_horizon_text=dynamic_horizon_text,
            dynamic_intervals=dynamic_intervals,
            dynamic_odes_text=dynamic_odes_text,
            dynamic_initial_conditions_text=dynamic_initial_conditions_text,
            dynamic_running_cost_text=dynamic_running_cost_text,
            dynamic_terminal_cost_text=dynamic_terminal_cost_text,
            dynamic_control_bounds_text=dynamic_control_bounds_text,
            dynamic_control_guess_text=dynamic_control_guess_text,
            dynamic_terminal_constraints_text=dynamic_terminal_constraints_text,
        )

    warnings = []

    if not objective_text:
        return {"status": "error", "message": "Objective function cannot be empty.", "warnings": warnings}

    variable_names = [f"x{i}" for i in range(1, num_vars + 1)]
    symbols = sp.symbols(" ".join(variable_names))
    if num_vars == 1:
        symbols = (symbols,)

    symbol_map = {name: sym for name, sym in zip(variable_names, symbols)}

    try:
        objective_expr = _parse_expression(objective_text, symbol_map)
        bounds = _parse_bounds(bounds_text, variable_names)
        integer_vars = _parse_integer_vars(integer_vars_text, variable_names)
        constraints = _parse_constraints(constraints_text, symbol_map)
        initial_guess = _parse_initial_guess(initial_guess_text, variable_names)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "warnings": warnings}

    warnings.extend(_denominator_warnings(objective_expr, constraints, bounds, variable_names))
    fatal_messages = _denominator_fatal_errors(problem_type, objective_expr, constraints, bounds, variable_names)
    if fatal_messages:
        return {
            "status": "error",
            "message": "\n".join(fatal_messages),
            "warnings": warnings,
        }

    if problem_type == "Linear":
        return _solve_linear_problem(
            symbols=symbols,
            variable_names=variable_names,
            objective_expr=objective_expr,
            objective_sense=objective_sense,
            constraints=constraints,
            bounds=bounds,
            integer_vars=integer_vars,
            warnings=warnings,
        )

    if problem_type == "Nonlinear":
        return _solve_nonlinear_problem(
            symbols=symbols,
            variable_names=variable_names,
            objective_expr=objective_expr,
            objective_sense=objective_sense,
            constraints=constraints,
            bounds=bounds,
            integer_vars=integer_vars,
            initial_guess=initial_guess,
            warnings=warnings,
        )

    return {"status": "error", "message": f"Unknown problem type: {problem_type}", "warnings": warnings}


def _solve_linear_problem(
    symbols,
    variable_names,
    objective_expr,
    objective_sense,
    constraints,
    bounds,
    integer_vars,
    warnings,
) -> dict:
    try:
        c, obj_const = _linear_coefficients(objective_expr, symbols)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Objective is not linear.\n{exc}",
            "warnings": warnings,
        }

    if objective_sense == "Maximize":
        c = -c

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for cons in constraints:
        lhs = cons["lhs"]
        rhs = cons["rhs"]
        op = cons["op"]

        try:
            if op == "<=":
                expr = sp.expand(lhs - rhs)
                coeffs, const = _linear_coefficients(expr, symbols)
                A_ub.append(coeffs)
                b_ub.append(-const)

            elif op == ">=":
                expr = sp.expand(rhs - lhs)
                coeffs, const = _linear_coefficients(expr, symbols)
                A_ub.append(coeffs)
                b_ub.append(-const)

            elif op in ("=", "=="):
                expr = sp.expand(lhs - rhs)
                coeffs, const = _linear_coefficients(expr, symbols)
                A_eq.append(coeffs)
                b_eq.append(-const)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Constraint is not linear:\n{cons['text']}\n{exc}",
                "warnings": warnings,
            }

    integrality = np.zeros(len(variable_names), dtype=int)
    for i, name in enumerate(variable_names):
        if name in integer_vars:
            integrality[i] = 1

    try:
        result = linprog(
            c=np.array(c, dtype=float),
            A_ub=np.array(A_ub, dtype=float) if A_ub else None,
            b_ub=np.array(b_ub, dtype=float) if b_ub else None,
            A_eq=np.array(A_eq, dtype=float) if A_eq else None,
            b_eq=np.array(b_eq, dtype=float) if b_eq else None,
            bounds=bounds,
            integrality=integrality,
            method="highs",
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc), "warnings": warnings}

    if not result.success:
        return {"status": "error", "message": result.message, "warnings": warnings}

    x_opt = np.array(result.x, dtype=float)

    objective_value = float(result.fun + obj_const)
    if objective_sense == "Maximize":
        objective_value = -float(result.fun) + obj_const

    return _build_success_result(
        variable_names=variable_names,
        x_opt=x_opt,
        objective_value=objective_value,
        solver_name="HiGHS (linprog)",
        message=result.message,
        success=bool(result.success),
        iterations=getattr(result, "nit", None),
        bounds=bounds,
        constraints=constraints,
        symbols=symbols,
        warnings=warnings,
        initial_guess_used={},
    )


def _solve_nonlinear_problem(
    symbols,
    variable_names,
    objective_expr,
    objective_sense,
    constraints,
    bounds,
    integer_vars,
    initial_guess,
    warnings,
) -> dict:
    if integer_vars:
        return {
            "status": "error",
            "message": "Integer restrictions are not supported yet for nonlinear problems.",
            "warnings": warnings,
        }

    try:
        objective_fn_raw = sp.lambdify(symbols, objective_expr, modules="numpy")
    except Exception as exc:
        return {"status": "error", "message": f"Could not build objective function.\n{exc}", "warnings": warnings}

    if objective_sense == "Minimize":
        def objective_fn(x):
            return float(objective_fn_raw(*x))
    else:
        def objective_fn(x):
            return -float(objective_fn_raw(*x))

    scipy_constraints = []
    for cons in constraints:
        lhs = cons["lhs"]
        rhs = cons["rhs"]
        op = cons["op"]

        try:
            if op == "<=":
                expr = sp.expand(rhs - lhs)
                fn_raw = sp.lambdify(symbols, expr, modules="numpy")
                scipy_constraints.append(
                    {"type": "ineq", "fun": lambda x, f=fn_raw: float(f(*x))}
                )

            elif op == ">=":
                expr = sp.expand(lhs - rhs)
                fn_raw = sp.lambdify(symbols, expr, modules="numpy")
                scipy_constraints.append(
                    {"type": "ineq", "fun": lambda x, f=fn_raw: float(f(*x))}
                )

            elif op in ("=", "=="):
                expr = sp.expand(lhs - rhs)
                fn_raw = sp.lambdify(symbols, expr, modules="numpy")
                scipy_constraints.append(
                    {"type": "eq", "fun": lambda x, f=fn_raw: float(f(*x))}
                )
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Could not build nonlinear constraint:\n{cons['text']}\n{exc}",
                "warnings": warnings,
            }

    x0 = np.array(_build_initial_guess(bounds, initial_guess, variable_names), dtype=float)
    initial_guess_used = {name: float(val) for name, val in zip(variable_names, x0)}

    try:
        result = minimize(
            fun=objective_fn,
            x0=x0,
            bounds=bounds,
            constraints=scipy_constraints,
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-9},
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc), "warnings": warnings}

    if not result.success:
        return {"status": "error", "message": result.message, "warnings": warnings}

    x_opt = np.array(result.x, dtype=float)
    objective_value = float(result.fun)
    if objective_sense == "Maximize":
        objective_value = -objective_value

    return _build_success_result(
        variable_names=variable_names,
        x_opt=x_opt,
        objective_value=objective_value,
        solver_name="SLSQP (scipy.optimize.minimize)",
        message=result.message,
        success=bool(result.success),
        iterations=getattr(result, "nit", None),
        bounds=bounds,
        constraints=constraints,
        symbols=symbols,
        warnings=warnings,
        initial_guess_used=initial_guess_used,
    )


def _solve_dynamic_problem(
    objective_sense: str,
    dynamic_state_vars_text: str,
    dynamic_control_vars_text: str,
    dynamic_horizon_text: str,
    dynamic_intervals: int,
    dynamic_odes_text: str,
    dynamic_initial_conditions_text: str,
    dynamic_running_cost_text: str,
    dynamic_terminal_cost_text: str,
    dynamic_control_bounds_text: str,
    dynamic_control_guess_text: str,
    dynamic_terminal_constraints_text: str,
) -> dict:
    warnings = []

    try:
        state_names = _parse_name_list(dynamic_state_vars_text, "state variables")
        control_names = _parse_name_list(dynamic_control_vars_text, "control variables")
    except Exception as exc:
        return {"status": "error", "message": str(exc), "warnings": warnings}

    sense_sign = 1.0 if objective_sense == "Minimize" else -1.0

    try:
        horizon = float(dynamic_horizon_text)
    except Exception:
        return {
            "status": "error",
            "message": "Time horizon must be a valid positive number.",
            "warnings": warnings,
        }

    if horizon <= 0:
        return {
            "status": "error",
            "message": "Time horizon must be greater than zero.",
            "warnings": warnings,
        }

    if dynamic_intervals < 1:
        return {
            "status": "error",
            "message": "Control intervals must be at least 1.",
            "warnings": warnings,
        }

    t_sym = sp.Symbol("t")
    state_symbols = sp.symbols(" ".join(state_names))
    control_symbols = sp.symbols(" ".join(control_names))

    if len(state_names) == 1:
        state_symbols = (state_symbols,)
    if len(control_names) == 1:
        control_symbols = (control_symbols,)

    dynamic_symbol_map = {name: sym for name, sym in zip(state_names, state_symbols)}
    dynamic_symbol_map.update({name: sym for name, sym in zip(control_names, control_symbols)})
    dynamic_symbol_map["t"] = t_sym

    terminal_symbol_map = {name: sym for name, sym in zip(state_names, state_symbols)}
    terminal_symbol_map["t"] = t_sym

    try:
        ode_exprs = _parse_dynamic_odes(dynamic_odes_text, state_names, dynamic_symbol_map)
        initial_conditions = _parse_required_value_map(
            dynamic_initial_conditions_text,
            state_names,
            "initial conditions",
        )
        running_cost_expr = (
            _parse_expression(dynamic_running_cost_text, dynamic_symbol_map)
            if dynamic_running_cost_text.strip()
            else sp.Integer(0)
        )
        terminal_cost_expr = (
            _parse_expression(dynamic_terminal_cost_text, terminal_symbol_map)
            if dynamic_terminal_cost_text.strip()
            else sp.Integer(0)
        )
        control_bounds = _parse_bounds(dynamic_control_bounds_text, control_names)
        control_guess = _parse_optional_value_map(
            dynamic_control_guess_text,
            control_names,
            "initial control guess",
        )
        terminal_constraints = _parse_constraints(dynamic_terminal_constraints_text, terminal_symbol_map)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "warnings": warnings}

    if any(lb is None and ub is None for lb, ub in control_bounds):
        warnings.append(
            "At least one control variable is unbounded. Dynamic optimization is usually more stable with finite control bounds."
        )

    try:
        ode_funcs = [
            sp.lambdify((*state_symbols, *control_symbols, t_sym), expr, modules="numpy")
            for expr in ode_exprs
        ]
        running_cost_fn = sp.lambdify(
            (*state_symbols, *control_symbols, t_sym),
            running_cost_expr,
            modules="numpy",
        )
        terminal_cost_fn = sp.lambdify((*state_symbols, t_sym), terminal_cost_expr, modules="numpy")

        terminal_constraint_data = []
        for cons in terminal_constraints:
            lhs_fn = sp.lambdify((*state_symbols, t_sym), cons["lhs"], modules="numpy")
            rhs_fn = sp.lambdify((*state_symbols, t_sym), cons["rhs"], modules="numpy")
            terminal_constraint_data.append(
                {
                    "text": cons["text"],
                    "op": cons["op"],
                    "lhs_fn": lhs_fn,
                    "rhs_fn": rhs_fn,
                }
            )
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Could not build dynamic equations or costs.\n{exc}",
            "warnings": warnings,
        }

    x0_state = np.array([initial_conditions[name] for name in state_names], dtype=float)
    decision_var_names, repeated_bounds = _repeat_control_bounds(control_names, control_bounds, dynamic_intervals)
    z0, initial_guess_used = _build_dynamic_initial_guess(
        control_names=control_names,
        control_bounds=control_bounds,
        control_guess=control_guess,
        intervals=dynamic_intervals,
    )

    cache = {"key": None, "value": None}

    def simulate_with_controls(z, tight=False):
        z_array = np.asarray(z, dtype=float)
        key = (tuple(float(val) for val in z_array), tight)

        if cache["key"] == key:
            return cache["value"]

        try:
            dt = horizon / dynamic_intervals
            t_grid = np.linspace(0.0, horizon, dynamic_intervals + 1)

            y_aug = np.concatenate([x0_state.copy(), np.array([0.0])])

            all_t = [0.0]
            all_x = [x0_state.copy()]
            all_running = [0.0]

            for k in range(dynamic_intervals):
                t0 = t_grid[k]
                t1 = t_grid[k + 1]
                u_vec = z_array[k * len(control_names):(k + 1) * len(control_names)]

                def ode_aug(t, y):
                    x_vec = y[:-1]
                    args = [*x_vec.tolist(), *u_vec.tolist(), t]
                    dxdt = np.array([float(fn(*args)) for fn in ode_funcs], dtype=float)
                    running = float(running_cost_fn(*args))
                    if not np.all(np.isfinite(dxdt)) or not np.isfinite(running):
                        raise ValueError("Non-finite value encountered in ODE or running cost.")
                    return np.concatenate([dxdt, np.array([running])])

                rtol = 1e-8 if tight else 1e-6
                atol = 1e-10 if tight else 1e-8
                max_step = max(dt / 20.0, 1e-3)
                sol = solve_ivp(
                    ode_aug,
                    (t0, t1),
                    y_aug,
                    method="RK45",
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step,
                )
                if not sol.success:
                    sol = solve_ivp(
                        ode_aug,
                        (t0, t1),
                        y_aug,
                        method="BDF",
                        rtol=rtol,
                        atol=atol,
                        max_step=max_step,
                    )

                if not sol.success:
                    raise RuntimeError(sol.message)

                y_aug = sol.y[:, -1]

                if not np.all(np.isfinite(y_aug)):
                    raise ValueError("Non-finite state encountered during simulation.")

                if k == 0:
                    interval_t = sol.t
                    interval_x = sol.y[:-1, :].T
                    interval_running = sol.y[-1, :]
                else:
                    interval_t = sol.t[1:]
                    interval_x = sol.y[:-1, 1:].T
                    interval_running = sol.y[-1, 1:]

                all_t.extend(interval_t.tolist())
                all_x.extend(interval_x.tolist())
                all_running.extend(interval_running.tolist())

            x_final = y_aug[:-1]
            running_cost_value = float(y_aug[-1])
            terminal_cost_value = float(terminal_cost_fn(*x_final.tolist(), horizon))
            raw_objective = running_cost_value + terminal_cost_value
            total_objective = sense_sign * raw_objective

            if not np.isfinite(total_objective):
                raise ValueError("Non-finite total objective encountered.")

            control_profiles = {}
            for i, name in enumerate(control_names):
                control_profiles[name] = [
                    float(z_array[k * len(control_names) + i]) for k in range(dynamic_intervals)
                ]

            state_trajectories = {}
            all_x_array = np.asarray(all_x, dtype=float)
            for i, name in enumerate(state_names):
                state_trajectories[name] = all_x_array[:, i].tolist()

            result = {
                "ok": True,
                "objective_value": float(total_objective),
                "raw_objective_value": float(raw_objective),
                "running_cost_value": float(running_cost_value),
                "terminal_cost_value": float(terminal_cost_value),
                "final_states": {name: float(val) for name, val in zip(state_names, x_final)},
                "control_profiles": control_profiles,
                "time_grid": t_grid.tolist(),
                "trajectory_time": [float(v) for v in all_t],
                "state_trajectories": state_trajectories,
                "running_cost_trajectory": [float(v) for v in all_running],
                "x_final_array": x_final,
            }
        except Exception as exc:
            result = {
                "ok": False,
                "message": str(exc),
            }

        cache["key"] = key
        cache["value"] = result
        return result

    initial_sim = simulate_with_controls(z0)
    if not initial_sim["ok"]:
        return {
            "status": "error",
            "message": f"Initial simulation failed.\n{initial_sim['message']}",
            "warnings": warnings,
        }

    def objective_fn(z):
        sim = simulate_with_controls(z)
        if not sim["ok"]:
            return 1e12
        return float(sim["objective_value"])

    scipy_constraints = []
    for cons in terminal_constraint_data:
        op = cons["op"]
        lhs_fn = cons["lhs_fn"]
        rhs_fn = cons["rhs_fn"]

        if op == "<=":
            def ineq_fun(z, lf=lhs_fn, rf=rhs_fn):
                sim = simulate_with_controls(z)
                if not sim["ok"]:
                    return -1e12
                x_final = sim["x_final_array"]
                return float(rf(*x_final.tolist(), horizon) - lf(*x_final.tolist(), horizon))
            scipy_constraints.append({"type": "ineq", "fun": ineq_fun})

        elif op == ">=":
            def ineq_fun(z, lf=lhs_fn, rf=rhs_fn):
                sim = simulate_with_controls(z)
                if not sim["ok"]:
                    return -1e12
                x_final = sim["x_final_array"]
                return float(lf(*x_final.tolist(), horizon) - rf(*x_final.tolist(), horizon))
            scipy_constraints.append({"type": "ineq", "fun": ineq_fun})

        elif op in ("=", "=="):
            def eq_fun(z, lf=lhs_fn, rf=rhs_fn):
                sim = simulate_with_controls(z)
                if not sim["ok"]:
                    return 1e12
                x_final = sim["x_final_array"]
                return float(lf(*x_final.tolist(), horizon) - rf(*x_final.tolist(), horizon))
            scipy_constraints.append({"type": "eq", "fun": eq_fun})

    try:
        result = minimize(
            fun=objective_fn,
            x0=np.array(z0, dtype=float),
            bounds=repeated_bounds,
            constraints=scipy_constraints,
            method="SLSQP",
            options={
                "maxiter": 500,
                "ftol": 1e-9,
                "disp": False,
            },
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc), "warnings": warnings}

    if not result.success:
        rng = np.random.default_rng(42)
        for _attempt in range(3):
            z_rand = _random_initial_guess(repeated_bounds, rng)
            try:
                result_retry = minimize(
                    fun=objective_fn,
                    x0=np.array(z_rand, dtype=float),
                    bounds=repeated_bounds,
                    constraints=scipy_constraints,
                    method="SLSQP",
                    options={"maxiter": 500, "ftol": 1e-9, "disp": False},
                )
            except Exception:
                continue
            if result_retry.success:
                result = result_retry
                warnings.append("Initial optimization failed; solution found via multi-start restart.")
                break
        else:
            return {"status": "error", "message": result.message, "warnings": warnings}

    sim_opt = simulate_with_controls(result.x, tight=True)
    if not sim_opt["ok"]:
        return {
            "status": "error",
            "message": f"Optimization finished, but final simulation failed.\n{sim_opt['message']}",
            "warnings": warnings,
        }

    active_bounds = _detect_active_bounds(decision_var_names, np.asarray(result.x, dtype=float), repeated_bounds)
    constraint_report = _evaluate_dynamic_terminal_constraints(
        terminal_constraint_data=terminal_constraint_data,
        x_final=sim_opt["x_final_array"],
        horizon=horizon,
    )

    if getattr(result, "nit", 0) <= 1:
        warnings.append(
            "Optimizer stopped after only 1 iteration. The objective may be poorly scaled, too flat, or the finite-difference step may still be too small."
        )

    controls_changed = bool(np.linalg.norm(np.asarray(result.x, dtype=float) - np.asarray(z0, dtype=float)) > 1e-8)
    if not controls_changed:
        warnings.append(
            "Optimal control is essentially identical to the initial guess."
        )

    if abs(float(initial_sim["objective_value"])) < 1e-10 and abs(float(sim_opt["objective_value"])) < 1e-10:
        warnings.append(
            "Objective values are extremely small in magnitude. Consider scaling the running or terminal cost for better optimizer sensitivity."
        )

    return {
        "status": "success",
        "success": bool(result.success),
        "message": result.message,
        "solver": "SLSQP dynamic single-shooting",
        "iterations": getattr(result, "nit", None),
        "objective_value": float(sim_opt["objective_value"]),
        "raw_objective_value": float(sim_opt["raw_objective_value"]),
        "initial_objective_value": float(initial_sim["objective_value"]),
        "initial_raw_objective_value": float(initial_sim["raw_objective_value"]),
        "warnings": warnings,
        "active_bounds": active_bounds,
        "constraint_report": constraint_report,
        "initial_guess_used": initial_guess_used,
        "final_states": sim_opt["final_states"],
        "control_profiles": sim_opt["control_profiles"],
        "time_horizon": float(horizon),
        "control_intervals": int(dynamic_intervals),
        "trajectory_time": sim_opt["trajectory_time"],
        "state_trajectories": sim_opt["state_trajectories"],
        "running_cost_trajectory": sim_opt["running_cost_trajectory"],
    }


def _build_success_result(
    variable_names,
    x_opt,
    objective_value,
    solver_name,
    message,
    success,
    iterations,
    bounds,
    constraints,
    symbols,
    warnings,
    initial_guess_used,
):
    constraint_report = _evaluate_constraints(constraints, symbols, x_opt)
    active_bounds = _detect_active_bounds(variable_names, x_opt, bounds)

    return {
        "status": "success",
        "success": success,
        "message": message,
        "solver": solver_name,
        "iterations": iterations,
        "variables": {name: float(val) for name, val in zip(variable_names, x_opt)},
        "objective_value": float(objective_value),
        "warnings": warnings,
        "active_bounds": active_bounds,
        "constraint_report": constraint_report,
        "initial_guess_used": initial_guess_used,
    }


def _parse_expression(expr_text: str, symbol_map: dict):
    try:
        return parse_expr(
            expr_text,
            local_dict=symbol_map,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse expression '{expr_text}': {exc}")


def _parse_constraints(text: str, symbol_map: dict):
    constraints = []
    for line in _split_constraint_lines(text):
        lhs, op, rhs = _parse_constraint_line(line, symbol_map)
        constraints.append(
            {
                "text": line,
                "lhs": lhs,
                "rhs": rhs,
                "op": op,
            }
        )
    return constraints


def _parse_constraint_line(line: str, symbol_map: dict):
    operators = ["<=", ">=", "==", "="]
    for op in operators:
        if op in line:
            lhs_text, rhs_text = line.split(op, 1)
            lhs = _parse_expression(lhs_text.strip(), symbol_map)
            rhs = _parse_expression(rhs_text.strip(), symbol_map)
            return lhs, op, rhs
    raise ValueError(f"Constraint must contain one of <=, >=, =, == : {line}")


def _split_constraint_lines(text: str):
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_name_list(text: str, label: str):
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError(f"You must enter at least one {label}.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label.capitalize()} contain duplicates.")
    for name in names:
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid identifier in {label}.")
    return names


def _parse_dynamic_odes(text: str, state_names: list[str], symbol_map: dict):
    ode_map = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not lines:
        raise ValueError("You must enter at least one differential equation.")

    for line in lines:
        if ":" not in line:
            raise ValueError(f"Invalid ODE line: {line}")

        state_name, expr_text = line.split(":", 1)
        state_name = state_name.strip()
        expr_text = expr_text.strip()

        if state_name not in state_names:
            raise ValueError(f"Unknown state variable in ODEs: {state_name}")

        ode_map[state_name] = _parse_expression(expr_text, symbol_map)

    missing = [name for name in state_names if name not in ode_map]
    if missing:
        raise ValueError(f"Missing ODE(s) for: {', '.join(missing)}")

    return [ode_map[name] for name in state_names]


def _parse_bounds(bounds_text: str, variable_names: list[str]):
    bounds = [(None, None) for _ in variable_names]

    for line in [ln.strip() for ln in bounds_text.splitlines() if ln.strip()]:
        if ":" not in line:
            raise ValueError(f"Invalid bound line: {line}")

        var_name, rest = line.split(":", 1)
        var_name = var_name.strip()

        if var_name not in variable_names:
            raise ValueError(f"Unknown variable in bounds: {var_name}")

        parts = [p.strip() for p in rest.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bounds must be in format 'x1: lower, upper' -> {line}")

        lower = None if parts[0].lower() == "none" or parts[0] == "" else float(parts[0])
        upper = None if parts[1].lower() == "none" or parts[1] == "" else float(parts[1])

        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound cannot exceed upper bound in: {line}")

        idx = variable_names.index(var_name)
        bounds[idx] = (lower, upper)

    return bounds


def _parse_integer_vars(text: str, variable_names: list[str]):
    if not text.strip():
        return set()

    names = {item.strip() for item in text.split(",") if item.strip()}
    unknown = names.difference(variable_names)
    if unknown:
        raise ValueError(f"Unknown integer variable(s): {', '.join(sorted(unknown))}")
    return names


def _parse_initial_guess(text: str, variable_names: list[str]):
    guess = {}

    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        if ":" not in line:
            raise ValueError(f"Invalid initial guess line: {line}")

        var_name, value = line.split(":", 1)
        var_name = var_name.strip()

        if var_name not in variable_names:
            raise ValueError(f"Unknown variable in initial guess: {var_name}")

        guess[var_name] = float(value.strip())

    return guess


def _parse_required_value_map(text: str, variable_names: list[str], label: str):
    values = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not lines:
        raise ValueError(f"You must enter {label}.")

    for line in lines:
        if ":" not in line:
            raise ValueError(f"Invalid line in {label}: {line}")

        var_name, value = line.split(":", 1)
        var_name = var_name.strip()

        if var_name not in variable_names:
            raise ValueError(f"Unknown variable in {label}: {var_name}")

        values[var_name] = float(value.strip())

    missing = [name for name in variable_names if name not in values]
    if missing:
        raise ValueError(f"Missing value(s) in {label}: {', '.join(missing)}")

    return values


def _parse_optional_value_map(text: str, variable_names: list[str], label: str):
    values = {}
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        if ":" not in line:
            raise ValueError(f"Invalid line in {label}: {line}")

        var_name, value = line.split(":", 1)
        var_name = var_name.strip()

        if var_name not in variable_names:
            raise ValueError(f"Unknown variable in {label}: {var_name}")

        values[var_name] = float(value.strip())

    return values


def _build_initial_guess(bounds, initial_guess_dict, variable_names):
    guess = []
    for i, name in enumerate(variable_names):
        if name in initial_guess_dict:
            value = initial_guess_dict[name]
        else:
            lower, upper = bounds[i]
            if lower is not None and upper is not None:
                value = 0.5 * (lower + upper)
            elif lower is not None and upper is None:
                value = lower + 1.0
            elif lower is None and upper is not None:
                value = upper - 1.0
            else:
                value = 0.0

        lower, upper = bounds[i]
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)

        guess.append(value)

    return guess


def _repeat_control_bounds(control_names, control_bounds, intervals):
    decision_var_names = []
    repeated_bounds = []

    for k in range(intervals):
        for i, name in enumerate(control_names):
            decision_var_names.append(f"{name}[{k + 1}]")
            repeated_bounds.append(control_bounds[i])

    return decision_var_names, repeated_bounds


def _build_dynamic_initial_guess(control_names, control_bounds, control_guess, intervals):
    initial_guess_used = {}

    for name, (lower, upper) in zip(control_names, control_bounds):
        if name in control_guess:
            value = control_guess[name]
        elif lower is not None and upper is not None:
            value = 0.5 * (lower + upper)
        elif lower is not None and upper is None:
            value = lower + 1.0
        elif lower is None and upper is not None:
            value = upper - 1.0
        else:
            value = 0.0

        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)

        initial_guess_used[name] = float(value)

    z0 = [
        initial_guess_used[name]
        for _k in range(intervals)
        for name in control_names
    ]

    return z0, initial_guess_used


def _linear_coefficients(expr, symbols):
    expr = sp.expand(expr)

    try:
        poly = sp.Poly(expr, *symbols)
    except Exception as exc:
        raise ValueError(f"Expression is not a valid linear polynomial: {expr}") from exc

    if poly.total_degree() > 1:
        raise ValueError(f"Expression is not linear: {expr}")

    zero_subs = {sym: 0 for sym in symbols}
    coeffs = [float(sp.diff(expr, sym).subs(zero_subs)) for sym in symbols]
    const = float(expr.subs(zero_subs))

    return coeffs, const


def _denominator_symbols(expr):
    try:
        _num, denominator = sp.fraction(sp.together(expr))
    except Exception:
        return set()

    if denominator == 1:
        return set()

    return {str(sym) for sym in denominator.free_symbols}


def _denominator_warnings(objective_expr, constraints, bounds, variable_names):
    warnings = []
    bounds_map = {name: bounds[i] for i, name in enumerate(variable_names)}

    all_exprs = [objective_expr]
    for cons in constraints:
        all_exprs.append(cons["lhs"])
        all_exprs.append(cons["rhs"])

    seen = set()
    for expr in all_exprs:
        denom_vars = _denominator_symbols(expr)
        for var_name in denom_vars:
            if var_name in seen:
                continue
            seen.add(var_name)
            lower, _ = bounds_map[var_name]
            if lower is None or lower <= 0:
                warnings.append(
                    f"{var_name} appears in a denominator but does not have a strictly positive lower bound."
                )
            else:
                warnings.append(
                    f"{var_name} appears in a denominator. Its lower bound is positive, which is good for solver stability."
                )

    return warnings


def _denominator_fatal_errors(problem_type, objective_expr, constraints, bounds, variable_names):
    if problem_type != "Nonlinear":
        return []

    bounds_map = {name: bounds[i] for i, name in enumerate(variable_names)}
    all_exprs = [objective_expr]
    for cons in constraints:
        all_exprs.append(cons["lhs"])
        all_exprs.append(cons["rhs"])

    errors = []
    seen = set()
    for expr in all_exprs:
        denom_vars = _denominator_symbols(expr)
        for var_name in denom_vars:
            if var_name in seen:
                continue
            seen.add(var_name)
            lower, _ = bounds_map[var_name]
            if lower is None or lower <= 0:
                errors.append(
                    f"Variable {var_name} appears in a denominator. For nonlinear solving, give it a strictly positive lower bound."
                )

    return errors


def _evaluate_constraints(constraints, symbols, x_opt):
    report = []

    for cons in constraints:
        lhs_fn = sp.lambdify(symbols, cons["lhs"], modules="numpy")
        rhs_fn = sp.lambdify(symbols, cons["rhs"], modules="numpy")

        lhs_val = float(lhs_fn(*x_opt))
        rhs_val = float(rhs_fn(*x_opt))
        op = cons["op"]

        if op == "<=":
            slack = rhs_val - lhs_val
            satisfied = slack >= -TOL
        elif op == ">=":
            slack = lhs_val - rhs_val
            satisfied = slack >= -TOL
        else:
            slack = abs(lhs_val - rhs_val)
            satisfied = slack <= 1e-5

        report.append(
            {
                "constraint": cons["text"],
                "lhs": lhs_val,
                "rhs": rhs_val,
                "slack": float(slack),
                "satisfied": bool(satisfied),
            }
        )

    return report


def _evaluate_dynamic_terminal_constraints(terminal_constraint_data, x_final, horizon):
    report = []

    for cons in terminal_constraint_data:
        lhs_val = float(cons["lhs_fn"](*x_final.tolist(), horizon))
        rhs_val = float(cons["rhs_fn"](*x_final.tolist(), horizon))
        op = cons["op"]

        if op == "<=":
            slack = rhs_val - lhs_val
            satisfied = slack >= -TOL
        elif op == ">=":
            slack = lhs_val - rhs_val
            satisfied = slack >= -TOL
        else:
            slack = abs(lhs_val - rhs_val)
            satisfied = slack <= 1e-5

        report.append(
            {
                "constraint": cons["text"],
                "lhs": lhs_val,
                "rhs": rhs_val,
                "slack": float(slack),
                "satisfied": bool(satisfied),
            }
        )

    return report


def _detect_active_bounds(variable_names, x_opt, bounds):
    active = []

    for name, value, (lower, upper) in zip(variable_names, x_opt, bounds):
        if lower is not None:
            tol = max(1e-6, 1e-6 * max(1.0, abs(value), abs(lower)))
            if abs(value - lower) <= tol:
                active.append(f"{name} is at lower bound ({lower})")

        if upper is not None:
            tol = max(1e-6, 1e-6 * max(1.0, abs(value), abs(upper)))
            if abs(value - upper) <= tol:
                active.append(f"{name} is at upper bound ({upper})")

    return active


def _random_initial_guess(bounds, rng):
    guess = []
    for lower, upper in bounds:
        if lower is not None and upper is not None:
            guess.append(float(rng.uniform(lower, upper)))
        elif lower is not None:
            guess.append(float(lower + rng.exponential(1.0)))
        elif upper is not None:
            guess.append(float(upper - rng.exponential(1.0)))
        else:
            guess.append(float(rng.normal(0.0, 1.0)))
    return guess