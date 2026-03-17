from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from core.optimization.solvers import solve_optimization_problem

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


class OptimizationController:
    def solve_problem(
        self,
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
    ) -> tuple[str, dict | None]:
        """Return (formatted_text, raw_result_or_None)."""
        result = solve_optimization_problem(
            problem_type=problem_type,
            objective_sense=objective_sense,
            num_vars=num_vars,
            objective_text=objective_text,
            constraints_text=constraints_text,
            bounds_text=bounds_text,
            integer_vars_text=integer_vars_text,
            initial_guess_text=initial_guess_text,
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

        if result["status"] == "not_implemented":
            return result["message"], None

        if result["status"] == "error":
            lines = ["Solver Error:", result["message"]]
            warnings = result.get("warnings", [])
            if warnings:
                lines.append("")
                lines.append("Warnings:")
                for item in warnings:
                    lines.append(f"- {item}")
            return "\n".join(lines), None

        if problem_type == "Dynamic":
            return self._format_dynamic_result(result, objective_sense), result

        return self._format_static_result(result, problem_type, objective_sense), result

    def _format_static_result(self, result: dict, problem_type: str, objective_sense: str) -> str:
        lines = [
            f"Problem Type: {problem_type}",
            f"Objective Sense: {objective_sense}",
            f"Solver: {result.get('solver', 'N/A')}",
            f"Success: {result['success']}",
            f"Message: {result['message']}",
        ]

        if result.get("iterations") is not None:
            lines.append(f"Iterations: {result['iterations']}")

        lines.extend(["", "Optimal Variables:"])
        for name, value in result["variables"].items():
            lines.append(f"  {name} = {value:.6f}")

        lines.append("")
        lines.append(f"Optimal Objective Value: {result['objective_value']:.6f}")

        initial_guess = result.get("initial_guess_used", {})
        if initial_guess:
            lines.append("")
            lines.append("Initial Guess Used:")
            for name, value in initial_guess.items():
                lines.append(f"  {name} = {value:.6f}")

        warnings = result.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for item in warnings:
                lines.append(f"- {item}")

        active_bounds = result.get("active_bounds", [])
        if active_bounds:
            lines.append("")
            lines.append("Active Bounds:")
            for item in active_bounds:
                lines.append(f"- {item}")

        diagnostics = result.get("constraint_report", [])
        if diagnostics:
            lines.append("")
            lines.append("Constraint Diagnostics:")
            for item in diagnostics:
                status = "OK" if item["satisfied"] else "VIOLATED"
                lines.append(
                    f"- {status} | {item['constraint']} | "
                    f"lhs={item['lhs']:.6f}, rhs={item['rhs']:.6f}, "
                    f"slack={item['slack']:.6f}"
                )

        return "\n".join(lines)

    def _format_dynamic_result(self, result: dict, objective_sense: str) -> str:
        lines = [
            "Problem Type: Dynamic",
            f"Objective Sense: {objective_sense}",
            f"Solver: {result.get('solver', 'N/A')}",
            f"Success: {result['success']}",
            f"Message: {result['message']}",
            f"Time Horizon: {result.get('time_horizon', 0):.6f}",
            f"Control Intervals: {result.get('control_intervals', 0)}",
        ]

        if result.get("iterations") is not None:
            lines.append(f"Iterations: {result['iterations']}")

        lines.append("")
        lines.append(f"Optimal Objective Value: {result['objective_value']:.6f}")

        final_states = result.get("final_states", {})
        if final_states:
            lines.append("")
            lines.append("Final States:")
            for name, value in final_states.items():
                lines.append(f"  {name}(T) = {value:.6f}")

        control_profiles = result.get("control_profiles", {})
        if control_profiles:
            lines.append("")
            lines.append("Optimal Control Profiles (piecewise constant):")
            for name, values in control_profiles.items():
                value_text = ", ".join(f"{v:.4f}" for v in values)
                lines.append(f"  {name}: [{value_text}]")

        initial_guess = result.get("initial_guess_used", {})
        if initial_guess:
            lines.append("")
            lines.append("Initial Control Guess Used:")
            for name, value in initial_guess.items():
                lines.append(f"  {name} = {value:.6f}")

        warnings = result.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for item in warnings:
                lines.append(f"- {item}")

        active_bounds = result.get("active_bounds", [])
        if active_bounds:
            lines.append("")
            lines.append("Active Bounds:")
            for item in active_bounds:
                lines.append(f"- {item}")

        diagnostics = result.get("constraint_report", [])
        if diagnostics:
            lines.append("")
            lines.append("Terminal Constraint Diagnostics:")
            for item in diagnostics:
                status = "OK" if item["satisfied"] else "VIOLATED"
                lines.append(
                    f"- {status} | {item['constraint']} | "
                    f"lhs={item['lhs']:.6f}, rhs={item['rhs']:.6f}, "
                    f"slack={item['slack']:.6f}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # MATLAB code generation
    # ------------------------------------------------------------------

    def generate_matlab_code(
        self,
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
        control_method: str = "Piecewise Constant (ZOH — Zero-Order Hold)",
        method_order: int = 3,
    ) -> str:
        if problem_type == "Dynamic":
            return self._gen_dynamic_matlab(
                objective_sense=objective_sense,
                state_vars_text=dynamic_state_vars_text,
                control_vars_text=dynamic_control_vars_text,
                horizon_text=dynamic_horizon_text,
                intervals=dynamic_intervals,
                odes_text=dynamic_odes_text,
                ic_text=dynamic_initial_conditions_text,
                running_cost_text=dynamic_running_cost_text,
                terminal_cost_text=dynamic_terminal_cost_text,
                control_bounds_text=dynamic_control_bounds_text,
                control_guess_text=dynamic_control_guess_text,
                terminal_constraints_text=dynamic_terminal_constraints_text,
                control_method=control_method,
                method_order=method_order,
            )
        if problem_type == "Linear":
            return self._gen_linear_matlab(
                objective_sense=objective_sense,
                num_vars=num_vars,
                objective_text=objective_text,
                constraints_text=constraints_text,
                bounds_text=bounds_text,
                integer_vars_text=integer_vars_text,
            )
        if problem_type == "Nonlinear":
            return self._gen_nonlinear_matlab(
                objective_sense=objective_sense,
                num_vars=num_vars,
                objective_text=objective_text,
                constraints_text=constraints_text,
                bounds_text=bounds_text,
                initial_guess_text=initial_guess_text,
            )
        return "% MATLAB code generation is not supported for this problem type."

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _var_names(num_vars: int) -> list[str]:
        return [f"x{i}" for i in range(1, num_vars + 1)]

    @staticmethod
    def _to_matlab_expr(expr: str, var_names: list[str]) -> str:
        """Convert a Python-style expression to a MATLAB-style expression."""
        result = expr.strip()
        for name in sorted(var_names, key=len, reverse=True):
            idx = var_names.index(name) + 1
            result = re.sub(r"\b" + re.escape(name) + r"\b", f"x({idx})", result)
        result = result.replace("**", ".^")
        result = result.replace("/", "./")
        return result

    @staticmethod
    def _parse_bounds_dict(bounds_text: str, var_names: list[str]) -> tuple[list, list]:
        """Return (lb, ub) lists aligned to var_names. None means Inf/-Inf."""
        lb = [None] * len(var_names)
        ub = [None] * len(var_names)
        for line in bounds_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, rest = line.split(":", 1)
            name = name.strip()
            if name not in var_names:
                continue
            idx = var_names.index(name)
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) >= 1 and parts[0].lower() not in ("none", ""):
                lb[idx] = parts[0]
            if len(parts) >= 2 and parts[1].lower() not in ("none", ""):
                ub[idx] = parts[1]
        return lb, ub

    @staticmethod
    def _matlab_bound_vec(values: list, default_inf: str) -> str:
        parts = [str(v) if v is not None else default_inf for v in values]
        return "[" + "; ".join(parts) + "]"

    # ---- linear ----------------------------------------------------------

    def _gen_linear_matlab(
        self,
        objective_sense: str,
        num_vars: int,
        objective_text: str,
        constraints_text: str,
        bounds_text: str,
        integer_vars_text: str,
    ) -> str:
        var_names = self._var_names(num_vars)
        symbols = sp.symbols(" ".join(var_names))
        if num_vars == 1:
            symbols = (symbols,)
        sym_map = {n: s for n, s in zip(var_names, symbols)}
        maximize = objective_sense.lower() == "maximize"

        # objective coefficients via sympy
        try:
            obj_expr = parse_expr(objective_text, local_dict=sym_map,
                                  transformations=_TRANSFORMATIONS)
            f_coeffs = [float(obj_expr.coeff(s)) for s in symbols]
        except Exception:
            f_coeffs = None

        # parse constraints into matrix form
        ineq_rows: list[tuple[list[float], float]] = []
        eq_rows:   list[tuple[list[float], float]] = []
        parse_ok = True
        for raw in constraints_text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            if ">=" in raw:
                lhs_s, rhs_s = raw.split(">=", 1)
                flip, op = True, "ineq"
            elif "<=" in raw:
                lhs_s, rhs_s = raw.split("<=", 1)
                flip, op = False, "ineq"
            elif "==" in raw or "=" in raw:
                sep = "==" if "==" in raw else "="
                lhs_s, rhs_s = raw.split(sep, 1)
                flip, op = False, "eq"
            else:
                parse_ok = False
                continue
            try:
                diff = (parse_expr(lhs_s, local_dict=sym_map, transformations=_TRANSFORMATIONS)
                        - parse_expr(rhs_s, local_dict=sym_map, transformations=_TRANSFORMATIONS))
                row = [float(diff.coeff(s)) for s in symbols]
                const = -float(diff.subs({s: 0 for s in symbols}))
                if flip:
                    row = [-c for c in row]
                    const = -const
                (ineq_rows if op == "ineq" else eq_rows).append((row, const))
            except Exception:
                parse_ok = False

        lb_vals, ub_vals = self._parse_bounds_dict(bounds_text, var_names)
        int_vars = [v.strip() for v in integer_vars_text.split(",") if v.strip()]
        intcon = [var_names.index(v) + 1 for v in int_vars if v in var_names]
        use_ilp = bool(intcon)
        solver_fn = "intlinprog" if use_ilp else "linprog"

        L: list[str] = []
        if not parse_ok:
            L.append("% WARNING: some constraints could not be parsed — verify A/b manually.")
            L.append("")
        L.append("% " + "=" * 60)
        L.append(f"% {'Integer ' if use_ilp else ''}Linear Programming  —  {objective_sense}")
        L.append("% Generated by ChemEng App  |  MATLAB " + solver_fn)
        L.append("% " + "=" * 60)
        L.append("")

        if f_coeffs is not None:
            sign = -1 if maximize else 1
            f_str = "[" + "; ".join(f"{sign * c:g}" for c in f_coeffs) + "]"
            if maximize:
                L.append("% Objective coefficients (negated — linprog minimises)")
            L.append(f"f = {f_str};")
        else:
            L.append(f"% TODO: replace with actual objective coefficient column vector")
            L.append(f"f = zeros({num_vars}, 1);")
        L.append("")

        if ineq_rows:
            L.append("A = [")
            for row, _ in ineq_rows:
                L.append("  " + ", ".join(f"{c:g}" for c in row) + ";")
            L.append("];")
            L.append("b = [" + "; ".join(f"{r:g}" for _, r in ineq_rows) + "];")
        else:
            L.append("A = [];  b = [];")
        L.append("")

        if eq_rows:
            L.append("Aeq = [")
            for row, _ in eq_rows:
                L.append("  " + ", ".join(f"{c:g}" for c in row) + ";")
            L.append("];")
            L.append("beq = [" + "; ".join(f"{r:g}" for _, r in eq_rows) + "];")
        else:
            L.append("Aeq = [];  beq = [];")
        L.append("")

        L.append(f"lb = {self._matlab_bound_vec(lb_vals, '-Inf')};")
        L.append(f"ub = {self._matlab_bound_vec(ub_vals, 'Inf')};")
        L.append("")

        if use_ilp:
            ic_str = "[" + ", ".join(str(i) for i in intcon) + "]"
            L.append(f"intcon = {ic_str};  % indices of integer variables")
            L.append("")
            L.append("options = optimoptions('intlinprog', 'Display', 'iter');")
            L.append("[x, fval, exitflag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);")
        else:
            L.append("options = optimoptions('linprog', 'Display', 'iter');")
            L.append("[x, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub, options);")
        L.append("")

        L.append("if exitflag == 1")
        L.append("    disp('Optimal solution found.');")
        for i, name in enumerate(var_names, 1):
            L.append(f"    fprintf('{name} = %.6f\\n', x({i}));")
        if maximize:
            L.append("    fprintf('Optimal objective value: %.6f\\n', -fval);  % un-negate")
        else:
            L.append("    fprintf('Optimal objective value: %.6f\\n', fval);")
        L.append("else")
        L.append("    fprintf('Solver did not converge. Exit flag: %d\\n', exitflag);")
        L.append("end")

        return "\n".join(L)

    # ---- nonlinear -------------------------------------------------------

    def _gen_nonlinear_matlab(
        self,
        objective_sense: str,
        num_vars: int,
        objective_text: str,
        constraints_text: str,
        bounds_text: str,
        initial_guess_text: str,
    ) -> str:
        var_names = self._var_names(num_vars)
        maximize = objective_sense.lower() == "maximize"

        obj_ml = self._to_matlab_expr(objective_text, var_names)

        ig: dict[str, str] = {}
        for line in initial_guess_text.splitlines():
            line = line.strip()
            if ":" in line:
                n, v = line.split(":", 1)
                ig[n.strip()] = v.strip()
        x0_str = "[" + "; ".join(ig.get(v, "0") for v in var_names) + "]"

        ineq_exprs: list[str] = []
        eq_exprs:   list[str] = []
        for raw in constraints_text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            if ">=" in raw:
                lhs_s, rhs_s = raw.split(">=", 1)
                ineq_exprs.append(self._to_matlab_expr(
                    f"({rhs_s.strip()}) - ({lhs_s.strip()})", var_names))
            elif "<=" in raw:
                lhs_s, rhs_s = raw.split("<=", 1)
                ineq_exprs.append(self._to_matlab_expr(
                    f"({lhs_s.strip()}) - ({rhs_s.strip()})", var_names))
            elif "==" in raw or "=" in raw:
                sep = "==" if "==" in raw else "="
                lhs_s, rhs_s = raw.split(sep, 1)
                eq_exprs.append(self._to_matlab_expr(
                    f"({lhs_s.strip()}) - ({rhs_s.strip()})", var_names))

        lb_vals, ub_vals = self._parse_bounds_dict(bounds_text, var_names)

        L: list[str] = []
        L.append("% " + "=" * 60)
        L.append(f"% Nonlinear Programming  —  {objective_sense}")
        L.append("% Generated by ChemEng App  |  MATLAB fmincon (interior-point)")
        L.append("% " + "=" * 60)
        L.append("")

        if maximize:
            L.append("% Negate objective — fmincon minimises")
            L.append(f"obj = @(x) -({obj_ml});")
        else:
            L.append(f"obj = @(x) {obj_ml};")
        L.append("")

        if ineq_exprs or eq_exprs:
            L.append("function [c, ceq] = nonlcon(x)")
            if ineq_exprs:
                L.append("    c = [" + "; ...\n         ".join(ineq_exprs) + "];")
            else:
                L.append("    c = [];")
            if eq_exprs:
                L.append("    ceq = [" + "; ...\n           ".join(eq_exprs) + "];")
            else:
                L.append("    ceq = [];")
            L.append("end")
            L.append("")
            nonlcon_arg = "@nonlcon"
        else:
            nonlcon_arg = "[]"

        L.append(f"lb = {self._matlab_bound_vec(lb_vals, '-Inf')};")
        L.append(f"ub = {self._matlab_bound_vec(ub_vals, 'Inf')};")
        L.append(f"x0 = {x0_str};")
        L.append("")
        L.append("options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');")
        L.append(f"[x, fval, exitflag, output] = fmincon(obj, x0, [], [], [], [], lb, ub, {nonlcon_arg}, options);")
        L.append("")
        L.append("if exitflag > 0")
        L.append("    disp('Optimal solution found.');")
        for i, name in enumerate(var_names, 1):
            L.append(f"    fprintf('{name} = %.6f\\n', x({i}));")
        if maximize:
            L.append("    fprintf('Optimal objective value: %.6f\\n', -fval);  % un-negate")
        else:
            L.append("    fprintf('Optimal objective value: %.6f\\n', fval);")
        L.append("else")
        L.append("    fprintf('Solver did not converge. Exit flag: %d\\n', exitflag);")
        L.append("end")

        return "\n".join(L)

    # ---- parametric dynamic methods --------------------------------------

    def _gen_parametric_matlab(
        self,
        control_method: str,
        method_order: int,
        objective_sense: str,
        state_names: list,
        control_names: list,
        T_str: str,
        N: int,
        maximize: bool,
        ode_map: dict,
        ic_map: dict,
        ctrl_lb: dict,
        ctrl_ub: dict,
        ctrl_guess: dict,
        running_cost_text: str,
        terminal_cost_text: str,
        term_ineq: list,
        term_eq: list,
        to_ml,
    ) -> str:
        n_s = len(state_names)
        n_c = len(control_names)
        I   = "    "
        d   = method_order
        sign = "-" if maximize else "+"
        nonlcon_arg = "@term_con" if (term_ineq or term_eq) else "[]"

        is_poly    = "Polynomial"  in control_method
        is_fourier = "Fourier"     in control_method
        # else: Exponential Basis

        if is_poly:
            method_label    = f"Polynomial Basis (degree {d})"
            n_dec_per       = d + 1
            formula_comment = f"% u_j(t) = c0 + c1*(t/T) + ... + c{d}*(t/T)^{d}"
        elif is_fourier:
            method_label    = f"Fourier Series ({d} harmonics)"
            n_dec_per       = 1 + 2 * d
            formula_comment = f"% u_j(t) = a0 + sum_k [ak*cos(k*w*t) + bk*sin(k*w*t)], k=1..{d}"
        else:
            method_label    = f"Exponential Basis ({d} terms)"
            n_dec_per       = d
            tail            = f" + c{d}*exp(-{d-1}*t/T)" if d > 1 else ""
            formula_comment = f"% u_j(t) = c1 + c2*exp(-t/T) + c3*exp(-2t/T) + ...{tail}"

        lb_parts  = [ctrl_lb.get(n, "-Inf") for n in control_names]
        ub_parts  = [ctrl_ub.get(n,  "Inf") for n in control_names]

        # Initial guess: use user's guess if given, else midpoint, else 0
        mid_parts = []
        for n in control_names:
            g  = ctrl_guess.get(n)
            lb = ctrl_lb.get(n)
            ub = ctrl_ub.get(n)
            if g:
                mid_parts.append(g)
            elif lb and ub and lb not in ("-Inf",) and ub not in ("Inf",):
                mid_parts.append(f"({lb} + {ub}) / 2")
            else:
                mid_parts.append("0")

        x0_parts = [ic_map.get(v, "0") for v in state_names]

        # Collect expression warnings before generating code
        all_exprs = [running_cost_text, terminal_cost_text] + list(ode_map.values())
        expr_warnings = []
        for raw in all_exprs:
            expr_warnings.extend(_expr_warnings(raw))

        L: list[str] = []
        L.append("% " + "=" * 60)
        L.append(f"% Dynamic Optimization  —  {objective_sense}  ({method_label})")
        L.append("% Generated by ChemEng App  |  MATLAB fmincon + ode45")
        L.append("% " + "=" * 60)
        L.append("%")
        L.append("% HOW TO USE:")
        L.append("%   1. Save this file as  dynamic_optimization.m  (function file, not script)")
        L.append("%   2. Run by typing:  dynamic_optimization  in the MATLAB Command Window.")
        L.append("%")
        L.append("% EXPRESSION SYNTAX REMINDER:")
        L.append("%   Powers:         CA^2  or  CA**2   (app accepts both)")
        L.append("%   Multiplication: 2*CA  or  2.*CA")
        L.append("%   Division:       CA/2  or  CA./2")
        L.append("%   Do NOT write:   CA(k), CA(end), (-CA)(10)")
        L.append("%   States/controls are scalars at each time step, not arrays.")
        L.append("%   To write -10*CA use:  -10*CA   (not  (-CA)(10)  or  -CA(10))")
        if expr_warnings:
            L.append("%")
            L.append("% *** EXPRESSION WARNINGS — please review before running ***")
            for w in expr_warnings:
                L.append(f"% {w}")
        L.append("%")
        L.append(f"% Control parameterised as: {method_label}")
        L.append("% Physical bounds enforced as linear constraints (A_ineq*u <= b_ineq)")
        L.append("% evaluated at the ODE time-grid nodes.")
        L.append("")
        L.append("function dynamic_optimization()")
        L.append("")

        # ---- Parameters ----
        L.append(f"{I}% ------- Problem parameters -------")
        L.append(f"{I}T_end  = {T_str};")
        L.append(f"{I}N      = {N};       % ODE simulation steps (grid density)")
        L.append(f"{I}dt     = T_end / N;")
        L.append(f"{I}t_grid = linspace(0, T_end, N + 1);")
        L.append(f"{I}n_s    = {n_s};    % states:   {', '.join(state_names)}")
        L.append(f"{I}n_c    = {n_c};    % controls: {', '.join(control_names)}")
        L.append(f"{I}")
        L.append(f"{I}x0_ic  = [{'; '.join(x0_parts)}];  % initial state")
        L.append(f"{I}")

        # ---- Parameterisation ----
        L.append(f"{I}% ------- Control parameterisation: {method_label} -------")
        L.append(f"{I}{formula_comment}")
        if is_poly:
            L.append(f"{I}d      = {d};        % polynomial degree")
            L.append(f"{I}n_dec  = d + 1;     % coefficients per control: c0 … c_d")
            L.append(f"{I}")
            L.append(f"{I}% Evaluation matrix Phi: (N+1) x n_dec  at time-grid nodes")
            L.append(f"{I}Phi = zeros(N+1, n_dec);")
            L.append(f"{I}for ki = 1:N+1")
            L.append(f"{I}    t_norm    = t_grid(ki) / T_end;")
            L.append(f"{I}    Phi(ki,:) = t_norm .^ (0:d);")
            L.append(f"{I}end")
        elif is_fourier:
            L.append(f"{I}n_harm = {d};               % number of harmonics")
            L.append(f"{I}omega  = 2*pi / T_end;      % fundamental angular frequency (rad/time)")
            L.append(f"{I}n_dec  = 1 + 2*n_harm;      % a0, a1, b1, …, a_n, b_n  per control")
            L.append(f"{I}")
            L.append(f"{I}% Evaluation matrix Phi: (N+1) x n_dec  at time-grid nodes")
            L.append(f"{I}Phi = zeros(N+1, n_dec);")
            L.append(f"{I}for ki = 1:N+1")
            L.append(f"{I}    t_k        = t_grid(ki);")
            L.append(f"{I}    Phi(ki,1)  = 1;")
            L.append(f"{I}    for h = 1:n_harm")
            L.append(f"{I}        Phi(ki, 1+2*(h-1)+1) = cos(h * omega * t_k);")
            L.append(f"{I}        Phi(ki, 1+2*(h-1)+2) = sin(h * omega * t_k);")
            L.append(f"{I}    end")
            L.append(f"{I}end")
        else:  # Exponential
            L.append(f"{I}n_exp  = {d};        % number of exponential terms")
            L.append(f"{I}n_dec  = n_exp;      % coefficients per control")
            L.append(f"{I}")
            L.append(f"{I}% Evaluation matrix Phi: (N+1) x n_dec")
            L.append(f"{I}% Phi(ki, i) = exp(-(i-1) * t_grid(ki) / T_end)")
            L.append(f"{I}Phi = zeros(N+1, n_dec);")
            L.append(f"{I}for ki = 1:N+1")
            L.append(f"{I}    Phi(ki,:) = exp(-(0:n_exp-1) .* (t_grid(ki) / T_end));")
            L.append(f"{I}end")

        L.append(f"{I}")
        L.append(f"{I}% Linear constraints: lb <= u(t_k) <= ub  at every grid node")
        L.append(f"{I}ub_col = [{'; '.join(ub_parts)}];   % upper bound per control (column vector)")
        L.append(f"{I}lb_col = [{'; '.join(lb_parts)}];   % lower bound per control")
        L.append(f"{I}A_ineq = [kron(eye(n_c), Phi); kron(eye(n_c), -Phi)];")
        L.append(f"{I}b_ineq = [repelem(ub_col(:), N+1, 1); repelem(-lb_col(:), N+1, 1)];")
        L.append(f"{I}")

        # Initial guess
        L.append(f"{I}% Initial guess: constant at midpoint (first coeff = midpoint, rest = 0)")
        L.append(f"{I}u0 = [];")
        for j, n in enumerate(control_names):
            L.append(f"{I}u0 = [u0; {mid_parts[j]}; zeros(n_dec-1, 1)];  % {n}")
        L.append(f"{I}")

        # Solve
        L.append(f"{I}% ------- Solve -------")
        L.append(f"{I}options = optimoptions('fmincon', 'Display', 'iter', ...")
        L.append(f"{I}    'Algorithm', 'interior-point', 'MaxFunctionEvaluations', 50000);")
        L.append(f"{I}[u_opt, J_opt, exitflag] = fmincon(@objective, u0, A_ineq, b_ineq, [], [], [], [], {nonlcon_arg}, options);")
        L.append(f"{I}")

        # Display
        L.append(f"{I}% ------- Display results -------")
        L.append(f"{I}if exitflag > 0")
        if maximize:
            L.append(f"{I}    fprintf('Optimal objective: %.6f\\n', -J_opt);")
        else:
            L.append(f"{I}    fprintf('Optimal objective: %.6f\\n', J_opt);")
        L.append(f"{I}    x_final = simulate_fwd(u_opt);")
        for i, sname in enumerate(state_names, 1):
            L.append(f"{I}    fprintf('{sname}(T) = %.6f\\n', x_final({i}, end));")
        L.append(f"{I}else")
        L.append(f"{I}    fprintf('Did not converge. exitflag = %d\\n', exitflag);")
        L.append(f"{I}end")
        L.append(f"{I}")

        # Plot (dense time grid for smooth control curve)
        n_plots = n_s + n_c
        L.append(f"{I}% ------- Plot -------")
        L.append(f"{I}x_opt  = simulate_fwd(u_opt);")
        L.append(f"{I}t_plot = linspace(0, T_end, 300);   % dense grid for smooth control curve")
        L.append(f"{I}u_plot = zeros(n_c, 300);")
        L.append(f"{I}for ki = 1:300")
        L.append(f"{I}    u_plot(:, ki) = eval_control(t_plot(ki), u_opt);")
        L.append(f"{I}end")
        L.append(f"{I}figure;")
        for i, sname in enumerate(state_names, 1):
            L.append(f"{I}subplot({n_plots}, 1, {i});")
            L.append(f"{I}plot(t_grid, x_opt({i},:), 'b-', 'LineWidth', 1.5);")
            L.append(f"{I}ylabel('{sname}'); title('State: {sname}'); grid on;")
        for i, cname in enumerate(control_names, 1):
            L.append(f"{I}subplot({n_plots}, 1, {n_s + i});")
            L.append(f"{I}plot(t_plot, u_plot({i},:), 'r-', 'LineWidth', 1.5);")
            L.append(f"{I}ylabel('{cname}'); title('Control: {cname}  ({method_label})'); grid on;")
        L.append(f"{I}xlabel('Time');")
        L.append("")

        # ---- Nested functions ----
        L.append(f"{I}% ================================================================")
        L.append(f"{I}% Nested helper functions — share workspace automatically.")
        L.append(f"{I}% ================================================================")
        L.append("")

        # eval_control: reconstruct control at time t from coefficient vector u_vec
        L.append(f"{I}function u_c = eval_control(t, u_vec)")
        if is_poly:
            L.append(f"{I}    coeffs = reshape(u_vec, [n_c, n_dec]);")
            L.append(f"{I}    t_norm = t / T_end;")
            L.append(f"{I}    basis  = (t_norm .^ (0:d))';")
            L.append(f"{I}    u_c    = coeffs * basis;")
        elif is_fourier:
            L.append(f"{I}    coeffs    = reshape(u_vec, [n_c, n_dec]);")
            L.append(f"{I}    basis     = zeros(n_dec, 1);")
            L.append(f"{I}    basis(1)  = 1;")
            L.append(f"{I}    for h = 1:n_harm")
            L.append(f"{I}        basis(1+2*(h-1)+1) = cos(h * omega * t);")
            L.append(f"{I}        basis(1+2*(h-1)+2) = sin(h * omega * t);")
            L.append(f"{I}    end")
            L.append(f"{I}    u_c = coeffs * basis;")
        else:
            L.append(f"{I}    coeffs = reshape(u_vec, [n_c, n_dec]);")
            L.append(f"{I}    t_norm = t / T_end;")
            L.append(f"{I}    basis  = exp(-(0:n_dec-1)' .* t_norm);")
            L.append(f"{I}    u_c    = coeffs * basis;")
        L.append(f"{I}end")
        L.append("")

        # ode_func
        L.append(f"{I}function dxdt = ode_func(~, x_s, u_c)")
        L.append(f"{I}    dxdt = zeros(n_s, 1);")
        for i, sname in enumerate(state_names, 1):
            rhs_ml = to_ml(ode_map.get(sname, "0"))
            L.append(f"{I}    dxdt({i}) = {rhs_ml};  % d{sname}/dt")
        L.append(f"{I}end")
        L.append("")

        # simulate_fwd: integrates ODEs using the continuous control profile
        L.append(f"{I}function x_traj = simulate_fwd(u_vec)")
        L.append(f"{I}    x_traj = zeros(n_s, N + 1);")
        L.append(f"{I}    x_traj(:, 1) = x0_ic;")
        L.append(f"{I}    for k = 1:N")
        L.append(f"{I}        ode_fcn = @(t, xs) ode_func(t, xs, eval_control(t, u_vec));")
        L.append(f"{I}        [~, xk] = ode45(ode_fcn, [t_grid(k), t_grid(k+1)], x_traj(:, k));")
        L.append(f"{I}        x_traj(:, k+1) = xk(end, :)';")
        L.append(f"{I}    end")
        L.append(f"{I}end")
        L.append("")

        # objective
        L.append(f"{I}function J = objective(u_vec)")
        L.append(f"{I}    x_traj = simulate_fwd(u_vec);")
        L.append(f"{I}    J = 0;")
        if running_cost_text.strip():
            running_ml = to_ml(running_cost_text.strip())
            L.append(f"{I}    for k = 1:N")
            L.append(f"{I}        x_s = x_traj(:, k);")
            L.append(f"{I}        u_c = eval_control(t_grid(k), u_vec);")
            L.append(f"{I}        J   = J {sign} dt * ({running_ml});")
            L.append(f"{I}    end")
        if terminal_cost_text.strip():
            terminal_ml = to_ml(terminal_cost_text.strip())
            L.append(f"{I}    x_s = x_traj(:, end);")
            L.append(f"{I}    u_c = eval_control(T_end, u_vec);")
            L.append(f"{I}    J   = J {sign} ({terminal_ml});")
        L.append(f"{I}end")
        L.append("")

        # term_con (only if needed)
        if term_ineq or term_eq:
            L.append(f"{I}function [c, ceq] = term_con(u_vec)")
            L.append(f"{I}    x_traj = simulate_fwd(u_vec);")
            L.append(f"{I}    x_s    = x_traj(:, end);")
            L.append(f"{I}    u_c    = eval_control(T_end, u_vec);")
            L.append(f"{I}    c    = [" + ("; ".join(term_ineq) if term_ineq else "") + "];")
            L.append(f"{I}    ceq  = [" + ("; ".join(term_eq)   if term_eq   else "") + "];")
            L.append(f"{I}end")
            L.append("")

        L.append("end  % dynamic_optimization")
        return "\n".join(L)

    # ---- dynamic ---------------------------------------------------------

    def _gen_dynamic_matlab(
        self,
        objective_sense: str,
        state_vars_text: str,
        control_vars_text: str,
        horizon_text: str,
        intervals: int,
        odes_text: str,
        ic_text: str,
        running_cost_text: str,
        terminal_cost_text: str,
        control_bounds_text: str,
        control_guess_text: str,
        terminal_constraints_text: str,
        control_method: str = "Piecewise Constant (ZOH — Zero-Order Hold)",
        method_order: int = 3,
    ) -> str:
        state_names   = [s.strip() for s in state_vars_text.split(",") if s.strip()]
        control_names = [s.strip() for s in control_vars_text.split(",") if s.strip()]
        n_s = len(state_names)
        n_c = len(control_names)
        T_str = horizon_text.strip() or "10"
        N = intervals
        maximize      = objective_sense.lower() == "maximize"
        is_parametric = any(m in control_method for m in ("Polynomial", "Fourier", "Exponential"))
        use_foh       = not is_parametric and "FOH" in control_method
        n_dec = N + 1 if use_foh else N    # used only for ZOH/FOH path

        def to_ml(expr: str) -> str:
            result = expr.strip()
            for name in sorted(state_names, key=len, reverse=True):
                result = re.sub(r"\b" + re.escape(name) + r"\b",
                                f"x_s({state_names.index(name) + 1})", result)
            for name in sorted(control_names, key=len, reverse=True):
                result = re.sub(r"\b" + re.escape(name) + r"\b",
                                f"u_c({control_names.index(name) + 1})", result)
            # Replace Python power operator and division
            result = result.replace("**", ".^")
            # Replace bare ^ with .^ only when not already preceded by a dot
            result = re.sub(r"(?<!\.)\^", ".^", result)
            # Element-wise division (scalar-safe)
            result = result.replace("/", "./")
            # Fix doubled ./ from "./" inputs already containing dot
            result = result.replace("../", "./")
            return result

        def _expr_warnings(raw: str) -> list:
            """Return MATLAB comment warnings for suspicious expression patterns."""
            warnings = []
            for name in state_names + control_names:
                # Detect stateName( or controlName( — user is indexing a scalar
                if re.search(r"\b" + re.escape(name) + r"\s*\(", raw):
                    warnings.append(
                        f"% WARNING: '{name}(' found in expression — states/controls are scalars "
                        f"at each time step, not arrays. Did you mean '*(' for multiplication? "
                        f"e.g. use  -{name}  not  (-{name})(10), and  2*{name}  not  {name}(2)."
                    )
            return warnings

        ode_map: dict[str, str] = {}
        for line in odes_text.splitlines():
            if ":" in line:
                name, rhs = line.split(":", 1)
                ode_map[name.strip()] = rhs.strip()

        ic_map: dict[str, str] = {}
        for line in ic_text.splitlines():
            if ":" in line:
                name, val = line.split(":", 1)
                ic_map[name.strip()] = val.strip()

        ctrl_lb: dict[str, str] = {}
        ctrl_ub: dict[str, str] = {}
        for line in control_bounds_text.splitlines():
            if ":" in line:
                name, rest = line.split(":", 1)
                parts = [p.strip() for p in rest.split(",")]
                cname = name.strip()
                if len(parts) >= 1 and parts[0].lower() not in ("none", ""):
                    ctrl_lb[cname] = parts[0]
                if len(parts) >= 2 and parts[1].lower() not in ("none", ""):
                    ctrl_ub[cname] = parts[1]

        ctrl_guess: dict[str, str] = {}
        for line in control_guess_text.splitlines():
            if ":" in line:
                name, val = line.split(":", 1)
                ctrl_guess[name.strip()] = val.strip()

        term_ineq: list[str] = []
        term_eq:   list[str] = []
        for raw in terminal_constraints_text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            if ">=" in raw:
                lhs_s, rhs_s = raw.split(">=", 1)
                term_ineq.append(to_ml(f"({rhs_s.strip()}) - ({lhs_s.strip()})"))
            elif "<=" in raw:
                lhs_s, rhs_s = raw.split("<=", 1)
                term_ineq.append(to_ml(f"({lhs_s.strip()}) - ({rhs_s.strip()})"))
            elif "==" in raw or "=" in raw:
                sep = "==" if "==" in raw else "="
                lhs_s, rhs_s = raw.split(sep, 1)
                term_eq.append(to_ml(f"({lhs_s.strip()}) - ({rhs_s.strip()})"))

        if is_parametric:
            return self._gen_parametric_matlab(
                control_method=control_method,
                method_order=method_order,
                objective_sense=objective_sense,
                state_names=state_names,
                control_names=control_names,
                T_str=T_str,
                N=N,
                maximize=maximize,
                ode_map=ode_map,
                ic_map=ic_map,
                ctrl_lb=ctrl_lb,
                ctrl_ub=ctrl_ub,
                ctrl_guess=ctrl_guess,
                running_cost_text=running_cost_text,
                terminal_cost_text=terminal_cost_text,
                term_ineq=term_ineq,
                term_eq=term_eq,
                to_ml=to_ml,
            )

        # ----------------------------------------------------------
        # Build MATLAB code using the NESTED-FUNCTION pattern.
        #
        # WHY: In MATLAB, a local function defined at the bottom of a
        # *script* file has its own isolated workspace — it cannot see
        # variables like n_c, N, x0_ic, etc.  The only clean fix is to
        # wrap everything in a *function file* and define helpers as
        # NESTED functions inside it.  Nested functions in MATLAB share
        # the parent function's workspace automatically.
        #
        # HOW TO USE: save the generated code as  dynamic_optimization.m
        # and run it by typing  dynamic_optimization  in MATLAB.
        # ----------------------------------------------------------

        I = "    "  # 4-space indent for nested function bodies
        method_label = "FOH (Piecewise Linear)" if use_foh else "ZOH (Piecewise Constant)"
        reshape_dim  = "N + 1" if use_foh else "N"

        # Collect expression warnings before writing code
        all_raw_exprs = ([running_cost_text, terminal_cost_text]
                         + list(ode_map.values())
                         + [s for s in terminal_constraints_text.splitlines()])
        expr_warnings: list[str] = []
        for raw in all_raw_exprs:
            expr_warnings.extend(_expr_warnings(raw))

        L: list[str] = []
        L.append("% " + "=" * 60)
        L.append(f"% Dynamic Optimization  —  {objective_sense}  ({method_label}, Direct Single Shooting)")
        L.append("% Generated by ChemEng App  |  MATLAB fmincon + ode45")
        L.append("% " + "=" * 60)
        L.append("%")
        L.append("% HOW TO USE:")
        L.append("%   1. Save this file as  dynamic_optimization.m  (function file, not script)")
        L.append("%   2. Run by typing:  dynamic_optimization")
        L.append("%      in the MATLAB Command Window or press Run.")
        L.append("%")
        L.append("% EXPRESSION SYNTAX REMINDER:")
        L.append("%   Powers:         CA^2  or  CA**2   (app accepts both)")
        L.append("%   Multiplication: 2*CA")
        L.append("%   Division:       CA/2  or  CA./2")
        L.append("%   WRONG:          CA(k), CA(end), (-CA)(10)  ← invalid array index on scalar!")
        L.append("%   CORRECT:        -CA, -10*CA, (CA - 0.5)^2")
        if expr_warnings:
            L.append("%")
            L.append("% *** EXPRESSION WARNINGS — fix before running ***")
            for w in expr_warnings:
                L.append(f"% {w}")
        L.append("%")
        L.append("% Nested-function pattern is used so all helpers share the workspace.")
        if use_foh:
            L.append("% FOH: control is linearly interpolated between N+1 nodes (smoother profiles).")
        else:
            L.append("% ZOH: control is held constant within each of the N intervals.")
        L.append("")
        L.append("function dynamic_optimization()")
        L.append("")

        # Parameters
        L.append(f"{I}% ------- Problem parameters -------")
        L.append(f"{I}T_end  = {T_str};")
        L.append(f"{I}N      = {N};       % control intervals")
        L.append(f"{I}dt     = T_end / N;")
        L.append(f"{I}t_grid = linspace(0, T_end, N + 1);")
        L.append(f"{I}n_s    = {n_s};    % states:   {', '.join(state_names)}")
        L.append(f"{I}n_c    = {n_c};    % controls: {', '.join(control_names)}")
        L.append(f"{I}")
        x0_parts = [ic_map.get(v, "0") for v in state_names]
        L.append(f"{I}x0_ic  = [{'; '.join(x0_parts)}];  % initial state")
        L.append(f"{I}")

        lb_parts    = [ctrl_lb.get(n, "-Inf") for n in control_names]
        ub_parts    = [ctrl_ub.get(n, "Inf")  for n in control_names]
        guess_parts = [ctrl_guess.get(n, "0") for n in control_names]
        n_rep = "N + 1" if use_foh else "N"
        L.append(f"{I}% Control bounds and initial guess  ({n_rep} values per control for {method_label})")
        L.append(f"{I}lb_ctrl = repmat([{'; '.join(lb_parts)}], {n_rep}, 1);")
        L.append(f"{I}ub_ctrl = repmat([{'; '.join(ub_parts)}], {n_rep}, 1);")
        L.append(f"{I}u0      = repmat([{'; '.join(guess_parts)}], {n_rep}, 1);")
        L.append(f"{I}")

        # Solve
        L.append(f"{I}% ------- Solve -------")
        L.append(f"{I}options = optimoptions('fmincon', 'Display', 'iter', ...")
        L.append(f"{I}    'Algorithm', 'interior-point', 'MaxFunctionEvaluations', 50000);")
        nonlcon_arg = "@term_con" if (term_ineq or term_eq) else "[]"
        L.append(f"{I}[u_opt, J_opt, exitflag] = fmincon(@objective, u0, [], [], [], [], lb_ctrl, ub_ctrl, {nonlcon_arg}, options);")
        L.append(f"{I}")

        # Display
        L.append(f"{I}% ------- Display results -------")
        L.append(f"{I}if exitflag > 0")
        L.append(f"{I}    u_opt_mat = reshape(u_opt, [n_c, {reshape_dim}]);")
        if maximize:
            L.append(f"{I}    fprintf('Optimal objective: %.6f\\n', -J_opt);")
        else:
            L.append(f"{I}    fprintf('Optimal objective: %.6f\\n', J_opt);")
        L.append(f"{I}    x_final = simulate_fwd(u_opt_mat);")
        for i, sname in enumerate(state_names, 1):
            L.append(f"{I}    fprintf('{sname}(T) = %.6f\\n', x_final({i}, end));")
        L.append(f"{I}else")
        L.append(f"{I}    fprintf('Did not converge. exitflag = %d\\n', exitflag);")
        L.append(f"{I}end")
        L.append(f"{I}")

        # Plot
        L.append(f"{I}% ------- Plot -------")
        L.append(f"{I}u_opt_mat = reshape(u_opt, [n_c, {reshape_dim}]);")
        L.append(f"{I}x_opt     = simulate_fwd(u_opt_mat);")
        L.append(f"{I}figure;")
        n_plots = n_s + n_c
        for i, sname in enumerate(state_names, 1):
            L.append(f"{I}subplot({n_plots}, 1, {i});")
            L.append(f"{I}plot(t_grid, x_opt({i},:), 'b-', 'LineWidth', 1.5);")
            L.append(f"{I}ylabel('{sname}'); title('State: {sname}'); grid on;")
        for i, cname in enumerate(control_names, 1):
            L.append(f"{I}subplot({n_plots}, 1, {n_s + i});")
            if use_foh:
                # FOH: control defined at all N+1 nodes → plain plot
                L.append(f"{I}plot(t_grid, u_opt_mat({i},:), 'r-', 'LineWidth', 1.5);")
            else:
                # ZOH: control constant per interval → stairs plot
                L.append(f"{I}stairs(t_grid(1:end-1), u_opt_mat({i},:), 'r-', 'LineWidth', 1.5);")
            L.append(f"{I}ylabel('{cname}'); title('Control: {cname}  ({method_label})'); grid on;")
        L.append(f"{I}xlabel('Time');")
        L.append("")

        # ---- nested functions ----
        L.append(f"{I}% ================================================================")
        L.append(f"{I}% Nested helper functions — share workspace automatically.")
        L.append(f"{I}% ================================================================")
        L.append("")

        # ode_func
        L.append(f"{I}function dxdt = ode_func(~, x_s, u_c)")
        L.append(f"{I}    dxdt = zeros(n_s, 1);")
        for i, sname in enumerate(state_names, 1):
            rhs_ml = to_ml(ode_map.get(sname, "0"))
            L.append(f"{I}    dxdt({i}) = {rhs_ml};  % d{sname}/dt")
        L.append(f"{I}end")
        L.append("")

        # simulate_fwd — ZOH or FOH interpolation
        L.append(f"{I}function x_traj = simulate_fwd(u_mat)")
        L.append(f"{I}    x_traj = zeros(n_s, N + 1);")
        L.append(f"{I}    x_traj(:, 1) = x0_ic;")
        L.append(f"{I}    for k = 1:N")
        if use_foh:
            L.append(f"{I}        u_s = u_mat(:, k);       % control at interval start")
            L.append(f"{I}        u_e = u_mat(:, k + 1);   % control at interval end")
            L.append(f"{I}        % Linearly interpolate control within [t_grid(k), t_grid(k+1)]")
            L.append(f"{I}        ode_fcn = @(t, xs) ode_func(t, xs, ...")
            L.append(f"{I}            u_s + (t - t_grid(k)) ./ dt .* (u_e - u_s));")
            L.append(f"{I}        [~, xk] = ode45(ode_fcn, [t_grid(k), t_grid(k+1)], x_traj(:, k));")
        else:
            L.append(f"{I}        u_k = u_mat(:, k);")
            L.append(f"{I}        [~, xk] = ode45(@(t, xs) ode_func(t, xs, u_k), [t_grid(k), t_grid(k+1)], x_traj(:, k));")
        L.append(f"{I}        x_traj(:, k+1) = xk(end, :)';")
        L.append(f"{I}    end")
        L.append(f"{I}end")
        L.append("")

        # objective
        L.append(f"{I}function J = objective(u_vec)")
        L.append(f"{I}    u_mat  = reshape(u_vec, [n_c, {reshape_dim}]);")
        L.append(f"{I}    x_traj = simulate_fwd(u_mat);")
        L.append(f"{I}    J = 0;")
        if running_cost_text.strip():
            running_ml = to_ml(running_cost_text.strip())
            sign = "-" if maximize else "+"
            L.append(f"{I}    for k = 1:N")
            L.append(f"{I}        x_s = x_traj(:, k);")
            if use_foh:
                L.append(f"{I}        u_c = (u_mat(:, k) + u_mat(:, k+1)) ./ 2;  % midpoint control (FOH)")
            else:
                L.append(f"{I}        u_c = u_mat(:, k);")
            L.append(f"{I}        J = J {sign} dt * ({running_ml});")
            L.append(f"{I}    end")
        if terminal_cost_text.strip():
            terminal_ml = to_ml(terminal_cost_text.strip())
            sign = "-" if maximize else "+"
            L.append(f"{I}    x_s = x_traj(:, end);")
            L.append(f"{I}    u_c = zeros(n_c, 1);")
            L.append(f"{I}    J = J {sign} ({terminal_ml});")
        L.append(f"{I}end")
        L.append("")

        # term_con (only if needed)
        if term_ineq or term_eq:
            L.append(f"{I}function [c, ceq] = term_con(u_vec)")
            L.append(f"{I}    u_mat  = reshape(u_vec, [n_c, {reshape_dim}]);")
            L.append(f"{I}    x_traj = simulate_fwd(u_mat);")
            L.append(f"{I}    x_s    = x_traj(:, end);")
            L.append(f"{I}    u_c    = zeros(n_c, 1);")
            L.append(f"{I}    c    = [" + ("; ".join(term_ineq) if term_ineq else "") + "];")
            L.append(f"{I}    ceq  = [" + ("; ".join(term_eq)   if term_eq   else "") + "];")
            L.append(f"{I}end")
            L.append("")

        # close main function
        L.append("end  % dynamic_optimization")

        return "\n".join(L)
