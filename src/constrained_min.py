import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rho, x0,
                t0=1.0, mu=10.0, tolerance=1e-8, max_iter=100):
    """
    Interior point method for constrained optimization using log-barrier method.

    Parameters:
    - func: callable, objective function with interface (x, True) -> (f, grad, hess)
    - ineq_constraints: list of callables, inequality constraints g_i(x) >= 0
    - eq_constraints_mat: matrix A for equality constraints Ax = b
    - eq_constraints_rhs: vector b for equality constraints Ax = b
    - x0: initial point (must be strictly feasible)
    - t0: initial barrier parameter
    - mu: barrier parameter increase factor
    - tolerance: convergence tolerance
    - max_iter: maximum outer iterations

    Returns:
    - x_opt: optimal point
    - f_opt: optimal objective value
    - success: convergence flag
    - path: list of points from outer iterations
    - obj_values: list of objective values from outer iterations
    """

    x = np.array(x0, dtype=float)
    t = t0
    path = [x.copy()]
    obj_values = []

    # Check if we have equality constraints
    has_eq_constraints = (eq_constraints_mat is not None and
                          eq_constraints_rho is not None and
                          eq_constraints_mat.size > 0)

    if has_eq_constraints:
        eq_constraints_mat = np.array(eq_constraints_mat)
        eq_constraints_rhs = np.array(eq_constraints_rho)

    for outer_iter in range(max_iter):
        # Define barrier function
        def barrier_func(x, compute_grad_hess=True):
            # Original objective
            f_val, f_grad, f_hess = func(x, compute_grad_hess)

            # Log-barrier terms
            barrier_val = 0.0
            barrier_grad = np.zeros_like(x)
            barrier_hess = np.zeros((len(x), len(x)))

            for constraint in ineq_constraints:
                g_val, g_grad, g_hess = constraint(x, compute_grad_hess)

                if g_val <= 0:
                    return np.inf, None, None

                # -log(g(x))
                barrier_val -= np.log(g_val)
                if compute_grad_hess:
                    barrier_grad -= g_grad / g_val
                    barrier_hess -= (g_hess * g_val - np.outer(g_grad, g_grad)) / (g_val ** 2)

            total_f = t * f_val + barrier_val
            total_grad = t * f_grad + barrier_grad if compute_grad_hess else None
            total_hess = t * f_hess + barrier_hess if compute_grad_hess else None

            return total_f, total_grad, total_hess

        # Solve barrier problem using Newton's method with equality constraints
        x_new = solve_barrier_problem(x, barrier_func, eq_constraints_mat,
                                      eq_constraints_rho if has_eq_constraints else None)

        # Store results
        f_val, _, _ = func(x_new, True)
        obj_values.append(f_val)
        path.append(x_new.copy())

        # Check convergence
        if len(ineq_constraints) / t < tolerance:
            break

        # Update barrier parameter and point
        t *= mu
        x = x_new

    f_opt, _, _ = func(x, True)
    return x, f_opt, True, path, obj_values


def solve_barrier_problem(x0, barrier_func, eq_mat=None, eq_rhs=None,
                          tolerance=1e-12, max_iter=100):
    """
    Solve the barrier subproblem using Newton's method with equality constraints.
    """
    x = np.array(x0, dtype=float)

    has_eq = eq_mat is not None and eq_rhs is not None

    for i in range(max_iter):
        f_val, grad, hess = barrier_func(x, True)

        if f_val == np.inf:
            break

        if has_eq:
            # Newton step with equality constraints
            # Solve KKT system: [H A^T; A 0] [dx; lambda] = [-grad; 0]
            A = eq_mat
            n = len(x)
            m = A.shape[0] if A.ndim > 1 else 1

            # Build KKT matrix
            kkt_matrix = np.zeros((n + m, n + m))
            kkt_matrix[:n, :n] = hess
            if A.ndim > 1:
                kkt_matrix[:n, n:] = A.T
                kkt_matrix[n:, :n] = A
            else:
                kkt_matrix[:n, n] = A
                kkt_matrix[n, :n] = A

            # Build RHS
            rhs = np.zeros(n + m)
            rhs[:n] = -grad

            try:
                solution = solve(kkt_matrix, rhs)
                dx = solution[:n]
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                dx = -np.linalg.pinv(hess) @ grad
        else:
            # Unconstrained Newton step
            try:
                dx = -solve(hess, grad)
            except np.linalg.LinAlgError:
                dx = -np.linalg.pinv(hess) @ grad

        # Line search (simple backtracking)
        alpha = 1.0
        while alpha > 1e-8:
            x_new = x + alpha * dx
            f_new, _, _ = barrier_func(x_new, False)
            if f_new < f_val:
                break
            alpha *= 0.5

        x = x + alpha * dx

        # Check convergence
        if np.linalg.norm(dx) < tolerance:
            break

    return x
