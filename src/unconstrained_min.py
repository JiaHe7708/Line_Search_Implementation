import numpy as np
import warnings


class LineSearchMinimizer:
    """
    Line search minimization with Gradient Descent and Newton methods.
    Supports backtracking line search with Wolfe conditions.
    """

    def __init__(self):
        self.path = []
        self.obj_values = []
        self.success = False

    def minimize(self, f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method='gd'):
        """
        Minimize function f starting from x0 using line search.

        Parameters:
        - f: function that returns (value, gradient, hessian) or (value, gradient)
        - x0: starting point (numpy array)
        - obj_tol: tolerance for objective function change
        - param_tol: tolerance for parameter change
        - max_iter: maximum number of iterations
        - method: 'gd' for gradient descent, 'newton' for Newton's method

        Returns:
        - final_x: final location
        - final_f: final objective value
        - success: boolean flag indicating convergence
        """

        # Initialize
        x = np.array(x0, dtype=float)
        self.path = [x.copy()]

        # Evaluate initial point
        if method == 'newton':
            f_val, grad, hess = f(x, True)
        else:
            f_val, grad = f(x, False)

        self.obj_values = [f_val]
        prev_f = f_val

        print(f"Iteration 0: x = {x}, f(x) = {f_val:.6e}")

        for i in range(1, max_iter + 1):
            # Compute search direction
            if method == 'newton':
                # Newton direction: solve H * p = -g
                try:
                    p = -np.linalg.solve(hess, grad)
                    # Check if Newton decrement is small enough
                    newton_decrement = np.sqrt(grad.T @ p)
                    if newton_decrement < np.sqrt(2 * obj_tol):
                        print(f"Converged due to small Newton decrement: {newton_decrement:.6e}")
                        self.success = True
                        break
                except np.linalg.LinAlgError:
                    # Deal with the potential error when Hessian is singular
                    p = -grad
                    warnings.warn("!!! Hessian is singular, using gradient descent step")
            else:
                # Gradient descent direction
                p = -grad


            # Line search with backtracking
            alpha = self._backtracking_line_search(f, x, f_val, grad, p, method == 'newton')

            # Update
            x_new = x + alpha * p

            # Evaluate new point
            if method == 'newton':
                f_new, grad_new, hess_new = f(x_new, True)
                hess = hess_new
            else:
                f_new, grad_new = f(x_new, False)
                # add check point
                if np.allclose(grad_new, 0):
                    print("!!! Zero gradient detected - linear function at minimum")
                    self.success = True
                    return x, f_val, True

            # Store path
            self.path.append(x_new.copy())
            self.obj_values.append(f_new)

            print(f"Iteration {i}: x = {x_new}, f(x) = {f_new:.6e}")

            # Check convergence criteria
            # Objective function change
            if abs(f_new - prev_f) < obj_tol:
                print(f"...Converged due to small objective change: {abs(f_new - prev_f):.6e}")
                self.success = True
                break

            # Parameter change
            if np.linalg.norm(x_new - x) < param_tol:
                print(f"...Converged due to small parameter change: {np.linalg.norm(x_new - x):.6e}")
                self.success = True
                break

            # Update for next iteration
            x = x_new
            f_val = f_new
            grad = grad_new
            prev_f = f_val

        if not self.success:
            print(f"Maximum iterations ({max_iter}) reached without convergence")

        return x, f_val, self.success

    def _backtracking_line_search(self, f, x, f_val, grad, p, need_hess=False,
                                  c1=0.01, rho=0.5, alpha_init=1.0):
        """
        Backtracking line search with Wolfe conditions.
        """
        alpha = alpha_init

        while True:
            x_new = x + alpha * p

            # Evaluate function at new point
            if need_hess:
                f_new, _, _ = f(x_new, True)
            else:
                f_new, _ = f(x_new, False)

            # Check Armijo condition
            if f_new <= f_val + c1 * alpha * (grad.T @ p):
                break

            # Reduce step size
            alpha *= rho

            # Prevent infinite loop with very small step
            if alpha < 1e-16:
                break

        return alpha


def minimize_function(f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method='gd'):
    """
    Convenience function for minimization.

    Parameters:
    - f: objective function
    - x0: starting point
    - obj_tol: objective tolerance
    - param_tol: parameter tolerance
    - max_iter: maximum iterations
    - method: 'gd' or 'newton'

    Returns:
    - minimizer: LineSearchMinimizer object with results
    """
    minimizer = LineSearchMinimizer()
    final_x, final_f, success = minimizer.minimize(f, x0, obj_tol, param_tol, max_iter, method)
    return minimizer