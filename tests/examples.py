import numpy as np


def quadratic_1(x, need_hessian=False):
    """
    Quadratic function with Q = [[1, 0], [0, 1]] (circles)
    f(x) = x^T Q x
    """
    Q = np.array([[1, 0], [0, 1]])

    f_val = x.T @ Q @ x
    grad = 2 * Q @ x

    if need_hessian:
        hess = 2 * Q
        return f_val, grad, hess
    else:
        return f_val, grad


def quadratic_2(x, need_hessian=False):
    """
    Quadratic function with Q = [[1, 0], [0, 100]] (axis-aligned ellipses)
    f(x) = x^T Q x
    """
    Q = np.array([[1, 0], [0, 100]])

    f_val = x.T @ Q @ x
    grad = 2 * Q @ x

    if need_hessian:
        hess = 2 * Q
        return f_val, grad, hess
    else:
        return f_val, grad


def quadratic_3(x, need_hessian=False):
    """
    Quadratic function with rotated ellipse contours
    Q = R^T * D * R where R is rotation matrix and D is diagonal
    """
    # Rotation matrix for sqrt(3)/2 and -0.5, 0.5 and sqrt(3)/2
    R = np.array([[np.sqrt(3) / 2, -0.5],
                  [0.5, np.sqrt(3) / 2]])
    D = np.array([[100, 0], [0, 1]])
    Q = R.T @ D @ R

    f_val = x.T @ Q @ x
    grad = 2 * Q @ x

    if need_hessian:
        hess = 2 * Q
        return f_val, grad, hess
    else:
        return f_val, grad


def rosenbrock(x, need_hessian=False):
    """
    Rosenbrock function: f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2
    Famous non-convex optimization benchmark with banana-shaped contours
    """
    x1, x2 = x[0], x[1]

    f_val = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    # Gradient
    df_dx1 = -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1 ** 2)
    grad = np.array([df_dx1, df_dx2])

    if need_hessian:
        # Hessian
        d2f_dx1dx1 = -400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2
        d2f_dx1dx2 = -400 * x1
        d2f_dx2dx1 = -400 * x1
        d2f_dx2dx2 = 200

        hess = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                         [d2f_dx2dx1, d2f_dx2dx2]])
        return f_val, grad, hess
    else:
        return f_val, grad


def linear_function(x, need_hessian=False):
    """
    Linear function: f(x) = a^T x
    Contour lines are straight lines
    """
    a = np.array([2, -3])  # Choose non-zero vector

    f_val = a.T @ x
    grad = a

    if need_hessian:
        hess = np.zeros((2, 2))
        return f_val, grad, hess
    else:
        return f_val, grad


def exponential_function(x, need_hessian=False):
    """
    Exponential function: f(x1, x2) = exp(x1 + 3*x2 - 0.1) + exp(x1 - 3*x2 - 0.1) + exp(-x1 - 0.1)
    Contour lines look like smoothed corner triangles
    """
    x1, x2 = x[0], x[1]

    exp1 = np.exp(x1 + 3 * x2 - 0.1)
    exp2 = np.exp(x1 - 3 * x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)

    f_val = exp1 + exp2 + exp3

    # Gradient
    df_dx1 = exp1 + exp2 - exp3
    df_dx2 = 3 * exp1 - 3 * exp2
    grad = np.array([df_dx1, df_dx2])

    if need_hessian:
        # Hessian
        d2f_dx1dx1 = exp1 + exp2 + exp3
        d2f_dx1dx2 = 3 * exp1 - 3 * exp2
        d2f_dx2dx1 = 3 * exp1 - 3 * exp2
        d2f_dx2dx2 = 9 * exp1 + 9 * exp2

        hess = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                         [d2f_dx2dx1, d2f_dx2dx2]])
        return f_val, grad, hess
    else:
        return f_val, grad


# Dictionary of all test functions for easy access
TEST_FUNCTIONS = {
    'quadratic_1': quadratic_1,
    'quadratic_2': quadratic_2,
    'quadratic_3': quadratic_3,
    'rosenbrock': rosenbrock,
    'linear': linear_function,
    'exponential': exponential_function
}