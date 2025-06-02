import numpy as np


def example1(x, need_hessian=False):
    """
    Contour lines are circles
    Q = [[1, 0], [0, 1]]
    Quadratic function: f(x) = x^T Q x
    """
    Q = np.array([[1, 0], [0, 1]])

    f_val = x.T @ Q @ x
    grad = 2 * Q @ x

    if need_hessian:
        hess = 2 * Q
        return f_val, grad, hess
    else:
        return f_val, grad


def example2(x, need_hessian=False):
    """
    Contour lines are axis aligned ellipses
    Q = [[1, 0], [0, 100]]
    Quadratic function: f(x) = x^T Q x
    """
    Q = np.array([[1, 0], [0, 100]])

    f_val = x.T @ Q @ x
    grad = 2 * Q @ x

    if need_hessian:
        hess = 2 * Q
        return f_val, grad, hess
    else:
        return f_val, grad


def example3(x, need_hessian=False):
    """
    Contour lines are rotated ellipses
    Quadratic function:
    Q = R^T * D * R where R is rotation matrix and D is diagonal
    """
    # calculate the rotation matrix
    R = np.array([[np.sqrt(3)/2, -0.5],
                  [0.5,         np.sqrt(3)/2]])
    D = np.array([[100, 0], [0, 1]])
    Q = R.T @ D @ R

    f_val = x.T @ Q @ x
    grad = 2 * Q @ x

    if need_hessian:
        hess = 2 * Q
        return f_val, grad, hess
    else:
        return f_val, grad


def rosenbrock_func(x, need_hessian=False):
    """
    Rosenbrock function: f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2
    """
    x1, x2 = x[0], x[1]

    f_val = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    # calculate the gradient
    df_dx1 = -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1 ** 2)
    grad = np.array([df_dx1, df_dx2])

    if need_hessian:
        # calculate the Hessian
        d2f_dx1dx1 = -400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2
        d2f_dx1dx2 = -400 * x1
        d2f_dx2dx1 = -400 * x1
        d2f_dx2dx2 = 200

        hess = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                         [d2f_dx2dx1, d2f_dx2dx2]])
        return f_val, grad, hess
    else:
        return f_val, grad


def linear_func(x, need_hessian=False):
    """Properly implemented linear function"""
    a = np.array([2, -3])
    f_val = a.T @ x
    grad = a.copy()

    if need_hessian:
        return f_val, grad, np.zeros((2, 2))
    return f_val, grad  # Note: no hessian returned here


def exponential_func(x, need_hessian=False):
    """
    Exponential function: f(x1, x2) = exp(x1 + 3*x2 - 0.1) + exp(x1 - 3*x2 - 0.1) + exp(-x1 - 0.1)
    Contour lines look like smoothed corner triangles
    """
    x1, x2 = x[0], x[1]

    exp1 = np.exp(x1 + 3 * x2 - 0.1)
    exp2 = np.exp(x1 - 3 * x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)

    f_val = exp1 + exp2 + exp3

    # calculate the gradient
    df_dx1 = exp1 + exp2 - exp3
    df_dx2 = 3 * exp1 - 3 * exp2
    grad = np.array([df_dx1, df_dx2])

    if need_hessian:
        # calculate the Hessian
        d2f_dx1dx1 = exp1 + exp2 + exp3
        d2f_dx1dx2 = 3 * exp1 - 3 * exp2
        d2f_dx2dx1 = 3 * exp1 - 3 * exp2
        d2f_dx2dx2 = 9 * exp1 + 9 * exp2

        hess = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                         [d2f_dx2dx1, d2f_dx2dx2]])
        return f_val, grad, hess
    else:
        return f_val, grad


# Dictionary of all test results
RESULTS_ = {
    'example1': example1,
    'example2': example2,
    'example3': example3,
    'rosenbrock': rosenbrock_func,
    'linear': linear_func,
    'exponential': exponential_func
}
