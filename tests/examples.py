import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Update June 30

np.seterr(all='ignore') # Ignores all floating-point errors and warnings

# Quadratic Programming Problem Functions
def qp_objective(x, compute_grad_hess=True):
    """
    Objective: x^2 + y^2 + (z+1)^2
    """
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2

    if not compute_grad_hess:
        return f, None, None

    grad = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    hess = np.array([[2, 0, 0],
                     [0, 2, 0],
                     [0, 0, 2]])

    return f, grad, hess


def qp_ineq_constraint_x(x, compute_grad_hess=True):
    """Inequality constraint: x >= 0"""
    g = x[0]

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([1, 0, 0])
    hess = np.zeros((3, 3))

    return g, grad, hess


def qp_ineq_constraint_y(x, compute_grad_hess=True):
    """Inequality constraint: y >= 0"""
    g = x[1]

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([0, 1, 0])
    hess = np.zeros((3, 3))

    return g, grad, hess


def qp_ineq_constraint_z(x, compute_grad_hess=True):
    """Inequality constraint: z >= 0"""
    g = x[2]

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([0, 0, 1])
    hess = np.zeros((3, 3))

    return g, grad, hess


# QP problem data
qp_ineq_constraints = [qp_ineq_constraint_x, qp_ineq_constraint_y, qp_ineq_constraint_z]
qp_eq_constraints_mat = np.array([[1, 1, 1]])  # x + y + z = 1
qp_eq_constraints_rhs = np.array([1])
qp_x0 = np.array([0.1, 0.2, 0.7])


# Linear Programming Problem Functions
def lp_objective(x, compute_grad_hess=True):
    """
    Objective: -(x + y) for maximization of x + y
    """
    f = -(x[0] + x[1])

    if not compute_grad_hess:
        return f, None, None

    grad = np.array([-1, -1])
    hess = np.zeros((2, 2))

    return f, grad, hess


def lp_ineq_constraint_1(x, compute_grad_hess=True):
    """Inequality constraint: y >= -x + 1  =>  x + y - 1 >= 0"""
    g = x[0] + x[1] - 1

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([1, 1])
    hess = np.zeros((2, 2))

    return g, grad, hess


def lp_ineq_constraint_2(x, compute_grad_hess=True):
    """Inequality constraint: y <= 1  =>  1 - y >= 0"""
    g = 1 - x[1]

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([0, -1])
    hess = np.zeros((2, 2))

    return g, grad, hess


def lp_ineq_constraint_3(x, compute_grad_hess=True):
    """Inequality constraint: x <= 2  =>  2 - x >= 0"""
    g = 2 - x[0]

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([-1, 0])
    hess = np.zeros((2, 2))

    return g, grad, hess


def lp_ineq_constraint_4(x, compute_grad_hess=True):
    """Inequality constraint: y >= 0"""
    g = x[1]

    if not compute_grad_hess:
        return g, None, None

    grad = np.array([0, 1])
    hess = np.zeros((2, 2))

    return g, grad, hess


# LP problem data
lp_ineq_constraints = [lp_ineq_constraint_1, lp_ineq_constraint_2,
                       lp_ineq_constraint_3, lp_ineq_constraint_4]
lp_eq_constraints_mat = None
lp_eq_constraints_rhs = None
lp_x0 = np.array([0.5, 0.75])



## June 2
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
