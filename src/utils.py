import numpy as np
import matplotlib.pyplot as plt

## June 30: Constrained line search plotting functions
## PLot the resutls
def plot_qp_results(path, obj_values, x_opt, f_opt):
    """Plot results for QP problem"""
    path = np.array(path)

    # First plot: 3D feasible region and central path
    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot feasible region (triangle)
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    ax1.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k-', linewidth=2, label='Feasible region')

    # Fill the triangle
    triangle = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax1.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], alpha=0.3, color='orange')

    # Plot central path
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-', linewidth=2, markersize=6, label='Central path')

    # Plot final solution
    ax1.scatter([x_opt[0]], [x_opt[1]], [x_opt[2]], color='red', s=100,
                label=f'Solution: ({x_opt[0]:.3f}, {x_opt[1]:.3f}, {x_opt[2]:.3f})')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('QP: Feasible Region and Central Path')
    ax1.legend(loc='upper right')

    # Second plot: objective vs iteration
    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(obj_values)), obj_values, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Outer Iteration')
    ax2.set_ylabel('Objective Value')
    ax2.set_title('QP: Objective Value vs Iteration')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"QP Final Results:")
    print(f"Optimal point: x = {x_opt[0]:.6f}, y = {x_opt[1]:.6f}, z = {x_opt[2]:.6f}")
    print(f"Optimal objective value: {f_opt:.6f}")
    print(f"Equality constraint (x+y+z=1): {np.sum(x_opt):.6f}")
    print(f"Inequality constraints: x={x_opt[0]:.6f} >= 0, y={x_opt[1]:.6f} >= 0, z={x_opt[2]:.6f} >= 0")


def plot_lp_results(path, obj_values, x_opt, f_opt):
    """Plot results for LP problem"""
    path = np.array(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # First plot: 2D feasible region and central path
    x_vals = np.linspace(-0.5, 2.5, 100)

    # Plot constraints
    # y >= -x + 1
    y1 = -x_vals + 1
    ax1.plot(x_vals, y1, 'g-', label='y = -x + 1')

    # y <= 1
    ax1.axhline(y=1, color='b', linestyle='-', label='y = 1')

    # x <= 2
    ax1.axvline(x=2, color='r', linestyle='-', label='x = 2')

    # y >= 0
    ax1.axhline(y=0, color='m', linestyle='-', label='y = 0')

    # Fill feasible region
    vertices = np.array([[0,1], [1,0], [2,0], [2,1]])

    ax1.fill(vertices[:, 0], vertices[:, 1], alpha=0.3, color='orange', label='Feasible region')

    # Plot central path
    ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6, label='Central path')

    # Plot final solution
    ax1.scatter([x_opt[0]], [x_opt[1]], color='red', s=100,
                label=f'Solution: ({x_opt[0]:.3f}, {x_opt[1]:.3f})')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('LP: Feasible Region and Central Path')
    ax1.legend(loc='lower left')
    ax1.grid(True)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 1.5)

    # Second plot: objective vs iteration
    # Convert back to maximization for plotting
    max_obj_values = [-val for val in obj_values]
    ax2.plot(range(len(max_obj_values)), max_obj_values, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Outer Iteration')
    ax2.set_ylabel('Objective Value (x + y)')
    ax2.set_title('LP: Objective Value vs Iteration')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print results
    actual_obj = x_opt[0] + x_opt[1]  # For maximization
    print(f"\nLP Final Results:")
    print(f"Optimal point: x = {x_opt[0]:.6f}, y = {x_opt[1]:.6f}")
    print(f"Optimal objective value (x + y): {actual_obj:.6f}")
    print(f"Constraint values:")
    print(f"Condition  y >= -x + 1: {x_opt[1]:.6f} >= {-x_opt[0] + 1:.6f} satisfied")
    print(f"Condition  y >= 0: {x_opt[1]:.6f} >= 0 satisfied")
    print(f"Condition  y <= 1: {x_opt[1]:.6f} <= 1 satisfied")
    print(f"Condition  x <= 2: {x_opt[0]:.6f} <= 2 satisfied")



# June 2: Unconstrained line search plotting functions
def plot_contour_lines(func, xlim, ylim, paths=None, method_names=None,
                             title="Contour Line", levels=20):
    """
    Plot contour lines

    Parameters:
    - func: function to plot
    - xlim: tuple of (xmin, xmax) for x-axis
    - ylim: tuple of (ymin, ymax) for y-axis
    - paths: list of paths
    - method_names: list of method names for legend
    - title: plot title
    - levels: number of contour levels
    """
    # Create grid
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            f_val = func(point, False)[0]
            Z[i, j] = f_val

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot contours
    contour = plt.contour(X, Y, Z, levels=levels, colors='black', alpha=0.6)
    plt.contourf(X, Y, Z, levels=levels, alpha=0.5, cmap='viridis')
    plt.colorbar(label='Function Value')

    # Plot paths if provided
    if paths is not None:
        colors = ['orange', 'blue', 'green', 'orange', 'red']
        markers = ['o', 's', '^', 'v', 'D']

        for i, path in enumerate(paths):
            if len(path) > 0:
                path_array = np.array(path)
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                # Plot path
                plt.plot(path_array[:, 0], path_array[:, 1],
                         color=color, marker=marker, markersize=4,
                         linewidth=2, alpha=0.8)

                # Mark start and end points
                plt.plot(path_array[0, 0], path_array[0, 1],
                         marker='*', color=color, markersize=10,
                         markeredgecolor='black', markeredgewidth=1)
                plt.plot(path_array[-1, 0], path_array[-1, 1],
                         marker='X', color=color, markersize=8,
                         markeredgecolor='black', markeredgewidth=1)

                # Add to legend
                method_name = method_names[i] if method_names else f'Method {i + 1}'
                plt.plot([], [], color=color, marker=marker,
                         linewidth=2, label=method_name)

    # Formatting
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if paths is not None:
        plt.legend()

    plt.tight_layout()
    return plt.gcf()


def plot_convergence(obj_values_list, method_names=None, title="Convergence Plot", log_scale=False):
    """
    Plot function values vs iteration number for comparison of methods.

    Parameters:
    - obj_values_list: list of objective value sequences (one per method)
    - method_names: list of method names for legend
    - title: plot title
    """
    plt.figure(figsize=(10, 8))

    colors = ['orange', 'blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'v', 'D']

    for i, obj_values in enumerate(obj_values_list):
        if len(obj_values) > 0:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            method_name = method_names[i] if method_names else f'Method {i + 1}'

            iterations = range(len(obj_values))
            plt.semilogy(iterations, obj_values,
                         color=color, marker=marker, linewidth=2,
                         markersize=4, label=method_name)
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Function Value (in log scale)')

    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def get_plot_limits(func_name):
    """
    Get appropriate plot limits for different functions.
    """
    limits = { # ((xlim),(ylim))
        'example1': ((-2, 3), (-2, 3)),
        'example2': ((-1, 2), (-1.5, 2)),
        'example3': ((-1.5, 2), (-1, 2)),
        'rosenbrock_func': ((-1.5, 2), (-1.5, 3)),
        'linear_func': ((-2, 2), (-1, 5)),
        'exponential_func': ((-0.8, 2), (-0.5, 2))
    }
    return limits.get(func_name, ((-1.5, 2), (-2, 3)))


def print_final_result(minimizer, method_name, func_name):
    """
    Print final iteration details in a formatted way.
    """
    if len(minimizer.path) > 0:
        final_x = minimizer.path[-1]
        final_f = minimizer.obj_values[-1]
        iterations = len(minimizer.path) - 1

        print(f"\n{method_name} Results for {func_name}:")
        print(f"Final iteration: {iterations}")
        print(f"Final x: {final_x}")
        print(f"Final f(x): {final_f:.6e}")
        print(f"Success: {minimizer.success}")
        print("-" * 50)
