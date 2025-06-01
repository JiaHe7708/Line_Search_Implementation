import numpy as np
import matplotlib.pyplot as plt


def plot_contours_with_paths(func, xlim, ylim, paths=None, method_names=None,
                             title="Contour Plot", levels=20):
    """
    Plot contour lines of a function and optionally overlay optimization paths.

    Parameters:
    - func: function to plot (should take x and return at least f_val)
    - xlim: tuple of (xmin, xmax) for x-axis
    - ylim: tuple of (ymin, ymax) for y-axis
    - paths: list of paths (each path is a list of points)
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
            try:
                f_val = func(point, False)[0]  # Get only function value
                Z[i, j] = f_val
            except:
                Z[i, j] = np.nan

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot contours
    contour = plt.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
    plt.contourf(X, Y, Z, levels=levels, alpha=0.3, cmap='viridis')
    plt.colorbar(label='Function Value')

    # Plot paths if provided
    if paths is not None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
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


def plot_convergence(obj_values_list, method_names=None, title="Convergence Plot"):
    """
    Plot function values vs iteration number for comparison of methods.

    Parameters:
    - obj_values_list: list of objective value sequences (one per method)
    - method_names: list of method names for legend
    - title: plot title
    """
    plt.figure(figsize=(10, 6))

    colors = ['red', 'blue', 'green', 'orange', 'purple']
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

    plt.xlabel('Iteration')
    plt.ylabel('Function Value (log scale)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def get_plot_limits(func_name):
    """
    Get appropriate plot limits for different functions.
    """
    limits = {
        'quadratic_1': ((-2, 2), (-2, 2)),
        'quadratic_2': ((-1, 1), (-0.5, 0.5)),
        'quadratic_3': ((-1, 1), (-1, 1)),
        'rosenbrock': ((-2, 2), (-1, 3)),
        'linear': ((-2, 2), (-2, 2)),
        'exponential': ((-1, 1), (-1, 1))
    }

    return limits.get(func_name, ((-2, 2), (-2, 2)))


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