import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.unconstrained_min import minimize_function
from src.utils import plot_contour_lines, plot_convergence, get_plot_limits, print_final_result
from tests.examples22 import RESULTS_


class TestUnconstrainedMinimization(unittest.TestCase):

    def setUp(self):
        """Set up test parameters"""
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.max_iter_rosenbrock_gd = 10000

        # Starting points
        self.x0_default = np.array([1.0, 1.0])
        self.x0_rosenbrock = np.array([-1.0, 2.0])

    def run_optimization_test(self, func_name, func, x0, max_iter_gd=None):
        """
        Run optimization test for a given function with both GD and Newton methods.
        """
        print(f"\n{'=' * 60}")
        print(f"Testing {func_name.upper()}")
        print(f"{'=' * 60}")

        # Set max iterations
        max_iter_gd = max_iter_gd or self.max_iter

        # Run Gradient Descent
        print(f"\nRunning Gradient Descent for {func_name}...")
        gd_minimizer = minimize_function(
            func, x0, self.obj_tol, self.param_tol, max_iter_gd, method='gd'
        )

        print(f"\nRunning Newton's Method for {func_name}...")
        # Run Newton's Method
        newton_minimizer = minimize_function(
            func, x0, self.obj_tol, self.param_tol, self.max_iter, method='newton'
        )

        # Print final results
        print_final_result(gd_minimizer, "Gradient Descent", func_name)
        print_final_result(newton_minimizer, "Newton Method", func_name)

        # Create plots
        self.create_plots(func_name, func, gd_minimizer, newton_minimizer)

        return gd_minimizer, newton_minimizer

    def create_plots(self, func_name, func, gd_minimizer, newton_minimizer):
        """Create and save plots for the optimization results."""

        # Get appropriate plot limits
        xlim, ylim = get_plot_limits(func_name)

        # Plot 1: Contours with paths
        paths = [gd_minimizer.path, newton_minimizer.path]
        method_names = ['Gradient Descent', "Newton Method"]

        fig1 = plot_contour_lines(
            func, xlim, ylim, paths, method_names,
            title=f'{func_name.replace("_", " ").title()} - Optimization Paths'
        )

        # Save contour plot
        plt.savefig(f'{func_name}_contour line.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot 2: Convergence
        obj_values_list = [gd_minimizer.obj_values, newton_minimizer.obj_values]

        fig2 = plot_convergence(
            obj_values_list, method_names,
            title=f'{func_name.replace("_", " ").title()} - Convergence',
            log_scale=False
        )

        # Save convergence plot
        plt.savefig(f'{func_name}_convergence.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    # Tests
    def test_example1(self):
        """Test quadratic function with circular contours"""
        func = RESULTS_['example1']
        self.run_optimization_test('example1', func, self.x0_default)

    def test_example2(self):
        """Test quadratic function with axis-aligned elliptical contours"""
        func = RESULTS_['example2']
        self.run_optimization_test('example2', func, self.x0_default)

    def test_example3(self):
        """Test quadratic function with rotated elliptical contours"""
        func = RESULTS_['example3']
        self.run_optimization_test('example3', func, self.x0_default)

    def test_rosenbrock(self):
        """Test Rosenbrock function"""
        func = RESULTS_['rosenbrock']
        self.run_optimization_test('rosenbrock', func, self.x0_rosenbrock,
                                   self.max_iter_rosenbrock_gd)

    def test_linear(self):
        """Test linear function"""
        func = RESULTS_['linear']
        self.run_optimization_test('linear', func, self.x0_default)

    def test_exponential(self):
        """Test exponential function"""
        func = RESULTS_['exponential']
        self.run_optimization_test('exponential', func, self.x0_default)


def run_all_tests():
    """
    Run all optimization tests and generate plots.
    This function can be called directly to run tests without unittest framework.
    """
    test_suite = TestUnconstrainedMinimization()
    test_suite.setUp()

    test_methods = {
        ('example1', test_suite.test_example1),
        ('example2', test_suite.test_example2),
        ('example3', test_suite.test_example3),
        ('rosenbrock', test_suite.test_rosenbrock),
        ('linear', test_suite.test_linear),
        ('exponential', test_suite.test_exponential)
    }
    for test_name, test_method in test_methods:
        try:
            test_method()
        except Exception as e:
            print(f"Error in {test_name}: {e}")


if __name__ == '__main__':
    # Run with unittest framework
    unittest.main()

    print("\n Complete")