import unittest
import numpy as np
import sys
import os
np.seterr(all='ignore') # Ignores all floating-point errors and warnings


# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import plot_qp_results,plot_lp_results

from src.constrained_min import interior_pt
from tests.examples import (qp_objective, qp_ineq_constraints, qp_eq_constraints_mat,
                            qp_eq_constraints_rhs, qp_x0,
                            lp_objective, lp_ineq_constraints, lp_eq_constraints_mat,
                            lp_eq_constraints_rhs, lp_x0)


class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        """Test quadratic programming problem"""
        # print("\n" + "=" * 50)
        print("Testing Quadratic Programming Problem")
        print("-" * 50)

        print("Problem: min x^2 + y^2 + (z+1)^2")
        print("Subject to: x + y + z = 1, x >= 0, y >= 0, z >= 0")
        print(f"Initial point: {qp_x0}")

        # Solve the problem
        x_opt, f_opt, success, path, obj_values = interior_pt(
            qp_objective,
            qp_ineq_constraints,
            qp_eq_constraints_mat,
            qp_eq_constraints_rhs,
            qp_x0
        )

        # Check convergence
        self.assertTrue(success, "QP optimization should converge")

        # Check feasibility
        # Equality constraint: x + y + z = 1
        eq_violation = abs(np.sum(x_opt) - 1.0)
        self.assertLess(eq_violation, 1e-6, f"Equality constraint violation: {eq_violation}")

        # Inequality constraints: x, y, z >= 0
        self.assertGreaterEqual(x_opt[0], -1e-6, f"x >= 0 violated: x = {x_opt[0]}")
        self.assertGreaterEqual(x_opt[1], -1e-6, f"y >= 0 violated: y = {x_opt[1]}")
        self.assertGreaterEqual(x_opt[2], -1e-6, f"z >= 0 violated: z = {x_opt[2]}")

        # # The theoretical solution should be close to (0, 0, 1)
        # # since we're minimizing distance to (0, 0, -1) subject to being on probability simplex
        # expected_solution = np.array([0.0, 0.0, 1.0])
        # solution_error = np.linalg.norm(x_opt - expected_solution)
        # # self.assertLess(solution_error, 0.1, f"Solution should be close to (0,0,1), got {x_opt}")

        print(">> Optimization converged")


        # Plot results
        plot_qp_results(path, obj_values, x_opt, f_opt)
        print("=" * 50)
        print()

    def test_lp(self):
        """Test linear programming problem"""
        # print("\n" + "=" * 50)
        print("Testing Linear Programming Problem")
        print("-" * 50)

        print("Problem: max x + y")
        print("Subject to: y >= -x + 1, y <= 1, x <= 2, y >= 0")
        print(f"Initial point: {lp_x0}")

        # Solve the problem
        x_opt, f_opt, success, path, obj_values = interior_pt(
            lp_objective,  # Minimizing -(x + y)
            lp_ineq_constraints,
            lp_eq_constraints_mat,
            lp_eq_constraints_rhs,
            lp_x0
        )

        # Check convergence
        self.assertTrue(success, "LP optimization should converge")

        # Check feasibility
        # y >= -x + 1
        constraint1 = x_opt[0] + x_opt[1] - 1
        self.assertGreaterEqual(constraint1, -1e-6, f"y >= -x + 1 violated: {constraint1}")

        # y <= 1
        self.assertLessEqual(x_opt[1], 1 + 1e-6, f"y <= 1 violated: y = {x_opt[1]}")

        # x <= 2
        self.assertLessEqual(x_opt[0], 2 + 1e-6, f"x <= 2 violated: x = {x_opt[0]}")

        # y >= 0
        self.assertGreaterEqual(x_opt[1], -1e-6, f"y >= 0 violated: y = {x_opt[1]}")

        # # The theoretical solution should be at the vertex (1, 1) or (2, 0)
        # # Actually, the maximum is at (1, 1) with objective value 2
        # actual_obj = x_opt[0] + x_opt[1]
        # # self.assertGreater(actual_obj, 1.8, f"Objective value should be close to 2, got {actual_obj}")

        print(">> Optimization converged")
        # print(f"✓ Optimization converged successfully")
        # print(f"✓ Solution satisfies all constraints")
        # print(f"✓ Solution achieves good objective value")

        # Plot results
        plot_lp_results(path, obj_values, x_opt, f_opt)
        print("=" * 50)
        print()

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()