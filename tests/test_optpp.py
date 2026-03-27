# ruff: noqa: RUF069

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.enums import ExitCode
from ropt.results import GradientResults, Results
from ropt.workflow import BasicOptimizer

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
            "lower_bounds": [-1.0] * 3,
            "upper_bounds": [1.0] * 3,
        },
        "backend": {
            "method": "update_this_in_the_test",
            "convergence_tolerance": 1e-6,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_optpp_unconstrained(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["backend"]["method"] = f"{external}q_newton"
    enopt_config["variables"]["lower_bounds"] = -np.inf
    enopt_config["variables"]["upper_bounds"] = np.inf
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


@pytest.mark.parametrize("method", ["bcq_newton", "q_nips"])
def test_optpp_bound_constraint(enopt_config: Any, method: str, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = f"everest_optimizers/{method}"
    enopt_config["variables"]["lower_bounds"] = -1.0
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.2], atol=0.02
    )


def test_optpp_eq_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


def test_optpp_ge_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_optpp_le_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_optpp_le_ge_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_optpp_le_ge_linear_constraints_two_sided(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )


def test_optpp_eq_nonlinear_constraint(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_optpp_ineq_nonlinear_constraint(
    enopt_config: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )
    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_optpp_ineq_nonlinear_constraints_two_sided(
    enopt_config: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[1],
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.01, 0.4], atol=0.02
    )


def test_optpp_ineq_nonlinear_constraints_eq_ineq(
    enopt_config: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[1],
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.01, 0.4], atol=0.02
    )


def test_optpp_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/bcq_newton"

    def func_p(_0: NDArray[np.float64], _1: int) -> float:
        return 1.0

    def func_q(_0: NDArray[np.float64], _1: int) -> float:
        return np.nan

    functions = [func_p, func_q]

    optimizer = BasicOptimizer(enopt_config, evaluator(functions))
    exit_code = optimizer.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_optpp_user_abort(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/bcq_newton"
    last_evaluation = 0

    def _abort() -> bool:
        nonlocal last_evaluation

        if last_evaluation == 2:
            return True
        last_evaluation += 1
        return False

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.set_abort_callback(_abort)
    exit_code = optimizer.run(initial_values)
    assert optimizer.results is not None
    assert last_evaluation == 2
    assert exit_code == ExitCode.USER_ABORT


def test_optpp_evaluation_policy_separate(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/bcq_newton"
    enopt_config["gradient"] = {"evaluation_policy": "separate"}
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )

    enopt_config["gradient"] = {"evaluation_policy": "separate"}
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optpp_optimizer_variables_subset(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["backend"]["method"] = "everest_optimizers/bcq_newton"
    enopt_config["variables"]["lower_bounds"] = -1.0
    enopt_config["variables"]["upper_bounds"] = 1.0

    # Fix the second variables, the test function still has the same optimal
    # values for the other parameters:
    enopt_config["variables"]["mask"] = [True, False, True]

    def assert_gradient(results: tuple[Results, ...]) -> None:
        for item in results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.target_objective[1] == 0.0
                assert np.all(np.equal(item.gradients.objectives[:, 1], 0.0))

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.set_results_callback(assert_gradient)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_optpp_optimizer_variables_subset_linear_constraints(
    enopt_config: Any, evaluator: Any
) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem: The
    # second and third constraints are dropped because they involve variables
    # that are not optimized. They are still checked by the monitor:
    enopt_config["backend"]["method"] = "everest_optimizers/q_nips"
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "lower_bounds": [1.0, 1.0, 2.0],
        "upper_bounds": [1.0, 1.0, 2.0],
    }
    enopt_config["variables"]["mask"] = [True, False, True]
    initial = initial_values.copy()
    initial[1] = 1.0
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 1.0, 0.75], atol=0.02
    )
