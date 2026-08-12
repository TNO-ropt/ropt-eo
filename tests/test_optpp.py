# ruff: file-ignore[float-equality-comparison]

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.components.event_handlers import CallbackHandler
from ropt.enums import EnOptEventType, ExitCode
from ropt.events import EnOptEvent
from ropt.results import GradientResults
from ropt.simple import optimize

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
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
def test_optpp_unconstrained(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}q_newton"
    config["variables"]["lower_bounds"] = -np.inf
    config["variables"]["upper_bounds"] = np.inf
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("method", ["bcq_newton", "q_nips"])
def test_optpp_bound_constraint(config: Any, method: str, eval_func: Any) -> None:
    config["backend"]["method"] = f"everest_optimizers/{method}"
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.2], atol=0.02)


def test_optpp_eq_linear_constraint(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_optpp_ge_linear_constraint(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_le_linear_constraint(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_le_ge_linear_constraints(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_le_ge_linear_constraints_two_sided(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)


def test_optpp_eq_nonlinear_constraint(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    def constraint_function(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_optpp_ineq_nonlinear_constraint(
    config: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0

    def constraint_function(variables: NDArray[np.float64], _: Any) -> float:
        return weight * float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_ineq_nonlinear_constraints_two_sided(
    config: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }

    def constraint_function_1(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[1])

    def constraint_function_2(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function_1, constraint_function_2]),
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.01, 0.4], atol=0.02)


def test_optpp_ineq_nonlinear_constraints_eq_ineq(
    config: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }

    def constraint_function_1(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[1])

    def constraint_function_2(variables: NDArray[np.float64], _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function_1, constraint_function_2]),
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.01, 0.4], atol=0.02)


def test_optpp_failed_realizations(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/bcq_newton"

    def func_p(_0: NDArray[np.float64], _1: int) -> float:
        return 1.0

    def func_q(_0: NDArray[np.float64], _1: int) -> float:
        return np.nan

    functions = [func_p, func_q]

    result = optimize(config, initial_values, eval_func(functions))
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_optpp_evaluation_policy_separate(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/bcq_newton"
    config["gradient"] = {"evaluation_policy": "separate"}
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)

    config["gradient"] = {"evaluation_policy": "separate"}
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optpp_optimizer_variables_subset(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "everest_optimizers/bcq_newton"
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = 1.0

    # Fix the second variables, the test function still has the same optimal
    # values for the other parameters:
    config["variables"]["mask"] = [True, False, True]

    def assert_gradient(event: EnOptEvent) -> None:
        for item in event.results or ():
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.target_objective[1] == 0.0
                assert np.all(np.equal(item.gradients.objectives[:, 1], 0.0))

    result = optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=assert_gradient,
            )
        ],
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optpp_optimizer_variables_subset_linear_constraints(
    config: Any, eval_func: Any
) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem: The
    # second and third constraints are dropped because they involve variables
    # that are not optimized. They are still checked by the monitor:
    config["backend"]["method"] = "everest_optimizers/q_nips"
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "lower_bounds": [1.0, 1.0, 2.0],
        "upper_bounds": [1.0, 1.0, 2.0],
    }
    config["variables"]["mask"] = [True, False, True]
    initial = initial_values.copy()
    initial[1] = 1.0
    result = optimize(config, initial, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 1.0, 0.75], atol=0.02)
