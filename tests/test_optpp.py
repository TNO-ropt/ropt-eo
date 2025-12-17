import operator
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from ropt.config import EnOptConfig
from ropt.enums import EventType, ExitCode
from ropt.results import FunctionResults, GradientResults, Results
from ropt.transforms import OptModelTransforms
from ropt.transforms.base import NonLinearConstraintTransform, ObjectiveTransform
from ropt.workflow import BasicOptimizer, Event

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
        "optimizer": {
            "method": "update_this_in_the_test",
            "tolerance": 1e-6,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_optpp_unconstrained(enopt_config: Any, evaluator: Any, external: str) -> None:
    enopt_config["optimizer"]["method"] = f"{external}q_newton"
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
    enopt_config["optimizer"]["method"] = f"everest_optimizers/{method}"
    enopt_config["variables"]["lower_bounds"] = -1.0
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.2], atol=0.02
    )


def test_optpp_eq_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    test_functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables: weight * variables[0] + weight * variables[2],
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }
    test_functions = (
        *test_functions,
        operator.itemgetter(1),
        lambda variables: variables[0] + variables[2],
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }
    test_functions = (
        *test_functions,
        operator.itemgetter(1),
        lambda variables: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.01, 0.4], atol=0.02
    )


def test_optpp_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = "everest_optimizers/bcq_newton"

    def func_p(_0: NDArray[np.float64]) -> float:
        return 1.0

    def func_q(_0: NDArray[np.float64]) -> float:
        return np.nan

    functions = [func_p, func_q]

    optimizer = BasicOptimizer(enopt_config, evaluator(functions))
    exit_code = optimizer.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_optpp_user_abort(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["method"] = "everest_optimizers/bcq_newton"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/bcq_newton"
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/bcq_newton"
    enopt_config["variables"]["lower_bounds"] = -1.0
    enopt_config["variables"]["upper_bounds"] = 1.0

    # Fix the second variables, the test function still has the same optimal
    # values for the other parameters:
    enopt_config["variables"]["mask"] = [True, False, True]

    def assert_gradient(results: tuple[Results, ...]) -> None:
        for item in results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.weighted_objective[1] == 0.0
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
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
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


class ObjectiveScaler(ObjectiveTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales


def test_optpp_objective_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
    optimizer1 = BasicOptimizer(enopt_config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.functions is not None
    variables1 = optimizer1.results.evaluations.variables
    objectives1 = optimizer1.results.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    def function1(variables: NDArray[np.float64]) -> float:
        return float(test_functions[0](variables))

    def function2(variables: NDArray[np.float64]) -> float:
        return float(test_functions[1](variables))

    init1 = test_functions[1](initial_values)
    transforms = OptModelTransforms(
        objectives=ObjectiveScaler(np.array([init1, init1]))
    )

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer2 = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer2._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(optimizer2.results.evaluations.variables, variables1, atol=0.02)
    assert optimizer2.results.functions is not None
    assert np.allclose(objectives1, optimizer2.results.functions.objectives, atol=0.025)


def test_optpp_objective_with_lazy_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["optimizer"]["method"] = "everest_optimizers/q_nips"
    optimizer1 = BasicOptimizer(enopt_config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.functions is not None
    variables1 = optimizer1.results.evaluations.variables
    objectives1 = optimizer1.results.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    objective_transform = ObjectiveScaler(np.array([1.0, 1.0]))
    transforms = OptModelTransforms(objectives=objective_transform)

    init1 = test_functions[1](initial_values)

    def function1(variables: NDArray[np.float64]) -> float:
        objective_transform.set_scales([init1, init1])
        return float(test_functions[0](variables))

    def function2(variables: NDArray[np.float64]) -> float:
        return float(test_functions[1](variables))

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer2 = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer2._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(optimizer2.results.evaluations.variables, variables1, atol=0.02)
    assert optimizer2.results.functions is not None
    assert np.allclose(objectives1, optimizer2.results.functions.objectives, atol=0.025)


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_bounds / self._scales, upper_bounds / self._scales

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints / self._scales

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scales

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_diffs * self._scales, upper_diffs * self._scales


def test_optpp_nonlinear_constraint_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["optimizer"]["method"] = "q_nips"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    optimizer1 = BasicOptimizer(enopt_config, evaluator(functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values), ndmin=1)
    transforms = OptModelTransforms(nonlinear_constraints=ConstraintScaler(scales))
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds is not None
    assert bounds[1] == 0.4 / scales

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer2 = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer2._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables,
        optimizer1.results.evaluations.variables,
        atol=0.02,
    )
    assert optimizer1.results.functions is not None
    assert optimizer2.results.functions is not None
    assert np.allclose(
        optimizer1.results.functions.objectives,
        optimizer2.results.functions.objectives,
        atol=0.025,
    )


def test_optpp_nonlinear_constraint_with_lazy_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["optimizer"]["method"] = "q_nips"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }

    functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    optimizer1 = BasicOptimizer(enopt_config, evaluator(functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert optimizer1.results.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values), ndmin=1)
    scaler = ConstraintScaler([1.0])
    transforms = OptModelTransforms(nonlinear_constraints=scaler)

    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds[1] == 0.4

    def constraint_function(variables: NDArray[np.float64]) -> float:
        scaler.set_scales(scales)
        return float(variables[0] + variables[2])

    functions = (*test_functions, constraint_function)

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        results = event.data.get("results", ())
        config = event.data["config"]
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert config.nonlinear_constraints is not None
                assert transforms.nonlinear_constraints is not None
                _, upper_bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
                    config.nonlinear_constraints.lower_bounds,
                    config.nonlinear_constraints.upper_bounds,
                )
                assert np.allclose(upper_bounds, 0.4 / scales)
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer2 = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer2._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables,
        optimizer1.results.evaluations.variables,
        atol=0.02,
    )
    assert optimizer1.results.functions is not None
    assert optimizer2.results.functions is not None
    assert np.allclose(
        optimizer1.results.functions.objectives,
        optimizer2.results.functions.objectives,
        atol=0.025,
    )
