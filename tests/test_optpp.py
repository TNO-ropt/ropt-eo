# ruff: file-ignore[float-equality-comparison, private-member-access]

from pathlib import Path
from typing import Any

import everest_optimizers
import numpy as np
import pytest
from everest_optimizers import EvaluationMode, Evaluator
from numpy.typing import NDArray
from pydantic import ValidationError
from ropt import (
    EvaluationFunctionContext,
    ExitCode,
    FunctionResults,
    GradientResults,
    UnsupportedError,
    optimize,
)
from ropt.backend import OptimizationProblem, OptimizerCallbackResult
from ropt.config import BackendConfig
from ropt.context import EnOptContext
from ropt.utils import validate_backend_options

from ropt_eo import everest_optimizers as backend_module
from ropt_eo.everest_optimizers import EverestOptimizers

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "backend": {
            "method": "eo/optpp_q_newton2",
            "convergence_tolerance": 1e-6,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.fixture(name="minimize_options")
def minimize_options_fixture(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    passed: list[Any] = []
    original_minimize = everest_optimizers.minimize

    def capture_minimize(*args: Any, **kwargs: Any) -> Any:
        passed.append(kwargs["options"])
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr(backend_module, "minimize", capture_minimize)
    return passed


def _parse_options(evaluation_policy: str = "auto", **config: Any) -> dict[str, Any]:
    backend = EverestOptimizers(
        BackendConfig.model_validate({"method": "eo/optpp_q_newton2"} | config)
    )
    backend._evaluation_policy = evaluation_policy  # type: ignore[assignment]
    return backend._parse_options()


def _evaluator_backend(
    evaluation_policy: str, calls: list[tuple[bool, bool]]
) -> EverestOptimizers:
    backend = EverestOptimizers(BackendConfig(method="eo/optpp_q_newton2"))
    backend._evaluation_policy = evaluation_policy  # type: ignore[assignment]

    def callback(
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> OptimizerCallbackResult:
        calls.append((return_functions, return_gradients))
        return OptimizerCallbackResult(
            functions=np.array([variables[0], 2.0]) if return_functions else None,
            gradients=np.ones((2, variables.size)) if return_gradients else None,
        )

    backend._optimizer_callback = callback
    return backend


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (EvaluationMode.VALUES, [(True, False)]),
        (EvaluationMode.GRADIENTS, [(False, True)]),
        (EvaluationMode.BOTH, [(True, True)]),
    ],
)
def test_optpp_evaluator_requests_mode(
    mode: EvaluationMode, expected: list[tuple[bool, bool]]
) -> None:
    calls: list[tuple[bool, bool]] = []
    backend = _evaluator_backend("auto", calls)

    functions, gradients = backend._evaluate(np.array([3.0]), mode)

    assert calls == expected
    assert (functions is not None) == bool(mode & EvaluationMode.VALUES)
    assert (gradients is not None) == bool(mode & EvaluationMode.GRADIENTS)


def test_optpp_evaluator_separate_policy_splits_requests() -> None:
    calls: list[tuple[bool, bool]] = []
    backend = _evaluator_backend("separate", calls)

    functions, gradients = backend._evaluate(np.array([3.0]), EvaluationMode.BOTH)

    assert calls == [(True, False), (False, True)]
    np.testing.assert_array_equal(functions, [3.0, 2.0])
    np.testing.assert_array_equal(gradients, np.ones((2, 1)))


@pytest.mark.parametrize("constrained", [False, True])
def test_optpp_numpy_problem_forwarded(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    constrained: bool,
) -> None:
    config["variables"]["mask"] = [True, False, True]
    if constrained:
        config["variables"]["lower_bounds"] = [-1.0, -2.0, -3.0]
        config["variables"]["upper_bounds"] = [1.0, 2.0, 3.0]
        config["linear_constraints"] = {
            "coefficients": [[1.0, 2.0, 3.0]],
            "lower_bounds": 1.0,
            "upper_bounds": 2.0,
        }
        config["nonlinear_constraints"] = {
            "lower_bounds": [0.0, 0.01, 0.0],
            "upper_bounds": [np.inf, 0.01, 0.3],
        }
    problem = OptimizationProblem(
        EnOptContext.model_validate(config), np.array(initial_values)
    )
    backend = _evaluator_backend("auto", [])
    minimize_calls: list[NDArray[np.float64]] = []

    def capture_minimize(
        x0: NDArray[np.float64],
        *,
        evaluate: Evaluator,
        bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
        linear_constraints: tuple[
            NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
        ]
        | None,
        nonlinear_equalities: NDArray[np.bool_] | None,
        output_file: Path | None,
        options: dict[str, Any] | list[str] | None,
    ) -> None:
        minimize_calls.append(x0)
        assert x0 is problem.initial_values
        assert evaluate == backend._evaluate
        assert bounds[0] is problem.lower_bounds
        assert bounds[1] is problem.upper_bounds
        if problem.linear_constraints is None:
            assert linear_constraints is None
        else:
            assert linear_constraints is not None
            for actual, expected in zip(
                linear_constraints, problem.linear_constraints[:3], strict=True
            ):
                assert actual is expected
        assert nonlinear_equalities is problem.nonlinear_equalities
        assert output_file is None
        assert options is None

    monkeypatch.setattr(backend_module, "minimize", capture_minimize)
    backend.start(
        problem,
        backend._optimizer_callback,
        evaluation_policy="auto",
        output_dir=None,
    )
    assert len(minimize_calls) == 1


@pytest.mark.parametrize("method", ["q_nips", "optpp_q_newton"])
def test_optpp_unsupported_method_rejected(method: str) -> None:
    with pytest.raises(UnsupportedError, match=f"'{method}' is not supported"):
        EverestOptimizers(BackendConfig(method=f"eo/{method}"))


def test_optpp_invalid_options(config: Any) -> None:
    config["backend"]["options"] = [
        "max_iterations = 0",
        "merit_function el_bakry",
    ]
    validate_backend_options("eo/optpp_q_newton2", config["backend"]["options"])

    config["backend"]["options"] = [
        "max_iterations = 0",
        "search_method = 1",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Input should be 'value_based_line_search',"
    ):
        validate_backend_options("eo/optpp_q_newton2", config["backend"]["options"])

    config["backend"]["options"] = [
        "max_iterations = 0",
        "foo = xyz",
        "bar",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`, `bar`"
    ):
        validate_backend_options("eo/optpp_q_newton2", config["backend"]["options"])


def test_optpp_invalid_dict_options() -> None:
    validate_backend_options(
        "eo/optpp_q_newton2", {"max_iterations": 0, "merit_function": "el_bakry"}
    )
    with pytest.raises(
        ValidationError, match=r"Input should be 'value_based_line_search',"
    ):
        validate_backend_options("eo/optpp_q_newton2", {"search_method": 1})
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `debug`"
    ):
        validate_backend_options("eo/optpp_q_newton2", {"debug": True})


@pytest.mark.parametrize(
    "options",
    [{"search_method": "tr_pds"}, ["search_method = tr_pds"]],
)
def test_optpp_tr_pds_rejected(options: dict[str, str] | list[str]) -> None:
    with pytest.raises(ValidationError, match="search_method"):
        validate_backend_options("eo/optpp_q_newton2", options)


def test_optpp_string_options_comments_ignored() -> None:
    validate_backend_options(
        "eo/optpp_q_newton2", ["max_step = 2  # comment", "# only a comment"]
    )


def test_optpp_string_options_converted_to_typed_dict() -> None:
    options = _parse_options(
        options=[
            "max_iterations = 5",
            "max_function_evaluations 7",
            "convergence_tolerance=1e-6",
            "max_step = 2  # comment",
            "# only a comment",
            "search_method trust_region",
        ]
    )
    assert options == {
        "max_iterations": 5,
        "max_function_evaluations": 7,
        "convergence_tolerance": 1e-6,
        "max_step": 2.0,
        "search_method": "trust_region",
    }
    assert isinstance(options, dict)
    assert type(options["max_iterations"]) is int
    assert type(options["max_step"]) is float


def test_optpp_string_options_duplicate_rejected() -> None:
    with pytest.raises(ValueError, match=r"Duplicate option\(s\): `max_step`"):
        validate_backend_options("eo/optpp_q_newton2", ["max_step 1", "max_step 2"])


@pytest.mark.parametrize("option", ["max_step", "max_iterations"])
def test_optpp_string_options_bare_keyword_requires_value(option: str) -> None:
    with pytest.raises(ValueError, match=rf"require a value: `{option}`"):
        validate_backend_options("eo/optpp_q_newton2", [option])


@pytest.mark.parametrize(
    "option", ["max_iterations = ten", "output_file", "output = loud"]
)
def test_optpp_string_options_invalid_value_rejected(option: str) -> None:
    with pytest.raises(ValidationError):
        validate_backend_options("eo/optpp_q_newton2", [option])


@pytest.mark.parametrize(
    "options",
    [
        {"max_step": 2.0, "search_method": "value_based_line_search"},
        ["max_step = 2", "search_method value_based_line_search"],
    ],
)
def test_optpp_options_passed_as_dict(
    config: Any, eval_func: Any, minimize_options: list[Any], options: Any
) -> None:
    del config["backend"]["convergence_tolerance"]
    config["backend"]["options"] = options
    optimize(config, initial_values, eval_func())
    assert minimize_options == [
        {"max_step": 2.0, "search_method": "value_based_line_search"}
    ]
    assert type(minimize_options[0]["max_step"]) is float


def test_optpp_string_and_dict_options_equivalent(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    config["backend"]["options"] = {"max_step": 0.5, "search_method": "trust_region"}
    dict_result = optimize(config, initial_values, eval_func())
    config["backend"]["options"] = ["max_step = 0.5", "search_method trust_region"]
    string_result = optimize(config, initial_values, eval_func())
    assert dict_result.results is not None
    assert string_result.results is not None
    assert np.allclose(dict_result.results.variables, [0.0, 0.0, 0.2], atol=0.02)
    np.testing.assert_array_equal(
        string_result.results.variables, dict_result.results.variables
    )


def test_optpp_config_overrides_dict_options() -> None:
    options = _parse_options(
        options={"max_iterations": 5, "convergence_tolerance": 0.1, "max_step": 2.0},
        max_iterations=10,
        convergence_tolerance=1e-6,
    )
    assert options == {
        "max_iterations": 10,
        "convergence_tolerance": 1e-6,
        "max_step": 2.0,
    }


def test_optpp_config_overrides_string_options() -> None:
    options = _parse_options(
        options=["max_iterations = 5", "convergence_tolerance 0.1", "max_step 2"],
        max_iterations=10,
        convergence_tolerance=1e-6,
    )
    assert options == {
        "max_iterations": 10,
        "convergence_tolerance": 1e-6,
        "max_step": 2.0,
    }


@pytest.mark.parametrize("verbose", [None, False, 0, True, 4])
def test_optpp_verbose_adds_no_output_option(verbose: Any) -> None:
    assert _parse_options(options={}, verbose=verbose) == {}
    assert _parse_options(options=[], verbose=verbose) == {}


def test_optpp_verbose_keeps_output_option() -> None:
    assert _parse_options(options={"output": "quiet"}, verbose=True) == {
        "output": "quiet"
    }
    assert _parse_options(options=["output quiet"], verbose=True) == {"output": "quiet"}


def test_optpp_speculative_policy_sets_speculative() -> None:
    assert _parse_options("speculative", options={}) == {"speculative": True}
    assert _parse_options("speculative", options=[]) == {"speculative": True}
    assert _parse_options("separate", options={}) == {}


@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_optpp_unconstrained(config: Any, eval_func: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}eo/optpp_q_newton2"
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optpp_default_method(config: Any, eval_func: Any) -> None:
    config["backend"]["method"] = "eo/default"
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optpp_bound_constraint(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.2], atol=0.02)


def test_optpp_output_dir_writes_report_independent_of_verbose(
    config: Any,
    eval_func: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "output"
    config["backend"]["verbose"] = True
    config["optimizer"] = {"output_dir": output_dir}
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    optimize(config, initial_values, eval_func())
    assert capfd.readouterr() == ("", "")
    assert [path.name for path in tmp_path.iterdir()] == ["output"]
    assert [path.name for path in output_dir.iterdir()] == ["optpp_q_newton.txt"]


def test_optpp_without_output_dir_writes_no_report(
    config: Any,
    eval_func: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    optimize(config, initial_values, eval_func())
    assert list(tmp_path.iterdir()) == []


def test_optpp_eq_linear_constraint(config: Any, eval_func: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_optpp_ge_linear_constraint(config: Any, eval_func: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_le_linear_constraint(config: Any, eval_func: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_le_ge_linear_constraints(config: Any, eval_func: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_le_ge_linear_constraints_two_sided(config: Any, eval_func: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.1, 0.0, 0.4], atol=0.02)

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.1, 0.0, 0.4], atol=0.02)


def test_optpp_eq_nonlinear_constraint(
    config: Any, eval_func: Any, test_functions: Any
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    def constraint_function(
        variables: NDArray[np.float64], _: EvaluationFunctionContext
    ) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.25, 0.0, 0.75], atol=0.02)


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
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0

    def constraint_function(
        variables: NDArray[np.float64], _: EvaluationFunctionContext
    ) -> float:
        return weight * float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_optpp_ineq_nonlinear_constraints_two_sided(
    config: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    def constraint_function(
        variables: NDArray[np.float64], _: EvaluationFunctionContext
    ) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("merit_function", ["el_bakry", "argaez_tapia", "van_shanno"])
def test_optpp_ineq_nonlinear_constraints_eq_ineq(
    config: Any,
    eval_func: Any,
    test_functions: Any,
    merit_function: str,
) -> None:
    config["backend"]["options"] = {"merit_function": merit_function}
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }

    def constraint_function_1(
        variables: NDArray[np.float64], _: EvaluationFunctionContext
    ) -> float:
        return float(variables[1])

    def constraint_function_2(
        variables: NDArray[np.float64], _: EvaluationFunctionContext
    ) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function_1, constraint_function_2]),
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, [-0.1, 0.01, 0.4], atol=0.02)


def test_optpp_failed_realizations(config: Any, eval_func: Any) -> None:
    def func_p(_0: NDArray[np.float64], _1: EvaluationFunctionContext) -> float:
        return 1.0

    def func_q(_0: NDArray[np.float64], _1: EvaluationFunctionContext) -> float:
        return np.nan

    functions = [func_p, func_q]

    result = optimize(config, initial_values, eval_func(functions))
    assert result.exit_code == ExitCode.TOO_FEW_REALIZATIONS


@pytest.mark.parametrize("evaluation_policy", ["separate", "speculative"])
def test_optpp_evaluation_policy(
    config: Any, eval_func: Any, evaluation_policy: str
) -> None:
    config["gradient"] = {"evaluation_policy": evaluation_policy}
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optpp_speculative_policy_requests_both(
    config: Any, eval_func: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes: list[EvaluationMode] = []
    original_evaluate = EverestOptimizers._evaluate

    def record_evaluate(
        self: EverestOptimizers, variables: NDArray[np.float64], mode: EvaluationMode
    ) -> Any:
        modes.append(EvaluationMode(mode))
        return original_evaluate(self, variables, mode)

    monkeypatch.setattr(EverestOptimizers, "_evaluate", record_evaluate)
    config["gradient"] = {"evaluation_policy": "speculative"}
    optimize(config, initial_values, eval_func())

    assert modes
    assert set(modes) == {EvaluationMode.BOTH}


def test_optpp_optimizer_variables_subset(config: Any, eval_func: Any) -> None:
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = 1.0

    # Fix the second variables, the test function still has the same optimal
    # values for the other parameters:
    config["variables"]["mask"] = [True, False, True]

    def assert_gradient(item: FunctionResults | GradientResults) -> None:
        if isinstance(item, GradientResults):
            assert item.gradients is not None
            assert item.target_gradient is not None
            assert item.target_gradient[1] == 0.0
            assert np.all(np.equal(item.gradients.objectives[:, 1], 0.0))

    result = optimize(
        config,
        initial_values,
        eval_func(),
        result_callback=assert_gradient,
        report_gradients=True,
    )
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_optpp_optimizer_variables_subset_linear_constraints(
    config: Any, eval_func: Any
) -> None:
    # The second constraint only involves the fixed variable.
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0]],
        "lower_bounds": [1.0, 1.0],
        "upper_bounds": [1.0, 1.0],
    }
    config["variables"]["mask"] = [True, False, True]

    result = optimize(config, [0.0, 1.0, 0.1], eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.25, 1.0, 0.75], atol=0.02)


def test_optpp_optimizer_variables_subset_linear_constraints_offset(
    config: Any, eval_func: Any
) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 1, 0]],
        "lower_bounds": [1.0, -np.inf],
        "upper_bounds": [1.0, 1.15],
    }
    config["variables"]["mask"] = [True, False, True]

    result = optimize(config, [0.0, 1.0, 0.1], eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.15, 1.0, 0.85], atol=0.02)


def test_optpp_optimizer_variables_subset_mixed_linear_constraint(
    config: Any, eval_func: Any
) -> None:
    # Fix the second variable. The second constraint involves only that
    # variable and is dropped; the third mixes it with a free one and is kept,
    # with the fixed variable's contribution moved into the bound, so
    # `x1 + x2 <= 1.6` becomes `x2 <= 0.6`. That binds, which is why the result
    # is not the [0.25, 0.75] the first constraint alone would give.
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [0, 1, 1]],
        "lower_bounds": [1.0, 1.0, -np.inf],
        "upper_bounds": [1.0, 1.0, 1.6],
    }
    config["variables"]["mask"] = [True, False, True]
    result = optimize(config, [0.0, 1.0, 0.1], eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.4, 1.0, 0.6], atol=0.02)
