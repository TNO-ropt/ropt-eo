"""This module implements the OPT++ optimization plugin."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np
from everest_optimizers import minimize  # type: ignore[import-untyped]
from numpy.typing import NDArray
from ropt.config.options import OptionsSchemaModel
from ropt.plugins.optimizer.base import Optimizer, OptimizerPlugin
from ropt.plugins.optimizer.utils import (
    NormalizedConstraints,
    get_masked_linear_constraints,
    validate_supported_constraints,
)
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

if TYPE_CHECKING:
    from ropt.config import EnOptConfig
    from ropt.optimization import OptimizerCallback

_SUPPORTED_METHODS: Final[set[str]] = {"q_newton", "bcq_newton", "q_nips"}
_DEFAULT_METHOD: Final = "q_nips"


# Categorize the methods by the types of constraint they support or require.

_CONSTRAINT_REQUIRES_BOUNDS: Final = {"bcq_newton", "q_nips"}
_CONSTRAINT_SUPPORT_BOUNDS: Final = {"bcq_newton", "q_nips"}
_CONSTRAINT_SUPPORT_LINEAR_EQ: Final = {"q_nips"}
_CONSTRAINT_SUPPORT_LINEAR_INEQ: Final = {"q_nips"}
_CONSTRAINT_SUPPORT_NONLINEAR_EQ: Final = {"q_nips"}
_CONSTRAINT_SUPPORT_NONLINEAR_INEQ: Final = {"q_nips"}

_ConstraintType = str | Callable[..., float] | Callable[..., NDArray[np.float64]]

_METHOD_MAP: Final = {
    "q_newton": "optpp_q_newton",
    "bcq_newton": "optpp_bcq_newton",
    "q_nips": "optpp_q_nips",
}


class EverestOptimizers(Optimizer):
    """OPT++ optimization backend for ropt.

    This class provides an interface to several optimization algorithms from
    the OPT++ library, enabling their use within `ropt`.

    To select an optimizer, set the `method` field within the
    [`optimizer`][ropt.config.OptimizerConfig] section of the
    [`EnOptConfig`][ropt.config.EnOptConfig] configuration object to the desired
    algorithm's name. Most methods support the general options defined in the
    [`EnOptConfig`][ropt.config.EnOptConfig] object. For algorithm-specific
    options, use the `options` dictionary within the
    [`optimizer`][ropt.config.OptimizerConfig] section.

    The table below lists the included methods together with the method-specific
    options that are supported:

    --8<-- "everest_optimizers.md"
    """

    _supported_constraints: ClassVar[dict[str, set[str]]] = {
        "bounds": _CONSTRAINT_SUPPORT_BOUNDS,
        "linear:eq": _CONSTRAINT_SUPPORT_LINEAR_EQ,
        "linear:ineq": _CONSTRAINT_SUPPORT_LINEAR_INEQ,
        "nonlinear:eq": _CONSTRAINT_SUPPORT_NONLINEAR_EQ,
        "nonlinear:ineq": _CONSTRAINT_SUPPORT_NONLINEAR_INEQ,
    }
    _required_constraints: ClassVar[dict[str, set[str]]] = {
        "bounds": _CONSTRAINT_REQUIRES_BOUNDS,
    }

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer implemented by the Optpp plugin.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._optimizer_callback = optimizer_callback
        self._config = config
        _, _, self._method = self._config.optimizer.method.lower().rpartition("/")
        if self._method == "default":
            self._method = _DEFAULT_METHOD
        if self._method not in _SUPPORTED_METHODS:
            msg = f"OPT++ optimizer algorithm {self._method} is not supported"
            raise NotImplementedError(msg)
        validate_supported_constraints(
            self._config,
            self._method,
            self._supported_constraints,
            self._required_constraints,
        )
        self._options = self._parse_options()
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._cached_gradient: NDArray[np.float64] | None = None

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None
        self._cached_gradient = None

        self._bounds = self._initialize_bounds()
        self._constraints = self._initialize_constraints(initial_values)

        minimize(
            fun=self._function,
            x0=initial_values[self._config.variables.mask],
            method=_METHOD_MAP[self._method],
            bounds=self._bounds,
            jac=self._gradient,
            constraints=self._constraints,
            options=self._options or None,
        )

    def _initialize_bounds(self) -> Bounds | None:
        if (
            np.isfinite(self._config.variables.lower_bounds).any()
            or np.isfinite(self._config.variables.upper_bounds).any()
        ):
            lower_bounds = self._config.variables.lower_bounds[
                self._config.variables.mask
            ]
            upper_bounds = self._config.variables.upper_bounds[
                self._config.variables.mask
            ]
            return Bounds(lower_bounds, upper_bounds)
        return None

    def _initialize_constraints(
        self, initial_values: NDArray[np.float64]
    ) -> list[NonlinearConstraint | LinearConstraint]:
        self._normalized_constraints = None

        lin_coef, lin_lower, lin_upper = None, None, None
        self._linear_constraint_bounds: (
            tuple[NDArray[np.float64], NDArray[np.float64]] | None
        ) = None
        if self._config.linear_constraints is not None:
            lin_coef, lin_lower, lin_upper = get_masked_linear_constraints(
                self._config, initial_values
            )
            self._linear_constraint_bounds = (lin_lower, lin_upper)
        nonlinear_bounds = (
            None
            if self._config.nonlinear_constraints is None
            else (
                self._config.nonlinear_constraints.lower_bounds,
                self._config.nonlinear_constraints.upper_bounds,
            )
        )
        if (bounds := _get_constraint_bounds(nonlinear_bounds)) is not None:
            self._normalized_constraints = NormalizedConstraints()
            self._normalized_constraints.set_bounds(*bounds)
        return self._initialize_constraints_object(lin_coef, lin_lower, lin_upper)

    def _fun_object(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        self._normalized_constraints.set_constraints(
            self._constraint_functions(variables).transpose()
        )
        assert self._normalized_constraints.constraints is not None
        return self._normalized_constraints.constraints[:, 0]

    def _jac_object(
        self,
        variables: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        self._normalized_constraints.set_gradients(
            self._constraint_gradients(variables)
        )
        assert self._normalized_constraints.gradients is not None
        return self._normalized_constraints.gradients

    def _initialize_constraints_object(
        self,
        lin_coef: NDArray[np.float64] | None,
        lin_lower: NDArray[np.float64] | None,
        lin_upper: NDArray[np.float64] | None,
    ) -> list[LinearConstraint | NonlinearConstraint]:
        constraints: list[LinearConstraint | NonlinearConstraint] = []
        if self._config.linear_constraints is not None:
            assert lin_coef is not None
            assert lin_lower is not None
            assert lin_upper is not None
            constraints.append(LinearConstraint(lin_coef, lin_lower, lin_upper))
        if self._normalized_constraints is not None:
            ub = [
                0.0 if is_eq else np.inf for is_eq in self._normalized_constraints.is_eq
            ]
            constraints.append(
                NonlinearConstraint(
                    fun=self._fun_object,
                    jac=self._jac_object,
                    lb=[0.0] * len(ub),
                    ub=ub,
                ),
            )
        return constraints

    def _function(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        if variables.ndim > 1 and variables.size == 0:
            return np.array([])
        functions, _ = self._get_function_or_gradient(
            variables, get_function=True, get_gradient=False
        )
        assert functions is not None
        if variables.ndim > 1:
            return functions[:, 0]
        return np.array(functions[0])

    def _gradient(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        _, gradients = self._get_function_or_gradient(
            variables, get_function=False, get_gradient=True
        )
        assert gradients is not None
        return gradients[0, :]

    def _constraint_functions(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions, _ = self._get_function_or_gradient(
            variables, get_function=True, get_gradient=False
        )
        assert functions is not None
        return np.array(functions[1:])

    def _constraint_gradients(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _, gradients = self._get_function_or_gradient(
            variables, get_function=False, get_gradient=True
        )
        assert gradients is not None
        return gradients[1:, :]

    def _get_function_or_gradient(
        self, variables: NDArray[np.float64], *, get_function: bool, get_gradient: bool
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        if (
            self._cached_variables is None
            or variables.shape != self._cached_variables.shape
            or not np.allclose(variables, self._cached_variables)
        ):
            self._cached_variables = None
            self._cached_function = None
            self._cached_gradient = None
            if self._normalized_constraints is not None:
                self._normalized_constraints.reset()

        function = self._cached_function if get_function else None
        gradient = self._cached_gradient if get_gradient else None

        compute_functions = get_function and function is None
        compute_gradients = get_gradient and gradient is None

        if compute_functions or compute_gradients:
            self._cached_variables = variables.copy()
            speculative = self._config.gradient.evaluation_policy == "speculative"
            compute_functions = compute_functions or speculative
            compute_gradients = compute_gradients or speculative
            new_function, new_gradient = self._compute_functions_and_gradients(
                variables,
                compute_functions=compute_functions,
                compute_gradients=compute_gradients,
            )
            if compute_functions:
                assert new_function is not None
                self._cached_function = new_function.copy()
                if get_function:
                    function = new_function
            if compute_gradients:
                assert new_gradient is not None
                self._cached_gradient = new_gradient.copy()
                if get_gradient:
                    gradient = new_gradient

        return function, gradient

    def _compute_functions_and_gradients(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool,
        compute_gradients: bool,
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        new_function = None
        new_gradient = None
        if (
            compute_functions
            and compute_gradients
            and self._config.gradient.evaluation_policy == "separate"
        ):
            callback_result = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            new_function = callback_result.functions
            callback_result = self._optimizer_callback(
                variables,
                return_functions=False,
                return_gradients=True,
            )
            new_gradient = callback_result.gradients
        else:
            callback_result = self._optimizer_callback(
                variables,
                return_functions=compute_functions,
                return_gradients=compute_gradients,
            )
            new_function = callback_result.functions
            new_gradient = callback_result.gradients

        # The optimizer callback may change non-linear constraint bounds:
        if (
            self._normalized_constraints is not None
            and callback_result.nonlinear_constraint_bounds is not None
        ):
            bounds = _get_constraint_bounds(callback_result.nonlinear_constraint_bounds)
            assert bounds is not None
            self._normalized_constraints.set_bounds(*bounds)

        if self.allow_nan and new_function is not None:
            new_function = np.where(np.isnan(new_function), np.inf, new_function)
        return new_function, new_gradient

    def _parse_options(self) -> dict[str, Any]:
        options = (
            copy.deepcopy(self._config.optimizer.options)
            if isinstance(self._config.optimizer.options, dict)
            else {}
        )
        if self._config.optimizer.max_iterations is not None:
            options["max_iterations"] = self._config.optimizer.max_iterations
        if self._config.optimizer.max_functions is not None:
            options["max_function_evaluations"] = self._config.optimizer.max_functions
        if self._config.optimizer.tolerance is not None:
            options["convergence_tolerance"] = self._config.optimizer.tolerance
        return options


def _get_constraint_bounds(
    nonlinear_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    bounds = []
    if nonlinear_bounds is not None:
        bounds.append(nonlinear_bounds)
    if bounds:
        lower_bounds, upper_bounds = zip(*bounds, strict=True)
        return np.concatenate(lower_bounds), np.concatenate(upper_bounds)
    return None


class EverestOptimizersPlugin(OptimizerPlugin):
    """The OPT++ optimizer plugin class."""

    @classmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> EverestOptimizers:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return EverestOptimizers(config, optimizer_callback)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (_SUPPORTED_METHODS | {"default"})

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC501
        if options is not None:
            if not isinstance(options, dict):
                msg = "OPT++ optimizer options must be a dictionary"
                raise ValueError(msg)
            *_, method = method.rpartition("/")
            OptionsSchemaModel.model_validate(_OPTIONS_SCHEMA).get_options_model(
                _DEFAULT_METHOD if method == "default" else method
            ).model_validate(options)


_DEFAULT_OPTIONS: dict[str, Any] = {
    "debug": bool,
    "output_file": str,
    "search_method": Literal["line_search", "trust_region", "trust_pds"],
    "search_pattern_size": int,
    "max_step": int,
    "gradient_multiplier": float,
    "max_iterations": int,
    "max_function_evaluations": int,
    "convergence_tolerance": float,
    "gradient_tolerance": float,
}

_OPTIONS_SCHEMA: dict[str, Any] = {
    "methods": {
        "q_newton": {"options": _DEFAULT_OPTIONS},
        "bcq_newton": {"options": _DEFAULT_OPTIONS},
        "q_nips": {
            "options": (
                _DEFAULT_OPTIONS
                | {
                    "merit_function": Literal["el_bakry", "argaez_tapia", "van_shanno"],
                    "mu": float,
                    "centering_parameter": float,
                    "steplength_to_boundary": float,
                    "constraint_tolerance": float,
                }
            )
        },
    },
}


if __name__ == "__main__":
    from ropt.config.options import gen_options_table

    Path("everest_optimizers.md").write_text(
        gen_options_table(_OPTIONS_SCHEMA), encoding="utf-8"
    )
