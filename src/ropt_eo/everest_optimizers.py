"""This module implements the OPT++ optimization plugin."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, override

from everest_optimizers import EvaluationMode, minimize
from ropt import UnsupportedError
from ropt.backend import Backend
from ropt.backend.utils import resolve_verbosity
from ropt.config.options import OptionsSchemaModel

if TYPE_CHECKING:
    import numpy as np
    from everest_optimizers import EvaluationResult
    from numpy.typing import NDArray
    from ropt.backend import OptimizationProblem, OptimizerCallback
    from ropt.config import BackendConfig
    from ropt.plugins import MethodSpec

_logger = logging.getLogger("ropt.backend.eo")

_METHOD: Final = "optpp_q_newton"


class EverestOptimizers(Backend):
    """OPT++ optimization backend for ropt.

    This class provides the `optpp_q_newton` method of `everest-optimizers`, a
    replacement for the Dakota method of the same name. Like Dakota, it runs
    an interior-point method if there are linear or nonlinear constraints, a
    bound-constrained quasi-Newton method if there are finite bounds, and an
    unconstrained quasi-Newton or limited-memory BFGS method otherwise.

    !!! warning "This backend cannot run concurrently in-process"
        OPT++ always creates an output file under a fixed name in the current
        working directory, which `everest-optimizers` deletes again once the run
        has started. Two runs at once in the same process therefore create and
        delete the same file, and each one's `output_file` is a path they share.
        To use this backend alongside anything else, prefix the method with
        `external/` and it runs in a process of its own, through the
        [`external`][ropt.backend.external.ExternalBackend] backend.

    To select the optimizer, set the `method` field within the
    [`backend`][ropt.config.BackendConfig] section of the
    [`EnOptContext`][ropt.context.EnOptContext] configuration object to
    `eo/optpp_q_newton` or `eo/default`. For algorithm-specific options, use the
    `options` field within the [`backend`][ropt.config.BackendConfig]
    section. It is either a dictionary keyed by option name, or a list of
    strings in the format of the Dakota
    [`optpp_q_newton`](https://snl-dakota.github.io/docs/latest_release/users/usingdakota/reference/method-optpp_q_newton.html)
    method, such as `["search_method = trust_region", "max_step 10"]`. Both
    forms accept the same options:

    --8<-- "everest_optimizers.md"

    The `tr_pds` search method is not supported.

    The `output_file` option is not a Dakota option: it names the file that
    receives the OPT++ output.

    Some fields of the [`backend`][ropt.config.BackendConfig] section set
    options as well:

    - `max_iterations` and `convergence_tolerance` replace the options of the
      same name.
    - `verbose` adds `output debug`, unless the options contain an `output`
      keyword. OPT++ is silent at all other `output` levels.
    - An `evaluation_policy` of `speculative` in the
      [`gradient`][ropt.config.GradientConfig] section adds the Dakota
      `speculative` keyword.
    """

    methods: ClassVar[MethodSpec] = {_METHOD, "default"}

    def __init__(self, backend_config: BackendConfig) -> None:
        """Initialize the Everest optimizer backend.

        Args:
            backend_config: The configuration for the backend, containing the
                            method name and options.
        """  # ruff: ignore[docstring-missing-exception]
        self._config = backend_config
        _, _, method = backend_config.method.lower().rpartition("/")
        if method not in {_METHOD, "default"}:
            msg = f"OPT++ optimizer algorithm '{method}' is not supported."
            raise UnsupportedError(msg)

    @override
    def start(
        self,
        problem: OptimizationProblem,
        optimizer_callback: OptimizerCallback,
        *,
        evaluation_policy: Literal["speculative", "separate", "auto"],
        output_dir: Path | None,
    ) -> None:
        """Start the optimization.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._optimizer_callback = optimizer_callback
        self._evaluation_policy = evaluation_policy
        options = self._parse_options()
        linear = problem.linear_constraints
        _logger.debug("Using OPT++ optimizer: %s", _METHOD)

        minimize(
            problem.initial_values,
            evaluate=self._evaluate,
            bounds=(problem.lower_bounds, problem.upper_bounds),
            linear_constraints=None if linear is None else linear[:3],
            nonlinear_equalities=problem.nonlinear_equalities,
            options=options or None,
        )

    @property
    @override
    def bypasses_python_output(self) -> bool:
        """Whether the optimizer prints without going through Python.

        OPT++ writes from its C++ implementation, so its output does not pass
        through `sys.stdout`. It is silent unless the `output` option is
        `debug` or `output_file` names a file, but the declaration covers the
        whole backend, which costs nothing since it cannot run concurrently
        in-process anyway.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        return True

    @override
    def validate_options(self) -> None:
        """Validate the options of a given method.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        _validate_options(self._config.options)

    def _evaluate(
        self, variables: NDArray[np.float64], mode: EvaluationMode
    ) -> EvaluationResult:
        return_functions = bool(mode & EvaluationMode.VALUES)
        return_gradients = bool(mode & EvaluationMode.GRADIENTS)
        if (
            return_functions
            and return_gradients
            and self._evaluation_policy == "separate"
        ):
            functions = self._optimizer_callback(
                variables, return_functions=True, return_gradients=False
            ).functions
            gradients = self._optimizer_callback(
                variables, return_functions=False, return_gradients=True
            ).gradients
            return functions, gradients
        result = self._optimizer_callback(
            variables,
            return_functions=return_functions,
            return_gradients=return_gradients,
        )
        return result.functions, result.gradients

    def _parse_options(self) -> dict[str, Any]:
        parsed = _validate_options(self._config.options)
        if "output" not in parsed and self._is_verbose():
            parsed["output"] = "debug"
        return parsed | self._get_overrides()

    def _get_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if self._config.max_iterations is not None:
            overrides["max_iterations"] = self._config.max_iterations
        if self._config.convergence_tolerance is not None:
            overrides["convergence_tolerance"] = self._config.convergence_tolerance
        if self._evaluation_policy == "speculative":
            overrides["speculative"] = True
        return overrides

    def _is_verbose(self) -> bool:
        # OPT++ is silent by default and offers no reporting levels.
        level = resolve_verbosity(verbose=self._config.verbose)
        return level is None or level > 0


def _split_option(option: str) -> tuple[str, str | None] | None:
    text = option.split("#", maxsplit=1)[0].strip()
    if not text:
        return None
    name, *value = re.split(r"\s*=\s*|\s+", text, maxsplit=1)
    return name, value[0] if value else None


def _split_options(options: list[str]) -> list[tuple[str, str | None]]:
    split = (_split_option(option) for option in options)
    return [item for item in split if item is not None]


def _validate_options(options: dict[str, Any] | list[str] | None) -> dict[str, Any]:
    flags: set[str] = set()
    if isinstance(options, list):
        options, flags = _options_from_strings(options)
    options_model = OptionsSchemaModel.model_validate(
        _OPTIONS_SCHEMA
    ).get_options_model(_METHOD)
    validated = options_model.model_validate(options or {}).model_dump(
        exclude_unset=True
    )
    # Lax validation would turn a bare keyword into a number, such as 1.0.
    missing = sorted(name for name in flags if validated[name] is not True)
    if missing:
        msg = f"Option(s) require a value: {', '.join(f'`{name}`' for name in missing)}"
        raise ValueError(msg)
    return validated


def _options_from_strings(options: list[str]) -> tuple[dict[str, Any], set[str]]:
    split = _split_options(options)
    names = [name for name, _ in split]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        msg = f"Duplicate option(s): {', '.join(f'`{name}`' for name in duplicates)}"
        raise ValueError(msg)
    flags = {name for name, value in split if value is None}
    return {name: True if value is None else value for name, value in split}, flags


_OPTIONS_SCHEMA: dict[str, Any] = {
    "methods": {
        _METHOD: {
            "options": {
                "search_method": Literal[
                    "value_based_line_search",
                    "gradient_based_line_search",
                    "trust_region",
                ],
                "merit_function": Literal["el_bakry", "argaez_tapia", "van_shanno"],
                "steplength_to_boundary": float,
                "centering_parameter": float,
                "max_step": float,
                "gradient_tolerance": float,
                "max_iterations": int,
                "convergence_tolerance": float,
                "max_function_evaluations": int,
                "output": Literal["silent", "quiet", "normal", "verbose", "debug"],
                "output_file": str,
            },
            "url": "https://snl-dakota.github.io/docs/latest_release/users/usingdakota/reference/method-optpp_q_newton.html",
        },
    },
}


if __name__ == "__main__":
    from ropt.config.options import gen_options_table

    Path("everest_optimizers.md").write_text(
        gen_options_table(_OPTIONS_SCHEMA), encoding="utf-8"
    )
