# An `everest-optimizers` plugin for `ropt`

The `ropt-eo` package extends the [`ropt`](https://tno-ropt.github.io/ropt/dev/)
module by providing the Dakota-compatible `optpp_q_newton` method from the
`everest-optimizers` package. `ropt` itself is a robust optimization framework
designed for both continuous and discrete optimization workflows and is
extensible through its plugin architecture. Installing `ropt-eo` makes this
method available as `eo/optpp_q_newton2` within `ropt`.

## Reference

::: ropt_eo.everest_optimizers.EverestOptimizers
    options:
        members: False
