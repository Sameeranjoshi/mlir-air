*(Original URL: https://sdk.cerebras.net/csl/code-examples/benchmark-power-method.html)*

---

::: {#sdkruntime-power-method}
:::

Note that the `allreduce` and `stencil_3d_7pts` modules used in this
code are identical to those used in
`sdkruntime-7pt-stencil-spmv`{.interpreted-text role="ref"}.

# layout.csl

::: {.literalinclude language="csl"}
benchmarks/power-method/src/layout.csl
:::

# kernel.csl

::: {.literalinclude language="csl"}
benchmarks/power-method/src/kernel.csl
:::

# blas.csl

::: {.literalinclude language="csl"}
benchmarks/power-method/src/blas.csl
:::

# run.py

::: {.literalinclude language="python"}
benchmarks/power-method/run.py
:::

# cmd_parser.py

::: {.literalinclude language="python"}
benchmarks/power-method/cmd_parser.py
:::

# util.py

::: {.literalinclude language="python"}
benchmarks/power-method/util.py
:::

# power_method.py

::: {.literalinclude language="python"}
benchmarks/power-method/power_method.py
:::

# commands.sh

::: {.literalinclude language="shell"}
benchmarks/power-method/commands_wse3.sh
:::

------------------------------------------------------------------------

# layout_power.csl

::: {.literalinclude language="csl"}
benchmarks/power-method/src/layout_power.csl
:::

# kernel_power.csl

::: {.literalinclude language="csl"}
benchmarks/power-method/src/kernel_power.csl
:::

# device_run.py

::: {.literalinclude language="python"}
benchmarks/power-method/device_run.py
:::
