*(Original URL: https://sdk.cerebras.net/csl/code-examples/benchmark-bicgstab.html)*

---

::: {#sdkruntime-bicgstab}
:::

Note that the `allreduce` and `stencil_3d_7pts` modules used in this
code are identical to those used in
`sdkruntime-7pt-stencil-spmv`{.interpreted-text role="ref"}.

# layout.csl

::: {.literalinclude language="csl"}
benchmarks/bicgstab/src/layout.csl
:::

# kernel.csl

::: {.literalinclude language="csl"}
benchmarks/bicgstab/src/kernel.csl
:::

# blas.csl

::: {.literalinclude language="csl"}
benchmarks/bicgstab/src/blas.csl
:::

# run.py

::: {.literalinclude language="python"}
benchmarks/bicgstab/run.py
:::

# cmd_parser.py

::: {.literalinclude language="python"}
benchmarks/bicgstab/cmd_parser.py
:::

# util.py

::: {.literalinclude language="python"}
benchmarks/bicgstab/util.py
:::

# bicgstab.py

::: {.literalinclude language="python"}
benchmarks/bicgstab/bicgstab.py
:::

# commands.sh

::: {.literalinclude language="shell"}
benchmarks/bicgstab/commands_wse3.sh
:::

------------------------------------------------------------------------

# layout_bicgstab.csl

::: {.literalinclude language="csl"}
benchmarks/bicgstab/src/layout_bicgstab.csl
:::

# kernel_bicgstab.csl

::: {.literalinclude language="csl"}
benchmarks/bicgstab/src/kernel_bicgstab.csl
:::

# device_run.py

::: {.literalinclude language="python"}
benchmarks/bicgstab/device_run.py
:::
