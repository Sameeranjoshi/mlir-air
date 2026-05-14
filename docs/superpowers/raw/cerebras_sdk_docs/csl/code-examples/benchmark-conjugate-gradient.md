*(Original URL: https://sdk.cerebras.net/csl/code-examples/benchmark-conjugate-gradient.html)*

---

::: {#sdkruntime-conjugate-gradient}
:::

Note that the `allreduce` and `stencil_3d_7pts` modules used in this
code are identical to those used in
`sdkruntime-7pt-stencil-spmv`{.interpreted-text role="ref"}.

# layout.csl

::: {.literalinclude language="csl"}
benchmarks/conjugate-gradient/src/layout.csl
:::

# kernel.csl

::: {.literalinclude language="csl"}
benchmarks/conjugate-gradient/src/kernel.csl
:::

# blas.csl

::: {.literalinclude language="csl"}
benchmarks/conjugate-gradient/src/blas.csl
:::

# run.py

::: {.literalinclude language="python"}
benchmarks/conjugate-gradient/run.py
:::

# cmd_parser.py

::: {.literalinclude language="python"}
benchmarks/conjugate-gradient/cmd_parser.py
:::

# util.py

::: {.literalinclude language="python"}
benchmarks/conjugate-gradient/util.py
:::

# cg.py

::: {.literalinclude language="python"}
benchmarks/conjugate-gradient/cg.py
:::

# commands.sh

::: {.literalinclude language="shell"}
benchmarks/conjugate-gradient/commands_wse3.sh
:::

------------------------------------------------------------------------

# layout_cg.csl

::: {.literalinclude language="csl"}
benchmarks/conjugate-gradient/src/layout_cg.csl
:::

# kernel_cg.csl

::: {.literalinclude language="csl"}
benchmarks/conjugate-gradient/src/kernel_cg.csl
:::

# device_run.py

::: {.literalinclude language="python"}
benchmarks/conjugate-gradient/device_run.py
:::
