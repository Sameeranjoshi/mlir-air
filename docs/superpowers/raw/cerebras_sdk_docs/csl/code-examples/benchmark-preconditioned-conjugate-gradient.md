*(Original URL: https://sdk.cerebras.net/csl/code-examples/benchmark-preconditioned-conjugate-gradient.html)*

---

::: {#sdkruntime-preconditioned-conjugate-gradient}
:::

Note that the `allreduce` and `stencil_3d_7pts` modules used in this
code are identical to those used in
`sdkruntime-7pt-stencil-spmv`{.interpreted-text role="ref"}.

# layout.csl

::: {.literalinclude language="csl"}
benchmarks/preconditioned-conjugate-gradient/src/layout.csl
:::

# kernel.csl

::: {.literalinclude language="csl"}
benchmarks/preconditioned-conjugate-gradient/src/kernel.csl
:::

# blas.csl

::: {.literalinclude language="csl"}
benchmarks/preconditioned-conjugate-gradient/src/blas.csl
:::

# run.py

::: {.literalinclude language="python"}
benchmarks/preconditioned-conjugate-gradient/run.py
:::

# cmd_parser.py

::: {.literalinclude language="python"}
benchmarks/preconditioned-conjugate-gradient/cmd_parser.py
:::

# util.py

::: {.literalinclude language="python"}
benchmarks/preconditioned-conjugate-gradient/util.py
:::

# pcg.py

::: {.literalinclude language="python"}
benchmarks/preconditioned-conjugate-gradient/pcg.py
:::

# commands.sh

::: {.literalinclude language="shell"}
benchmarks/preconditioned-conjugate-gradient/commands_wse3.sh
:::

------------------------------------------------------------------------

# layout_pcg.csl

::: {.literalinclude language="csl"}
benchmarks/preconditioned-conjugate-gradient/src/layout_pcg.csl
:::

# kernel_pcg.csl

::: {.literalinclude language="csl"}
benchmarks/preconditioned-conjugate-gradient/src/kernel_pcg.csl
:::

# device_run.py

::: {.literalinclude language="python"}
benchmarks/preconditioned-conjugate-gradient/device_run.py
:::
