*(Original URL: https://sdk.cerebras.net/csl/code-examples/benchmark-7pt-stencil-spmv.html)*

---

::: {#sdkruntime-7pt-stencil-spmv}
:::

# layout.csl

::: {.literalinclude language="csl"}
benchmarks/7pt-stencil-spmv/src/layout.csl
:::

# kernel.csl

::: {.literalinclude language="csl"}
benchmarks/7pt-stencil-spmv/src/kernel.csl
:::

# run.py

::: {.literalinclude language="python"}
benchmarks/7pt-stencil-spmv/run.py
:::

# cmd_parser.py

::: {.literalinclude language="python"}
benchmarks/7pt-stencil-spmv/cmd_parser.py
:::

# benchmark-libs/stencil_3d_7pts/layout.csl

::: {.literalinclude language="csl"}
benchmarks/benchmark-libs/stencil_3d_7pts/layout.csl
:::

# benchmark-libs/stencil_3d_7pts/pe.csl

::: {.literalinclude language="csl"}
benchmarks/benchmark-libs/stencil_3d_7pts/pe.csl
:::

# benchmark-libs/allreduce/layout.csl

::: {.literalinclude language="csl"}
benchmarks/benchmark-libs/allreduce/layout.csl
:::

# benchmark-libs/allreduce/pe.csl

::: {.literalinclude language="csl"}
benchmarks/benchmark-libs/allreduce/pe.csl
:::

# commands.sh

::: {.literalinclude language="shell"}
benchmarks/7pt-stencil-spmv/commands_wse3.sh
:::
