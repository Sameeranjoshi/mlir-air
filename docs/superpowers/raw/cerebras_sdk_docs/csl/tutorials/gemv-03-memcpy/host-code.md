*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-03-memcpy/host-code.html)*

---

The host code is largely similar to the previous tutorials, except we
now must copy `A`, `x`, and `b` to the device after initializing them on
the host. We do this with `memcpy_h2d`, which has similar syntax to the
previously introduced `memcpy_d2h`.

We include our modified `run.py` below.

::: {.literalinclude language="python"}
../../code-examples/tutorials/gemv-03-memcpy/run.py
:::

This code introduces three `memcpy_h2d` calls, one for each of `A`, `x`,
and `b`:

``` python
runner.memcpy_h2d(A_symbol, A, 0, 0, 1, 1, M*N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(x_symbol, x, 0, 0, 1, 1, N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(b_symbol, b, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

These calls have quite a few arguments, but they\'re identical to those
used by `memcpy_d2h`, other than the first two. For `memcpy_h2d`, the
first argument is the symbol on device that points to the array to which
you want to copy. The next argument is the `numpy` array from which you
are copying. Note that the arrays passed to memcpy must be 1D.

See `tut-gemv-01-complete-program`{.interpreted-text role="ref"} for an
explanation of the remaining arguments.
