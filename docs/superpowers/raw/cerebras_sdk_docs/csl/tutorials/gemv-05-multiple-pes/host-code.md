*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-05-multiple-pes/host-code.html)*

---

Our host code must now copy `A`, `x` and `b` to multiple PEs, and must
copy back `y` from multiple PEs. Let\'s take a look at how we must
modify our `memcpy_h2d` and `memcpy_d2h` calls in `run.py` to do this:

::: {.literalinclude language="python"}
../../code-examples/tutorials/gemv-05-multiple-pes/run.py
:::

First, note that we read one more parameter from the compile output,
`width`. Our host code uses this to specify how many PEs it must copy
tensors to and from.

Now let\'s take a closer at the `mempcy_h2d` calls:

``` python
runner.memcpy_h2d(A_symbol, np.tile(A, width), 0, 0, width, 1, M*N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(x_symbol, np.tile(x, width), 0, 0, width, 1, N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(b_symbol, np.tile(b, width), 0, 0, width, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

We want to copy each of `A`, `x`, and `b` to each PE in our program
rectangle. But `memcpy_h2d` does not perform a broadcast; it takes its
input array and distributes it within the region of interest (ROI) based
on the `order` parameter. Here, we use `np.tile` to duplicate each array
`width` times.

In the first `memcpy_h2d`, the input array `np.tile(A, width)` is a 1D
array formed by duplicating `A` `width` times, so the full input
array\'s size is `M*N*width`. Our ROI is specified by `0, 0, width, 1`,
meaning that we copy to the a row of `width` PEs beginning at PE (0, 0).
We copy `M*N` elements to each PE.

Because our order is `ROW_MAJOR`, the result is that PE (0, 0) will
receive the first `M*N` elements of the tiled array, PE (1, 0) will
receive the next `M*N` elements, and so on. Thus, each PE will receive
an identical `M*N` elements corresponding to a copy of `A`.

When we copy `y` back from the device, `memcpy_d2h` proceeds similarly:

``` python
y_result = np.zeros([M*width], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 0, 0, width, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

Our output array `y_result` has size `M*width`, since each of the
`width` PEs copies back the `M` elements of `y`.

We test that our copied-back result is correct for all PEs by comparing
`y_result` to a tiled `y_expected`. See
`tut-gemv-01-complete-program`{.interpreted-text role="ref"} for an
explanation of the remaining arguments.
