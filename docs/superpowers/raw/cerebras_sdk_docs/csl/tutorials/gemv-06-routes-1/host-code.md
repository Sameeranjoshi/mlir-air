*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-06-routes-1/host-code.html)*

---

Our new host code must:

1.  Copy `b` into the left PE\'s `y` array
2.  Copy the left halves of `A` and `x` to the left PE, and the right
    halves to the right PE
3.  After the device kernel completes, copy `y` back from the right PE

We explain some features of our new `run.py` below.

::: {.literalinclude language="python"}
../../code-examples/tutorials/gemv-06-routes-1/run.py
:::

# Copying `b` into `y` of left PE

We copy `b` into `y` of the left PE here:

``` python
runner.memcpy_h2d(y_symbol, b, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

Notice that the ROI is a single PE, located at (0, 0) in the program
rectangle. The right PE (1, 0) is omitted from this `memcpy` call.

# Copying `A` and `x`

We copy `A` and `x` to the device as follows:

``` python
runner.memcpy_h2d(A_symbol, A.transpose().ravel(), 0, 0, 2, 1, M*N_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_h2d(x_symbol, x, 0, 0, 2, 1, N_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

Notice that the ROI is now both PEs, so the `memcpy` calls copy data
into both the left and right PE.

Because we now store `A` column-major on the PEs, we transpose our `A`
matrix, and then flatten it to a 1D array with `ravel()`. Each PE gets
`M*N_per_PE` elements, so each PE gets `N_per_PE` columms of `A`.

Similarly, each PE gets `N_per_PE` elements of `x`.

# Copying back result

We copy back `y` from the right PE as follows:

``` python
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 1, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

Notice that our ROI now begins at (1, 0), and contains a single PE.
Thus, this `memcpy` call copies back the `M` elements of `y` only from
the right PE.

Once this call is complete, we then, as in our previous tutorials, check
that the received result is correct.
