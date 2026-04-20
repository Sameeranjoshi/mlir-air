*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/host-code.html)*

---

What does our host code need to do?

1.  Import needed libraries
2.  Specify paths to compiled code and instantiate runner object
3.  Run device kernel `init_and_compute`
4.  Copy back `y` and check result

We explain some features of our `run.py` file containing the host code
below.

::: {.literalinclude language="python"}
../../code-examples/tutorials/gemv-01-complete-program/run.py
:::

# Imports

`SdkRuntime` is the library containing the functionality necessary for
loading and running the device code, as well as copying data on and off
the wafer.

Along with `SdkRuntime`, we import `MemcpyDataType` and `MemcpyOrder`,
which are enums containing types for use with memcpy calls. We explain
this in more detail below.

# Instantiating runner

This script contains two arguments: `name` and `cmaddr`. We use `name`
to specify the directory containing the compilation output. We will
discuss `cmaddr` later, but for now, we leave it unspecified.

We instantiate a runner object using `SdkRuntime`\'s constructor:

``` python
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
```

We then load the program onto the device and begin running with
`runner.load()` and `runner.run()`.

We also grab a handle for later copying `y` off the device, with the
call to `runner.get_id('y')`.

# Running device kernel

Next, we launch our device kernel `init_and_compute`:

``` python
runner.launch('init_and_compute', nonblock=False) 
```

The `nonblock=False` flag simply specifies that this call will wait to
return control to the host program until after the kernel has been
launched. Otherwise, this call will return control to the host
immediately.

# Copying back result

We use a call to `memcpy_d2h` to copy the result `y` back from the
device. First, we must allocate space on the host to hold the result:

``` python
y_result = np.zeros([1*1*M], dtype=np.float32)
```

Then, we copy `y` from the device into this array:

``` python
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

This call has quite a few arguments, so let\'s walk through them. The
first argument is the array on the host to hold the result, which we
just allocated on the previous line. The next argument, `y_symbol`, is
the symbol on device that points to the `y` array.

The next four arguments specify the location of the rectangle of PEs
from which to copy, which is referred to as the \"region of interest\"
or ROI. The first two, `0, 0`, specify that the northwest corner of the
ROI begins at PE (0, 0) within the program rectangle. Thus, it begins at
the northwesternmost corner of the program rectangle.

The next two specify the width and height of the ROI. We only copy the
result back from a single PE, so the width and height of our ROI is
simply `1, 1`.

:::: warning
::: title
Warning
:::

Note that we specify the ROI based on its position in the program
rectangle, NOT on its position in the device fabric.
::::

The next argument specifies how many elements to copy back from each PE
in the ROI. In this case, the result `y` has `M` elements.

The next four arguments are all keyword arguments specifying certain
attributes of this copy operation. We\'ll defer discussion of the
`streaming` keyword to a future tutorial. Note, however, that any copy
between host-to-device which copies to or from a device symbol uses
`streaming=False`.

The `order` keyword specifies the layout of the data copied back to
`y_result`. `memcpy_d2h` always copies into a 1D array on the host.
`ROW_MAJOR` specifies that the data is ordered by (ROI height, ROI
width, elements per PE). Thus, the data copied back from each PE is
contiguous in the result array. `COLUMN_MAJOR`, on the other hand,
specifies that the data is ordered by (elements per PE, ROI width, ROI
height). Thus, the result array will contain the 0th element from each
PE, followed by the 1st element from each PE, and so on.

For this tutorial, because we are copying back from a single PE,
`ROW_MAJOR` and `COLUMN_MAJOR` are identical. In general, for copies
over larger fabrics, `COLUMN_MAJOR` is more performant than `ROW_MAJOR`.

The `data_type` keyword specifies the width of the data copied back. We
are copying back single-precision floating point numbers, so the data
width is 32 bit.

`nonblock=False` specifies that this call will not return control to the
host until the copy into `y_result` has finished.

:::: note
::: title
Note
:::

How does the program ensure that this copy does not happen until
`init_and_compute` has finished? The memcpy infrastructure in the CSL
program can only execute one command at a time. After a device kernel is
launched, `unblock_cmd_stream` must be called before a `memcpy_d2h` can
proceed. The call to `unblock_cmd_stream` at the end of the
`init_and_compute` function in `pe_program.csl` guarantees that
`init_and_compute` finishes before the `memcpy_d2h` occurs.
::::

# Finishing program and checking result

The call to `runner.stop()` stops the execution of the program on
device.

We then check that the `y_result` we copied back from the device matches
the `y_expected` we pre-computed on the host. If they indeed match, we
print a `SUCCESS` message.
