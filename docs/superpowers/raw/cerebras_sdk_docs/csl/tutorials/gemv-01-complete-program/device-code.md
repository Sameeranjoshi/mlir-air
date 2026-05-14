*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/device-code.html)*

---

In the previous tutorial, we declared arrays and wrote functions
`initialize` and `gemv` to initialize and compute `y = Ax + b`. What
else do we need for our device code to form a complete program?

1.  We need a top-level \"layout\" file, which will define the program
    rectangle on which our kernel will run, and assign a code file to
    the single PE in our rectangle.
2.  We need to initialize the infrastructure of the memcpy library,
    which allows the host to launch kernels and copy data to and from
    the device.

We first walk through `layout.csl`, which defines our program layout. We
include this code below.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-01-complete-program/layout.csl
:::

# Initializing memcpy infrastructure

At the very top of this file is an `@import_module` call, which imports
the top-level memcpy infrastructure. This module import requires width
and height parameters which correspond to the dimensions of the program
rectangle. This program only uses a single PE, so width and height are
both 1.

Module imports in CSL act like unique struct types. Thus, the code in
the CSL standard library file `memcpy/get_params` can be used like a
struct named `memcpy`.

# Defining layout

The layout block is evaluated at compile time. We use it to define the
number of PEs used in our program and assign code to each of those PEs.

`@set_rectangle` defines the shape of our program. Because our program
will run on a single PE, we are compiling this program for a 1x1
rectangle of PEs.

Our single PE has coordinate (0,0), and we assign to it the code file
`pe_program.csl`, which we will explore later. We also pass some
parameters related to memcpy to this program.

The `memcpy` struct contains a function named `get_params`, which
returns some parameters for the memcpy infrastructure that each PE\'s
code file must include. This function takes as an argument the column
number of the PE; thus, for this program, the appropriate parameters are
returned by `memcpy.get_params(0)`.

# Exporting symbols

Our host program will directly launch a device kernel, and copy back the
result `y`. The two `@export_name` calls make the symbols visible to the
host program.

The first `@export_name` call makes the symbol named `y` visible to the
host, as a pointer to an array of type `f32`. Its mutability is set to
`false`, meaning that the host can only read from and not write to the
symbol.

The second `@export_name` call makes the symbol `init_and_compute`
visible to the host; this is the function which we will launch from the
host to compute the GEMV. This function takes no arguments, so its type
is `fn()void`.

# Adding memcpy to the PE program

Now, let\'s take a look at `pe_program.csl`, which defines the code that
we assign to our single PE. This program is largely the same as the
preceding tutorial\'s `code.csl` file, but with some additional
infrastructure related to `memcpy`.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-01-complete-program/pe_program.csl
:::

At the top, we declare a parameter named `memcpy_params`: this
parameter\'s value is set at compile time by `@set_tile_code` in
`layout.csl`.

Next is another memcpy-related `@import_module`, this time importing the
PE-specific `<memcpy/memcpy>` standard library file as a struct named
`sys_mod`.

Our functions `initialize` and `gemv` are identical to the previous
tutorial. However, note one addition to `init_and_compute`. After `gemv`
finishes, we must notify the memcpy infrastructure that additional
commands from the host can proceed. Thus, we must call
`sys_mod.unblock_cmd_stream()` at the end of our function. The control
flow of every host-callable function in a CSL program must end with a
call to `unblock_cmd_stream()`.

Everything inside of `comptime` block is evaluated at compile time. This
comptime block exports symbols so they can be advertised to the host. In
particular, `y_ptr`, which is a pointer to the array `y`, is exported
with the name `y`. The `init_and_compute` function is also exported.
