*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax/device-code.html)*

---

What does our code need to do?

1.  Define the dimensions of our matrix
2.  Define arrays for holding `A`, `x`, `b`, and `y`
3.  Define a function that initialize the arrays
4.  Define a function to compute `A*x + b` and store the result in `y`

We explain the sections of our `code.csl` file containing this code
below. This code file, along with all tutorials and examples, are
available in the `csl-extras` directory contained within the SDK
tarball.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-00-basic-syntax/code.csl
:::

Reading the above CSL code, we can see the following:

# Defining our variables and constants

We first define two constants, `N` and `M`, that give our matrix and
vector dimensions. For the purposes of this and the next few tutorials,
we\'ll use `M = 4` and `N = 6`.

We then declare the arrays `A`, `x`, `b`, and `y`, which hold our matrix
and vectors. `A` stores `N*M` single-precision floating point elements.
We will store the matrix in `A` in a row-major fashion. These constants
and arrays are declared in global scope: they will be visible to all
functions in this code file.

Note that all data items must explicitly be declared as variables or
constants, with the `var` or `const` keywords.

# Defining our initialize function

Next, we define a function named `initialize` that we will call to
initialize the values stored in `A`, `x`, `b`, and `y`. `A` will be
initialized such that each element `i` holds the value `i`, all values
of `x` will be initialized to `1.0`, and all values of `b` will be
initialized to `2.0`. `y` will be zero-initialized.

To initialize `A`, we use a `for` loop with the `range` syntax.
`@range(i16, M*N)` returns the sequence of integers
`0, 1, 2, ..., M*N-1`, in `i16`, or half-precision signed, format. The
`for` loop iterates over this sequence of integers, and the variable
`idx` stores the index of the current loop iteration. On each loop
iteration, `@as(f32, idx)` casts the integer value `idx` to type `f32`,
or single precision float, and assigns this value to the element
`A[idx]`.

We also use a range-for loop to initialize all elements of `x` to the
value `1.0`.

To initialize `y` and `b`, we demonstrate the syntax of a while loop
with an assignment expression that acts as a loop index. At each loop
iteration, the variable `i` is incremented by 1. This assignment
expression is executed at the end of each loop iteration.

# Defining our gemv function

The function `gemv` actually computes `y = A*x + b`. The outer loop
iterates over `M`, i.e., over the rows of the matrix. For each row `i`
in the matrix, the inner loop over `N` computes the dot product of that
row with the vector `x` by incrementing the variable `tmp`. After
completing the inner loop, the final value of `y[i]` is computed from
`tmp` and `b[i]`.

This code sample ends with a function named `init_and_compute`, which
simply calls `initialize` followed by `gemv`.
