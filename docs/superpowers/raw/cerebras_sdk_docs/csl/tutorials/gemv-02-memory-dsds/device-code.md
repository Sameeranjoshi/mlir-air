*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-02-memory-dsds/device-code.html)*

---

In the previous tutorial, we created a complete CSL program using a
single PE to initialize and compute `y = Ax + b`. What do we need to do
in `pe_program.csl` to take advantage of memory DSDs and builtin
operations on tensors?

1.  We need to define DSDs for accessing our tensors
2.  We need to rewrite the `gemv` function to operate on these DSDs

We previously walked through `layout.csl`, which is the same for this
tutorial as the previous one. We include the new `pe_program.csl` below,
and highlight the changes in this code.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-02-memory-dsds/pe_program.csl
:::

# Defining our memory DSDs

First, let\'s take a look at the DSDs we define for accessing `b` and
`y`:

``` csl
var b_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> b[i] }); 
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] }); 
```

`b_dsd` and `y_dsd` are the memory DSDs for accessing `b`, and `y`,
respectively.

The `tensor_access` field defines the access pattern of these DSDs.
`|i|` specifies the induction variable, and `{M}` specifies the loop
bound; i.e., these DSDs will access `M` elements. After `->`, an
expression is given for accessing a memory location using the induction
variable. This expression must be **affine**, or linear plus a constant.

The access pattern for these DSDs is straightforward: these DSDs loop
over all `M` elements, in order, of their respective tensors.

Now let\'s take a look at the DSD for accessing `A`:

``` csl
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i*N] });
```

This DSD accesses `M` elements of `A`, but strided by `N` elements;
i.e., `A_dsd` accesses elements `0, N, 2*N, ... (M-1)*N`. Because `A` is
stored in row major format, this means that `A_dsd` as defined here
accesses the 0th column of `A`.

:::: note
::: title
Note
:::

These memory DSDs are of type `mem1d_dsd`, which are one-dimensional
memory DSDs. CSL also provides `mem4d_dsd`, multidimensional memory DSDs
for up to four dimensions.

You can learn more about memory DSDs in our language reference guide
`language-dsds`{.interpreted-text role="ref"}.
::::

# Using our DSDs to compute GEMV

Now that we\'ve defined our DSDs, let\'s take a look at how to use them
to compute GEMV.

Recall that our previous `gemv()` function was defined as follows:

``` csl
fn gemv() void {                                                                                           
  for (@range(i16, M)) |i| {                                                                               
    var tmp: f32 = 0.0;                                                                                    
    for (@range(i16, N)) |j| {                                                                             
      tmp += A[i*N + j] * x[j];                                                                            
    }                                                                                                      
    y[i] = tmp + b[i];                                                                                     
  }                                                                                                        
}
```

Now, our `gemv()` looks like this:

``` csl
fn gemv() void {
  for (@range(u16, N)) |i| {
    @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
    A_dsd = @increment_dsd_offset(A_dsd, 1, f32);
  }
  @fadds(y_dsd, y_dsd, b_dsd);
}
```

Notice that we now only have one explicit loop over `N`, instead of two
explicit loops. At each iteration, this `@fmacs` operation does the
following:

- performs a vector-scalar multiplication between the column of `A`
  referenced by `A_dsd` and the scalar `x[i]`,
- performs an elementwise vector addition between this result and the
  vector `y`,
- and stores this final result into `y`.

Thus, each `@fmacs` operation increments the `M` elements of `y` by the
vector-scalar product of column `i` of `A` and element `i` of `x`.

The `@increment_dsd_offset` operation at each loop iteration increments
`A_dsd` to reference the next column of `A`. This builtin operation
takes `A_dsd` and creates a new DSD by offseting its access by 1 `f32`
element.

For instance, the first time this operation occurs, `A_dsd` will now
access elements `1, N+1, 2*N+1, ... (M-1)*N+1` of `A`. Again, because
`A` is stored row major, this will access the 1st column of `A`.

Once this loop over the `N` columns of `A` is complete, `y` contains the
result of `A*x`. The `@fadds` operation performs an elementwise vector
addition between `y` and `b`, storing the result back in `y`. Now `y`
contains the result of `A*x + b`.

# Using builtins to initialize tensors

You may have noticed one other slight change to this code. Instead of
initializing `x`, `b`, and `y`, in the `initialize` function, we make
use of builtins to provide values for them at declaration:

``` csl
var x = @constants([N]f32, 1.0);
var b = @constants([M]f32, 2.0);
var y = @zeros([M]f32);
```

The `@constants` builtin returns a tensor of the specified type, with
all elements initialized to the specified value. Thus, `x` is
initialized as an `N` element tensor of all ones, and `b` is initialized
as an `M` element tensor of all twos.

The `@zeros` builtin is rather obvious. `y` is initialized as an `M`
element tensor of all zeros.
