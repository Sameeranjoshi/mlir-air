*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-03-memcpy/device-code.html)*

---

Our previous tutorials initialized `A`, `x`, and `b` on device before
computing GEMV. What else do we need for our device code to support a
host-to-device memcpy of `A`, `x`, and `b`, so that we need only
initialize them on the host?

1.  We need our layout file to export the symbol names for `A`, `x`, and
    `b`.
2.  We need our PE program to export pointers to `A`, `x`, and `b`. The
    PE program no longer needs to initialize these tensors.

We include the new `layout.csl` below, and highlight the changes.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-03-memcpy/layout.csl
:::

As described previously, `@export_name` makes symbol names visible to
the host program.

Notice that we now have `@export_name` calls for `A`, `x`, and `b`.
Unlike `y`, the mutability of these symbols is set to `true`, since the
host will write to these symbols.

Now let\'s take a look at `pe_program.csl`.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-03-memcpy/pe_program.csl
:::

Notice that we no longer need an `initialize` function. When
`init_and_compute` is called, we assume `A`, `x`, and `b` have already
been initialized.

We additionally now define pointers `A_ptr`, `x_ptr`, and `b_ptr` to
`A`, `x`, and `b`, respectively. These pointers are exported with
`@export_symbol`, so that they will be visible to the host.
