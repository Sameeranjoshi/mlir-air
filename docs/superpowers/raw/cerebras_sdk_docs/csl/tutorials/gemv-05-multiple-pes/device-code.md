*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-05-multiple-pes/device-code.html)*

---

How do we need to modify our layout file to support running the program
on multiple PEs?

1.  We need to modify `@set_rectangle` to reflect our new program
    rectangle.
2.  We need to modify our `memcpy` infrastructure to reflect the use of
    multiple PEs.
3.  We need to call `@set_tile_code` for each coordinate inside this
    program rectangle.

`pe_program.csl` remains largely the same; we simply assign it to more
PEs. We include the new `layout.csl` below, and highlight the changes.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-05-multiple-pes/layout.csl
:::

Notice that we define a new compile time parameter `width`, whose value
will be set in our compile command. We will use this value to set the
number of PEs in our row of PEs used by the program.

When we import `<memcpy_multi/get_params>`, we use `width` to specify
the width of the program rectangle for which memcpy infrastructure will
be generated. The `height` is still 1.

Inside of our layout block, our program rectangle is now specified with
`@set_rectangle(width, 1)`. For each of the PEs in this rectangle, we
must call `@set_tile_code`, so we do this in a loop. The loop coordinate
is the PE\'s `x`-coordinate, or column number, which is needed to set
the correct `memcpy_params` for each PE.
