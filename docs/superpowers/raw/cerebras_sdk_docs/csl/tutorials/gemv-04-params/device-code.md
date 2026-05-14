*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-04-params/device-code.html)*

---

How must we modify our layout code to support compile time parmeters for
`M` and `N`?

1.  We need to define top level parameters for `M` and `N` that will be
    set by the compile command
2.  We need to pass these parameters along to our PE program

Let\'s take a look at the modified `layout.csl`:

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-04-params/layout.csl
:::

Notice that we\'ve defined two parameters at the top of the file:

``` csl
param M: i16;
param N: i16; 
```

Additionally, we pass these parameters along to our PE program inside of
our `@set_tile_code` call:

``` csl
@set_tile_code(0, 0, "pe_program.csl", .{
  .memcpy_params = memcpy.get_params(0),
  .M = M,
  .N = N
});
```

Now let\'s take a look at the modified `pe_program.csl`:

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-04-params/pe_program.csl
:::

`pe_program.csl` must also contain parameter declarations for `M` and
`N`. When this file is compiled, it uses the values passed to it by
`layout.csl`\'s `@set_tile_code` call to bind them.

`M` and `N` are no longer hard-coded in this file.
