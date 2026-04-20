*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/compiling.html)*

---

We compile this code for the CS-2 simulator using:

``` bash
$ cslc layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out
```

This command will produce multiple ELF files, in a directory named
`out`. Let\'s walk through several aspects of this command.

First, we specify the top level file to be compiled, in this case
`layout.csl`. `pe_program.csl` does not have to be specified in the
compilation command, because it is included by `layout.csl`.

We also must specify the fabric dimensions of our target device, and the
fabric offset at which we place our program. As we specified above, this
tutorial is using an 8 x 3 simulated fabric, and we place the program\'s
lone PE at column 4, row 1 of the fabric.

:::: warning
::: title
Warning
:::

Every program using memcpy **must** use a fabric offset of `4,1`, and if
compiling for a simulated fabric, must use a fabric dimension of at
least `width+7,height+1`, where `width` and `height` are the dimensions
of the program. These additional PEs are used by memcpy to route data on
and off the wafer.
::::

Last, note that flag specifying `memcpy` and `channels`. Every program
using memcpy must include the `--memcpy` flag. When running on a real
system, the `channels` flag determines the max throughput for
transferring data on and off the wafer. Its value can be no larger than
the width of the program rectangle, and maxes out at 16. Typically,
performance improvements are minimal past 8 channels.

This program is also compatible with the CS-3 architecture. We can
specify the `--arch` flag to determine for which architecture we
compile. The default value is `--arch=wse2`, where WSE-2 is the
processor architecture used in the CS-2. We specify the value
`--arch=wse3` to compile for WSE-3, the processor architecture used in
the CS-3.
