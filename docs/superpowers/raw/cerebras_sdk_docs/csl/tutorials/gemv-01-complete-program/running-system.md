*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/running-system.html)*

---

We\'ve compiled and run this program using the fabric simulator, but
with a few modest changes, we can also compile and run on a real
Cerebras system.

First, we must modify the compile command to replace the `fabric-dims`
with the actual dimensions of our target fabric. Most CS-2s will have a
fabric dimension of 757 x 996, so our compile command becomes:

``` bash
$ cslc layout.csl --fabric-dims=757,996 --fabric-offsets=4,1 --memcpy --channels=1 -o out
```

This program is also compatible with the CS-3, which has a fabric
dimension of 762 x 1176. Compiling for the CS-3 requires specifying the
WSE-3 architecture:

``` bash
$ cslc layout.csl --arch=wse3 --fabric-dims=762,1176 --fabric-offsets=4,1 --memcpy --channels=1 -o out
```

The Cerebras system is a network attached accelerator. When targeting a
real system for running a program, we must know its IP address. This is
the purpose of the `SdkRuntime` constructor\'s `cmaddr` keyword
argument.

If the IP address is stored in an environment variable named
`$CS_IP_ADDR`, then you can run on the system with:

``` bash
$ cs_python run.py --name out --cmaddr $CS_IP_ADDR:9000
```

We use port 9000 to connect to the system and launch our program.

:::: note
::: title
Note
:::

The compile and run commands above are used when running the SDK
directly from a host node connected to the CS system. If using a
Wafer-Scale Cluster in appliance mode, see
`appliance-mode`{.interpreted-text role="ref"}.
::::
