*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-03-memcpy/index.html)*

---

# GEMV Tutorial 3: Memcpy {#tut-gemv-03-memcpy}

We\'ve already written a program that launches a kernel and copies the
result back to the host, so lets extend this to copying the initial
tensors from the host to the device.

This program will now have three phases:

1.  Host-to-device memcpy of `A`, `x`, and `b`
2.  Kernel launch
3.  Device-to-host memcpy of `y`

## Learning objectives

## Example overview

## Modifying the CSL

## Modifying the host code

## Compiling and running the program

## Exercises

## Next

In the next tutorial, we expand this program to use data structure
descriptors (DSDs), a core language feature of CSL.
