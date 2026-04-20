*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/exercises.html)*

---

We initialize `A`, `x`, and `b` on the host to the same values we
initialize them on the device, manually.

Instead of initializing them like this, we could also use `memcpy_d2h`
calls to copy them from the device just as we do with `y`. Create
exported symbols for `A`, `x`, and `b`, and use them to copy these
arrays back to the host and compute an expected result for `y`.

Note that `A`, `x`, and `b` are not initialized until the
`init_and_compute` device kernel executes. We can also break up
`init_and_compute` into two device kernel calls. Create separate device
kernel calls for `initialize` and `gemv` which are launched separately
on the host, and copy back `A`, `x`, and `b` after you launch
`initialize` but before you launch `gemv`.
