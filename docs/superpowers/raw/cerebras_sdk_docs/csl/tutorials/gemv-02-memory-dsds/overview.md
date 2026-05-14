*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-02-memory-dsds/overview.html)*

---

Our program will run on a single processing element (PE). Like the
previous tutorial, we will demonstrate the program with a simulated
fabric consisting of an 8 x 3 block of PEs.

Our problem steps are identical to the previous tutorial. Our layout
file, host code, and compile and run commands are also identical. We
only need to modify `pe_program.csl`, and we\'ll take a closer look at
changes to this file.
