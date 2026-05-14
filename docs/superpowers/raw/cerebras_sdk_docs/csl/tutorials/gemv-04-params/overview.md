*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-04-params/overview.html)*

---

Our program will run on a single processing element (PE). Like the
previous tutorials, we will demonstrate the program with a simulated
fabric consisting of an 8 x 3 block of PEs.

Our problem steps are identical to the previous tutorial. We need to
modify our device code to replace the constants `M` and `N` with
parameters, and modify our compile command to set these parameter
values. Our host code must be modified to read these values from compile
output.
