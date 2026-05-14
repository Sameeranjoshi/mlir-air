*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax/overview.html)*

---

This simple code computes the general matrix-vector product
`y = Ax + b`, where `A` has dimensions `M x N`, `x` is `N x 1`, and `b`
and `y` are `M x 1`.

Our code will store `A` in a one-dimensional array of size `M*N`, using
a row-major ordering. This computation will be performed with 32-bit
arithmetic.
