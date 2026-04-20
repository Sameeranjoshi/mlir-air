*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-05-multiple-pes/overview.html)*

---

Our program will now run on four processing elements (PE). We will
demonstrate the program with a simulated fabric consisting of an 11 x 3
block of PEs.

For this program, each PE will perform the exact same work; that is, we
will copy `A`, `x`, and `b` to each of the four PEs, the four PEs will
each perform a GEMV, and then we will copy back the result `y` from each
PE.

`pe_program.csl` does not change. We simply need to modify `layout.csl`
to assign it to multiple PEs. We also need to modify our host code to
copy to and from multiple PEs instead of just one.

# Problem Steps

Visually, this program consists of the following steps:

**1. Host copies A, x, b to four PEs on device.**

<figure class="align-center">
<img src="images/tutorial_gemv_5_1.png" width="500"
alt="images/tutorial_gemv_5_1.png" />
</figure>

**2. Host launches function on each PE to compute y.**

<figure class="align-center">
<img src="images/tutorial_gemv_5_2.png" width="500"
alt="images/tutorial_gemv_5_2.png" />
</figure>

**3. Host copies result y from each PE.**

<figure class="align-center">
<img src="images/tutorial_gemv_5_3.png" width="500"
alt="images/tutorial_gemv_5_3.png" />
</figure>
