*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-03-memcpy/overview.html)*

---

Our program will run on a single processing element (PE). Like the
previous tutorials, we will demonstrate the program with a simulated
fabric consisting of an 8 x 3 block of PEs.

Our problem steps are nearly identical to the previous tutorials, except
we now copy `A`, `x`, and `b` to the device after initializing them on
the host.

`pe_program.csl` no longer needs to initialize `A`, `x`, and `b`, but
both CSL files will need to be updated to export symbols for these
tensors. The host code will need to introduce three `memcpy_h2d` calls
to copy the tensors to the device.

# Problem Steps

Visually, this program consists of the following steps:

**1. Host copies A, x, b to device.**

<figure class="align-center">
<img src="images/tutorial_gemv_3_1.png" width="500"
alt="images/tutorial_gemv_3_1.png" />
</figure>

**2. Host launches function to compute y.**

<figure class="align-center">
<img src="images/tutorial_gemv_3_2.png" width="500"
alt="images/tutorial_gemv_3_2.png" />
</figure>

**3. Host copies result y from device.**

<figure class="align-center">
<img src="images/tutorial_gemv_3_3.png" width="500"
alt="images/tutorial_gemv_3_3.png" />
</figure>
