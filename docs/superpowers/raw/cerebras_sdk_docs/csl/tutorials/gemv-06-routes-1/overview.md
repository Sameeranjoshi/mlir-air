*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-06-routes-1/overview.html)*

---

Our program will run on two processing elements (PE).

We will demonstrate the program with a simulated fabric consisting of a
9 x 3 block of PEs.

The program will first copy `b` into the left PE\'s `y` array. Then, it
will copy the left half of `A`\'s columns into the left PE, and the
right half of `A`\'s columns into the right PE. Similarly, it will copy
the the first `N/2` elements of `x` into the left PE, and the last `N/2`
elements of `x` into the right PE.

Each PE will then compute `A*x` for its local pieces of `A` and `x`.
Thus, both PEs perform a matrix-vector product for an `M x N/2` matrix.
The PEs will increment their local `y` arrays by this result.

The left PE then sends its `y` array to the right PE, and the right PE
increments its local `y` array by the received values. Because the left
`y` array contained the contribution from `b`, the final summed `y` on
the right PE is our GEMV result.

The host then copies `y` off of the right PE.

# Problem Steps

Visually, this program consists of the following steps:

**1. Host copies b into y array of left PE.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_1.png" width="500"
alt="images/tutorial_gemv_6_1.png" />
</figure>

**2. Host copies left N/2 columns of A to left PE, right N/2 columns to
right PE.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_2.png" width="500"
alt="images/tutorial_gemv_6_2.png" />
</figure>

**3. Host copies first N/2 elements of x to left PE, last N/2 elements
to right PE.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_3.png" width="500"
alt="images/tutorial_gemv_6_3.png" />
</figure>

**4. Host launches function to compute GEMV.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_4.png" width="500"
alt="images/tutorial_gemv_6_4.png" />
</figure>

**5. Each PE increments local y by local portion of matrix-vector
product Ax.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_5.png" width="500"
alt="images/tutorial_gemv_6_5.png" />
</figure>

**6. Left PE sends local y to right PE, and right PE increments y by
received values.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_6.png" width="500"
alt="images/tutorial_gemv_6_6.png" />
</figure>

**6. Right PE now contains final result y. Host copies back y from right
PE.**

<figure class="align-center">
<img src="images/tutorial_gemv_6_7.png" width="500"
alt="images/tutorial_gemv_6_7.png" />
</figure>
