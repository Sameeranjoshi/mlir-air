*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/overview.html)*

---

Our program will run on a single processing element (PE).

We will demonstrate the program with a simulated fabric consisting of an
8 x 3 block of PEs.

:::: warning
::: title
Warning
:::

The coordinates of PEs are always specified (column, row). The
dimensions of a grid of PEs are specified (width, height), or,
equivalently, (number of columns, number of rows).
::::

<figure class="align-center">
<img src="images/tutorial_gemv_1_0.png" width="500"
alt="images/tutorial_gemv_1_0.png" />
</figure>

# Problem Steps

Visually, this program consists of the following steps:

**1. Host launches function on PE.**

<figure class="align-center">
<img src="images/tutorial_gemv_1_1.png" width="500"
alt="images/tutorial_gemv_1_1.png" />
</figure>

**2. Function initializes A, x, b, and computes y.**

<figure class="align-center">
<img src="images/tutorial_gemv_1_2.png" width="500"
alt="images/tutorial_gemv_1_2.png" />
</figure>

**3. Host copies result y from device.**

<figure class="align-center">
<img src="images/tutorial_gemv_1_3.png" width="500"
alt="images/tutorial_gemv_1_3.png" />
</figure>
