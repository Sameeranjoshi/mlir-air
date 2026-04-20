*(Original URL: https://sdk.cerebras.net/csl/tutorials/index.html)*

---

# Tutorials {#csl-tutorials}

This series of tutorials serves as an introduction to bulding programs
written in CSL using the Cerebras SDK. In each successive tutorial, we
introduce additional language features, using a general matrix-vector
product (GEMV) as our core computation.

<div class="container-fluid">

<div class="row">
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 0</h3>
                <p class="card-text">Basic Syntax</p>
                <a href="gemv-00-basic-syntax/index.html" class="card-link">Learn preliminaries of CSL syntax.</a>
            </div>
        </div>
    </div>

    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 1</h3>
                <p class="card-text">A Complete Program</p>
                <a href="gemv-01-complete-program/index.html" class="card-link">Write a complete CSL program.</a>
            </div>
        </div>
    </div>

    <!-- break -->

</div>
<br/>
<div class="row">
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 2</h3>
                <p class="card-text">Memory DSDs</p>
                <a href="gemv-02-memory-dsds/index.html" class="card-link">Use memory data structure descriptors (DSDs) for efficient operations on tensors.</a>
            </div>
        </div>
    </div>
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 3</h3>
                <p class="card-text">Memcpy</p>
                <a href="gemv-03-memcpy/index.html" class="card-link">Copy tensors from device to host, and vice versa. </a>
            </div>
        </div>
    </div>

    <!-- break -->

</div>
<br/>
<div class="row">
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 4</h3>
                <p class="card-text">Parameters</p>
                <a href="gemv-04-params/index.html" class="card-link">Use parameters for compile-time specification of your program.</a>
            </div>
        </div>
    </div>
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 5</h3>
                <p class="card-text">Multiple PEs</p>
                <a href="gemv-05-multiple-pes/index.html" class="card-link">Run your program on multiple PEs. </a>
            </div>
        </div>
    </div>

    <!-- break -->

</div>
<br/>
<div class="row">
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">GEMV Tutorial 6</h3>
                <p class="card-text">Routes and Fabric DSDs</p>
                <a href="gemv-06-routes-1/index.html" class="card-link">Use routes and colors to distribute a single GEMV across multiple PEs.</a>
            </div>
        </div>
    </div>
    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
            </div>
        </div>
    </div>

    <!-- break -->

</div>
</div>
<hr/>

::: {.toctree maxdepth="2" caption="SDK Tutorials" hidden=""}
gemv-00-basic-syntax/index.rst gemv-01-complete-program/index.rst
gemv-02-memory-dsds/index.rst gemv-03-memcpy/index.rst
gemv-04-params/index.rst gemv-05-multiple-pes/index.rst
gemv-06-routes-1/index.rst
:::
