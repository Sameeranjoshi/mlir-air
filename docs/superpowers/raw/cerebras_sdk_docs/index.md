*(Original URL: https://sdk.cerebras.net/index.html)*

---

# Documentation for Developing with CSL

This is the documentation for developing kernels for Cerebras system.
Here you will find getting started guides, quickstarts, tutorials, code
samples, release notes, and more.

<div class="container-fluid">

<div class="row">

    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">Start Here</h3>
                <p class="card-text">Computing with Cerebras</p>
                <a href="computing-with-cerebras.html" class="card-link">A conceptual, "mental model" view.</a>
            </div>
        </div>
    </div>

    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">Installation Guide</h3>
                <p class="card-text">Installing the Cerebras SDK</p>
                <a href="installation-guide.html" class="card-link">Setup your environment for using the fabric simulator or a real CS system.</a>
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
                <h3 class="card-title">Introductory Tutorials</h3>
                <p class="card-text">Step-by-step instruction in CSL</p>
                <a href="csl/tutorials/index.html" class="card-link">Get started writing your first programs in CSL using our SDK.</a>
            </div>
        </div>
    </div>

    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">Working with Code Samples</h3>
                <p class="card-text">Learn how to run the code samples</p>
                <a href="csl/working-with-code-samples.html" class="card-link">A detailed look into compiling and running the provided code samples.</a>
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
                <h3 class="card-title">CSL Code Samples</h3>
                <p class="card-text">Explore CSL programs</p>
                <a href="csl/code-examples/index.html" class="card-link">From simple single-PE programs to full-wafer conjugate gradients.</a>
            </div>
        </div>
    </div>

    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">CSL Language Guide</h3>
                <p class="card-text">See how to use CSL</p>
                <a href="csl/language_index.html" class="card-link">Reference for the CSL language.</a>
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
                <h3 class="card-title">SdkRuntime API Reference</h3>
                <p class="card-text">High-performance SDK host API</p>
                <a href="api-docs/sdkruntime-api.html" class="card-link">Run binaries on Cerebras systems using our high-performance runtime.</a>
            </div>
        </div>
    </div>

    <div class="col">
        <div class="card h-100" >
            <div class="card-body">
                <h3 class="card-title">SDK Appliance API Reference</h3>
                <p class="card-text">Wafer-Scale Cluster SDK appliance API</p>
                <a href="api-docs/appliance-api.html" class="card-link">Compile and run binaries on Cerebras Wafer-Scale Clusters in appliance mode.</a>
            </div>
        </div>
    </div>

    <!-- break -->

</div>
</div>
<hr/>

::: {.toctree maxdepth="3" hidden=""}
sdk-release-notes/sdk-rel-notes-cumulative.rst
sdk-release-notes/sdk-doc-updates.rst
:::

::: {.toctree maxdepth="3" caption="Start Here" hidden=""}
computing-with-cerebras.rst tensor-streaming.rst installation-guide.rst
csl/tutorials/index.rst
:::

::: {.toctree caption="Development Guides" hidden=""}
csl/csl-compiler.rst csl/working-with-code-samples.rst
csl/code-examples/index.rst csl/language_index.rst appliance-mode.rst
:::

::: {.toctree caption="Debugging" hidden=""}
debug/debugging.rst debug/sdk-gui.rst
:::

::: {.toctree caption="Host API Reference" hidden=""}
api-docs/sdkruntime-api.rst api-docs/sdklayout-api.rst
api-docs/appliance-api.rst
:::
