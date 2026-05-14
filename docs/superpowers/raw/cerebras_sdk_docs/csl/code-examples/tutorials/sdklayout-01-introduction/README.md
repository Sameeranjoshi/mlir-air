*(Original URL: https://sdk.cerebras.net/csl/code-examples/tutorials/sdklayout-01-introduction/README.html)*

---

# SdkLayout 1: Introduction

This tutorial introduces the `SdkLayout` API. `SdkLayout` allows us to
define and compile multi-PE WSE programs. Specifically, it consists of
the following main features:

- Creation of CSL code regions: rectangular CSL code regions can be
  instantiated given a CSL source code file path, a name, and the width
  and height dimensions.
- Routing and switching: for a given CSL code region we can specify
  routing and switching information on a single PE within the code
  region, on a rectangular sub-region, or on the entire code region. See
  tutorial `sdkruntime-sdklayout-02-routing`{.interpreted-text
  role="ref"}.
- Automatic color allocation: routing can be done based on symbolic
  colors. The `SdkLayout` engine will then allocate physical values
  automatically. See tutorials
  `sdkruntime-sdklayout-02-routing`{.interpreted-text role="ref"} and
  `sdkruntime-sdklayout-03-ports-and-connections`{.interpreted-text
  role="ref"}.
- Automatic routing between code regions: users can create input and
  output ports on code regions and connect them. The `SdkLayout` engine
  will automatically find optimal routes between them. See tutorial
  `sdkruntime-sdklayout-03-ports-and-connections`{.interpreted-text
  role="ref"}.
- Host-to-device and device-to-host connections: an input or output port
  can be connected to the host to create an input or output stream
  respectively. See tutorial
  `sdkruntime-sdklayout-04-h2d-d2h`{.interpreted-text role="ref"}.

This tutorial demonstrates the most basic compilation flow, where a
single-PE program with no colors and no routing sets the value of a
global variable in device memory based on the value of a parameter.
