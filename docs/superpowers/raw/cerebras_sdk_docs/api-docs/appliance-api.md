*(Original URL: https://sdk.cerebras.net/api-docs/appliance-api.html)*

---

# SDK Appliance API Reference

This section presents the SDK Appliance API reference for running SDK
programs on a Cerebras Wafer-Scale Cluster (WSC).

See `appliance-mode`{.interpreted-text role="ref"} for an introduction
to running in appliance mode on a Wafer-Scale Cluster.

## SdkCompiler {#sdk-appliance-sdkcompiler}

Python API for compiling SDK programs on a Cerebras Wafer-Scale Cluster.

## SdkLauncher {#sdk-appliance-sdklauncher}

## SdkRuntime {#sdk-appliance-sdkruntime}

:::: note
::: title
Note
:::

The `SdkRuntime` appliance bindings are deprecated. Use `SdkLauncher` to
wrap an SDK host Python script instead.
::::

## sdk_utils {#sdk-appliance-sdk-utils}

Utility functions for common operations with
`SdkRuntime`{.interpreted-text role="py:class"}. Import from
`cerebras.sdk.client.sdk_utils`.

See `sdkruntime-sdk-utils`{.interpreted-text role="ref"}.

## debug_util {#sdk-appliance-debug-util}

Utilities for parsing debug output and core files of a simulator run.
Import from `cerebras.sdk.client.debug_util`.

See `sdkruntime-debug-util`{.interpreted-text role="ref"}.
