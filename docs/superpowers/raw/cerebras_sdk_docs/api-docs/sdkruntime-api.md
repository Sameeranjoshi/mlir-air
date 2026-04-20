*(Original URL: https://sdk.cerebras.net/api-docs/sdkruntime-api.html)*

---

# SdkRuntime API Reference

This section presents the `SdkRuntime` Python host API reference and
associated utilities to develop kernels for the Cerebras Wafer-Scale
Engine.

## sdkruntimepybind module

Python API for `SdkRuntime` functions.

### MemcpyDataType

### MemcpyOrder

### SdkCompileArtifacts

### SdkExecutionPlatform

### SdkRuntime

### SdkTarget

### SimfabConfig

### Task

### get_platform

### get_simulator

### get_system

## sdk_utils module {#sdkruntime-sdk-utils}

Utility functions for common operations with `SdkRuntime`. Import from
`cerebras.sdk.sdk_utils`.

### calculate_cycles

### input_array_to_u32

### memcpy_view

## debug_util module {#sdkruntime-debug-util}

Utilities for parsing debug output and core files of a simulator run.
Import from `cerebras.sdk.debug.debug_util`.

### debug_util
