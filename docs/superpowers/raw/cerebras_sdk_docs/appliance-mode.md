*(Original URL: https://sdk.cerebras.net/appliance-mode.html)*

---

# Running SDK on a Wafer-Scale Cluster {#appliance-mode}

:::: note
::: title
Note
:::

The Cerebras Wafer-Scale Cluster appliance running Cerebras ML Software
2.4 supports SDK 1.3. [See here for SDK 1.3
documentation](https://cerebras-sdk-docs-130.netlify.app).

The Cerebras Wafer-Scale Cluster appliance running Cerebras ML Software
2.5 supports SDK 1.4, the current version of SDK software.
::::

In addition to the containerized Singularity build of the Cerebras SDK
(see `install-guide`{.interpreted-text role="ref"}) for installation
information), the SDK is also supported on Cerebras Wafer-Scale Clusters
(WSC) running in appliance mode.

This page documents some modifications needed to your code to run on a
Wafer-Scale Cluster. For more information about setting up and using a
Wafer-Scale Cluster, see [the documention
here](https://training-docs.cerebras.ai/getting-started/overview).

## Summary

The [Cerebras Wafer-Scale Cluster](https://cerebras.ai/product-cluster/)
is our solution to training massive neural networks with near-linear
scaling. The Wafer-Scale Cluster consists of one or more CS systems,
together with special CPU nodes, memory servers, and interconnects,
presented to the end user as a single system, or appliance. The
appliance is responsible for job scheduling and allocation of the
systems.

There are two types of SDK jobs that can run on the appliance: compile
jobs, which are used to compile code on a worker node, and run jobs,
which either run the compiled code on a worker node using the simulator,
or run the code on a real CS system within the appliance.

We will walk through some changes necessary to compile and run your code
on a Wafer-Scale Cluster. Modified code examples for supporting a
Wafer-Scale Cluster can be requested from <developer@cerebras.net>.

Note that there are currently some limitations for running SDK jobs on a
Wafer-Scale Cluster. Unlike ML jobs, SDK jobs can only use a single
worker node and CS system.

See `sdk-appliance-api-reference`{.interpreted-text role="ref"} for the
full API documentation.

## Setup and Installation

First, learn about the components of the Cerebras Wafer-Scale Cluster
[here](https://training-docs.cerebras.ai/concepts/cerebras-wafer-scale-cluster).

The user interacts with the Wafer-Scale Cluster via a user node. Start
by setting up a Python virtual environment on the user node:

``` bash
$ python3.8 -m venv sdk_venv
$ source sdk_venv/bin/activate
```

Next, install the `cerebras_appliance` and `cerebras_sdk` Python wheels
in the virtual enviroment, specifying the proper Cerebras Software
release:

``` bash
(sdk_venv) $ pip install --upgrade pip
(sdk_venv) $ pip install cerebras_appliance==2.5.0
(sdk_venv) $ pip install cerebras_sdk==2.5.0
```

## Compiling

As an example, we\'ll walk through porting
`sdkruntime-gemv-01-complete-program`{.interpreted-text role="ref"}. In
the containerized SDK setup, this code is compiled with the following
command:

``` bash
cslc ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out
```

To compile for the Wafer-Scale cluster, we use a Python script which
launches a compile job:

``` python
import json
from cerebras.sdk.client import SdkCompiler

# Instantiate copmiler using a context manager
# Disable version check to ignore appliance client and server version differences.
with SdkCompiler(disable_version_check=True) as compiler:

    # Launch compile job
    artifact_path = compiler.compile(
        ".",
        "layout.csl",
        "--fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out",
        "."
    )

    # Write the artifact_path to a JSON file
    with open("artifact_path.json", "w", encoding="utf8") as f:
        json.dump({"artifact_path": artifact_path,}, f)
```

The `SdkCompiler::compile` function takes four arguments:

> - the directory containing the CSL code files,
> - the name of the top level CSL code file that contains the layout
>   block,
> - the compiler arguments,
> - and the output directory or output file for the compile artifacts.

The last argument can either be a directory, specifying the location to
which compile artifacts will be copied with default file name; or a file
name, explicitly specifying the name and location for the compile
artifacts. The function returns the compile artifact path. We write this
artifact path to a JSON file which will be read by the runner object in
the Python host code.

Just as before, simply pass the full dimensions of the target system to
the `--fabric-dims` argument to compile for a real hardware run.

The `SdkCompiler()` constructor can take a few optional `kwargs`,
including:

> - `resource_cpu`: amount of CPU cores on the WSC\'s management node
>   used by the compile job in units of 1/1000 CPU (default: 24000, or
>   24 cores)
> - `resource_mem`: number of bytes of memory requested from the
>   management node for the compile job (default: `64 << 30`, or 64 GiB)
> - `disable_version_check`: specifies whether to ignore version
>   differences between appliance client and server

If SDK compilation jobs on the WSC are often waiting in the queue behind
other jobs, such as ML execute or run jobs, this is typically because
not enough resources are available on the management node. These
`kwargs` can be used to request less resources from the management node
and increase the number of simultaneously running jobs.

## Running with SdkLauncher {#appliance-sdklauncher}

In the containerized SDK setup, our Python host code for running is as
follows:

``` python
import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Matrix dimensions
M = 4
N = 6

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M, N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Load and run the program
runner.load()
runner.run()

# Launch the init_and_compute function on device
runner.launch('init_and_compute', nonblock=False)

# Copy y back from device
y_symbol = runner.get_id('y')
y_result = np.zeros([1*1*M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)
print("SUCCESS!")
```

If this file is named `run.py` and the compilation output is in the
directory `out`, we run it with the command:

``` bash
cs_python run.py --name out
```

To run on hardware, we then specify an IP address with the `--cmaddr`
flag.

We provide an `SdkLauncher` class for running host code directly from a
worker node within the appliance. This class allows files to be staged
on the appliance before running the same host code that you would with
the Singularity container.

We demonstrate below using `SdkLauncher` to run the host code for the
example above. We also include a demonstration of `stage` for
transferring a file to the appliance.

To pass a system address to a run script when using `SdkLauncher`, the
user must use the `%CMADDR%` template string, as demonstrated below.

``` python
import json
import os

from cerebras.sdk.client import SdkLauncher

# read the compile artifact_path from the json file
with open("artifact_path.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_path = data["artifact_path"]

# artifact_path contains the path to the compiled artifact.
# It will be transferred and extracted in the appliance.
# The extracted directory will be the working directory.
# Set simulator=False if running on CS system within appliance.
# Disable version check to ignore appliance client and server version differences.
with SdkLauncher(artifact_path, simulator=True, disable_version_check=True) as launcher:

    # Transfer an additional file to the appliance,
    # then write contents to stdout on appliance
    launcher.stage("additional_artifact.txt")
    response = launcher.run(
        "echo \"ABOUT TO RUN IN THE APPLIANCE\"",
        "cat additional_artifact.txt",
    )
    print("Test response: ", response)

    # Run the original host code as-is on the appliance,
    # using the same cmd as when using the Singularity container
    response = launcher.run("cs_python run.py --name out --cmaddr %CMADDR%")
    print("Host code execution response: ", response)

    # Fetch files from the appliance
    launcher.download_artifact("sim.log", "./output_dir/sim.log")
```

## Running with SdkRuntime Bindings

:::: note
::: title
Note
:::

The `SdkRuntime` appliance bindings are deprecated. Use `SdkLauncher` to
wrap an SDK host Python script instead.
::::

For appliance mode, we can also make some modifications to our original
`run.py` script and run this on the appliance:

``` python
import json
import os

import numpy as np

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime

# Matrix dimensions
M = 4
N = 6

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M, N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Read the artifact_path from the JSON file
with open("artifact_path.json", "r", encoding="utf8") as f:
    data = json.load(f)
    artifact_path = data["artifact_path"]

# Instantiate a runner object using a context manager.
# Set simulator=False if running on CS system within appliance.
# Disable version check to ignore appliance client and server version differences.
with SdkRuntime(artifact_path, simulator=True, disable_version_check=True) as runner:
    # Launch the init_and_compute function on device
    runner.launch('init_and_compute', nonblock=False)

    # Copy y back from device
    y_symbol = runner.get_id('y')
    y_result = np.zeros([1*1*M], dtype=np.float32)
    runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)
print("SUCCESS!")
```

In particular, note that:

> - The imports have changed to reflect appliance modules.
> - We read the path of our compile artifacts from the JSON file
>   generated when compiling.
> - We no longer need to specify a CM address when running on real
>   hardware. Instead, we simply pass a flag to the `SdkRuntime`
>   constructor specifying whether to run in the simulator or on
>   hardware.
> - `load()` and `run()` are replaced by `start()`.
> - We must use a [context
>   manager](https://docs.python.org/3/reference/datamodel.md#context-managers)
>   for the runner object. If we do so, the `start()` and `stop()`
>   functions are implicit, and we do not need to explicitly call them.

## Appliance Logging

When running with the appliance, you can control the level of
appliance-related logging printed to the console.

By default, the logger for the appliance is set to `WARNING`, which
means only `WARNING` and higher level messages will be enabled. You can
set the level of the logger directly to enable other levels, such as
`INFO` or `DEBUG`. For example:

``` python
import logging
from cerebras.appliance import logger

logging.basicConfig(level=logging.INFO)
```

## Appliance Job Monitoring and Management

Jobs can be monitored on the cluster with the `cstcl` CLI tool. Find
more information on cluster job monitoring and `csctl` in the [Cerebras
training
docs](https://training-docs.cerebras.ai/rel-2.5.0/cluster-monitoring/cerebras-job-scheduling-and-monitoring/cli-for-job-monitoring-csctl).

`Ctrl-C` can be used to cancel a running job.
