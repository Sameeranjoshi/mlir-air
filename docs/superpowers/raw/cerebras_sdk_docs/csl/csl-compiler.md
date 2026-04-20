*(Original URL: https://sdk.cerebras.net/csl/csl-compiler.html)*

---

# CSL Compiler

The CSL compiler can be invoked with the command `cslc` on your
terminal. See the following documentation describing the available
options. See `working-with-code-samples`{.interpreted-text role="ref"}
for usage examples.

``` text
usage: cslc [-h] [-o OUTPUT_NAME] [--params PARAMS] [--colors COLORS] [--memcpy] [--channels CHANNELS]
            [--import-path IMPORT_PATH] [--width-west-buf WIDTH_WEST_BUF] [--width-east-buf WIDTH_EAST_BUF] [--verbose]
            csl_filename

Frontend for cslc-driver. Creates a directory and then calls cslc-driver which will write its output files to the created
directory.

positional arguments:
  csl_filename          Input CSL file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_NAME        Output directory name (default: out)
  --params PARAMS       Comma-separated list of param-to-value mappings where a mapping is a `name:value` pair where name
                        is a string and value is an unsigned integer. The parameter list is passed on to cslc-driver as-is.
  --colors COLORS       Comma-separated list of color-to-value mappings where a mapping is a `color:value` pair where color
                        is a string and value is an unsigned integer. The parameter list is passed on to cslc-driver as-is.
  --memcpy              Add memcpy support to this program
  --channels CHANNELS   Number of memcpy I/O channels to use when memcpy support is compiled with this program.
  --import-path IMPORT_PATH
                        Add the given directory to the list of directories searched for <...> paths in @import_module and
                        @set_tile_code statements.
  --width-west-buf WIDTH_WEST_BUF
                        width of west buffer (default is zero, i.e. no buffer to mitigate slow input)
  --width-east-buf WIDTH_EAST_BUF
                        width of east buffer (default is zero, i.e. no buffer to mitigate slow output)
  --verbose             Verbose output
```
