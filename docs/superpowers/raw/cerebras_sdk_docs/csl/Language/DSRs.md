*(Original URL: https://sdk.cerebras.net/csl/Language/DSRs.html)*

---

# Data Structure Registers {#language-dsrs}

Data Structure Registers (DSRs) are physical registers that are used to
store DSD values. Each DSR belongs to one of three DSR files, namely the
`dest`, `src0` and `src1` DSR files. All DSD operations will actually
operate on DSRs behind the scenes and therefore, all DSD operands to DSD
operations must be loaded to DSRs before executing the respective DSD
operation.

Certain operations on DSRs require Extended DSRs (XDSRs). In this case,
the DSRs are augmented by the use of an XDSR, a physical register
belonging to a separate file.

The allocation of DSRs and XDSRs, and the loading of DSDs to DSRs, is
typically done automatically by the compiler.

However, it is also possible to create and use DSRs and XDSRs directly.
This chapter describes how users can allocate DSRs and XDSRs and then
load DSDs to DSRs explicitly without the compiler\'s assistance.

## DSR Types

There are 5 types of DSRs supported in CSL, each corresponding to one of
the three DSR files. These are the following:

- `dsr_dest` represents a DSR value that can only be used to store a
  destination operand to a DSD operation.
- `dsr_src0` represents a DSR value that can be used to store a source
  as well as a destination operand to a DSD operation.
- `dsr_src1` represents a DSR value that can be only be used to store a
  source operand to a DSD operation.
- `dsr_fifo_dest` represents a `dsr_dest` DSR that is expected to store
  a FIFO (See `language-fifo-dsrs`{.interpreted-text role="ref"}).
- `dsr_fifo_src1` represents a `dsr_src1` DSR that is expected to store
  a FIFO.

XDSR values are represented by the `xdsr` type.

### FIFO DSR types {#language-fifo-dsrs}

The `dsr_fifo_dest` and `dsr_fifo_src1` types can be used instead of
`dsr_dest` and `dsr_src1`, respectively, to represent DSRs that are
known to store a FIFO if one does not have access to a FIFO object. Like
FIFO objects, non-asynchronous DSD operations on FIFO DSRs will
terminate and return `false` when reading from an empty FIFO or writing
to a full FIFO. Otherwise, FIFO DSR-typed values have the same semantics
as the corresponding non-FIFO DSR types.

Behavior is undefined if a FIFO DSR-typed value is not initialized as
part of a FIFO when it is used in a DSD operation.

If a non-asynchronous DSD operation has a DSR operand that does **not**
have FIFO DSR type, but that DSR holds a FIFO, behavior is undefined if
that FIFO experiences a FIFO full or FIFO empty event. It is the
programmer\'s responsibility to avoid such FIFO full or FIFO empty
events.

## DSR Builtins

### \@get_dsr

Create a unique DSR identifier value. This value will uniquely identify
a physical DSR along with its DSR file.

#### Syntax

``` csl
@get_dsr(dsr_type, dsr_id);
@get_dsr(fifo_dsr_type, non_fifo_dsr);
```

Where:

- `dsr_type` is an expression of type `type` and whose value must be one
  of the DSR types.
- `dsr_id` is a comptime-known expression of integer type.
- `fifo_dsr_type` is an expression of type `type` whose value is one of
  the FIFO DSR types (`dsr_fifo_dest` or `dsr_fifo_src1`).
- `non_fifo_dsr` is a comptime-known expression of a non-FIFO DSR type.
- Returns a value of `dsr_type`.

#### Example

``` csl
const dsr1 = @get_dsr(dsr_dest, 0);
const dsr2 = @get_dsr(dsr_src0, 1);
const dsr3 = @get_dsr(dsr_src1, 6);
const dsr4 = @get_dsr(dsr_fifo_dest, 4);
const dsr5 = @get_dsr(dsr_fifo_src1, dsr3);
```

#### Semantics

Creates a unique DSR identifier value of `dsr_type` type using the
specified integer identifier. This builtin must be evaluated at
comptime. If the provided integer identifier has already been used for
the given DSR type or it is outside the valid range of values for the
given target architecture, then an error will be emitted.

The type of `non_fifo_dsr` must correspond to `fifo_dsr_type`. If
`fifo_dsr_type` is `dsr_fifo_dest`, then `non_fifo_dsr` must have type
`dsr_dest`, and if `fifo_dsr_type` is `dsr_fifo_src1`, then
`non_fifo_dsr` must have type `dsr_src1`.

### \@get_xdsr

Create a unique XDSR identifier value. This value will uniquely identify
a physical XDSR.

#### Syntax

``` csl
@get_xdsr(xdsr_id);
```

Where:

- `xdsr_id` is a comptime-known expression of integer type.
- Returns a value of type `xdsr`.

#### Example

``` csl
const my_xdsr = @get_xdsr(4);
```

#### Semantics

Creates a unique XDSR identifier value using the specified integer
identifier. This builtin must be evaluated at comptime. If the provided
integer identifier has already been used or it exceeds the valid range
of values for the given target architecture, then an error will be
emitted.

### \@load_to_dsr {#language-load-dsrs}

Load a DSD value into a DSR.

#### Syntax

``` csl
@load_to_dsr(dsr_value, dsd_value);
@load_to_dsr(dsr_value, dsd_value, config_struct);
```

Where:

- `dsr_value` a comptime-known expression of a DSR type.
- `dsd_value` an expression of DSD type.
- `config_struct` optional anonymous struct consisting of either of the
  following:
  - Asynchronous configuration setting fields as explained in
    `language-dsds-async`{.interpreted-text role="ref"}. These are
    allowed only for fabric DSDs. The supported settings are:
    - `async`
    - `activate`
    - `unblock`
    - `on_control`
  - The `save_address` setting field. This is allowed only for `mem1d`
    and `mem4d` DSDs. See `language-dsrs-save-address`{.interpreted-text
    role="ref"} for more details.
  - The `single_step` setting field.

#### Example

``` csl
const dsr1 = @get_dsr(dsr_dest, 0);
const dsr2 = @get_dsr(dsr_src0, 1);

fn foo(mem_dsd: mem1d_dsd, fab_dsd: fabin_dsd) void {
  // Loads a memory DSD to a DSR.
  @load_to_dsr(dsr1, mem_dsd);

  // Loads a fabric DSD to a DSR while specifying that the
  // input DSD is asynchronous with activation and on_control settings.
  @load_to_dsr(dsr2, fab_dsd, .{.async = true,
                                .activate = callback,
                                .on_control = .{.terminate = true}});
}

const A = @zeros([10]f16);
const mem_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{10} -> A[i]});
const fab_dsd = @get_dsd(fabin_dsd, .{.extent = 10,
                                      .fabric_color = @get_color(1),
                                      .input_queue = @get_input_queue(1)});

comptime {
  // The DSD will be loaded to the DSR at comptime, i.e., before the
  // program begins its execution.
  @load_to_dsr(dsr1, mem_dsd);

  // A fabric DSD with asynchronous properties will be loaded at comptime.
  @load_to_dsr(dsr2, fab_dsd, .{.async = true,
                                .activate = callback,
                                .on_control = .{.terminate = true}});

  // fab_dsd uses color 1 with input queue 1. When using explicit DSRs,
  // we must explicitly initialize the queue.
  @initialize_queue(@get_input_queue(1), .{ .color = @get_color(1) });
}
```

#### Semantics

The `@load_to_dsr` builtin can be called at comptime or runtime.

If it is called at runtime it will load the input DSD to the specified
DSR at runtime.

If it is called at comptime, the specified DSD will be loaded to the DSR
before the program begins executing.

A DSD of type `fabin_dsd` cannot be loaded to a `dsr_dest` DSR.

A DSD of type `fabout_dsd` cannot be loaded to a `dsr_src0` or
`dsr_src1` DSRs.

FIFO DSRs are not permitted in `@load_to_dsr`.

When using a `fabin_dsd` loaded to a DSR, the input queue used by the
`fabin_dsd` must be explicitly initialized with the associated color via
`@initialize_queue`.

On WSE-3, when using a `fabout_dsd` loaded to a DSR, the output queue
used by the `fabout_dsd` must be explicitly initialized with the
associated color via `@initialize_queue`.

### save_address {#language-dsrs-save-address}

The `save_address` option may be supplied to `@load_to_dsr` if the DSD
is of the type `mem1d_dsd` or `mem4d_dsd`. This causes subsequent DSD
operations on the DSR to update the DSR\'s base address for the
outermost (slowest-varying) dimension after termination to point one
position past the end of the range covered by the DSD operation. The
next operation on the DSR will effectively pick up where the previous
one ended.

#### Example

``` csl
const CHUNK_LENGTH = 4;
const N_CHUNKS = 3;

var chunks_in =
  [CHUNK_LENGTH * N_CHUNKS]i16 {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11
  };

var chunks_out = @zeros(
  [CHUNK_LENGTH * N_CHUNKS]i16
);

const chunks_in_dsd = @get_dsd(
  mem1d_dsd,
  .{ .tensor_access = |i|{CHUNK_LENGTH} -> chunks_in[i] }
);

const chunks_out_dsd = @get_dsd(
  mem1d_dsd,
  .{ .tensor_access = |i|{CHUNK_LENGTH} -> chunks_out[i] }
);

comptime {
  @load_to_dsr(chunks_out_dsr, chunks_out_dsd, .{ .save_address = true });
  @load_to_dsr(chunks_in_dsr, chunks_in_dsd, .{ .save_address = true });
}

task main() void {
  //
  // Each call to @mov16 will copy a chunk of size CHUNK_LENGTH, as
  // specified in the .tensor_access expression. The base address for both
  // the source and target operations will be incremented by CHUNK_LENGTH
  // each time.
  //
  // Thus the following loop is semantically equivalent to:
  //
  //     for (@range(i16, N_CHUNKS)) |i| {
  //       @mov16(chunks_out_dsr, chunks_in_dsr);
  //       @set_dsr_base_addr(chunks_out_dsr,
  //                          &chunks_out[CHUNK_LENGTH * (i+1)]);
  //       @set_dsr_base_addr(chunks_in_dsr,
  //                        &chunks_in[CHUNK_LENGTH * (i+1)]);
  //     }
  //
  for (@range(i16, N_CHUNKS)) |i| {
    @mov16(chunks_out_dsr, chunks_in_dsr);
  }
}
```

### single_step {#language-dsrs-single-step}

The `single_step` option may be supplied to `@load_to_dsr` to support
use with the `@map` builtin. When a DSR is used as an argument to
`@map`, it should be loaded with a DSD value where
`.single_step = true`, otherwise the behavior is unspecified. If a DSR
loaded with a DSD value where `.single_step = true` is used as an
argument to DSD builtins other than `@map`, the behavior is undefined.

#### Example

``` csl
const math_lib = @import_module("<math>");
const memDSD = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> A[i, i]);
const faboutDSD = @get_dsd(fabout_dsd,
                           .{.extent = size, .fabric_color = blue});
param inDSR: dsr_src1;
param outDSR: dsr_dest;

task foo() void {
  // Compute the square-root of each element of `memDSD` and
  // send it out to `faboutDSD`.
  @load_to_dsr(inDSR, memDSD, .{.single_step = true});
  @load_to_dsr(outDSR, faboutDSD, .{.single_step = true});
  @map(math_lib.sqrt_f16, inDSR, outDSR);
}
```

### \@allocate_fifo with DSRs

By default, the DSRs and XDSR used by `@allocate_fifo` (see
`language-dsds-fifos`{.interpreted-text role="ref"}) are allocated by
the compiler. However, it supports the use of user-specified DSRs and
XDSR as well, using the following syntax:

``` csl
@allocate_fifo(fifo_buffer, config_struct);
```

Where, in order to allocate a FIFO with user-specified DSRs and XDSR,
`config_struct` must contain the fields:

- `dest`: a comptime-known expression of `dsr_dest` type.
- `src`: a comptime-known expression of `dsr_src1` type.
- `xdsr`: a comptime-known expression of `xdsr` type.

The fields `dest`, `src`, and `xdsr` must all be specified together, or
all absent, otherwise an error will be emitted. The integer identifiers
of `dest` and `src` must match. If the provided DSR and XDSR identifiers
have already been used for their respective types or exceed the valid
range of values for the given target architecture, then an error will be
emitted.

Other fields of `config_struct` described in
`language-dsds-fifo-activate-push-pop`{.interpreted-text role="ref"}
retain their same semantics when the DSRs and XDSR are specified.

#### Example

``` csl
var buf = @zeros([240]u16);
const my_fifo = @allocate_fifo(buf, .{
   .dest = @get_dsr(dsr_dest, 4),
   .src = @get_dsr(dsr_src1, 4),
   .xdsr = @get_xdsr(1)});
```
