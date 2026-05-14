*(Original URL: https://sdk.cerebras.net/csl/Language/Libraries.html)*

---

# Libraries {#language-libraries}

Libraries are imported by enclosing the name of the library with angled
brackets.

``` csl
/// main.csl

const math = @import_module("<math>");

fn distance(x0 : f16, y0 : f16, x1 : f16, x1 : f16) f16 {
  return math.sqrt((x0-x1)*(x0-x1) + (x0-x1)*(x0-x1));
}
```

## \<complex\> {#language-libraries-complex}

The `complex` library provides structs containing `real` and `imag`
components and basic complex functions.

`complex` is a generic struct parameterized by its field type. The
`complex_32` and `complex_64` non-generic names are also provided; these
define a complex number using two `f16` values and a complex number
using two `f32` values, respectively.

`get_complex` is a generic constructor that returns a complex struct
based on the type of its inputs. The non-generic `get_complex_32` and
`get_complex_64` constructor functions are provided as well:

``` csl
// Returns struct {real: T, imag: T} where T can be f16 or f32
fn complex(comptime T: type) type
const complex_32 = complex(f16); // struct {real: f16, imag: f16}
const complex_64 = complex(f32); // struct {real: f32, imag: f32}

// Can operate on f16 or f32
fn get_complex(r: anytype, i: @type_of(r)) complex(@type_of(r))
fn get_complex_32(r : f16, i : f16) complex_32
fn get_complex_64(r : f32, i : f32) complex_64
```

The following functions are provided for operating on complex numbers.
They are written as generic functions to facilitate use in other
libraries or abstractions. In addition, non-generic `complex_32` and
`complex_64` functions are provided. These functions have names suffixed
with `_32` and `_64`, respectively.

``` csl
// x, y can be complex_32 or complex_64
fn add_complex(x: anytype, y: @type_of(x)) @type_of(x)
fn subtract_complex(x: anytype, y: @type_of(x)) @type_of(x)
fn multiply_complex(x: anytype, y: @type_of(x)) @type_of(x)
```

## \<control\> {#language-libraries-control}

The `control` library provides utilities for constructing control
wavelets.

The following functions and enums are provided by the library:

``` csl
// Max commands that can be encoded in a control wavelet
const MAX_CMDS = 8;

// Struct for representing switching opcodes
const opcode = enum(u32) {
  NOP = 0,
  SWITCH_ADV = 1,
  SWITCH_RST = 2,
  TEARDOWN = 3
};

// Encode payload that activates a control task with no argument
fn encode_control_task_payload(entrypoint: control_task_id) u32;

// Encode payload with one switch command, plus control task entrypoint.
fn encode_single_payload(cmd: opcode, ce_ignore: bool,
                         comptime entrypoint: control_task_id, data: u16) u32;

// Encode general control wavelet payload
fn encode_payload(comptime N: u16, comptime cmds: [N]opcode,
                  comptime ce_ignore: [N]bool,
                  ce_ignore_remaining: bool,
                  comptime entrypoint: anytype) u32;
```

All functions construct a payload returned as a 32-bit unsigned integer
which can be sent in a control wavelet.

`encode_control_task_payload` returns a control wavelet payload which
activates a control task on all receiving PEs. It has one argument:

- `entrypoint`: a `control_task_id` which is bound to the control task
  activated on a CE by the receipt of this wavelet.

`encode_single_payload` returns a control wavelet payload containing one
switch command, along with an optional control task entrypoint with
16-bit data argument. The function has the following arguments:

- `cmd`: a switching opcode to be consumed by the receiving PE router.
  This command will instruct the router to modify the configuration of
  the color on which the control wavelet is sent. This command can
  advance the switch position, reset the switch position, teardown the
  color, or do nothing. If the router of the PE on which the control
  wavelet is sent pops this command, then no additional receiving PEs
  will receive a switching opcode.
- `ce_ignore`: a boolean which determines whether this control wavelet
  is to be ignored by the CE of PEs which receive it. If `true`, the
  control wavelet will not be forwarded to the CE. If `false`, and the
  receiving color is configured to transmit down the `RAMP`, the control
  wavelet will be forwarded to the CE. `ce_ignore` must be `false` for
  an `entrypoint` to be activated by a receiving PE.
- `entrypoint`: a `control_task_id` will be activated on a CE by the
  receipt of this wavelet. Passing `{}` indicates that no control task
  activation on receiving PEs is desired. The control task will only be
  activated on a CE if `ce_ignore` is `false`, and the receiving color
  is configured to transmit down the `RAMP`.
- `data`: The control task activated by `entrypoint` may take a single
  16-bit argument. If the control task takes no argument, then this
  value will be ignored.

`encode_payload` can encode a general control wavelet payload with up to
eight switching commands. The function has the following arguments:

- `N`: number of commands to encode in the control wavelet. Maximum
  number of commands is eight.
- `cmd`: an array of switching opcodes to be consumed by PE routers.
  Each command will instruct the router to modify the configuration of
  the color on which the control wavelet is sent. Each command can
  advance the switch position, reset the switch position, teardown the
  color, or do nothing. If the router of the PE on which a command is
  executed pops the command, then the *next* command will be executed by
  the next receiving router.
- `ce_ignore`: an array of booleans which determines whether this
  control wavelet is to be ignored by the CE of PEs which receive it.
  Each `ce_ignore` value is processed along with the associated `cmd`,
  i.e., the same rules for popping commands apply. If the processed
  value is `true`, the control wavelet will not be forwarded to the CE.
  If `false`, and the receiving color is configured to transmit down the
  `RAMP`, the control wavelet will be forwarded to the CE. `ce_ignore`
  must be `false` for an `entrypoint` to be activated by a receiving PE.
- `ce_ignore_remaining`: a boolean which determines whether all other
  commands contained in this control wavelet are to be ignored by the CE
  of PEs receiving it. When `ce_ignore_remaining` is set to `false`,
  each unspecified command will travel down the `RAMP` and reach the CE
  (as a `NOP` command).
- `entrypoint`: a `control_task_id` which is bound to the control task
  activated on a CE by the receipt of this wavelet. Passing `{}`
  indicates that no control task activation on receiving PEs is desired.
  The control task will only be activated on a CE if `ce_ignore` is
  `false`, and the receiving color is configured to transmit down the
  `RAMP`. Because this function can encode up to eight switching
  commands, no data payload can be provided for this control task.

Unlike `encode_single_payload`, `encode_payload` does not take a `data`
argument. If a control payload only contains a single switching command,
then a 16-bit data argument can be supplied as an argument to the
control task activated on receipt of the wavelet. `data` is not
meaningful if there is more than one switching command in the control
wavelet, because the bits that would encode `data` encode the additional
switching commands instead.

A control task that declares no arguments will ignore `data`, and
furthermore, `data` is ignored if the wavelet is not forwarded to the CE
(the current command\'s `ce_ignore` value is `true`).

### Example

The task `main_task` sends out a control wavelet along the color `comm`,
which encodes a control task ID:

``` csl
const ctrl = @import_module("<control>");

const comm = @get_color(0);
const comm_out_queue = @get_output_queue(2);
const ctrl_entrypt_id = @get_control_task_id(40);

task main_task() void {
  const comm_out_dsd = @get_dsd(fabout_dsd, .{
                         .extent = 1,
                         .fabric_color = comm,
                         .control = true,
                         .output_queue = comm_out_queue,
                       });
  @mov32(comm_out_dsd, ctrl.encode_control_task_payload(ctrl_entrypt_id));
}
```

PEs which receive this wavelet along the color `comm` will activate a
control task bound to this control task ID. For instance, if the
receiving PE has the following code, then upon receipt of the control
wavelet, it will activate a task which increments the value `my_global`:

``` csl
const my_ctrl_id = @get_control_task_id(40);
var my_global: u32 = 0;

task my_ctrl_task() void {
  my_global += 1;
}

comptime {
  @bind_control_task(my_ctrl_task, my_ctrl_id);
}
```

## \<debug\> {#language-libraries-debug}

The `debug` library provides a tracing mechanism to record tagged
values.

``` csl
// Record values of the specified type
fn trace_bool(x : bool) void
fn trace_u8(x : u8) void
fn trace_i8(x : i8) void
fn trace_u16(x : u16) void
fn trace_i16(x : i16) void
fn trace_f16(x : @fp16()) void
fn trace_u32(x : u32) void
fn trace_i32(x : i32) void
fn trace_f32(x : f32) void

// Record a compile-time string
fn trace_string(comptime str : comptime_string) void

// Generic version
fn trace(x : anytype) void

// Record timestamp using the <time> library
fn trace_timestamp() void

// These functions are for internal use, recording raw words
fn tagged_put_u8(tag : u8, x : u8) void
fn put_u16(x : u16) void
fn put_u32(x : u32) void
```

A minimal example of a PE program recording timestamps and values using
an imported instance of the `<debug>` library:

``` csl
// pe_program.csl
// When importing an instance of the <debug> module, two things must be
// specified:
//
//   * (key: comptime_string) user-specified key that can be used to
//       retrieve the contents of the trace buffer after execution
//   * (buffer_size: comptime_int) size of buffer for recording traces
const trace = @import_module(
  "<debug>",
  .{ .key = "debug_example",
     .buffer_size = 100,
   }
);

var global : i16 = 0;

task main_task() void {
  // Record timestamp for beginning of task
  trace.trace_timestamp();

  // Record a compile-time string
  trace.trace_string("Hello, world");

  // Update global variable and record
  global = 5;
  trace.trace_i16(global);

  // Record timestamp for end of task
  trace.trace_timestamp();
}
```

## \<directions\> {#language-libraries-directions}

The `directions` library provides utility functions for manipulating
directions.

``` csl
fn rotate_clockwise(d : direction) direction
fn rotate_counterclockwise(d : direction) direction
fn flip_vertical(d : direction) direction
fn flip_horizontal(d : direction) direction
fn flip(d : direction) direction
```

## \<dsd_ops\> {#language-libraries-dsd-ops}

The `dsd_ops` library provides wrappers around DSD op builtins that
select an appropriate builtin depending on argument indicating the types
of the underlying data. These wrappers are guaranteed to expand to a
single call to a DSD op builtin. The wrappers may be used with any
combination of DSD, DSR, scalar, or pointer-to-scalar operands that is
supported by the underlying builtin operation.

Each function operates on a limited set of types. For DSD operations,
the programmer must ensure that the specified type accurately reflects
the type of the data being accessed in memory or streamed via the DSD.

The final argument, named `config`, is a configuration struct for the
underlying DSD op builtin. See `language-builtins`{.interpreted-text
role="ref"} for more details on the builtins underlying these functions.

Note that the `config` argument must be completely comptime-known. This
means that runtime `.activate` or `.unblock` values are not allowed with
these wrapper functions. We hope to lift this limitation in a future
release.

``` csl
// Data movement.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fmovh(dst, src, config)
//     f32  @fmovs(dst, src, config)
//     i16  @mov16(dst, src, config)
//     u16  @mov16(dst, src, config)
//     i32  @mov32(dst, src, config)
//     u32  @mov32(dst, src, config)
inline fn mov(T: type, dst: anytype, src: anytype,
              comptime config: anytype) bool;

// Conversion between data types.
//
//    Tdst     Tsrc  Resulting Operation
// -------  -------  -------------------
// @fp16()      f32  @fs2h(dst, src, config)
// @fp16()      i16  @xp162fh(dst, src, config)
// @fp16()  @fp16()  @fmovh(dst, src, config)
//     f32  @fp16()  @fh2s(dst, src, config)
//     f32     i16   @xp162fs(dst, src, config)
//     f32     f32   @fmovs(dst, src, config)
//     i16  @fp16()  @fh2xp16(dst, src, config)
//     i16     f32   @fs2xp16(dst, src, config)
//     i16     i16   @mov16(dst, src, config)
inline fn convert(Tdst: type, Tsrc: type,
                  dst: anytype, src: anytype, comptime config: anytype) bool;

// Addition.
//
//    Tdst     Tsrc  Resulting Operation
// -------  -------  -------------------
// @fp16()  @fp16()  @faddh(dst, src0, src1, config)
//     f32  @fp16()  @faddhs(dst, src0, src1, config)
//     f32      f32  @fadds(dst, src0, src1, config)
//     i16      i16  @add16(dst, src0, src1, config)
//     u16      u16  @add16(dst, src0, src1, config)
inline fn add(Tdst: type, Tsrc: type,
              dst: anytype, src0: anytype, src1: anytype,
              comptime config: anytype) bool;

// Subtraction.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fsubh(dst, src0, src1, config)
//     f32  @fsubs(dst, src0, src1, config)
//     i16  @sub16(dst, src0, src1, config)
inline fn sub(T: type, dst: anytype, src0: anytype, src1: anytype,
              comptime config: anytype) bool;

// Multiplication.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fmulh(dst, src0, src1, config)
//     f32  @fmuls(dst, src0, src1, config)
inline fn mul(T: type, dst: anytype, src0: anytype, src1: anytype,
              comptime config: anytype) bool;

// Fused multiply and accumulate.
//
//    Tdst     Tsrc  Resulting Operation
// -------  -------  -------------------
// @fp16()  @fp16()  @fmach(dst, src0, src1, x, config)
//     f32  @fp16()  @fmachs(dst, src0, src1, x, config)
//     f32      f32  @fmacs(dst, src0, src1, x, config)
inline fn fmac(Tdst: type, Tsrc: type, dst: anytype,
               src0: anytype, src1: anytype, x: anytype,
               comptime config: anytype) bool;

// Arithmetic negation.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fnegh(dst, src, config)
//     f32  @fnegs(dst, src, config)
inline fn neg(T: type, dst: anytype, src: anytype,
              comptime config: anytype) bool;

// Absolute value.
//
// See also 'math.abs', which is more appropriate for scalar data.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fabsh(dst, src, config)
//     f32  @fabss(dst, src, config)
inline fn abs(T: type, dst: anytype, src: anytype,
              comptime config: anytype) bool;

// Floating point normalization.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fnormh(dst, src, config)
//     f32  @fnorms(dst, src, config)
inline fn norm(T: type, dst: anytype, src: anytype,
               comptime config: anytype) bool;

// Floating-point exponent scaling.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fscaleh(dst, src0, src1, config)
//     f32  @scales(dst, src0, src1, config)
inline fn scale(T: type, dst: anytype, src0: anytype, src1: anytype,
                comptime config: anytype) bool;

// Elementwise maximum.
//
// See also 'math.max', which is more appropriate for scalar data.
//
//       T  Resulting Operation
// -------  -------------------
// @fp16()  @fmaxh(dst, src0, src1, config)
//     f32  @fmaxs(dst, src0, src1, config)
inline fn max(T: type, dst: anytype, src0: anytype, src1: anytype,
              comptime config: anytype) bool;
```

### Example

The following example illustrates the use of `dsd_ops` to build a
generic module that instantiates a local task with a given ID, and moves
data from the given input color via the given input queue, into a
user-specified buffer `buf`.

``` csl
// Filename: reader.csl
param buf;
param taskId: local_task_id;
param inputColor: color;
param inputQueue: input_queue;

const dsd_ops = @import_module("<dsd_ops>");

comptime {
  @comptime_print(bufElemCount);
  @comptime_print(bufElemType);
}
const bufType = @type_of(buf.*);
const bufElemCount = @element_count(bufType);
const bufElemType = @element_type(bufType);
const bufDSD = @get_dsd(mem1d_dsd, .{ .base_address = buf,
                                      .extent = bufElemCount });

const inputDSD = @get_dsd(fabin_dsd, .{ .extent = bufElemCount,
                                        .fabric_color = inputColor,
                                        .input_queue = inputQueue });

task t() void {
  dsd_ops.mov(bufElemType, bufDSD, inputDSD, .{});
}

fn bind_and_activate() void {
  @bind_local_task(t, taskId);
  @activate(taskId);
}
```

``` csl
// Filename: main.csl
var bufOfInt16 = @zeros([16]i16);
var bufOfFloat32 = @zeros([32]f32);

const readerInt16 = @import_module(
  "reader.csl",
  .{
    .buf = &bufOfInt16,
    .taskId = @get_local_task_id(8),
    .inputColor = @get_color(2),
    .inputQueue = @get_input_queue(2)
  }
);

const readerFloat32 = @import_module(
  "reader.csl",
  .{
    .buf = &bufOfFloat32,
    .taskId = @get_local_task_id(9),
    .inputColor = @get_color(3),
    .inputQueue = @get_input_queue(3)
  }
);

comptime {
  readerInt16.bind_and_activate();
  readerFloat32.bind_and_activate();
}
```

## \<empty\> {#language-libraries-empty}

This library is empty on purpose. This allows a conditional module
import as follows:

``` csl
const cache = @import_module(if (stage == 0) "<empty>" else cache_name);
```

## \<layout\> {#language-libraries-layout}

This library provides access to information about where the PE is
located. Specifically, the `x` and `y` coordinates in the rectangle can
be accessed at runtime, allowing code to be shared between PEs at
different locations.

``` csl
const layout_module = @import_module("<layout>");

// Return the 0-indexed x and y coord
layout_module.get_x_coord() u16;
layout_module.get_y_coord() u16;
```

## \<malloc\> {#language-libraries-malloc}

The `malloc` library implements an arena allocator using a statically
allocated buffer.

In arena allocators, a single buffer (arena) is used to ensure that all
objects are allocated sequentially in memory. Allocating and
deallocating memory are fast operations, requiring an addition and/or
assignment. The free operation frees all allocated objects at once.

The parameter `buffer_num_words` specifies the number of words of the
statically allocated buffer.

If the param `asserts_enabled` is true, all allocations assert that the
buffer has enough free memory. The default is false.

``` csl
// specify buffer size
const mem = @import_module("<malloc>", .{buffer_num_words = <buffer_size>});

// mem provides the following API:

// returns pointer to num_values of type T
fn malloc(comptime T: type, num_values: u16) [*]T

// non-generic versions, return pointer to num_values of corresponding type
fn malloc_i16(num_values:u16) [*]i16
fn malloc_u16(num_values:u16) [*]u16
fn malloc_f16(num_values:u16) [*]@fp16()
fn malloc_i32(num_values:u16) [*]i32
fn malloc_u32(num_values:u16) [*]u32
fn malloc_f32(num_values:u16) [*]f32

// returns true if an allocation of num_words elements of type T would
// succeed
fn has_enough_space(comptime T: type, num_values: u16) bool

// non-generic versions
// returns true if an allocation of num_words elements of type
// i16, u16, @fp16() would succeed:
fn has_enough_words(num_words:u16) bool
// returns true if an allocation of num_words elements of type
// i32, u32, f32 would succeed:
fn has_enough_double_words(num_double_words:u16) bool

// frees the entire buffer
fn free() void;
```

## \<math\> {#language-libraries-math}

### Math constants

The following can be used anywhere a floating point number is needed.

``` csl
const PI : comptime_float
const E : comptime_float
```

### Math functions

The `math` library provides standard mathematical functions. They are
written as generic functions to facilitate use in other libraries or
abstractions. In addition, non-generic `@fp16()` and `f32` functions are
provided. These functions have names suffixed with `_f16` and `_f32`,
respectively.

The following functions are provided:

``` csl
// T can be f16 or f32
fn POSITIVE_INF(comptime T: type) : T
fn NEGATIVE_INF(comptime T: type) : T
fn NaN(comptime T: type) : T

// x can be f16, cb16, bf16, f32, i8, i16, i32, i64,
// u8, u16, u32, or u64
fn abs(x: anytype) @type_of(x)
fn max(x: anytype, y: @type_of(x)) @type_of(x)
fn min(x: anytype, y: @type_of(x)) @type_of(x)
fn sign(x: anytype) @type_of(x)

// x can be f16, cb16, bf16, or f32
fn ceil(x: anytype) @type_of(x)
fn floor(x: anytype) @type_of(x)
fn fscale(f: anytype, s: i16) @type_of(f)
fn isNaN(x: anytype) bool
fn isInf(x: anytype) bool
fn isFinite(x: anytype) bool
fn isSignaling(x: anytype) bool
fn sig(x: anytype) @type_of(x)
fn signbit(x: anytype) bool
fn tanh(x: anytype) @type_of(x)

// x can be @fp16() or f32; function can only be evaluated at runtime
fn cos(x: anytype) @type_of(x)
fn exp(x: anytype) @type_of(x)
fn inv(x: anytype) @type_of(x)
fn invsqrt(x: anytype) @type_of(x)
fn log(x: anytype) @type_of(x)
fn pow(x: anytype, y: @type_of(x)) @type_of(x)
fn sin(x: anytype) @type_of(x)
fn sqrt(x: anytype) @type_of(x)
```

Corresponding non-generic functions are:

``` csl
const POSITIVE_INF_f16 : @fp16()
const POSITIVE_INF_f32 : f32

const NEGATIVE_INF_f32 : f32
const NEGATIVE_INF_f16 : @fp16()

const NaN_f16 : @fp16()
const NaN_f32 : f32

fn abs_f16(x: @fp16()) @fp16()
fn abs_f32(x: f32) f32

fn max_f16(x: @fp16(), y: @fp16()) @fp16()
fn max_f32(x: f32, y: f32) f32

fn min_f16(x: @fp16(), y: @fp16()) @fp16()
fn min_f32(x: f32, y: f32) f32

fn sign_f16(x: @fp16()) @fp16()
fn sign_f32(x: f32) f32

fn ceil_f16(x: @fp16()) @fp16()
fn ceil_f32(x: f32) f32

fn floor_f16(x: @fp16()) @fp16()
fn floor_f32(x: f32) f32

fn fscale_f16(f: @fp16(), s: i16) @fp16()
fn fscale_f32(f: f32, s: i16) f32

fn isNaN_f16(x: @fp16()) bool
fn isNaN_f32(x: f32) bool

fn isInf_f16(x: @fp16()) bool
fn isInf_f32(x: f32) bool

fn isFinite_f16(x: @fp16()) bool
fn isFinite_f32(x: f32) bool

fn isSignaling_f16(x: @fp16()) bool
fn isSignaling_f32(x: f32) bool

fn sig_f16(x: @fp16()) @fp16()
fn sig_f32(x: f32) f32

fn signbit_f16(x: @fp16()) bool
fn signbit_f32(x: f32) bool

fn tanh_f16(x: @fp16()) @fp16()
fn tanh_f32(x: f32) f32

// The following functions can only be evaluated at runtime
fn cos_f16(x: @fp16()) @fp16()
fn cos_f32(x: f32) f32

fn exp_f16(x: @fp16()) @fp16()
fn exp_f32(x: f32) f32

fn inv_f16(x: @fp16()) @fp16()
fn inv_f32(x: f32) f32

fn invsqrt_f16(x: @fp16()) @fp16()
fn invsqrt_f32(x: f32) f32

fn log_f16(x: @fp16()) @fp16()
fn log_f32(x: f32) f32

fn pow_f16(x: @fp16(), y: @fp16()) @fp16()
fn pow_f32(x: f32, y: f32) f32

fn sin_f16(x: @fp16()) @fp16()
fn sin_f32(x: f32) f32

fn sqrt_f16(x: @fp16()) @fp16()
fn sqrt_f32(x: f32) f32
```

### Example

``` csl
const math = @import_module("<math>");

var x: f16;

task t() void {
  if (!math.isFinite(x)) {
    x = 0.0;
  }
  var one = math.pow(math.sin(x), 2.0) + math.pow(math.cos(x), 2.0);
  if (math.abs(math.log(one) - 1.0) > 0.001) {
    x = math.NaN(f16);
  }
}
```

The same code can be written using non-generic functions:

``` csl
task t() void {
  if (!math.isFinite_f16(x)) {
    x = 0.0;
  }
  var one = math.pow_f16(math.sin_f16(x), 2.0) +
    math.pow_f16(math.cos_f16(x), 2.0);
  if (math.abs_f16(math.log_f16(one) - 1.0) > 0.001) {
    x = math.NaN_f16;
  }
}
```

### Note on `sin` and `cos` accuracy

Both `f16` and `f32` versions of `sin` and `cos` will produce incorrect
results when abs(x) ≥ 16384π (approximately 51472).

## \<random\> {#language-libraries-random}

The `random` library provides utility functions that wrap the
`@random16` builtin to create random values across various ranges and
distributions.

See `language-builtins-random16`{.interpreted-text role="ref"} for
information on the PRNG used by these functions.

``` csl
// sets the global state of the PRNG number `prng_id` to `seed`
fn set_global_prng_seed(seed: u32) void

// generate a random 16-bit number in the range [lower, upper)
fn random_f16(lower: @fp16(), upper: @fp16()) @fp16()

// generate a random 32-bit number in the ragne [lower, upper)
fn random_f32(lower: f32, upper: f32) f32

// generate a uniform random number in the range [0, 2^pow)
fn random_pow_u32(pow : u16) u32

// generate a normally distributed number using the Box-Muller transform
fn random_normal_f32() f32
```

## \<simprint\> {#language-libraries-simprint}

The `simprint` library contains functions to print strings and various
numeric data types to the simulator logs. This is intended primarily for
debugging, as the printed output is not visible when running on
hardware.

Messages produced by the `simprint` library are stored by the simulator
in fixed-size buffers, with one buffer per PE. A buffer will be flushed,
with its contents printed to the simulator logs, when the buffer is full
*or* a `"\n"` newline character is produced. Any data remaining in a
PE\'s print buffer at the end of simulator execution will be silently
discarded.

### Basic printing functions

``` csl
// Prints a comptime string `s` to the simulator logs.
//
// Note that if the string contains a zero (NUL) byte, the output will be
// truncated at the NUL.
fn print_string(comptime s: comptime_string) void;

// Prints an unsigned 16-bit integer `x` in binary. The output will be 16
// characters wide, with zero-padding inserted on the left if needed.
fn print_u16_binary(x: u16) void;

// Prints an unsigned 16-bit integer `x` in decimal.
fn print_u16_decimal(x: u16) void;

// Prints an unsigned 16-bit integer `x` in hex. The output will be 4
// characters wide, with zero-padding inserted on the left if needed.
fn print_u16_hex(x: u16) void;

// Prints a 16-bit floating-point value `x` in decimal.
fn print_f16(x: @fp16()) void;

// Prints an unsigned 32-bit integer `x` in binary. The output will be 32
// characters wide, with zero-padding inserted on the left if needed.
fn print_u32_binary(x: u32) void;

// Prints an unsigned 32-bit integer `x` in decimal.
fn print_u32_decimal(x: u32) void;

// Prints an unsigned 32-bit integer `x` in hex. The output will be 8
// characters wide, with zero-padding inserted on the left if needed.
fn print_u32_hex(x: u32) void;

// Prints a 32-bit floating-point value `x` in decimal.
fn print_f32(x: f32) void;
```

For example:

``` csl
// Assume the print buffer is empty to start with.

// "42" will _not_ immediately be displayed in the simulator logs by the
// following statement.
simprint.print_u16_decimal(42);

// The following statement will force the buffer to flush, so the "42"
// will be visible in the logs.
simprint.print_string("\n");
```

### Format strings

Two functions are provided to print formatted strings:

``` csl
// Prints a formatted string to the simulator logs. The format string is
// a compile-time string, and the arguments are the values to be inserted
// into the format string. A newline is automatically printed after the
// formatted string.
fn fmt(comptime fstr: comptime_string, args: anytype) void;

// As above, but does not print a newline after the formatted string.
// Note that output is not flushed to the simulator logs until a newline
// is encountered or an internal buffer fills up, so it is recommended to
// follow up with something that will print a newline.
fn fmt_no_nl(comptime fstr: comptime_string, args: anytype) void;

// Like 'fmt', but prepends the coordinates of the current PE to the
// output line, in the format "PE(X,Y): ".
fmt fmt_with_coords(comptime fstr: comptime_string, args: anytype) void;
```

Format specifiers are wrapped in curly braces, and correspond
positionaly to the arguments in `args`. Available format specifiers are:

- `{d}`: print the argument as a decimal number. Argument must have type
  `u16` or `u32`.
- `{X}`: print the argument as a hexadecimal number in upper case.
  Argument must have type `u16` or `u32`.
- `{b}`: print the argument as a binary number. Argument must have type
  `u16` or `u32`.
- `{f}`: print the argument as a floating-point number. Argument must
  have type `@fp16()` or `f32`.

A literal `{` character may be escaped by doubling it. For example,
`{{hello}` will print as `{hello}`.

:::: warning
::: title
Warning
:::

Code that uses `fmt` or `fmt_no_nl` is likely to exceed the compiler\'s
default limit on `inline` loop unrolling. If you encounter the error
`exceeded the maximum of 50 inlined iterations` when compiling, you can
add the flag `--max-inlined-iterations=1000000` to the compile command
to bypass this issue.
::::

For example:

``` csl
simprint.fmt(
  "{d} {X} {b} {f}",
  .{ @as(u16,42), @as(u16,42), @as(u16,42), @as(f16,42.0) }
);

// The above will print:
// 42 002A 0000000000101010 42.0

simprint.fmt_no_nl(
  "{d} {X}",
  .{ @as(u16,42), @as(u16,42) }
);
simprint.print_string(" ");
simprint.fmt_no_nl(
  "{b} {f}",
  .{ @as(u16,42), @as(f16,42.0) }
);
simprint.print_string("\n");

// The above will also print:
// 42 002A 0000000000101010 42.0
```

### Disabling output

Sometimes it is useful to disable all of the debug prints produced by a
particular instance of the `simprint` module, while keeping the option
to turn them back on later. This helps save on runtime and space
overhead, and can also be used to conditionally enable or disable debug
printing on certain PEs. Prints originating from a specific `simprint`
instance can be disabled by setting the `enable` parameter to `false` at
import time.

``` csl
const simprint = @import_module("<simprint>", .{ .enable = false });
```

The `enable` parameter is optional. Its default value is `true`, which
means that printing is enabled.

## \<tile_config\> {#language-libraries-tile-config}

The `tile_config` library contains APIs relating to the hardware
configuration of a PE. It contains the following top-level constants:

``` csl
// The base addresses of memory-mapped registers
const addresses: enum(reg_type)

// The type of a word-sized memory-mapped register
const reg_type: type

// The type of a memory-mapped register occupying two words
const double_reg_type: type

// The name of the target architecture, such as "wse2"
const target_name: comptime_string

// The size of a word in bytes
const word_size: comptime_int

// The minimum alignment for FIFO buffers in bytes
const FIFO_MIN_ALIGNMENT: comptime_int
```

The `tile_config` library also contains an API to access the PE\'s
coordinates in the rectangle at runtime.

``` csl
const fabric_coord: enum(reg_type) {
  X,
  Y
};
fn get_fabric_coord(dimension: fabric_coord) u16
```

### color_config {#language-libraries-tile-config-color-config}

This submodule of `tile_config` contains APIs and an enum type for
changing the configuration of a given color during a teardown phase.

First of all, the `color_config` submodule defines the following enum
type:

``` csl
const fabric_io = enum(u16) {
  TX_WEST,
  TX_EAST,
  TX_SOUTH,
  TX_NORTH,
  TX_RAMP,

  RX_WEST,
  RX_EAST,
  RX_SOUTH,
  RX_NORTH,
  RX_RAMP
};
```

This enum consists of all the input and output routing directions which
can be used to specify the routing direction we wish to modify.

Specifically, the `color_config` library consists of the following
functions:

``` csl
// Returns the word address of a color configuration that corresponds to
// color `c`.
fn get_color_config_addr(c: color) reg_type

// Enables `dir` direction for a color `c` or a word address `c`
// of a color configuration register.
fn set_io_direction(c: anytype, dir: fabric_io) void

// For a given color `c` or a word address `c` of a color configuration
// register, toggle the I/O direction `dir`.
fn toggle_io_direction(c: anytype, dir: fabric_io) void

// For a given color `c` or a word address `c` of a color configuration
// register, clear the setting for `dir`.
fn clear_io_direction(c: anytype, dir: fabric_io) void

// For a given color `c` or a word address `c` of a color configuration
// register, clear all I/O routes and reset them according to `new_routes`.
// The input routes are specified as the `rx` field of `new_routes` which
// must be an array of `direction` values or a single `direction` value.
// Similarly, the output routes are specified as the `tx` field of
// `new_routes` which must be an array of `direction` values or a single
// `direction` value. In `WSE3`, if `rx` is an array, it must have a
// single element (i.e., it must hold a value of type [1]direction).
fn reset_routes(c: anytype, comptime new_routes: comptime_struct) void
```

These functions can be used as follows:

``` csl
param red: color;
var blue: color;
const tile_config = @import_module("<tile_config>");
const color_config = tile_config.color_config;
const red_addr = color_config.get_color_config_addr(red);

task teardown() void {
  // Color `blue` is a `var` and therefore not known until runtime.
  const addr = color_config.get_color_config_addr(blue);

  // We can manipulate the I/O routing configuration for a
  // comptime-known color, a runtime color or an address to a
  // color configuration register.
  const offramp = color_config.fabric_io.TX_RAMP;
  const onramp = color_config.fabric_io.RX_RAMP;
  const rx_north = color_config.fabric_io.RX_NORTH;
  color_config.set_io_direction(red, offramp);
  color_config.set_io_direction(blue, onramp);
  color_config.set_io_direction(addr, rx_north);

  // Reset the routes for color `red` using multiple directions.
  color_config.reset_routes(red, .{.tx = [3]direction{NORTH, SOUTH, WEST},
                                   .rx = [2]direction{RAMP, EAST}
                                  });
  // Reset the routes for color `red` using single directions.
  color_config.reset_routes(red, .{.tx = NORTH, .rx = EAST});
}
```

### control_transform {#language-libraries-tile-config-control-transform}

This submodule of `tile_config` contains a function for setting the mask
for transforming the index part of control wavelets. This function is to
be used together with the DSD property `control_transform` to `XOR` the
first six bits of the index portion of a wavelet with the specified
mask.

``` csl
fn set_mask(mask: reg_type) void
```

This function can be used like:

``` csl
const tile_config = @import_module("<tile_config>");
const ctrl_xform = tile_config.control_transform;

var in_dsd = @get_dsd(fabin_dsd, .{ .fabric_color = recv_channel,
                                    .extent = 100,
                                    .input_queue = @get_input_queue(0),
                                    .control_transform = true });
const out_dsd = @get_dsd(fabout_dsd, .{ .extent = 100,
                                        .fabric_color = send_channel,
                                        .output_queue = @get_output_queue(1),
                                        .control_transform = true });

var buf = @zeros([5]u32);
const fifo = @allocate_fifo(buf);

task buffer() void {
  @mov32(fifo, in_dsd, .{ .async = true });
  @mov32(out_dsd, fifo, .{ .async = true });
}

comptime {
  ctrl_xform.set_mask(2);
}
```

The `set_mask` function can be used either at comptime or runtime. Only
the first six bits of the mask are taken into account.

### exceptions {#language-libraries-tile-config-exceptions}

This submodule of `tile_config` contains functions for setting values in
the exception mask register. The exception mask register determines
which exceptions cause the processor to stop. An unmasked exception
causes the processor to immediately stop execution. A masked exception
allows execution to continue. By default, all exceptions are masked. The
functions in this submodule can be used to unmask them.

``` csl
// Exceptions which can be unmasked with the below functions
const PERF_CNT_0_OVERFLOW = ...;
const PERF_CNT_1_OVERFLOW = ...;
const SW_EXCEPTION = ...;
const FP_UNDERFLOW = ...;
const FP_OVERFLOW = ...;
const FP_INEXACT = ...;
const FP_INVALID = ...;
const FP_DIV_BY_0 = ...;

// Unmask one of the exceptions above
fn set_exception_mask(exception_mask: reg_type) void;
```

This submodule can be used as follows:

``` csl
const tile_config = @import_module("<tile_config>");
const exceptions = tile_config.exceptions;

fn fp_div_by_0() f32 {
  // Set exception mask for FP_DIV_BY_0.
  // When floating point divide by zero occurs,
  // processor will stop execution.
  exceptions.set_exception_mask(exceptions.FP_DIV_BY_0);

  var x : f32 = 42.0;
  var y : f32 = 0.0;

  // This operation is a divide by zero, so processor should hang
  return x / y;
}
```

Each call to `set_exception_mask` overwrites the exception mask
register. Multiple exceptions can be unmasked simultaneously as follows:

``` csl
// FP_DIV_BY_0 is unmasked.
exceptions.set_exception_mask(exceptions.FP_DIV_BY_0);

// FP_DIV_BY_0 is masked again.
// FP_OVERFLOW and FP_UNDERFLOW are now unmasked.
exceptions.set_exception_mask(exceptions.FP_OVERFLOW
                            & exceptions.FP_UNDERFLOW);
```

### filters {#language-libraries-tile-config-filters}

This submodule of `tile_config` contains APIs for configuring filters:

``` csl
// The number of filters provided by the architecture.
const num_filters: comptime_int

// Set the active limit of a counter filter identified by `filter_id`
// to `limit`.
fn set_active_limit(filter_id: u16, limit: reg_type) void

// Set the maximum counter value of a counter filter identified by
// `filter_id` to `max_counter`.
fn set_max_counter(filter_id: u16, max_counter: reg_type) void

// Set the counter value of a counter filter identified by `filter_id`
// to `counter`.
fn set_counter(filter_id: u16, counter: reg_type) void

const counter_filter_state = enum(reg_type) {
  PASS,
  DROP
};
```

On wse3, the following APIs are also provided:

``` csl
// For the counter filter identified by `filter_id`, set its pass/drop state
// to `state` and counter value to `counter` half-wavelets.
fn set_counter_and_state(filter_id: u16, counter: double_reg_type,
                         state: counter_filter_state) void

// Set the counter value of the counter filter identified by `filter_id`
// to `counter` half-wavelets.
fn set_counter_value(filter_id: u16, counter: double_reg_type)

// Get the counter value of the counter filter identified by `filter_id` as
// a number of half-wavelets.
fn get_counter_value(filter_id: u16) double_reg_type

// Set the pass/drop state of the counter filter identified by `filter_id`
// to `state`.
fn set_state(filter_id: u16, state: counter_filter_state) void

// Get the pass/drop state of the counter filter identified by `filter_id`.
fn get_state(filter_id: u16) counter_filter_state

// Set the pass count of the counter filter identified by `filter_id` to
// `count` half-wavelets.
fn set_pass_count(filter_id: u16, count: double_reg_type) void

// Get the pass count of the counter filter identified by `filter_id` as a
// number of half-wavelets.
fn get_pass_count(filter_id: u16) double_reg_type

// Set the drop count of the counter filter identified by `filter_id` to
// `count` half-wavelets.
fn set_drop_count(filter_id: u16, count: double_reg_type) void

// Get the drop count of the counter filter identified by `filter_id` as a
// number of half-wavelets.
fn get_drop_count(filter_id: u16) double_reg_type
```

These functions can be used like:

``` csl
const config = @import_module("<tile_config>");
// Set the counter of filter ID 1 to 0
config.filters.set_counter(1, 0);
```

### input_queue_status {#language-libraries-tile-config-input-queue-status}

This submodule of `tile_config` contains APIs for inspecting input queue
status.

``` csl
// Type representing input queue status.
const status = struct {
  // Bit mask representing queue-empty status. The queue with index 'i' is
  // empty if the 'i'th bit is 1.
  empty: u8,
  // Bit mask representing queue-full status. The queue with index 'i' is
  // if the 'i'th bit is 1.
  full: u8,
};

// Reads and decodes the input queue status register, returning an object of
// type 'status' representing the input queue status.
fn get() status;

// Returns true if 's' indicates that 'q' is full.
fn is_full(s: status, q: input_queue) bool;

// Returns true if 's' indicates that 'q' is empty.
fn is_empty(s: status, q: input_queue) bool;

// Returns true if 's' indicates that all input queues are full.
fn all_full(s: status) bool;

// Returns true if 's' indicates that all input queues are empty.
fn all_empty(s: status) bool;

// Returns true if 's' indicates that no input queues are full.
fn none_full(s: status) bool;

// Returns true if 's' indicates that no input queues are empty.
fn none_empty(s: status) bool;

// Reads the input queue status register and returns its raw value.
fn get_raw() reg_type;
```

### main_thread_priority {#language-libraries-tile-config-main-thread-priority}

This submodule of `tile_config` contains APIs for configuring main
thread priority. The main thread is the thread that executes non-`async`
operations. Operations tagged with `async` execute on a microthread,
which is associated with a fabric input or output queue. Main thread
priority and microthread priority determine the relative scheduling
priority of the threads.

``` csl
// Enum for main thread priorities. The meanings of main thread priority
// levels are relative to microthread priorities, as follows:
//
//   MEDIUM_LOW: Between low- and medium-priority microthreads.
//   MEDIUM: Same priority as medium-priority microthreads.
//   MEDIUM_HIGH: Between medium- and high-priority microthreads.
//   HIGH: Same priority as high-priority microthreads.
const level = enum(u16) {
  MEDIUM_LOW = ...,
  MEDIUM = ...,
  MEDIUM_HIGH = ...,
  HIGH = ...
};

// Updates the priority for the main thread to `priority`. Note that updates
// to main thread priority made at runtime may take a few clock cycles to
// take effect. This function may be used at comptime or at runtime.
fn update_main_thread_priority(priority: level) void;
```

This function can be used like:

``` csl
const config = @import_module("<tile_config>");
const mt_priority = config.main_thread_priority;

comptime {
  mt_priority.update_main_thread_priority(mt_priority.level.MEDIUM);
}

task main() void {
  mt_priority.update_main_thread_priority(mt_priority.level.MEDIUM_HIGH);
}
```

### output_queue_status {#language-libraries-tile-config-output-queue-status}

This submodule of `tile_config` contains APIs for inspecting output
queue status.

``` csl
// Type representing output queue status.
const status = struct {
  // Bit mask representing queue-empty status. The queue with index 'i' is
  // empty if the 'i'th bit is 1.
  empty: u8,
  // Bit mask representing queue-full status. The queue with index 'i' is
  // full if the 'i'th bit is 1.
  full: u8,
};

// Reads and decodes the output queue status register, returning an object
// of type 'status' representing the output queue status.
fn get() status;

// Returns true if 's' indicates that 'q' is full.
fn is_full(s: status, q: output_queue) bool;

// Returns true if 's' indicates that 'q' is empty.
fn is_empty(s: status, q: output_queue) bool;

// Returns true if 's' indicates that all output queues are full.
fn all_full(s: status) bool;

// Returns true if 's' indicates that all output queues are empty.
fn all_empty(s: status) bool;

// Returns true if 's' indicates that no output queues are full.
fn none_full(s: status) bool;

// Returns true if 's' indicates that no output queues are empty.
fn none_empty(s: status) bool;

// Reads the output queue status register and returns its raw value.
fn get_raw() reg_type;
```

### switch_config {#language-libraries-tile-config-switch-config}

This submodule of `tile_config` contains APIs and enum types that can be
used to change the switch configuration of a given color during a
teardown phase.

First of all, the `switch_config` submodule defines the following enum
types:

``` csl
const pop_mode = enum(u16) {
  NO_POP,
  ALWAYS_POP,
  POP_ON_ADVANCE,

  CLEAR_MASK
};

const switch_status = enum(u16) {
  CLEAR_CURRENT_POS
};

const switch_pos = enum(u16) {
  POS1,
  POS2,
  POS3
};

const switch_select = enum(reg_type) {
  INPUT,
  OUTPUT
};
```

These enum types represent specific setting categories like pop mode and
switch status and they can be used to specify the settings that we want
to modify in a per-category manner.

In addition, the `switch_config` submodule consists of the following
functions:

``` csl
// Returns the word address of a given color `c` and switch setting type
// `setting_ty`.
fn get_switch_config_addr(c: color, comptime setting_ty: type) reg_type

// Returns the word address of a color configuration that corresponds to
// color `c`.
fn get_color_config_addr(c: color) reg_type

// For a given color `c` or a word address `c` of a color configuration
// register, clear its current switch position.
fn clear_current_position(c: anytype) void

// For a given color `c` or a word address `c` of a color configuration
// register, set the given ring mode to NO_RING_MODE or RING_MODE.
fn set_ring_mode(c: anytype, comptime mode: ring_mode) void

// For a given color `c` or a word address `c` of a color configuration
// register, set the given pop mode `mode` after clearing the previous one.
fn set_pop_mode(c: anytype, mode: pop_mode) void

// For a given color `c` or a word address `c` of a color configuration
// register, make all switch positions invalid.
fn set_invalid_for_all_switch_positions(c: anytype) void

// For a given color `c` or a word address `c` of a color configuration
// register, set the input or output direction of the specified switch
// position to `dir`.
fn set_switch_pos(pos: switch_pos, c: anytype, comptime dir: direction,
                  switch_kind: switch_select) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of the specified switch position to `dir_rx`,
// and TX direction of the specified switch position to `dir_tx`.
// NOTE: This function is supported on wse3 and beyond only.
fn set_rxtx_switch_pos(pos: switch_pos, c: anytype,
                       comptime dir_rx: direction,
                       comptime dir_tx: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of switch position 1 to `dir`.
fn set_rx_switch_pos1(c: anytype, comptime dir: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set TX direction of switch position 1 to `dir`.
fn set_tx_switch_pos1(c: anytype, comptime dir: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of switch position 1 to `dir_rx`, and
// TX direction of switch position 1 to `dir_tx`.
// NOTE: This function is supported on wse3 and beyond only.
fn set_rxtx_switch_pos1(c: anytype, comptime dir_rx: direction,
                        comptime dir_tx: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of switch position 2 to `dir`.
fn set_rx_switch_pos2(c: anytype, comptime dir: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set TX direction of switch position 2 to `dir`.
fn set_tx_switch_pos2(c: anytype, comptime dir: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of switch position 2 to `dir_rx`, and
// TX direction of switch position 2 to `dir_tx`.
// NOTE: This function is supported on wse3 and beyond only.
fn set_rxtx_switch_pos2(c: anytype, comptime dir_rx: direction,
                        comptime dir_tx: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of switch position 3 to `dir`.
fn set_rx_switch_pos3(c: anytype, comptime dir: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set TX direction of switch position 3 to `dir`.
fn set_tx_switch_pos3(c: anytype, comptime dir: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set RX direction of switch position 3 to `dir_rx`, and
// TX direction of switch position 3 to `dir_tx`.
// NOTE: This function is supported on wse3 and beyond only.
fn set_rxtx_switch_pos3(c: anytype, comptime dir_rx: direction,
                        comptime dir_tx: direction) void

// For a given color `c` or a word address `c` of a color configuration
// register, set the switch position to `posn`,
// where `posn` is 0, 1, 2, or 3.
fn set_current_switch_position(c: anytype, posn: u16) void
```

:::: warning
::: title
Warning
:::

On WSE-2, all colors support switch configuration. On WSE-3, only a
subset of colors support switch configuration: 0, 1, 2, 3, 4, 5, 6, 7,
8, 9, 12, 13, 16, 17, 20, and 21. `switch_config` APIs should only be
used with these colors on WSE-3.
::::

These functions can be used as follows:

``` csl
param red: color;
var blue: color;
const tile_config = @import_module("<tile_config>");
const switch_config = tile_config.switch_config;
const switch_status_addr =
      switch_config.get_switch_config_addr(red,
                                          switch_config.switch_status);

task teardown() void {
  // Color `blue` is a `var` and therefore not known until runtime.
  const addr = switch_config.get_switch_config_addr(blue,
                                                    switch_config.pop_mode);

  // We can manipulate the switch configuration using a
  // comptime-known color, a runtime color or an address to
  // a color configuration register.
  switch_config.clear_current_position(red);
  switch_config.clear_current_position(blue);
  switch_config.clear_current_position(switch_status_addr);

  // Reset the pop mode to a new setting.
  switch_config.set_pop_mode(red, switch_config.pop_mode.ALWAYS_POP);
}
```

### task_priority {#language-libraries-tile-config-task-priority}

This submodule of `tile_config` contains APIs for configuring task
priority:

``` csl
// Enum for task priorities: either HIGH or LOW.
const level = enum(u16) {
  LOW = 0,
  HIGH = 1
};

// Updates the task priority associated with `task_id` to `priority`.
fn update_task_priority(task_id: anytype, priority: level) void

// Sets the task priority associated with `task_id` to high.
fn set_task_priority(task_id: anytype) void

// Sets the task priority associated with `task_id` to low.
fn clear_task_priority(task_id: anytype) void
```

The provided `task_id` can be a `data_task_id` or `local_task_id` to set
the priority of the associated task.

In addition, the priority of tasks activated by wavelets, including
tasks bound to a `control_task_id`, can be specified using the `color`
on WSE-2, or the `input_queue` on WSE-3, that carries the wavelets.

Note that updates to task priority made at runtime may take a few clock
cycles to take effect. These functions may be used at comptime or at
runtime.

These functions can be used like:

``` csl
const config = @import_module("<tile_config>");
const task_priority = config.task_priority;
const task_priority_level = task_priority.level;

param high_id: data_task_id;
param low_id: local_task_id;

comptime {
  // Equivalent to:
  //   task_priority.update_task_priority(
  //     high_id,
  //     task_priority_level.HIGH);
  task_priority.set_task_priority(high_id);
}

task main() void {
  // Equivalent to:
  //   task_priority.update_task_priority(
  //     low_id,
  //     task_priority_level.LOW);
  task_priority.clear_task_priority(low_id);
}
```

### teardown {#language-libraries-tile-config-teardown}

This submodule of `tile_config` contains teardown APIs:

``` csl
// Returns the task ID that is reserved for the teardown handler.
fn get_task_id() local_task_id

// Return the values of the "teardown-pending" registers combined into
// one value. Only the first invocation of this function per-task is
// guaranteed to return the correct value. Any additional calls per-task
// will have undefined results.
fn get_pending() double_reg_type

// Given a value that represents the "teardown-pending" state, which has 1
// bit per routable color indicating the ones that are currently in
// teardown state, return `true` iff the input color `c` is in teardown
// state.
fn is_pending(value: double_reg_type, c: color) bool

// Exit the teardown state for a given color `c`.
fn exit(c: color) void
```

These functions can be used like:

``` csl
const config = @import_module("<tile_config>");
// Check if teardown is pending on color 8 or 9
var pendings = config.teardown.get_pending();
bool pending_8_or_9 = config.teardown.is_pending(pendings, @get_color(8)) or
    config.teardown.is_pending(pendings, @get_color(9));
```

## \<time\> {#language-libraries-time}

The time library returns the current 48-bit timestamp counter as three
16-bit unsigned integers in little endian form.

``` csl
// enable tsc registers for capturing timestamps
fn enable_tsc() void;

// disable tsc registers for capturing timestamps
fn disable_tsc() void;

// write timestamp to array of three u16 values
fn get_timestamp(result : *[3]u16) void;

// reset tsc register to 0
fn reset_tsc_counter() void;
```

Addionally, it is also possible to collect HW performance counters. The
hardware can store two performance counters. They are also enabled /
disabled with [enable_tsc]{.title-ref} and [disable_tsc]{.title-ref}.
These counters cannot be reset.

``` csl
// write the value of the HW performance counter with the given id (0 or 1)
// to array of three u16 values
fn get_perf_cntr(result : *[3]u16, id : i16) void;
```

## \<timer\> {#language-libraries-timer}

The timer library provides some additional utilities for managing
multiple timers in a program and calculating elapsed time.

``` csl
// library parameter specifying number of available timers
param timerCount;

// convenience type for [3]u16, used to hold single timestamp
// or elapsed time
const timerType: type = [3]u16;

// start timer with ID timerId
fn start(timerId: i16) void;

// stop timer with ID timerId
fn stop(timerId: i16) void;

// calculate elapsed time for timer with ID timerId
// the elapsed time is stored in result in an internal format
fn elapsed(timerId: i16, result: *timerType) void;
```

## \<kernels\> {#language-libraries-kernels}

This library differs from all other libraries in that it provides
kernels, as opposed to individual functions. The tally kernel implements
a two-phase tally, used to coordinate the work done by multiple PEs. The
fft kernel library implements a 3D FFT.

### \<fft\> {#language-libraries-kernels-fft}

The FFT library implements a 3D FFT across a rectangle of PEs.

The library consists of several modules:

1.  `<kernels/fft/fft3d_layout>`: Provides a full implementation of the
    3D FFT, including host exported functions for launching FFT and iFFT
    computations. Imported once in a program\'s `layout` file.
2.  `<kernels/fft/fft3d>`: Underlying implementation of 3D FFT. If using
    the `fft3d_layout` module, then this module is not necessary to
    import. Using this module requires the user to manually construct
    the layout and host exported functions.
3.  `<kernels/fft/get_params>`: Imported once in a program\'s `layout`
    file to provide correct FFT parameters for the `fft3d` module. If
    using the `fft3d_layout` module, then this module is not necessary
    to import.

A minimal example of using the `fft3d_layout` module in a program is as
follows:

``` csl
param N: u16;                   // FFT size in each dimension
param NUM_PENCILS_PER_DIM: u16; // Pencils in each dimension per PE
param WIDTH: i16 = N / NUM_PENCILS_PER_DIM; // # PEs in each dimension

const memcpy = @import_module("<memcpy/get_params>", .{
    .width = WIDTH,
    .height = WIDTH,
});

const fft_helper = @import_module("<kernels/fft/fft3d_layout>", .{
    .width = WIDTH,
    .memcpy = memcpy,
});

layout {
  @set_rectangle(WIDTH, WIDTH);
  fft_helper.FFT_kernel(0, 0, WIDTH, N, f32);
}
```

See the 3D FFT example program for a complete usage demonstration of the
`fft3d_layout` module.

### \<tally\> {#language-libraries-kernels-tally}

The tally library implements a two-phase tally kernel that allows PEs
within a rectangle to communicate progress/completion to the host.

The library consists of two modules:

1.  `<kernels/tally/layout>`: imported once and use in the `layout`
    block to parameterize each PE\'s tally behavior.
2.  `<kernels/tally/pe>`: imported once by each PE, consuming the
    parameters generated by the layout module.

A minimal example of importing and using both modules, starting with the
layout module:

``` csl
// code.csl

const tally = @import_module("<kernels/tally/layout>", .{
  .kernel_height=8,
  .kernel_width=4,
  .phase2_tally=0,
  .colors=[3]color{@get_color(1), @get_color(2), @get_color(3)},
  .output_color=@get_color(0),
});

layout {
  @set_rectangle(4, 8);

  for (@range(u16, 4)) |i| {
    for (@range(u16, 8)) |j| {
      @set_tile_code(i, j, "pe.csl", .{
        .tally_params = tally.get_params(i, j),
      });
    }
  }
}
```

And the per-PE module:

``` csl
// pe.csl

param tally_params: comptime_struct;

// On WSE-2, input_queues and output_queues can be the same.
// On WSE-3, they must be different.
const tally = @import_module("<kernels/tally/pe>",
  @concat_structs(tally_params, .{
    .input_queues=[2]u16{0, 1},
    .output_queues=[2]u16{0, 1},
  }));

task done() void {
  tally.signal_completion();
}

...
```

The tally kernel operates in two phases.

In the first phase, every PE must signal completion at least once. For
kernels where each PE knows when it is finished, this is the only phase
needed.

The first phase ends when every PE has signaled completion at least
once. During the second phase, PEs can bump (increase) the global tally.
When the global tally meets or exceeds the `phase2_tally` parameter, the
kernel signals completion by sending the total to the North on
`output_color` from the PE at (kernel_width - 1, 0).

The second phase is optional. If `phase2_tally == 0`, the second phase
will be skipped and the output signal on `output_color` will be 0.

## \<collectives_2d\> {#language-libraries-kernels-collectives-2d}

This library implements collective communication directives that allows
PEs to communicate data with one another.

The library consists of two modules:

1.  `<collectives_2d/params>`: Imported once to parameterize each PE in
    the `layout` block.
2.  `<collectives_2d/pe>`: Imported once per dimension per PE. Contains
    collective communication directives for a single axis.

### \<collectives_2d/params\>

The parameter module exposes a compile-time helper function for
configuring PEs to use `<collectives_2d>`

``` csl
fn get_params(Px: u16, Py: u16, ids: comptime_struct) comptime_struct
```

- `Px` is the PE\'s x-coordinate.
- `Py` is the PE\'s y-coordinate.
- `ids` is a struct that is expected to have either the `x`-related
  fields, the `y`-related fields, or all four, of the following:
  - `x_colors`: a struct containing 2 distinct colors as anonymous
    fields
  - `x_entrypoints`: a struct containing 2 distinct local task IDs as
    anonymous fields
  - `y_colors`: a struct containing 2 distinct colors as anonymous
    fields
  - `y_entrypoints`: a struct containing 2 distinct local task IDs as
    anonymous fields
- Returns a struct containing the parameters necessary to import library
  modules for the specified PE. This struct contains:
  - `x`: an opaque struct containing parameters needed to configure
    collective communications in the x-dimension.
  - `y`: an opaque struct containing parameters needed to configure
    collective communications in the y-dimension.

### \<collectives_2d/pe\>

The following directives are currently supported:

``` csl
fn init() void
fn broadcast(root: u16, buf: [*]u32, count: u16, callback: local_task_id) void
fn scatter(root: u16, send_buf: [*]u32, recv_buf: [*]u32, count: u16,
           callback: local_task_id) void
fn gather(root: u16, send_buf: [*]u32, recv_buf: [*]u32, count: u16,
          callback: local_task_id) void
fn reduce_fadds(root: u16, send_buf: [*]f32, recv_buf: [*]f32, count: u16,
                callback: local_task_id) void
```

`init` initializes the library. It must be invoked for each axis.

`broadcast` transmits the contents of `buf` from the root PE to the
`buf` of other PEs in the row or column. `count` should be the length of
`buf`. It is akin to `MPI_Bcast`.

`scatter` transmits `count`-many elements from `send_buf` from the root
PE to the `recv_buf` of other PEs in the row/column. It is akin to
`MPI_Scatter`.

`gather` accumulates `count`-many elements from `send_buf` of other PEs
into the `recv_buf` of the root PE. It is akin to `MPI_Gather`.

When distributing or aggregating elements using `scatter` or `gather`
for `N` PEs, the `send_buf` or `recv_buf` should have space for
`count * N` elements, respectively.

`reduce_fadds` computes an `MPI_Sum` for buffers of `f32`.

In general, all PEs must call the same directive with same `root` and
`count`. The primitives have the following common parameters:

- `root` is the root PE for network configuration,
- `send_buf` is a buffer containing data to be transmitted,
- `recv_buf` is a buffer for holding data received,
- `count` is the number of elements to be transmitted,
- `callback` is activated when the primitive completes.

The user can configure the resources of `collectives_2d`. Each imported
module must be assigned queue IDs (`queues`) and DSR IDs
(`dest_dsr_ids`, `src0_dsr_ids`, `src1_dsr_ids`). If the user does not
specify these parameters explicitly, the default values apply. The
following example shows the default values of queue IDs and DSR IDs of
`collectives_2d`.

A minimal example that sets up PEs to broadcast 10 elements from the
root PE to every other PE in the row/column consists of the following
layout code:

``` csl
// code.csl

param width: u16;
param height: u16;
param root: u16;

const c2d = @import_module("<collectives_2d/params>");

layout {
  @set_rectangle(width, height);

  var x: u16 = 0;
  while (x < width) : (x += 1) {
    var y: u16 = 0;
    while (y < height) : (y += 1) {
      const c2d_params = c2d.get_params(x, y, .{
        .x_colors = .{
          @get_color(0),
          @get_color(1)
        },
        .x_entrypoints = .{
          @get_local_task_id(2),
          @get_local_task_id(3)
        },
        .y_colors = .{
          @get_color(4),
          @get_color(5)
        },
        .y_entrypoints = .{
          @get_local_task_id(6),
          @get_local_task_id(7)
        },
      });
      @set_tile_code(
        x,
        y,
        "pe.csl",
        .{ .root = root, .c2d_params = c2d_params }
      );
    }
  }
}
```

And the per-PE module:

``` csl
// pe.csl

param c2d_params: comptime_struct;

const rect_height = @get_rectangle().height;
const rect_width = @get_rectangle().width;

// Pick two task IDs not used in the library for callbacks
const x_task_id = @get_local_task_id(15);
const y_task_id = @get_local_task_id(16);

const len = 10;
var x_data = @zeros([len]u32);
var y_data = @zeros([len]u32);

const mpi_x = @import_module(
  "<collectives_2d/pe>",
  .{ .dim_params = c2d_params.x,
     .queues = [2]u16{2,4},
     .dest_dsr_ids = [1]u16{1},
     .src0_dsr_ids = [1]u16{1},
     .src1_dsr_ids = [1]u16{1}
   }
);

const mpi_y = @import_module(
  "<collectives_2d/pe>",
  .{ .dim_params = c2d_params.y,
     .queues = [2]u16{3,5},
     .dest_dsr_ids = [1]u16{2},
     .src0_dsr_ids = [1]u16{2},
     .src1_dsr_ids = [1]u16{2}
   }
);

task x_task() void {
  var send_buf = @ptrcast([*]u32, &x_data);
  var recv_buf = @ptrcast([*]u32, &@zeros[len]u32);

  if (root == mpi_x.pe_id) {
    mpi_x.broadcast(root, send_buf, len, x_task_id);
  } else {
    mpi_x.broadcast(root, recv_buf, len, x_task_id);
  }
}

task y_task() void {
  var send_buf = @ptrcast([*]u32, &y_data);
  var recv_buf = @ptrcast([*]u32, &@zeros[len]u32);

  if (root == mpi_y.pe_id) {
    mpi_y.broadcast(root, send_buf, len, y_task_id);
  } else {
    mpi_y.broadcast(root, recv_buf, len, y_task_id);
  }
}
```
