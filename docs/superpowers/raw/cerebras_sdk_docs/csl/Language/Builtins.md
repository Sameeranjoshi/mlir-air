*(Original URL: https://sdk.cerebras.net/csl/Language/Builtins.html)*

---

# Builtins {#language-builtins}

This section documents the builtins available in CSL. Builtins related
to remote procedure calls (RPC) are documented in
`language-builtins-rpc`{.interpreted-text role="ref"}, and builtins for
operation on DSDs are documented in
`language-builtins-for-dsd-operations`{.interpreted-text role="ref"}.

## \@activate {#language-builtins-activate}

Set the status of a local task to *Active*, allowing it to be picked by
the task picker if it is also unblocked.

### Syntax

``` csl
@activate(id);
```

Where:

- `id` is an expression of type `local_task_id` that is bound to a local
  task.

### Example

``` csl
const task_id: local_task_id = @get_local_task_id(10);
comptime {
  @bind_local_task(my_task, task_id);
}

fn foo() void {
  // make my_task eligible to be picked by task picker
  @activate(task_id);
}
```

## \@allocate_fifo

Create a FIFO DSD value. See `language-dsds`{.interpreted-text
role="ref"} for details. See `language-dsrs`{.interpreted-text
role="ref"} for details about use with DSRs.

## \@as

Coerce an input value from one numeric or boolean type to another.

### Syntax

``` csl
@as(result_type, value);
```

Where:

- `result_type` is a numeric (i.e. boolean, integer, or float) type.
- `value` is numeric value.

### Example

``` csl
// Convert the integer literal 10 into the 16-bit float value 10.0.
@as(f16, 10);

// Convert the float literal 10.2 into the 16-bit integer value 10.
@as(i16, 10.2);
```

### Semantics

Float-to-integer type coercion rounds the value towards zero. For
example:

- `@as(i16, 11.2) == 11`
- `@as(i16, 10.8) == 10`
- `@as(i16, -10.8) == -10`

Float-to-bool and integer-to-bool type coercions are equivalent to
(unordered, not-equals) comparisons with zero. Thus:

- `@as(bool, 0) == false`
- `@as(bool, -0.0) == false`
- `@as(bool, -5) == true`
- `@as(bool, nan) == true`

## \@assert

Asserts that a condition is true.

### Syntax

``` csl
@assert(cond);
```

Where:

- `cond` is an expression of type `bool`.

### Example

``` csl
task t(wavelet : i16) void {
  @assert(wavelet > 10);
}
```

### Semantics

Causes program execution to abort if the assert condition is false.
Note: aborting will only happen in simulation, hardware executions of
the program will ignore the assert.

If the assert expression is encountered in a `comptime` context, the
builtin is equivalent to `@comptime_assert`.

## \@bitcast

Reinterpret the raw bits of the input value as a value of another type.

### Syntax

``` csl
@bitcast(result_type, value);
```

Where:

- `result_type` is a pointer or numeric (i.e. boolean, integer, or
  float) type.
- `value` is numeric value. Must be an integer if `result_type` is a
  pointer.
- the bit width of `value` matches the bit width of values of type
  `result_type`.

### Example

``` csl
// Convert the IEEE-754 binary16 value 1.0 into its hex representation
// 0x3c00.
const one: f16 = 1.0;
@bitcast(u16, one);

// Produce a 16-bit NaN value.
const all_ones: u16 = 0xffff;
@bitcast(f16, all_ones);
```

## \@bind_control_task {#language-builtins-bind-control-task}

Bind a task to a `control_task_id`, so that each time a control wavelet
containing this ID in its payload is received, the task is activated and
can be scheduled for activation.

### Syntax

``` csl
@bind_control_task(this_control_task, this_control_task_id);
```

Where:

- `this_control_task` is the name of a task.
- `this_control_task_id` is an identifier of type `control_task_id`.

### Example

``` csl
const my_task_id: control_task_id = @get_control_task_id(35);
task my_task() void {}

comptime {
  @bind_control_task(my_task, my_task_id);
}
```

### Semantics

The `@bind_control_task` builtin must appear in a top-level `comptime`
block.

## \@bind_data_task {#language-builtins-bind-data-task}

Bind a task to a `data_task_id`, so that each time a wavelet is received
along the routable color underlying `data_task_id`, the task is
activated and can be scheduled for execution.

### Syntax

``` csl
@bind_data_task(this_data_task, this_data_task_id);
```

Where:

- `this_data_task` is the name of a task.
- `this_data_task_id` is an identifier of type `data_task_id`.

### Example

``` csl
// On WSE-2, data_task_ids are created from routable colors
const my_task_id: data_task_id = @get_data_task_id(@get_color(0));

// A data task takes payload of wavelet as an argument
task my_task(data: f32) void {}

comptime {
  @bind_data_task(my_task, my_task_id);
}
```

### Semantics

The `@bind_data_task` builtin must appear in a top-level `comptime`
block. Tasks passed into this builtin must take at least one argument.

## \@bind_local_task {#language-builtins-bind-local-task}

Bind a task to a `local_task_id`, so that each time that `local_task_id`
is unblocked and activated, the task is activated and can be scheduled
for execution.

### Syntax

``` csl
@bind_local_task(this_local_task, this_local_task_id);
```

Where:

- `this_local_task` is the name of a task.
- `this_local_task_id` is an identifier of type `local_task_id`.

### Example

``` csl
const my_task_id: local_task_id = @get_local_task_id(10);
task my_task() void {}

comptime {
  @bind_local_task(my_task, my_task_id);
}
```

### Semantics

The `@bind_local_task` builtin must appear in a top-level `comptime`
block. Tasks passed into this builtin cannot take any arguments.

## \@block {#language-builtins-block}

Block the task associated with the input `color`, `data_task_id`, or
`local_task_id` so that the task is prevented from running when the task
identifier is activated.

For `color` and `data_task_id` inputs, `@block` prevents incoming
wavelets on the associated color from activating tasks. This also
applies to control wavelets carried by a color, preventing a control
task bound to the ID in a control wavelet\'s payload from activating.

### Syntax

``` csl
@block(id);
```

Where:

- `id` is an expression of type
  - WSE-2: `color`, `data_task_id`, or `local_task_id`.
  - WSE-3: `input_queue`, `data_task_id`, `local_task_id`, or `ut_id`.

### Example

``` csl
// WSE-2 example
const task_id: local_task_id = @get_local_task_id(10);
comptime {
  @bind_local_task(my_task, task_id);
}

fn foo() void {
  // Prevent my_task from running whenever my_task is activated
  @block(task_id);
}
```

## \@comptime_assert

Assert a compile-time condition to be true; abort compilation if
otherwise.

### Syntax

``` csl
@comptime_assert(cond);
@comptime_assert(cond, message);
```

Where:

- `cond` is a `comptime` expression of type `bool` and
- `message` is an expression of type `comptime_string`.

### Example

``` csl
param size: u16;

fn foo() void {
  @comptime_assert(size > 0 and size < 16);
  @comptime_assert(size > 0 and size < 16,
                   "size should be between 0 and 16");
}
```

## \@comptime_print

Prints values at compile-time whenever the compiler evaluates this
statement.

### Syntax

``` csl
@comptime_print(val1, val2, ...);
```

Where:

- all arguments are `comptime` expressions.

This builtin is overloaded for an arbitrary number of arguments.

### Example

``` csl
fn foo() void {
  const my_struct = .{.x = 0, .dir = NORTH};
  @comptime_print(my_struct); // prints the contents of `my_struct`.

  if (false) {
    @comptime_print("hello"); // not printed.
  }

  for ([2]i16 {1, 2}) |val| {
    @comptime_print("hello world"); // printed once.
    @comptime_print(val); // error: val is not comptime.
  }

  comptime {
    for ([2]i16 {1, 2}) |val| {
      @comptime_print(val); // all values are printed.
    }
  };
}
```

### Semantics

A `comptime_print` statement causes the compiler to print information
for its arguments whenever the compiler evaluates such builtin.

During `comptime` evaluation, the builtin is evaluated whenever
control-flow reaches that line.

Whenever the compiler is analysing reachable non-`comptime` code, the
builtin is evaluated exactly once. For instance, a `@comptime_print`
builtin inside a non-`comptime` loop causes the compiler to evaluate it
exactly once.

## \@concat_structs

Concatenate two anonymous structs into one.

### Syntax

``` csl
@concat_structs(this_struct, another_struct);
```

Where:

- `this_struct` and `another_struct` are `comptime` expressions of
  anonymous struct type.

### Example

``` csl
const x = .{ .foo = 10 };
const y = .{ .bar = -1 };

// Produce an anonymous struct z, with the value .{ .foo = 10, .bar = -1 }.
const z = @concat_structs(x, y);
```

Attempting to concatenate a struct with named fields and a struct with
nameless fields (e.g. `.{1, 2}`) results in an error.

Attempting to concatenate two structs with overlapping named fields also
results in an error.

## \@constants

Initialize a tensor with a value.

### Syntax

``` csl
@constants(tensor_type, value);
```

Where:

- `tensor_type` is a `comptime` tensor type.
- The type of `value` is the same as the base type of `tensor_type`.

### Example

``` csl
// Initialize a tensor of four rows and five columns with the same value 10.
const matrix = @constants([4,5]i16, 10);
```

## \@dimensions

Returns a 1D array in which the i\'th element equals the size of the
i\'th dimension of the input array type. The length of the returned
array equals the rank of the input array type. The type of each element
in the returned array is `u32`.

### Syntax

``` csl
@dimensions(array_type);
```

Where:

- `array_type` is a `type` defining an array.

### Example

``` csl
const array_type_3d = [3, 5, 7]f16;
const dims = @dimensions(array_type_3d);
// dims is a 1D array of length 3 with values [3, 5, 7]
```

## \@element_count

Returns the total number of elements in the input array type as an
`u32`.

### Syntax

``` csl
@element_count(array_type);
```

Where:

- `array_type` is a `type` defining an array.

### Example

``` csl
const array_type_3d = [3, 5, 7]f16;
const num_elements = @element_count(array_type_3d);
// num_elements == 105
```

## \@element_type

Returns the element type of the input array type as a `type`.

### Syntax

``` csl
@element_type(array_type);
```

Where:

- `array_type` is a `type` defining an array.

### Example

``` csl
const array_type_3d = [3, 5, 7]f16;
const elem_type = @element_type(array_type_3d);
// elem_type == f16
```

## \@field

Access the value of a given struct field.

### Syntax

``` csl
@field(some_struct, field_name);
```

where:

- `some_struct` is a value of a struct type.
- `field_name` is a string.

### Example

``` csl
var my_struct = .{.a = 10};

// returns the value of field 'a'
const a = @field(my_struct, "a");

// The 'a' field of 'my_struct' will be assigned the value '20'
@field(my_struct, "a") = 20;
```

### Semantics

The builtin returns the value stored in the `field_name` field of
`some_struct` if and only if such field exists.

A call to `@field` can also be used as the left-hand side of an
assignment as shown in the example. In this scenario, the underlying
field of `some_struct` named `field_name` will be updated.

## \@fp16 {#language-builtins-fp16}

Returns the selected runtime FP16 format.

### Syntax

``` csl
@fp16();
```

### Semantics

The builtin returns a `type` representing the runtime FP16 format
specified by the `--fp16-format` command line option: `f16`, `cb16`, or
`bf16`.

### Example

``` csl
if (@is_same_type(@fp16(), cb16)) {
   @comptime_print("compiling with --fp16-format=cb16");
}

// A simple function defined in terms of the selected runtime FP16 format.
// For example, if the code is compiled with --fp16-format=bf16, square will
// have type 'fn(bf16) bf16'
fn square(x: @fp16()) @fp16() {
   return x * x;
}
```

## \@get_array

Convert a string to an array of bytes.

### Syntax

``` csl
@get_array(string);
```

Where:

- `string` is an expression of type `comptime_string`.

### Semantics

Given a value `s` of type `comptime_string`, `@get_array` returns an
array of type `[@strlen(s)]u8`. This array contains the bytes inside the
string.

Note that:

- Strings in CSL are *not* null-terminated, so the length of the array
  returned by `@get_array(s)` is `@strlen(s)`, not `@strlen(s)+1`. If a
  null-terminated array is required, this can be constructed by
  concatenating the string `"\x00"` onto the end of the string before
  passing it to `@get_array`.
- Strings in CSL are strings of *bytes*, not of characters. String
  literals are interpreted as UTF-8, so if a string contains non-ASCII
  Unicode characters, the length of the array returned by `@get_array`
  will not match the number of characters in the string.

### Example

``` csl
const s = "abc";

// The type of 'arr' will be [3]u8.
const arr = @get_array(s);

// 'a', 'b', 'c' are 97, 98, 99 in UTF-8.
@comptime_assert(arr[0] == 97);
@comptime_assert(arr[1] == 98);
@comptime_assert(arr[2] == 99);

// The type of 'arr_with_terminator' will be [4]u8.
const arr_with_terminator = @get_array(@strcat(s, "\x00"));
@comptime_assert(arr[0] == 97);
@comptime_assert(arr[1] == 98);
@comptime_assert(arr[2] == 99);
@comptime_assert(arr[3] == 0);

// Although 'has_unicode' only has five characters, its UTF-8 encoding is 15
// bytes in length, because each Japanese hiragana character takes 3 bytes
// in UTF-8 encoding.
const has_unicode = "こんにちは";

// The type of 'unicode_arr' will be [15]u8.
const unicode_arr = @get_array(has_unicode);

// The first character in the string is "HIRAGANA LETTER KO", which happens
// to be encoded in UTF-8 as the bytes E3 81 93.
@comptime_assert(unicode_arr[0] == 0xe3);
@comptime_assert(unicode_arr[1] == 0x81);
@comptime_assert(unicode_arr[2] == 0x93);

// The last character in the string is "HIRAGANA LETTER HA", which happens
// to be encoded in UTF-8 as the bytes E3 81 AF.
@comptime_assert(unicode_arr[12] == 0xe3);
@comptime_assert(unicode_arr[13] == 0x81);
@comptime_assert(unicode_arr[14] == 0xaf);
```

## \@get_color {#language-builtins-get-color}

Create a value of type `color` with the provided identifier.

### Syntax

``` csl
@get_color(color_id);
```

Where:

- `color_id` is an integer value

### Semantics

If `color_id` is comptime-known then it must be within the range of
valid routable colors as defined by the target architecture.

If `color_id` is not comptime-known its type must be a 16-bit unsigned
integer. No runtime checks are performed in this case to ensure that the
color id is within the range of valid colors.

## \@get_config {#language-builtins-get-config}

Read the value of a PE configuration register.

### Syntax

``` csl
// The type of 'config' becomes a machine-word-sized unsigned
// integer type.
var config = @get_config(addr);

@get_config(addr, accessed_range);
@get_config(addr, accessed_ranges);
```

Where:

- `addr` is a machine-word-sized unsigned integer expression that
  represents the word-address of the configuration register.
- `access_range`, if specified, is a comptime-known 2-element tuple of
  integers specifying an *inclusive* range of addresses that `addr`
  falls within.
- `accessed_ranges`, if specified, is a tuple of comptime-known
  2-element tuples of integers that each specify an *inclusive* range of
  addresses that `addr` may fall within.

### Example

``` csl
comptime {
  // The value '42' will be stored in the configuration address '0x7e00'
  // before the program begins execution.
  @set_config(0x7e00, 42);

  // The previously-set value '42' will be retrieved from the configuration
  // address '0x7e00'.
  const old_value = @get_config(0x7e00);

  // This call will overwrite the previously set value in the configuration
  // address `0x7e00`.
  const new_value = old_value + 3;
  @set_config(0x7e00, new_value);
}

var config: [N]u16;
const base_addr = 0x7e00;
task foo(i: u16) void {
  // Read a configuration register at runtime.
  config[i] = @get_config(base_addr + i);

  // Read a configuration register that is known to occur within the range
  // [base_addr, base_addr + N].
  config[i] = @get_config(base_addr + i, .{base_addr, base_addr + N});

  // Read a configuration register that is known to occur within the ranges
  // [base_addr, base_addr + 4] or [base_addr + 8, base_addr + N].
  config[i] = @get_config(base_addr + i, .{.{base_addr, base_addr + 4},
                                           .{base_addr + 8, base_addr + N}});
}
```

### Semantics

The `@get_config` builtin can only be called at runtime and during the
evaluation of a top-level `comptime` block.

It cannot be evaluated at comptime unless it is during the evaluation of
a top-level `comptime` block.

If `@get_config` is encountered during the evaluation of a top-level
`comptime` block then it will retrieve any configuration value that was
previously stored at `addr`. If no user-defined value has previously
been written to `addr`, and a default value exists for the register at
`addr`, the default value will be returned. Otherwise, `@get_config`
will raise an error at compile time.

A call to `@get_config` at runtime will become a volatile runtime read
operation (i.e., a read that should never be optimized by the compiler)
that will return any configuration value stored to `addr`. In that
scenario the `addr` expression does not have to be comptime-known.

If `addr` is comptime-known then it must be a comptime-known integer
value that falls within the valid configuration address range for the
selected target architecture.

In cases where `addr` may be runtime but is known to occur within a
specific range or set of ranges, a second argument can be provided to
communicate this assumption.

A single, contiguous range may be specified as a tuple of integers,
`.{access_start, access_end}`. In this case, it is required that
`access_start <= addr <= access_end`.

Multiple ranges may be specified as a nested tuple,
`.{.{access_start_1, access_end_1}, ..., .{access_start_N, access_end_N}}`.
In this case, each inner tuple `.{access_start_i, access_end_i}` must
satisfy `access_start_i <= access_end_i`, and `addr` must fall within
one of the specified ranges.

An error is emitted if the compiler is able to detect a violation of the
above requirements. If a violation occurs at runtime that the compiler
cannot detect, behavior is undefined.

In addition, if `@get_config` is called during the evaluation of a
top-level `comptime` block then it is not allowed to specify an address
that falls within a configuration range that is reserved by the
compiler. These ranges correspond to the following configurations:

- All DSRs
- Filters
- Basic routing
- Switches
- Input queues
- Task table

`addr` must be coercible to a machine-word-sized unsigned integer
expression regardless of whether it\'s comptime-known or not.

## \@get_config_unchecked

Read the value of a PE configuration register unsafely.

:::: warning
::: title
Warning
:::

`@get_config_unchecked` is, by design, dangerous to use. `@get_config`
(see `language-builtins-get-config`{.interpreted-text role="ref"})
should generally be preferred.
::::

### Syntax

``` csl
// The type of 'config' becomes a machine-word-sized unsigned
// integer type.
var config = @get_config_unchecked(addr);
```

Where:

- `addr` is a machine-word-sized unsigned integer expression that
  represents the word-address of the configuration register.

### Example

``` csl
var config: [N]u16;
const base_addr = 0x7e00;
task foo(i: u16) void {
  config[i] = @get_config_unchecked(base_addr + i);
}
```

### Semantics

`@get_config_unchecked` is identical to `@get_config` (see
`language-builtins-get-config`{.interpreted-text role="ref"}) with two
exceptions:

- The compiler will not attempt to check if a reserved address is
  accessed by `@get_config_unchecked`. Accessing a reserved address may
  result in undefined behavior.
- The code generated for `@get_config` may insert delays to guarantee
  that it observes effects of preceding writes to configuration space.
  In some cases, this may be overly conservative.
  `@get_config_unchecked` will not cause such delays to be inserted. It
  is the programmer\'s responsibility to ensure that
  `@get_config_unchecked` does not observe indeterminate states of
  configuration space.

## \@get_control_task_id {#language-builtins-get-control-task-id}

Create a value of type `control_task_id` with the provided identifier.

### Syntax

``` csl
@get_control_task_id(id);
```

Where:

- `id` is a comptime-known expression of any unsigned integer type, or a
  runtime expression of type `u16`.

### Semantics

The builtin will only accept integers in the corresponding target
architecture\'s valid range for control task IDs.

If `id` is comptime-known, the builtin will only accept integers in the
corresponding target architecture\'s valid range for control task IDs.

If `id` is not comptime-known, its type must be `u16`. No runtime checks
are performed in this case to ensure that `id` is within the range of
valid control task IDs.

## \@get_data_task_id {#language-builtins-get-data-task-id}

Create a value of type `data_task_id` with the provided identifier.

### Syntax

``` csl
@get_data_task_id(id);
```

Where:

- `id` is an expression of type
  - WSE-2: `color`.
  - WSE-3: `input_queue`.

### Semantics

On WSE-2, if `id` is comptime-known, it must be within the range of
valid routable colors as defined by the target architecture. If `id` is
not comptime-known, no runtime checks are performed in this case to
ensure that `id` is within the range of valid routable colors.

## \@get_dsd

Create either a memory or fabric DSD value. See
`language-dsds`{.interpreted-text role="ref"} for details.

## \@get_dsr

Create a unique DSR identifier value. This value will uniquely identify
a physical DSR along with its DSR file. See
`language-dsrs`{.interpreted-text role="ref"} for details.

## \@get_filter_id

Get the integer identifier of the filter associated with a given color.

### Syntax

``` csl
@get_filter_id(color_value);
```

Where:

- `color_value` is a value of type `color`

### Semantics

The input `color_value` must be comptime-known and the builtin is
guaranteed to be evaluated at compile-time.

It returns the filter\'s identifier (if any) as an unsigned 16-bit
integer value. If there is no filter set for `color_value`, an error is
emitted. An error is also emitted if the compiler is unable to determine
a unique filter identifier for all the PEs that share the same code and
parameter values.

## \@get_input_queue

Create a value of type \'input_queue\' with the provided identifier.

### Syntax

``` csl
@get_input_queue(queue_id);
```

Where:

- `queue_id` is a comptime-known non-negative integer expression

### Semantics

The provided comptime-known `queue_id` must be a non-negative
comptime-known integer expression that is within the range of valid
input queue ids as defined by the target architecture.

## \@get_int

For types containing an underlying integer, return that integer value.

### Syntax

``` csl
@get_int(value);
```

Where:

- `value` is an expression with any of the following types:

  > - `color`
  > - `control_task_id`
  > - `data_task_id`
  > - any `enum` type
  > - `input_queue`
  > - any integer type
  > - `local_task_id`
  > - `output_queue`
  > - `ut_id`

### Semantics

The `@get_int` builtin must have a single argument `value` having one of
the types listed above. The underlying integer value of `value` is
returned. `@get_int` can be evaluated at both comptime and runtime.

- If `value` has `enum` type, a value of the enum\'s underlying integer
  type is returned.
- If `value` has integer type, it is returned unchanged.
- A `u16` is returned if `value` has type `color`, `control_task_id`,
  `data_task_id`, `input_queue`, `local_task_id`, or `output_queue`.

### Example

``` csl
const an_id = @get_local_task_id(29);
const a_num : i8 = -10;
const an_enum = enum(u32) {
  FOO = 1,
  BAR = 2,
  BAZ = 3
};

comptime {
   const an_id_int = @get_int(an_id);
   @comptime_assert(@is_same_type(@type_of(an_id_int), u16));
   @comptime_assert(an_int == 29);

   const another_num = @get_int(a_num);
   @comptime_assert(@is_same_type(@type_of(another_num), i8));
   @comptime_assert(another_num == -10);

   const foo = @get_int(an_enum.FOO);
   @comptime_assert(@is_same_type(@type_of(foo), u32));
   @comptime_assert(foo == 1);
}
```

## \@get_local_task_id {#language-builtins-get-local-task-id}

Create a value of type `local_task_id` with the provided identifier.

### Syntax

``` csl
@get_local_task_id(id);
```

Where:

- `id` is a comptime-known expression of any unsigned integer type, or a
  runtime expression of type `u16`.

### Semantics

If `id` is comptime-known, the builtin will only accept integers in the
corresponding target architecture\'s valid range for local task IDs.

If `id` is not comptime-known, its type must be `u16`. No runtime checks
are performed in this case to ensure that `id` is within the range of
valid local task IDs.

## \@get_output_queue

Create a value of type \'output_queue\' with the provided identifier.

### Syntax

``` csl
@get_output_queue(queue_id);
```

Where:

- `queue_id` is a comptime-known non-negative integer expression

### Semantics

The provided comptime-known `queue_id` must be a non-negative
comptime-known integer expression that is within the range of valid
output queue ids as defined by the target architecture.

## \@get_rectangle

Access the size of the rectangular region that was given to
`set_rectangle`, and other layout information.

### Syntax

``` csl
@get_rectangle();
```

Returns a `comptime_struct` with `u16` fields `width` and `height`, and
additional information about the underlying fabric and offsets.

### Example

``` csl
comptime {
  const rectangle = @get_rectangle();
  // rectangle = .{
  //   .width = <width>, .height = <height>,
  //   .fabric = {
  //      .width = <fabric_width>, .height = <fabric_height>,
  //   },
  //   .offsets = {
  //      .width = <offset_width>, .height = <offset_height>,
  //   }
  // }
}
```

### Semantics

`get_rectangle` returns the `width` and `height` provided to
`set_rectangle`, as a `comptime_struct`. This struct also contains
`fabric`, a nested struct that contains the `width` and `height` of the
underlying fabric, and `offsets`, a nested struct that contains the
`width` and `height` of the offset of the rectangle.

The `@get_rectangle` builtin can be used anywhere. In a `layout` block,
`@get_rectangle` is only valid *after* the call to `@set_rectangle`.

## \@get_string_from_byte

Given a comptime-known, non-negative integer small enough to fit in one
byte, returns a one-byte `comptime_string` containing only that byte.

### Syntax

``` csl
@get_string_from_byte(byte);
```

where:

- `byte` is a non-negative integer that fits in one byte (i.e., is in
  the range \[0, 255\]).

### Example

``` csl
const s = @get_string_from_byte(65);   // s == "A"
const t = @get_string_from_byte('A');  // t == "A"
const u = @get_string_from_byte('\n'); // u == "\n"
const v = @get_string_from_byte(0);    // v == "\x00"
```

## \@has_field

Checks whether a given struct value has a field with a given name.

### Syntax

``` csl
@has_field(some_struct, field_name);
```

where:

- `some_struct` is a value of a struct type.
- `field_name` is a string.

### Example

``` csl
@has_field(.{.blah = 10}, "blah"); // returns true.
```

### Semantics

The builtin returns true if and only if the struct `some_struct` has a
field called `field_name`.

The builtin is guaranteed to be evaluated at compile-time. The input
expressions are guaranteed to have no run-time effects.

## \@import_comptime_value

Import a customized JSON file as a comptime value.

### Syntax

``` csl
@import_comptime_value(type, comptime_string);
```

Where:

- `type` is the type of the imported value, and
- `comptime_string` is the name of the JSON file representing the
  imported value.

### Example

``` csl
// array.json
[1,2,3,4,5,6,7,8]

// csl example
const json = @import_comptime_value([8]u16, "array.json");

comptime {
   for(@range(u16, 8)) |i| {
      @comptime_assert(json[i] == i+1);
   }
}
```

### Semantics

The result type is provided as an argument, and the JSON must be
compatible with given type. Only scalars, `bool`, `color`, `comptime`
values (`comptime_int`, `comptime_float`, `comptime_string`,
`comptime_struct`), and arrays of these types, are supported.
`comptime_struct` can contain values of any of these types.

We use the following mappings when transliterating the JSON to CSL:

- Boolean map to `bool`.
- Strings map to `comptime_string`.
- Numbers map to integer types (`i8`, `u8`, \..., `comptime_int`), float
  types (`f16`, `f32`, `bf16`, `cb16`, `comptime_float`), or `color`.
- `null` in JSON is an error
- Arrays map to typed CSL arrays. Arrays can not be nested, and all
  elements in the JSON must map to the same type. The type of the
  generated CSL array can have multiple dimensions, provided the total
  number of elements is the same as the length of the one-dimensional
  JSON array.
- Objects map to `comptime_struct`. All fields MUST use the syntax
  [name:type]{.title-ref}`, to map to the name`.name`. All builtin types are supported, as well as one dimensional arrays. Examples of labels include`\"foo:bool\"`and`\"arr:\[8\]i16\"\`\`.
  The size of any array inside a Object map must be specified.

## \@import_module

Import a group of global symbols defined in a CSL file, while optionally
initializing parameters in the imported file. See
`language-modules`{.interpreted-text role="ref"} for details.

## \@increment_dsd_offset

Set the offset of a memory DSD value. See
`language-dsds`{.interpreted-text role="ref"} for details.

## \@initialize_queue {#language-builtins-initialize-queue}

Associates a routable color with a queue ID, and optionally sets the
priority of the microthread associated with the queue.

### Syntax

``` csl
@initialize_queue(queue);
@initialize_queue(queue, config);
```

Where:

- `queue` is a comptime-known expression of type `input_queue` or
  `output_queue`.
- `config` is a comptime-known struct expression with the following
  fields:
  - `color` is a comptime-known expression of type `color`.
  - `priority` is an optional field that can be either
    `.{ .high = true }`, `.{ .medium = true }`, or `.{ .low = true }`.
  - `ctrl_table_id` is
    - WSE-2: not supported.
    - WSE-3: an optional field that must be a comptime-known integer
      expression.

### Example

``` csl
const rx = @get_color(4);
const tx = @get_color(5);

comptime {
  // associates queue ID 3 with the color rx (4)
  @initialize_queue(@get_input_queue(3), .{ .color = rx });

  // ensures that output queue 5 is properly initialized at startup
  if (@is_arch("wse2")) {
    @initialize_queue(@get_output_queue(5));

  } else if (@is_arch("wse3")) {
    // Associates output queue ID 4 with the color tx (5)
    @initialize_queue(@get_output_queue(4), .{ .color = tx });

    // Associates input queue ID 0 with color rx (4). In addition,
    // it assigns a control task table ID of '4' to this input queue.
    // Control wavelets arriving through this input queue will be
    // directed to a separate control task table which is identified
    // by the `.ctrl_table_id` field.
    @initialize_queue(@get_input_queue(0),
      .{ .color = rx, .ctrl_table_id = 4 });
 }
```

``` csl
const rx = @get_color(4);
const rxq = @get_input_queue(3);

comptime {
  // associates the queue rxq (3) with the color rx (4) and high microthread
  // priority
  if (@is_arch("wse2"))
    @initialize_queue(rxq, .{ .color = rx, .priority = .{ .high = true } });
}
```

### Semantics

The `@initialize_queue` builtin will initialize the input or output
queue configuration associated with the input or output queue ID `queue`
respectively.

The builtin can only be called at most once per queue during the
evaluation of a top-level comptime block.

On WSE-2:

- If the argument `queue` is an expression of type `output_queue` then
  the builtin must have no more than a single argument (i.e., the
  `queue` argument).
- If the argument `queue` is an expression of type `input_queue` then
  the `config` argument must be supplied, and must be a comptime-known
  struct with fields `color` (required) and `priority` (optional).
- The `color` field is required and specifies the routable fabric color
  to which the input queue with ID `queue` will be bound.
- The `priority` field is optional and can be used to specify the
  priority of the microthread that will be attached to the respective
  input queue with ID `queue`. See
  `language-dsds-microthread-priority`{.interpreted-text role="ref"} for
  more information on microthread priority. The default value is
  `.{ .high = true }`.

On WSE-3:

- Both input and output queues require both `queue` and `config`
  arguments.
- The `config` comptime-known struct argument must have the `color`
  field but not the `priority` field.
- The `color` field specifies the routable fabric color that the input
  or output queue with ID `queue` will be bound to.
- The `ctrl_table_id` field is optional and allowed on input queues
  only. It can be used to specify an index identifier that represents a
  per-queue local control task table. What this means is that control
  wavelets arriving through the input queue with ID `queue` will be
  associated with a per-queue local control task table identified by the
  `ctrl_table_id` value. The default value is `0`.
- Multiple input queues can have the same value for `ctr_table_id` which
  means that they will be sharing the same control task table. For
  example, if we never use the `ctrl_table_id` for any of our input
  queues then the default behavior is that they will all share the same
  control task table with `ctrl_table_id=0` which is the same behavior
  as on WSE-2.

## \@is_arch

Returns true if the current CSL program is being compiled for the given
target architecture.

### Syntax

``` csl
@is_arch(an_arch);
```

Where:

- `an_arch` is a comptime-known string value that represents the
  architecture mnemonic. The available mnemonics are:
  - `"wse2"`: for the WSE-2 architecture
  - `"wse3"`: for the WSE-3 architecture

### Example

``` csl
/// Enable logic that is only valid if we are compiling
/// for the WSE-2 architecture.
if (@is_arch("wse2")) {
  ... WSE-2-specific logic ...
}
```

## \@is_comptime

Returns `true` if this expression is being evaluated as a comptime
expression, and `false` otherwise. (See
[`is_constant_evaluated`](https://en.cppreference.com/w/cpp/types/is_constant_evaluated)
for the details about the same function in C++).

### Syntax

``` csl
@is_comptime();
```

### Example

``` csl
fn foo() i32 {
  if (@is_comptime()) {
    // This branch is always comptime.
    return 1;
  } else {
    // This branch is never comptime,
    // so can call an external library,
    // and use architecture primitives.
    return 2;
  }
}

var comptime_result = @as(i32, 0);
var non_comptime_result = @as(i32, 0);
var init_result = foo(); // returns 1

task mytask() void {
  comptime_result = comptime foo(); // returns 1
  non_comptime_result = foo();   // returns 2
}

comptime {
  const comptime_block_result = foo(); // return 1
  ...
}
```

## \@is_same_type

Returns true if the two type arguments to this function are the same.

### Syntax

``` csl
@is_same_type(this_type, another_type);
```

Where:

- `this_type` and `another_type` are values of type `type`.

### Example

``` csl
param myType: type;

// This function uses the appropriate DSD operation based on the `myType`
// param.
fn mov(dst: mem1d_dsd, src: mem1d_dsd) void {
  if (@is_same_type(myType, f16)) {
    @fmovh(dst, src);
  } else {
    @comptime_assert(@is_same_type(myType, f32));
    @fmovs(dst, src);
  }
}
```

## \@load_to_dsr

Load a DSD value into a DSR. See `language-dsrs`{.interpreted-text
role="ref"} for details.

## \@map

Given a function, a list of input arguments and an optional output
argument, perform a mapping of the input arguments to the output
argument (if any) using the provided function.

### Syntax

``` csl
@map(callback, Input...);
@map(callback, Input..., Output);
```

Where:

- `callback` is a function that accepts as many arguments as the number
  of `Input` arguments. It may optionally produce a value.
- `Input` is a list of zero or more input arguments.
- `Output` is an output argument.

### Example

``` csl
const math_lib = @import_module("<math>");
const memDSD = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> A[i, i]);
const faboutDSD = @get_dsd(fabout_dsd,
                           .{.extent = size, .fabric_color = blue});

task foo() void {
  // Compute the square-root of each element of `memDSD` and
  // send it out to `faboutDSD`.
  @map(math_lib.sqrt_f16, memDSD, faboutDSD);
}
```

### Semantics

The `@map` builtin requires at least one of its arguments to be a DSD or
DSR (input or output).

If `callback` returns a non-void value, the `Output` argument is
mandatory and must be either a DSD, DSR of type `dsr_dest`, or a
non-const pointer value whose base-type must match the return type of
`callback`. A `fabin_dsd` value is not allowed as the `Output` argument.

The `Input` arguments may include non-DSD/DSR values whose types must be
compatible with the corresponding parameter types of `callback`. Values
of type `fabout_dsd` are not allowed as `Input` arguments. If a DSR is
used as an `Input` argument, it must be of type `dsr_src1`.

For each DSD or DSR argument to `@map`, the corresponding parameter type
or return type of `callback` must be a *DSD compatible* type. DSD
compatible types are scalar types with a bitwidth that is a multiple of
the machine word bitwidth. Currently, these types are: `i16`, `i32`,
`u16`, `u32`, `@fp16()`, `f32`. Note that `@fp16()` gives the type of
the selected runtime FP16 format (see
`language-builtins-fp16`{.interpreted-text role="ref"}).

DSR arguments to `@map` are expected to be loaded with the `single_step`
property (see `language-dsrs-single-step`{.interpreted-text
role="ref"}).

### Execution semantics

The `@map` builtin repeatedly calls the `callback` function for each
element of the DSD/DSR argument(s). Before each call to `callback`, the
next available value from each `Input` DSD/DSR is read and passed to
`callback` while the non-DSD/DSR `Input` arguments are forwarded to
`callback`. The value returned from the `callback` call - if any - is
written back to the `Output` DSD/DSR or to the memory address that is
specified by the `Output` non-const pointer.

After reading or writing a DSD/DSR element value the *length* (or
*extent* for fabric DSDs) of the respective DSD/DSR is decremented by
one. If the length/extent is zero then the read/write operation fails
and the implicit `@map` loop terminates. If DSD/DSR operands have
different lengths/extents, it is possible for values to be read and
discarded. Similarly, the computed value from `callback` may be
discarded.

## \@ptrcast

Casts a value of pointer type to a different pointer type.

### Syntax

``` csl
@ptrcast(destination_ptr_type, ptr);
```

Where:

- `destination_ptr_type` is a pointer type.
- `ptr` is a value of pointer type.

### Semantics

The builtin returns a pointer with the same memory address as `ptr`, but
whose type is `destination_ptr_type`.

The `destination_ptr_type` must not be a pointer whose base type is only
valid in `comptime` expressions. See
`language-comptime`{.interpreted-text role="ref"}.

This builtin is not valid in `comptime` expressions.

### Example

``` csl
const x:u32 = 10;
const new_ptr = @ptrcast(i16*, &x);
```

## \@random16 {#language-builtins-random16}

Generates a 16-bit pseudo-random value.

### Syntax

``` csl
@random16();
```

### Semantics

The builtin returns an `u16` value, generated through the LFSR algorithm
with polynomial $x^23 + x^18 + 1$. The LFSR state is advanced 128
iterations after every use. The initial state of the algorithm, when the
program starts, is set to `0xdeadbeef`. The state is not shared between
PEs, but it is shared between tasks.

The builtin is not valid in `comptime` expressions.

### Example

``` csl
const x:u16 = @random16();
const y:i16 = @as(i16, @random16());
```

## \@range

Generates a sequence of evenly spaced numbers.

### Syntax

``` csl
@range(elem_type, start, stop, step)
@range(elem_type, stop)
```

where:

- `elem_type` is an integer type.
- `start`, `stop` and `step` are numeric values.

### Examples

``` csl
@range(i16, 1, 5, 1)       // generates the sequence 1, 2, 3, 4. Note that 5
                           // is not included
@range(i32, 2, -3, 1)      // generates 2, 1, 0, -1, -2
@range(u16, 2, 7, -1)      // empty sequence
@range(comptime_int, 4)    // generates 0, 1, 2, 3
```

### Semantics

The range of elements is defined as follows:

- `start` defines the first element of the sequence.
- `step` defines how to generate the next element of the sequence given
  the previous element: `next = previous + step`.
- `stop` defines an upper bound on the sequence such that all elements
  in the sequence are strictly less than `stop`.

`start`, `stop` and `step` are coerced to the type `elem_type`. If this
it not possible, a compilation error is issued.

`step != 0` is required. If `step > 0 and stop <= start` or
`step < 0 and stop >= start`, then the resulting sequence is empty.

The two-argument version of `@range` is equivalent to the common
scenario where `start == 0` and `step == 1`.

## \@range_start, \@range_stop, \@range_step

Returns the `start`, `stop` or `step` value of a given range.

### Syntax

``` csl
var r = @range(elem_type, start, stop, step)
var first = @range_start(r);
var last = @range_stop(r);
var inc = @range_step(r);
```

### Examples

``` csl
const r = @range(i32, 3, 9, 2);
const start = @range_start(r);
// start == 3 of type i32

var stop_arg : u32 = 13;
var r2 = @range(u32, stop_arg);
var stop = @range_stop(r2);
// stop == stop_arg
```

## \@rank

Returns the rank (number of dimensions) of the input array type as a
`u16`.

### Syntax

``` csl
@rank(array_type);
```

Where:

- `array_type` is a `type` defining an array.

### Example

``` csl
const array_type_3d = [3, 5, 7]f16;
const rank = @rank(array_type_3d);
// rank == 3
```

## \@set_active_prng

Sets the active PRNG (Pseudo-Random Number Generator).

### Syntax

``` csl
@set_active_prng(prng_id);
```

Where:

- `prng_id` is a 16-bit unsigned integer expression that specifies the
  PRNG ID to be set.

### Semantics

The input integer expression `prng_id` specifies the active PRNG ID as
`prng_id % N` where `N` is the total number of PRNGs for the given
architecture.

This builtin cannot be evaluated at comptime.

### Example

``` csl
var prng_id: u16 = 2;
@set_active_prng(prng_id);
```

## \@set_color_config, \@set_local_color_config {#language-builtins-set-color-config}

Specify the color configuration for a specific color at a specific
processing element (PE) from a layout block (`@set_color_config`) or
from a processing element\'s top-level comptime block
(`@set_local_color_config`). A color configuration includes routing,
switching and filter configurations.

### Syntax

``` csl
@set_color_config(x_coord, y_coord, this_color, config);
@set_local_color_config(this_color, config);
```

Where:

- `x_coord` and `y_coord` are comptime-known integers indicating the PE
  coordinates.
- `this_color` is a comptime-known expression yielding a color value.
- `config` is a comptime-known anonymous struct with the following
  fields and sub-fields:
  - `routes`
    - `rx`
    - `tx`
    - `pop_mode` (deprecated, moved to `switches` field)
    - `color_swap_x`
    - `color_swap_y`
  - `switches`
    - `pos1`
    - `pos2`
    - `pos3`
    - `current_switch_pos`
    - `ring_mode`
    - `pop_mode`
  - `filter`
  - `teardown`

### Example

``` csl
color main_color;
color other_color;

layout {
  // Route wavelets of color main_color from west to ramp and east.
  const routes = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST } };
  @set_color_config(0, 0, main_color, .{ .routes = routes });
}

comptime {
  // Route wavelets of color main_color from west to ramp and east.
  const main_route = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST } };
  @set_local_color_config(main_color, .{ .routes = main_route });
}
```

### Semantics

Both `@set_color_config` and `@set_local_color_config` builtins will set
the color configuration - provided by the `config` field - to the input
color value of one or more PEs.

Calls to `@set_color_config` are only allowed during the evaluation of a
layout block. As a result they always refer to a specific PE that is
specified by the coordinate fields `x_coord` and `y_coord`.

Calls to `@set_local_color_config` are only allowed during the
evaluation of a top-level `comptime` block that belongs to a specific
PE\'s code and thus explicit coordinates are not needed as in calls to
`@set_color_config`. However, since one or more PEs may share the same
code - and thus the same top-level `comptime` block - a call to
`@set_local_color_config` may be associated with multiple PEs depending
on the rectangle\'s PE-to-code mapping defined by calls to the
`@set_tile_code` builtin.

Any two calls to `@set_color_config` and/or `set_local_color_config`
that refer to the same combination of PE and color are not allowed.

Finally, a color configuration without a `routes` field is not allowed
and will result in an error. That\'s because both the `switches` and
`filter` configurations are not valid without `routes`. If needed (e.g.,
for testing), a user can specify an empty `routes` field as
`.routes = .{}`.

#### Routing Configuration Semantics

##### rx and tx

The `rx` and `tx` fields specify the *receive* and *transmit* route
configurations for the given color. In particular, `rx` specifies the
direction(s) (i.e., `EAST`, `WEST`, `SOUTH`, `NORTH` and `RAMP`) from
which we are expecting to *receive* data and `tx` specifies the
direction(s) in which we wish to *transmit* data for the given color.
The example above demonstrates how these fields can be used in calls to
the `@set_color_config` and `@set_local_color_config` builtins. Both
`rx` and `tx` fields expect comptime-known structs with nameless fields
of unique direction values (e.g., `.tx = .{WEST, EAST, NORTH}`) or a
single direction (e.g., `.tx = WEST` and `.rx = RAMP`).

Note that it is only safe to enable multiple input directions for a
given color if it is known that wavelets will never arrive from multiple
directions on this color at once. If wavelets arrive from two enabled
directions on the same color at once, the behavior of the hardware
router is undefined.

##### pop_mode

This field is deprecated as a route setting and therefore it is moved to
the switch configuration semantics section. See
`switch-configuration-semantics`{.interpreted-text role="ref"}.

##### color_swap_x and color_swap_y

Both `color_swap_x` and `color_swap_y` fields expect a boolean value
that indicates whether we want to enable color swapping for the
horizontal and vertical direction respectively. More details about color
swapping can be found in
`language-advanced-features-color-swapping`{.interpreted-text
role="ref"}.

#### Switching Configuration Semantics {#switch-configuration-semantics}

##### pos1, pos2, pos3

The `pos1`, `pos2` and `pos3` fields expect an anonymous struct value
with only one of the following fields:

- `rx`
- `tx`
- `invalid`

A route configuration for a given color can change dynamically through
*control wavelets*; special wavelets that may carry route configuration
instructions. If we consider the `rx` and `tx` fields as the initial
configuration of the *receive* and *transmit* routes respectively, then
`pos1`, `pos2` and `pos3` are additional configurations we can switch to
in-sequence from `pos1` to `pos3` whenever we receive route-switching
control wavelets on the given color. In particular, the first
route-switching control wavelet will set the `pos1` configuration. The
next one will cause an advance to the `pos2` configuration and the third
will cause an advance to the `pos3` configuration. Any additional
route-switching control wavelets will either have no effect or go back
to the initial configuration (defined by the `rx` and `tx` fields)
depending on whether the `ring_mode` field is specified (see next
section about the [`ring_mode`](#ring_mode)). Unlike the top-level `rx`
field, the one nested under the `pos1`, `pos2` and `pos3` fields can
only have a single direction value (i.e., `EAST`, `WEST`, `NORTH`,
`SOUTH` or `RAMP`). On the other hand, the `tx` field can accept a
single direction value or more than one (if the target supports it) just
like the top-level `tx` field. In the following example, the call to
`@set_local_color config` builtin will configure routing for color `red`
such that receiving a route-switching control wavelet will change the
*receive* direction to `EAST`:

``` csl
const route = .{ .rx = .{ WEST }, .tx = .{ RAMP },
                 .pos1 = .{ .rx = EAST } };
@set_local_color_config(red, route);
```

The `invalid` field expects a boolean that must always be `true` which
will indicate that we can never advance to the corresponding switch
position and we will either remain on the previous one (if `ring_mode`
is not enabled) or advance back to the original switch position
indicated by the top-level `rx` and `tx` fields (if `ring_mode` is
enabled). All switch positions will default to `.{ .invalid = true }`.

##### pop_mode

The `pop_mode` field expects an anonymous struct value with only one of
the following fields:

- `no_pop`
- `always_pop`
- `pop_on_advance`
- `pop_on_advance_nop`

Each of these fields expects a boolean that must be `true`. In other
words, the `pop_mode` field can be viewed as an enum value that can take
one of the 4 possible values above. By specifying the `pop_mode` field
in a color configuration we can effectively mutate the sequence of
instructions carried by control wavelets as they pass through a PE on a
given color. In particular, when we select the `no_pop` mode the
instruction sequence of control wavelets remains as is. If we select the
`always_pop` mode then the first instruction in the sequence is always
popped every time a control wavelet arrives and the respective
instruction executed. If we select `pop_on_advance` then the first
instruction is popped only if the control wavelet has advanced the route
configuration to a new switch position (see section about `pos1`, `pos2`
and `pos3` [`fields`](#pos1-pos2-pos3)). If we select
`pop_on_advance_nop` then the first instruction is popped if the control
wavelet has advanced the route configuration to a new switch position or
is a no-op.

##### ring_mode

The `ring_mode` field expects a boolean value. If `true` then
route-switching control wavelets will cause the route configuration to
loop-back to the original configuration (specified by the `tx` and `rx`
fields) once all valid switch positions have been visited (see previous
section about `pos1`, `pos2` and `pos3` fields). If `false` then
route-switching control wavelets will have no effect on the routing
configuration once we reach the last valid switch position. In the
following example, if at a given point in time, the route configuration
for color `red` is at switch position 3 (specified by the `pos3` field)
then receiving a route-switching control wavelet will cause the route
configuration for `red` to loop-back into the initial one specified by
fields `rx` and `tx`:

``` csl
const route = .{ .rx = .{ WEST }, .tx = .{ RAMP },
                 .pos3 = .{ .rx = EAST }, .ring_mode = true };
@set_local_color_config(red, route);
```

##### current_switch_pos

The `current_switch_pos` field expects a non-negative integer in the
range \[0-3\] with `0` representing the initial route configuration
(specified by the `tx` and `rx` fields) and `1`, `2` and `3`
representing switch positions `1`, `2` and `3` respectively specified by
`pos1`, `pos2` and `pos3` fields. The switch position pointed to by the
`current_switch_pos` field will specify the initial route configuration
for the given color.

:::: warning
::: title
Warning
:::

On WSE-2, all colors support switch configuration. On WSE-3, only a
subset of colors support switch configuration: 0, 1, 2, 3, 4, 5, 6, 7,
8, 9, 12, 13, 16, 17, 20, and 21. Switches should only be configured for
these colors on WSE-3.
::::

#### Filter Configuration Semantics {#language-builtins-filters}

The `filter` field expects an anonymous struct value with the following
fields:

- `kind`
- `count_data`
- `count_control`
- `init_counter`
- `limit1`
- `limit2`
- `max_counter`
- `filter_control`
- `max_idx`
- `min_idx`

The `kind` field specifies the kind of the filter (or *filter mode*)
which is an anonymous struct value with a single boolean field that is
always true. The name of the boolean field represents the filter kind
mnemonic which is one of:

- `counter`
- `sparse_counter`
- `range`

The `kind` field can be viewed as an enum value that can take one of the
3 values above. The kind of the filter will determine the subset of
filter fields that are legal for that kind. Lets look at each one of the
three possible filter kinds separately.

##### Counter filter

The legal filter fields for a counter filter are the following:

- `count_data`
- `count_control`
- `init_counter`
- `limit1`
- `max_counter`
- `filter_control`

A counter filter consists of an active wavelet counter that gets
incremented every time a wavelet arrives at the given color. We can
configure the counter filter so that the active wavelet counter gets
incremented for every data wavelet or control wavelet or both. This is
done by setting the `count_data` and/or `count_control` fields that
expect a boolean value where `true` means that we enable data and
control wavelet counting respectively. The default is `false` for both
fields meaning that neither data nor control wavelets will cause the
counter to get incremented. We can initialize the active wavelet counter
by specifying the `init_counter` field that expects a non-negative
integer value (the default value is zero). The filter counter will get
incremented up to a certain value (inclusive) and then get reset to
zero. That value is specified by the `limit1` field that expects a
non-negative integer value that defaults to zero. The counter filter
will reject all wavelets whose active counter is greater than a maximum
value (exclusive) specified by the `max_counter` field that expects a
non-negative integer value that defaults to zero. Finally, the
`filter_control` field expects a boolean value. If `true` then only
control wavelets that arrive when the value of the active counter is
equal to `max_counter` are allowed to pass. All other wavelets will be
rejected by the filter.

In the following example, we set a counter filter for color `red` so
that every 5th data wavelet is rejected:

``` csl
const filter = .{.kind = .{.counter = true}, .count_data = true,
                 .limit1 = 5, .max_counter = 4};
@set_local_color_config(red, filter);
```

##### Sparse counter filter

The legal filter fields for a sparse counter filter are the following:

- `count_data`
- `count_control`
- `init_counter`
- `limit1`
- `limit2`
- `max_counter`

Sparse counter filters are similar to counter filters in that they both
rely on an active wavelet counter and the semantics of the `count_data`,
`count_control` and `max_counter` fields are identical. The difference
is that when the `limit2` field is specified then when the active
counter reaches the `limit1` value (inclusive) then the `limit2` value
is copied to `limit1` before the active counter gets reset back to zero.
When this happens `limit2` is effectively disabled and won\'t be copied
to `limit1` again. If the wavelet counter reaches `limit1` a second
time - after `limit2` was copied to `limit1` - then the filter will no
longer filter any wavelets and thus it will be effectively disabled.
Both `limit1` and `limit2` counters expect non-negative integer values
that default to zero. However, it is important to note that when
`limit2` is not specified and the active counter reaches `limit1` then
no other wavelets are filtered which means that the sparse counter
filter is effectively disabled. The same is true when no `limit1` value
is provided, i.e., the filter is effectively disabled and thus no
filtering is done.

##### Range filter

The legal filter fields for a range filter are the following:

- `max_idx`
- `min_idx`

When the filter\'s kind is set to `range` then all control wavelets are
accepted. However, data wavelets are filtered based on their index
value. In particular, a data wavelet will be rejected iff its index
value is not within the range \[`min_idx`, `max_idx`\].

#### Teardown Configuration Semantics

The `teardown` field expects a comptime-known boolean expression. If
`true` then the color associated with the given configuration will be
set to teardown mode when the program starts. This means that all
traffic will be suspended on that color until the teardown mode is
exited explicitly at runtime through the `<tile_config/teardown.csl>`
standard library API.

While a color is in teardown mode, all the configuration settings can be
re-set at runtime using a standard library API. For example, filters can
be configured using the `<tile_config/filters.csl>` standard library
API.

## \@set_config {#language-builtins-set-config}

Write the value of a PE configuration register.

### Syntax

``` csl
@set_config(addr, config_value);
@set_config(addr, config_value, accessed_range);
@set_config(addr, config_value, accessed_ranges);
```

Where:

- `addr` is a machine-word-sized unsigned integer expression that
  represents the word-address of the configuration register.
- `config_value` is a machine-word-sized unsigned integer expressions
  that represents the new configuration value.
- `access_range`, if specified, is a comptime-known 2-element tuple of
  integers specifying an *inclusive* range of addresses that `addr`
  falls within.
- `accessed_ranges`, if specified, is a tuple of comptime-known
  2-element tuples of integers that each specify an *inclusive* range of
  addresses that `addr` may fall within.

### Example

``` csl
comptime {
  // The value '42' will be stored in the configuration address '0x7e00'
  // before the program begins execution.
  @set_config(0x7e00, 42);

  // This call will overwrite the previously set value in the configuration
  // address `0x7e00`.
  const new_value = old_value + 3;
  @set_config(0x7e00, new_value);
}

var addr: u16;
task foo(value: u16) void {
  // Writes 'value' at the configuration address 'addr' at runtime.
  @set_config(addr, value);

  // Re-writes 'value'. The compiler will not optimize away the
  // first write because it is considered volatile.
  @set_config(addr, value);

  // Write 'value' at the configuration address 'addr', where 'addr' is known
  // to occur within the range [0x7e00, 0x7e0f].
  @set_config(addr, value, .{0x7e00, 0x7e0f});

  // Write 'value' at the configuration address 'addr', where 'addr' is known
  // to occur within one of the ranges [0x7e00, 0x7e0f], [0x7e20, 0x7e2f], or
  // [0x7e40, 0x7e4f].
  @set_config(addr, value, .{.{0x7e00, 0x7e0f}, .{0x7e20, 0x7e2f},
                             .{0x7e40, 0x7e4f}});
}
```

### Semantics

The `@set_config` builtin can be called at runtime and during the
evaluation of a top-level `comptime` block.

It cannot be evaluated at comptime unless it is during the evaluation of
a top-level `comptime` block.

If `@set_config` is encountered during the evaluation of a top-level
`comptime` block then it will store `config_value` to `addr` during link
time and therefore the configuration value is guaranteed to be present
before the program begins execution.

A call to `@set_config` during top-level comptime evaluation will always
overwrite any configuration value that was previously stored at `addr`.

A call to `@set_config` at runtime will become a volatile runtime write
operation that will store `config_value` to `addr`. In that scenario,
both `config_value` and `addr` expressions don\'t have to be
comptime-known.

If `addr` is comptime-known then it must be a comptime-known integer
value that falls within the valid configuration address range for the
selected target architecture.

In cases where `addr` may be runtime but is known to occur within a
specific range or set of ranges, a second argument can be provided to
communicate this assumption.

A single, contiguous range may be specified as a tuple of integers,
`.{access_start, access_end}`. In this case, it is required that
`access_start <= addr <= access_end`.

Multiple ranges may be specified as a nested tuple,
`.{.{access_start_1, access_end_1}, ..., .{access_start_N, access_end_N}}`.
In this case, each inner tuple `.{access_start_i, access_end_i}` must
satisfy `access_start_i <= access_end_i`, and `addr` must fall within
one of the specified ranges.

An error is emitted if the compiler is able to detect a violation of the
above requirements. If a violation occurs at runtime that the compiler
cannot detect, behavior is undefined.

In addition, if `@set_config` is called during the evaluation of a
top-level `comptime` block then it is not allowed to specify an address
that falls within a configuration range that is reserved by the
compiler. These ranges correspond to the following configurations:

- All DSRs
- Filters
- Basic routing
- Switches
- Input queues
- Task table

Most of the configurations in the list above can be changed through
other means (see `@set_color_config`, `@set_local_color_config` and
`@initialize_queue` builtins) while the rest are managed automatically
by the compiler (i.e., DSRs and task tables).

Both `addr` and `config_value` must be coercible to a machine-word-sized
unsigned integer expressions regardless of whether they are
comptime-known or not.

## \@set_config_unchecked

Write the value of a PE configuration register unsafely.

:::: warning
::: title
Warning
:::

`@set_config_unchecked` is, by design, dangerous to use. `@set_config`
(see `language-builtins-set-config`{.interpreted-text role="ref"})
should generally be preferred.
::::

### Syntax

``` csl
@set_config_unchecked(addr, config_value);
```

Where:

- `addr` is a machine-word-sized unsigned integer expression that
  represents the word-address of the configuration register.
- `config_value` is a machine-word-sized unsigned integer expressions
  that represents the new configuration value.

### Example

``` csl
var addr: u16;
task foo(value: u16) void {
  @set_config_unchecked(addr, value);
}
```

### Semantics

`@set_config_unchecked` is identical to `@set_config` (see
`language-builtins-set-config`{.interpreted-text role="ref"}) with two
exceptions:

- The compiler will not attempt to check if a reserved address is
  accessed by `@set_config_unchecked`. Accessing a reserved address may
  result in undefined behavior.
- The code generated for `@set_config` may insert delays to guarantee
  that its effects are observed by subsequent writes to configuration
  space. In some cases, this may be overly conservative.
  `@set_config_unchecked` will not cause such delays to be inserted. It
  is the programmer\'s responsibility to ensure that
  `@set_config_unchecked` does not cause indeterminate states of
  configuration space to be observed.

## \@set_dsd_base_addr

Set the base-address of a memory DSD value. See
`language-dsds`{.interpreted-text role="ref"} for details.

## \@set_dsd_length

Set the length of a 1D memory DSD value. See
`language-dsds`{.interpreted-text role="ref"} for details.

## \@set_dsd_stride

Set the stride of a 1D memory DSD value. See
`language-dsds`{.interpreted-text role="ref"} for details.

## \@set_fifo_read_length

Set the read length of a FIFO DSD. See `language-dsds`{.interpreted-text
role="ref"} for details.

## \@set_fifo_write_length

Set the write length of a FIFO DSD. See
`language-dsds`{.interpreted-text role="ref"} for details.

## \@set_rectangle

Specify the size of the rectangular region of processing element that
will execute this code.

### Syntax

``` csl
@set_rectangle(width, height);
```

Where:

- `width` and `height` are `comptime` integers.

### Example

``` csl
layout {
  // Use just one processing element for running this kernel.
  @set_rectangle(1, 1);
}
```

### Semantics

The `@set_rectangle` builtin must appear only in a `layout` block.
Additionally, there must be exactly one call to `@set_rectangle` in a
`layout` block.

## \@set_teardown_handler

Set a function to be the teardown handler for a given color.

### Syntax

``` csl
@set_teardown_handler(this_func, this_color);
```

Where:

- `this_func` is the name of a function with no input parameters and
  \'void\' return type.
- `this_color` is a comptime-known expression of type \'color\'.

### Example

``` csl
fn foo() void {
  ...
}
const blue = @get_color(0);

comptime {
  @set_teardown_handler(foo, blue);
}
```

### Semantics

The `@set_teardown_handler` builtin must appear in a top-level
`comptime` block. When a color goes into *teardown* mode at runtime,
then the function associated with that color, through a call to
`@set_teardown_handler`, will be executed.

Calling `@set_teardown_handler` for the same color more than once is not
allowed and will result in an error.

The color will not automatically exit from the teardown mode. The user
is responsible for exiting teardown mode explicitly from within the
respective teardown handler function.

The color that is passed to a `@set_teardown_handler` call must be
within the range of routable colors for the given target architecture.

If there is at least 1 call to `@set_teardown_handler` in the program
then no task is allowed to be bound to the *teardown task ID* and
vice-versa. The teardown task ID is the value returned by the CSL
standard library through the teardown API (see
`language-libraries-tile-config-teardown`{.interpreted-text
role="ref"}).

## \@set_tile_code

Specify the file that contains instructions to execute on a specific
processing element, while optionally initializing parameters defined in
the file.

### Syntax

``` csl
@set_tile_code(x_coord, y_coord);
@set_tile_code(x_coord, y_coord, filename);
@set_tile_code(x_coord, y_coord, filename, param_binding);
```

Where:

- `x_coord` and `y_coord` are `comptime` integers.
- `filename` is a `comptime` string.
- `param_binding` is a `comptime` anonymous struct.

### Example

``` csl
layout {
  // Specify to the compiler that *this* file contains code for PE #0,0.
  @set_tile_code(0, 0);

  // Inform compiler that code for PE #0,1 is in file "program.csl", relative
  // to the file that contains this `layout` block.
  @set_tile_code(0, 1, "program.csl");

  // Inform compiler about the location of the code using an absolute path.
  @set_tile_code(0, 2, "/var/lib/csl/modules/code.csl");

  // Instruct the compiler to use the file "other.csl" for code for PE #0,3.
  // Also, initialize the parameters `foo` and `bar` in that file.
  @set_tile_code(0, 3, "other.csl", .{ .foo = 10, .bar = -1 });
}
```

### Semantics

The `@set_tile_code` builtin must appear only in a `layout` block.
Additionally, there must be exactly one call to `@set_tile_code` for
each coordinate contained in the dimensions specified in the call to
`@set_rectangle`. Unless the specified file path is an absolute path, it
is interpreted as relative to the path of the file that contains the
`@set_tile_code()` builtin call.

## \@strcat

Concatenates compile-time strings.

### Syntax

``` csl
@strcat(str1, str2, ..., strN);
```

where:

- each argument is an expression of type `comptime_string`.

### Semantics

The `@strcat` builtin returns a value of `comptime_string` that results
from concatenating its arguments.

### Example

``` csl
@strcat("abc", "123");           // returns "abc123"
@strcat("hello", " ", "world!"); // returns "hello world!"
@strcat("abc");                  // returns "abc"
@strcat("");                     // returns ""
@strcat();                       // returns ""
```

## \@strlen

Returns the length of a compile-time string.

### Syntax

``` csl
@strlen(str);
```

where:

- `str` is an expression of type `comptime_string`.

### Semantics

The `@strlen` builtin returns a value of type `comptime_int` equal to
the length of its argument, i.e., the number of characters in the
string.

### Example

``` csl
@strlen("");                                 // returns 0
@strlen("abc123");                           // returns 6
@strlen(if (42 == 42) "abc" else "abc123");  // returns 3
```

## \@type_of

Returns the type of an expression.

### Syntax

``` csl
@type_of(any_expression);
```

where:

- `any_expression` is any valid expression.

### Semantics

The `@type_of` builtin returns a value of type `type` describing the
evaluated type of the input expression.

The builtin is always evaluated at compile-time, but the input
expression does not need to be `comptime`.

No code is generated for the input expression; as such, this expression
will not have run-time effects on the program.

## \@unblock {#language-builtins-unblock}

Unblock the task associated with the input `color`, `data_task_id`, or
`local_task_id` so that the task can be run when the task identifier is
activated.

### Syntax

``` csl
@unblock(id);
```

Where:

- `id` is an expression of type
  - WSE-2: `color`, `data_task_id`, or `local_task_id`.
  - WSE-3: `input_queue`, `data_task_id`, `local_task_id`, or `ut_id`.

### Example

``` csl
const task_id: local_task_id = @get_local_task_id(10);
comptime {
  @bind_local_task(my_task, task_id);
  @block(task_id);
}

fn foo() void {
  // allow my_task to be run whenever my_task is activated
  @unblock(task_id);
}
```

## \@zeros

Initialize a tensor with zeros.

### Syntax

``` csl
@zeros(tensor_type);
```

Where:

- `tensor_type` is a comptime-known numeric tensor type.

### Example

``` csl
// Initialize a tensor of four rows and five columns with all zeros.
const matrix = @zeros([4,5]f16);
```

## Builtins for Supporting Remote Procedure Calls (RPC) {#language-builtins-rpc}

This category includes builtins that enable users to advertise device
symbols to the host so that the host can interact with them through
wavelets akin to RPC. The advertised symbols could be data or functions
forming a host-callable API. In addition, this builtin category includes
builtins that allow users to interpret incoming wavelets by associating
them with the respective advertised symbols.

### \@export_name

Declare that a given symbolic name can be advertised from one or more
processing elements with a specific type and mutability.

#### Syntax

``` csl
@export_name(name, type);
@export_name(name, type, isMutable);
```

Where:

- `name` is an expression of type `comptime_string`
- `type` is an expression of type `type`
- `isMutable` is a comptime-known expression of type `bool`

#### Example

``` csl
layout {
  // Declare that symbolic name "A" can only be advertised
  // with type 'i16' by one or more PEs. The advertised
  // symbol must also be mutable (i.e., declared as 'var').
  @export_name("A", i16, true);

  // Declare that symbol name "foo" can only be advertised
  // as a function with type 'fn(f32)void' by one or more
  // PEs.
  @export_name("foo", fn(f32)void);
}
```

#### Semantics

Calls to the `@export_name` builtin can only appear during the
evaluation of a layout block. A given name can only be exported once
with `@export_name`. The third `isMutable` parameter must be provided
unless `type` is a function type.

The `type` parameter cannot be a comptime-only type (e.g.,
`comptime_int`, `comptime_float` etc.) with the exception of function
types.

In addition, the `type` parameter cannot be an aggregate type like an
array or struct.

The `type` parameter cannot be an enum type as well.

If `type` is a function type, then the same rules apply to the
respective function parameter types and return type.

If `type` is a function type, then it can have a maximum of 15 input
parameters.

### \@export_symbol

Advertise a device symbol to the host with a given name, if provided.

#### Syntax

``` csl
@export_symbol(symbol);
@export_symbol(symbol, name);
```

Where:

- `symbol` is a reference to a global device symbol.
- `name` is an expression of type `comptime_string`.

#### Example

``` csl
var A: i16;
fn bar(a: f32) void {...}

comptime {
  // Advertise symbol 'A' to the host. Since no 'name'
  // is provided, the default name would be the symbol's
  // name, i.e., 'A'.
  @export_symbol(A);

  // Advertise function 'bar' as 'foo' to the host.
  @export_symbol(bar, "foo");
}
```

#### Semantics

Calls to the `@export_symbol` builtin can only appear during the
evaluation of a top-level comptime block.

Its first argument must be a global symbol that is used at least once by
code that is not comptime evaluated.

A given symbol can be exported multiple times as long as each time the
`name` argument is provided and it is unique. If `name` is not provided
then the name of the symbol is used as the advertised name instead.

The advertised name (whether it is explicitly provided or defaulted to
the symbol\'s actual name) must always correspond to a name that was
exported during layout evaluation using the `@export_name` builtin. That
is, there must be a name exported with `@export_name` during layout
evaluation that has the same name, type and mutability.

The compiler will collect all exported symbols and advertise them to the
host by producing a JSON file containing meta-data for each one. The
schema of the produced JSON file is as follows:

``` json
{
  "rpc_symbols": [
    {
      "id": "Unique integer identifier for the exported symbol."

      "comment1": "Boolean indicating whether the exported symbol is",
      "comment2": "immutable or not. Not specified for functions."
      "immutable": "True/False",

      "name": "The name that is advertised to the host.",

      "type": "Type of the advertised symbol. Return-type for functions.",

      "kind": "One of Var/Func/Stream.",

      "color": "Integer representing the color of a stream if kind=Stream",

      "inputs": [
         "comment1": "The parameters of the function.",
         "comment2": "Not specified for data.",

         "name": "The name of the function parameter.",

         "type": "The type of the function parameter."
      ]
    }
  ]
}
```

### \@get_symbol_id

Returns the unique integer identifier for an advertised symbol.

#### Syntax

``` csl
@get_symbol_id(symbol);
```

Where:

- `symbol` is a reference to an advertised global device symbol.

#### Example

``` csl
var A: i16;
fn bar(a: f32) void {...}

task main(idx: u16) void {
  // If the incoming wavelet corresponds to the unique integer
  // id for 'A' then use 'A'.
  // If the incoming wavelet corresponds to the unique integer
  // id for 'bar' the call 'bar'.
  if (@get_symbol_id(A) == idx) {
    A = 42;
  } else if (@get_symbol_id(bar) == idx) {
    bar(...);
  }
}
```

#### Semantics

The `@get_symbol_id` builtin can be called at comptime or runtime but it
is not allowed to appear during layout evaluation. The input `symbol`
must have been advertised using `@export_symbol`.

### \@get_symbol_value

Returns the value of an advertised global symbol given a runtime integer
identifier value.

#### Syntax

``` csl
@get_symbol_value(type, id);
```

Where:

- `type` an expression of type `type`.
- `id` is a runtime-only integer identifier value.

#### Example

``` csl
var A: i16;
var B: i16;

task main(idx: u16) void {
  // Given an integer identifier passed with a wavelet,
  // return the value of 'A' or 'B' depending on the value
  // of 'idx'.
  var value = @get_symbol_value(i16, idx);
  ...
}
```

#### Semantics

The `@get_symbol_value` builtin can only be called at runtime.

If no symbol was advertised with the given integer identifier then the
behavior is undefined.

If there is a symbol advertised with the given integer identifier, then
the builtin will return a copy of its value.

There has to be at least one global symbol advertised with type `type`.

### \@get_tensor_ptr

Returns the value of an exported tensor pointer given a runtime integer
identifier value.

#### Syntax

``` csl
@get_tensor_ptr(id);
```

Where:

- `id` is a runtime-determined integer identifier value.

#### Example

``` csl
var A_ptr: [*]f16 = &A;
var B_ptr: *[size]i32 = &B;

comptime {
  @export_symbol(A_ptr);
  @export_symbol(B_ptr);
}

task main(idx: u16) void {
  // Given an integer identifier passed with a wavelet,
  // return the address of 'A' or 'B' depending on the value
  // of 'idx'.
  var ptr = @get_tensor_ptr(idx);
  ...
}
```

#### Semantics

The `@get_tensor_ptr` builtin can only be called at runtime.

If no tensor pointer has been advertised with the given integer
identifier then the behavior is undefined.

If there is a tensor pointer advertised with the given integer
identifier, then the builtin will return a copy of the pointer
bit-casted into a `[*]u16` type.

### \@get_xdsr

Create a unique XDSR identifier value. This value will uniquely identify
a physical XDSR. See `language-dsrs`{.interpreted-text role="ref"} for
details.

### \@has_exported_tensors

Returns `true` iff there is at least 1 tensor pointer exported and
`false` otherwise.

#### Syntax

``` csl
@has_exported_tensors();
```

#### Example

``` csl
task main(idx: u16) void {
  // If there are no exported tensors, the body
  // of the conditional branch will be removed.
  if (@has_exported_tensors()) {
      ...
      var ptr = @get_tensor_ptr(idx);
      ...
  }
}
```

#### Semantics

The `@has_exported_tensors` builtin cannot be called from a top-level
comptime block or a layout block.

It is guaranteed to be evaluated at comptime.

It will return `true` iff there is at least 1 exported tensor pointer
and `false` otherwise.

### \@rpc

Creates an RPC server listening to a given color.

#### Syntax

``` csl
@rpc(task_id);
```

Where:

- `task_id` is an expression of type `data_task_id`.

#### Example

``` csl
const rpc_color = @get_color(22);
const rpc_task_id = @get_data_task_id(rpc_color);

fn foo(a: u16, b: f32) void {...}

comptime {
  @export_symbol(foo);
  // Creates an RPC server listening to color '22' through
  // a wavelet-triggered task bound to data-task ID 'rpc_task_id'.
  // This RPC server can only dispatch calls to 'foo'.
  // Any other RPCs will be ignored.
  @rpc(rpc_task_id);

  // The user needs to ensure that traffic on color '22'
  // is directed into the RPC server.
  @set_local_color_config(rpc_color, .{.routes =
                                     .{.rx = .{WEST},
                                       .tx = .{RAMP}
                                      }});
}
```

#### Semantics

The `@rpc` builtin can only be called during the evaluation of a
top-level comptime block.

A call to `@rpc` will produce a wavelet-triggered task (WTT) that is
bound to `task_id` and would receive data from the underlying routable
color. Note that the user is responsible for routing the data through
that color such that they are received by the produced WTT.

No other task may be bound to `task_id` and vice-versa.

The WTT-based RPC server is expected to receive sequences of wavelets.
Each one of these sequences corresponds to a single RPC that consists of
a unique integer identifier corresponding to an exported function along
with its input arguments.

If the input arguments of an RPC do not match the expected number of
arguments for a given exported function, a runtime assertion is
triggered.

If the unique integer identifier of an RPC does not match any exported
function, the call will be ignored and the server will be ready for the
next RPC sequence.

If the function called by the RPC server returns a value, then this
value will be ignored.

No more than 1 call to `@rpc` is allowed for a given tile code which
means that we can always have up to 1 RPC server per PE.

## Builtins for DSD Operations {#language-builtins-for-dsd-operations}

These builtins perform bulk operations on a set of elements described by
DSDs, by exploiting native hardware instructions. The destination
operand is always the first argument, and the subsequent arguments are
either DSDs, scalars, or pointers.

Additionally, many of these builtins have a SIMD (single instruction,
multiple data) mode. For more information, see
`language-appendix-simd`{.interpreted-text role="ref"}.

### Syntax

For the DSD operation builtins below, the arguments are labeled as
follows:

- `dest_dsd`, `src_dsd1`, and `src_dsd2` are constants or variables
  created using the `@get_dsd` builtin or the `get_dsr` builtin.
- `dest_dsd` is the destination DSD or DSR. If this is a DSR value it
  must be of type `dsr_dest` or `dsr_src0`.
- `src_dsd1` and `src_dsd2` are source DSDs or DSRs or any combination
  of them. If any of the source operands are DSRs then they cannot be of
  type `dsr_dest`.
- `i16_value` is a value of type `i16`.
- `i32_value` is a value of type `i32`.
- `u16_value` is a value of type `u16`.
- `u32_value` is a value of type `u32`.
- `fp16_value` is a value of type `@fp16()`, where `@fp16()` gives the
  type of the selected runtime FP16 format (see
  `language-builtins-fp16`{.interpreted-text role="ref"}).
- `f32_value` is a value of type `f32`.
- `i16_pointer` is a pointer to a value of type `i16`.
- `i32_pointer` is a pointer to a value of type `i32`.
- `u16_pointer` is a pointer to a value of type `u16`.
- `u32_pointer` is a pointer to a value of type `u32`.
- `fp16_pointer` is a pointer to a value of type `@fp16()`.
- `f32_pointer` is a pointer to a value of type `f32`.

### \@add16 {#@add16}

Add two 16-bit integers.

``` csl
@add16(dest_dsd, src_dsd1,  src_dsd2);
@add16(dest_dsd, i16_value, src_dsd1);
@add16(dest_dsd, u16_value, src_dsd1);
@add16(dest_dsd, src_dsd1,  i16_value);
@add16(dest_dsd, src_dsd1,  u16_value);
```

### \@addc16 {#@addc16}

Add two 16-bit integers, with carry.

``` csl
@addc16(dest_dsd, src_dsd1,  src_dsd2);
@addc16(dest_dsd, i16_value, src_dsd1);
@addc16(dest_dsd, u16_value, src_dsd1);
@addc16(dest_dsd, src_dsd1,  i16_value);
@addc16(dest_dsd, src_dsd1,  u16_value);
```

### \@and16 {#@and16}

Bitwise-and two 16-bit integers.

``` csl
@and16(dest_dsd, src_dsd1,  src_dsd2);
@and16(dest_dsd, i16_value, src_dsd1);
@and16(dest_dsd, u16_value, src_dsd1);
@and16(dest_dsd, src_dsd1,  i16_value);
@and16(dest_dsd, src_dsd1,  u16_value);
```

### \@clz {#@clz}

Count leading zeros.

``` csl
// count leading zeros
@clz(dest_dsd, src_dsd1);
@clz(dest_dsd, i16_value);
@clz(dest_dsd, u16_value);
```

### \@ctz {#@ctz}

Count trailing zeros.

``` csl
// count trailing zeros
@ctz(dest_dsd, src_dsd1);
@ctz(dest_dsd, i16_value);
@ctz(dest_dsd, u16_value);
```

### \@fabsh {#@fabsh}

Absolute value of a 16-bit floating point.

``` csl
@fabsh(dest_dsd, src_dsd1);
@fabsh(dest_dsd, fp16_value);
```

### \@fabss {#@fabss}

Absolute value of a 32-bit floating point.

``` csl
@fabss(dest_dsd, src_dsd1);
@fabss(dest_dsd, f32_value);
```

### \@faddh {#@faddh}

Add two 16-bit floating point values.

``` csl
@faddh(dest_dsd,    src_dsd1,  src_dsd2);
@faddh(dest_dsd,    fp16_value, src_dsd1);
@faddh(dest_dsd,    src_dsd1,  fp16_value);
@faddh(fp16_pointer, fp16_value, src_dsd1);
```

### \@faddhs {#@faddhs}

Add a 16-bit and 32-bit floating point value.

``` csl
@faddhs(dest_dsd,    src_dsd1,  src_dsd2);
@faddhs(dest_dsd,    fp16_value, src_dsd1);
@faddhs(dest_dsd,    src_dsd1,  fp16_value);
@faddhs(f32_pointer, f32_value, src_dsd1);
```

### \@fadds {#@fadds}

Add two 32-bit floating point values.

``` csl
@fadds(dest_dsd,    src_dsd1,  src_dsd2);
@fadds(dest_dsd,    f32_value, src_dsd1);
@fadds(dest_dsd,    src_dsd1,  f32_value);
@fadds(f32_pointer, f32_value, src_dsd1);
```

### \@fh2s {#@fh2s}

Convert a 16-bit floating point value to a 32-bit floating point value.

``` csl
@fh2s(dest_dsd, src_dsd1);
@fh2s(dest_dsd, fp16_value);
```

### \@fh2xp16 {#@fh2xp16}

Convert a 16-bit floating point value to a 16-bit integer.

``` csl
@fh2xp16(dest_dsd,    src_dsd1);
@fh2xp16(dest_dsd,    fp16_value);
@fh2xp16(i16_pointer, fp16_value);
```

### \@fmach {#@fmach}

16-bit floating point multiply-add.

``` csl
@fmach(dest_dsd, src_dsd1, src_dsd2, fp16_value);
```

### \@fmachs {#@fmachs}

16-bit floating point multiply with 32-bit addition.

``` csl
@fmachs(dest_dsd, src_dsd1, src_dsd2, fp16_value);
```

### \@fmacs {#@fmacs}

32-bit floating point multiply-add.

``` csl
@fmacs(dest_dsd, src_dsd1, src_dsd2, f32_value);
```

### \@fmaxh {#@fmaxh}

16-bit floating point max.

``` csl
@fmaxh(dest_dsd,    src_dsd1,  src_dsd2);
@fmaxh(dest_dsd,    fp16_value, src_dsd1);
@fmaxh(dest_dsd,    src_dsd1,  fp16_value);
@fmaxh(fp16_pointer, fp16_value, src_dsd1);
```

### \@fmaxs {#@fmaxs}

32-bit floating point max.

``` csl
@fmaxs(dest_dsd,    src_dsd1,  src_dsd2);
@fmaxs(dest_dsd,    f32_value, src_dsd1);
@fmaxs(dest_dsd,    src_dsd1,  f32_value);
@fmaxs(f32_pointer, f32_value, src_dsd1);
```

### \@fmovh {#@fmovh}

Move a 16-bit floating point value.

``` csl
@fmovh(dest_dsd,    src_dsd1);
@fmovh(fp16_pointer, src_dsd1);
@fmovh(dest_dsd,    fp16_value);
```

### \@fmovs {#@fmovs}

Move a 32-bit floating point value.

``` csl
@fmovs(dest_dsd,    src_dsd1);
@fmovs(f32_pointer, src_dsd1)
@fmovs(dest_dsd,    f32_value);
```

### \@fmulh {#@fmulh}

Multiply 16-bit floating point values.

``` csl
@fmulh(dest_dsd,    src_dsd1,  src_dsd2);
@fmulh(dest_dsd,    fp16_value, src_dsd1);
@fmulh(dest_dsd,    src_dsd1,  fp16_value);
@fmulh(fp16_pointer, fp16_value, src_dsd1);
```

### \@fmuls {#@fmuls}

Multiply 32-bit floating point values.

``` csl
@fmuls(dest_dsd,    src_dsd1,  src_dsd2);
@fmuls(dest_dsd,    f32_value, src_dsd1);
@fmuls(dest_dsd,    src_dsd1,  f32_value);
@fmuls(f32_pointer, f32_value, src_dsd1);
```

### \@fnegh {#@fnegh}

Negate a 16-bit floating point value.

``` csl
@fnegh(dest_dsd, src_dsd1);
@fnegh(dest_dsd, fp16_value);
```

### \@fnegs {#@fnegs}

Negate a 32-bit floating point value.

``` csl
@fnegs(dest_dsd, src_dsd1);
@fnegs(dest_dsd, f32_value);
```

### \@fnormh {#@fnormh}

Normalize a 16-bit floating point value.

``` csl
@fnormh(fp16_pointer, fp16_value);
```

### \@fnorms {#@fnorms}

Normalize a 32-bit floating point value.

``` csl
@fnorms(f32_pointer, f32_value);
```

### \@fs2h {#@fs2h}

Convert a 32-bit floating point value to a 16-bit floating point value.

``` csl
@fs2h(dest_dsd, src_dsd1);
@fs2h(dest_dsd, f32_value);
```

### \@fs2xp16 {#@fs2xp16}

Convert a 32-bit floating point value to a 16-bit integer.

``` csl
@fs2xp16(dest_dsd,    src_dsd1);
@fs2xp16(dest_dsd,    f32_value);
@fs2xp16(i16_pointer, f32_value);
```

### \@fscaleh {#@fscaleh}

16-bit floating point multiplied by a constant.

``` csl
@fscaleh(fp16_pointer, fp16_value, i16_value);
```

### \@fscales {#@fscales}

32-bit floating point multiplied by a constant.

``` csl
@fscales(f32_pointer, f32_value, i16_value);
```

### \@fsubh {#@fsubh}

Subtract two 16-bit floating point values.

``` csl
@fsubh(dest_dsd,    src_dsd1,  src_dsd2);
@fsubh(dest_dsd,    fp16_value, src_dsd1);
@fsubh(dest_dsd,    src_dsd1,  fp16_value);
@fsubh(fp16_pointer, fp16_value, src_dsd1);
```

### \@fsubs {#@fsubs}

Subtract two 32-bit floating point values.

``` csl
@fsubs(dest_dsd,    src_dsd1,  src_dsd2);
@fsubs(dest_dsd,    f32_value, src_dsd1);
@fsubs(dest_dsd,    src_dsd1,  f32_value);
@fsubs(f32_pointer, f32_value, src_dsd1);
```

### \@mov16 {#@mov16}

Move a 16-bit integer.

``` csl
@mov16(dest_dsd,    src_dsd1);
@mov16(i16_pointer, src_dsd1);
@mov16(u16_pointer, src_dsd1);
@mov16(dest_dsd,    i16_value);
@mov16(dest_dsd,    u16_value);
```

### \@mov32 {#@mov32}

Move a 32-bit integer.

``` csl
@mov32(dest_dsd,    src_dsd1);
@mov32(i32_pointer, src_dsd1);
@mov32(u32_pointer, src_dsd1);
@mov32(dest_dsd,    i32_value);
@mov32(dest_dsd,    u32_value);
```

### \@or16 {#@or16}

Bitwise-or on two 16-bit integers.

``` csl
@or16(dest_dsd, src_dsd1,  src_dsd2);
@or16(dest_dsd, i16_value, src_dsd1);
@or16(dest_dsd, u16_value, src_dsd1);
@or16(dest_dsd, src_dsd1,  i16_value);
@or16(dest_dsd, src_dsd1,  u16_value);
```

### \@popcnt {#@podcnt}

Population count of an integer.

``` csl
@popcnt(dest_dsd, src_dsd1);
@popcnt(dest_dsd, i16_value);
@popcnt(dest_dsd, u16_value);
```

### \@sar16 {#@sar16}

Arithmetic shift right of a 16-bit integer.

``` csl
@sar16(dest_dsd, src_dsd1,  src_dsd2);
@sar16(dest_dsd, i16_value, src_dsd1);
@sar16(dest_dsd, u16_value, src_dsd1);
@sar16(dest_dsd, src_dsd1,  i16_value);
@sar16(dest_dsd, src_dsd1,  u16_value);
```

### \@sll16 {#@sll16}

Logical shift left of a 16-bit integer.

``` csl
@sll16(dest_dsd, src_dsd1,  src_dsd2);
@sll16(dest_dsd, i16_value, src_dsd1);
@sll16(dest_dsd, u16_value, src_dsd1);
@sll16(dest_dsd, src_dsd1,  i16_value);
@sll16(dest_dsd, src_dsd1,  u16_value);
```

### \@slr16 {#@slr16}

Logical shift right of a 16-bit integer.

``` csl
@slr16(dest_dsd, src_dsd1,  src_dsd2);
@slr16(dest_dsd, i16_value, src_dsd1);
@slr16(dest_dsd, u16_value, src_dsd1);
@slr16(dest_dsd, src_dsd1,  i16_value);
@slr16(dest_dsd, src_dsd1,  u16_value);
```

### \@sub16 {#@sub16}

Substract two 16-bit integers.

``` csl
@sub16(dest_dsd, src_dsd1, src_dsd2);
@sub16(dest_dsd, src_dsd1, i16_value);
@sub16(dest_dsd, src_dsd1, u16_value);
```

### \@xor16 {#@xor16}

Xor two 16-bit integers.

``` csl
@xor16(dest_dsd, src_dsd1,  src_dsd2);
@xor16(dest_dsd, i16_value, src_dsd1);
@xor16(dest_dsd, u16_value, src_dsd1);
@xor16(dest_dsd, src_dsd1,  i16_value);
@xor16(dest_dsd, src_dsd1,  u16_value);
```

### \@xp162fh {#@xp162fh}

Convert a 16-bit integer into a 16-bit floating point value.

``` csl
@xp162fh(dest_dsd, src_dsd1);
@xp162fh(dest_dsd, i16_value);
@xp162fh(dest_dsd, u16_value);
```

### \@xp162fs {#@xp162fs}

Convert a 16-bit integer into a 32-bit floating point value.

``` csl
@xp162fs(dest_dsd, src_dsd1);
@xp162fs(dest_dsd, i16_value);
@xp162fs(dest_dsd, u16_value);
```

### Example

``` csl
var tensor = [5]i16 {1, 2, 3, 4, 5};
const dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{5} -> tensor[i] });

fn foo() void {
  // Add the constant 10 to each element of `tensor`.
  // After executing this operation, `tensor` contains 11, 12, 13, 14, 15.
  @add16(dsd, dsd, 10);
}
```

### \@dfilt

Instructs an input queue to drop all data wavelets until a certain
number of control wavelets are encountered.

#### Syntax

``` csl
@dfilt(dsd, configuration);
```

Where:

- `dsd` is a `fabin_dsd` or DSR that contains a `fabin_DSD`.

  > - If a DSR is used, it must have type `dsr_src1` and be loaded with
  >   the `async` configuration (see `language-dsrs`{.interpreted-text
  >   role="ref"}). Behavior is undefined if `@dfilt` is used with a DSR
  >   that does not meet these conditions.

- `configuration` is the configuration struct that is optionally
  provided to other DSD operations (see
  `language-dsds`{.interpreted-text role="ref"}).

#### Semantics

The first argument to `@dfilt` must be a `fabin_dsd` or a DSR
representing an \'async\' `fabin_dsd`. A call to `@dfilt` will drop data
wavelets arriving on the input queue associated with the input DSD. The
`extent` of the DSD determines the number of control wavelets the
operation expects. The input queue will drop all data wavelets until the
specified number of control wavelets is encountered.

Unlike other DSD operations, the configuration struct is required, and
the `async` configuration must be `true`. `@dfilt` does not support the
`on_control` or `index` configurations.

#### Example

``` csl
var dsd = @get_dsd(fabin_dsd, .{ .fabric_color = 2, .extent = 10,
                                 .input_queue = @get_input_queue(1) });

fn foo() void {
   // Executing this operation causes input queue 1 to drop data wavelets
   // until 10 control wavelets have been encountered.
   @dfilt(dsd, .{ .async = true });
}
```
