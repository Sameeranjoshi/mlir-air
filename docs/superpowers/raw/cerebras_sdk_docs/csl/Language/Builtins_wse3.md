*(Original URL: https://sdk.cerebras.net/csl/Language/Builtins_wse3.html)*

---

# Builtins for WSE-3 {#language-builtins-wse3}

This section documents CSL builtins that are only supported for WSE-3.

## \@get_ut_id {#language-builtins-wse3-get-ut-id}

Create a value of type `ut_id` from the provided integer identifier.

### Syntax

``` csl
@get_ut_id(id);
```

Where:

- `id` is a comptime-known expression of any unsigned integer type, or a
  runtime expression of type `u16`.

### Semantics

If `id` is comptime-known, the builtin will only accept integers in the
target architecture\'s valid range for microthread IDs.

If `id` is not comptime-known, its type must be `u16`. In this case no
runtime checks are performed to ensure that `id` is within the range of
valid microthread IDs.

## \@queue_flush {#language-builtins-wse3-qflush}

Activates the teardown task ID (see
`language-libraries-tile-config-teardown`{.interpreted-text role="ref"})
once the given queue becomes empty.

### Syntax

``` csl
@queue_flush(queue_id);
```

Where:

- `queue_id` is an expression of type `input_queue` or `output_queue`.

### Example

``` csl
const out_q = @get_output_queue(2);
const fabout = @get_dsd(fabout_dsd, .{..., .output_queue = out_q});

task foo() void {
  @mov16(fabout, ..., .{.async = true});

  // Will activate the teardown task once 'out_q' is empty.
  // Note that 'out_q' may not be empty when the microthread
  // above is done.
  @queue_flush(out_q);
}
```

### Semantics

The builtin can only be evaluated at runtime, and therefore it cannot
appear during comptime evaluation or during the evaluation of a
top-level comptime or layout block.

The builtin\'s return type is `void`.

Calling the builtin will cause the teardown task to be activated once
the input (or output) queue associated with `queue_id` becomes empty or
is already empty.

It is guaranteed that if the teardown task is activated because of a
call to `@queue_flush` then the respective queue will be empty.

From within the teardown task the `queue_flush` library (see
`language-libraries-wse3-tile-config-queue-flush`{.interpreted-text
role="ref"}) can be used to check whether the teardown task was
activated due to a call to `queue_flush` for a given queue or not.

## \@set_control_task_table

Create a separate control task table.

### Syntax

``` csl
@set_control_task_table();
@set_control_task_table(config);
```

Where:

- `config` is a comptime-known (anonymous) struct with the following
  optional fields:
  - `instructions`
  - `stride`

### Example

``` csl
task foo() void {}
task bar() void {}

comptime {
  @bind_control_task(foo, @get_control_task_id(10));
  @bind_local_task(bar, @get_local_task_id(10));

  // Even though 'foo' and 'bar' have the same IDs they do not
  // clash because this call to @set_control_task_table will
  // decouple control tasks from data and local tasks. Control
  // tasks now have their own separate task table.
  @set_control_task_table(.{.instructions = 8, .stride = 4});
}
```

### Semantics

The `@set_control_task_table` builtin will decouple control tasks from
data and local tasks by creating a separate task table that is dedicated
to control tasks.

The builtin can only be called at most once during the evaluation of a
top-level comptime block.

It can have an optional argument that must be a comptime-known struct
with two optional fields: `instructions` and `stride`.

If `@set_control_task_table` is called without an argument then the
default values for `instructions` and `stride` will be used (see below).

The `instructions` field can be used to specify the number of
instructions for each entry point in the new control task table. The
number of instructions must be a comptime-known integer value within the
valid set of options which are `2`, `4` and `8`. The default value is
`4`.

The `stride` field requires a comptime-known integer value that
represents the stride - in number of entry points - per input queue\'s
local control table index (see
`language-builtins-initialize-queue`{.interpreted-text role="ref"}). Its
value should be in the range `[1, 7]` and the default value is `1`.

## \@set_empty_queue_handler

Set a function to be the empty queue handler for a given queue.

### Syntax

``` csl
@set_empty_queue_handler(func, queue_id);
```

Where:

- `func` is the name of a function with no input parameters and `void`
  return type.
- `queue_id` is a comptime-known expression of type `input_queue` or
  `output_queue`.

### Example

``` csl
const tile_config = @import_module("<tile_config>");

fn foo() void {
  // Reconfiguration of empty queue takes place here.
  ...
  // At the end, ensure that the queue flush status register is reset
  // to prevent re-entrancy.
  tile_config.queue_flush.exit();
}
const in_q = @get_input_queue(4);

comptime {
  // Specifies function 'foo' to be executed when 'in_q' is flushed
  // (i.e., becomes empty) after calling '@queue_flush(in_q)'.
  @set_empty_queue_handler(foo, in_q);
}
```

### Semantics

The `@set_empty_queue_handler` builtin must appear in a top-level
`comptime` block. When `@queue_flush` (see
`language-builtins-wse3-qflush`{.interpreted-text role="ref"}) has been
called for a qiven queue (input or output) and the teardown task is
activated due to that queue becoming empty, then the function associated
with that queue, through a call to `@set_empty_queue_handler`, will be
executed.

The user is responsible for resetting the status of the queue flush
status register through the `queue_flush` submodule of the
`<tile_config>` library (see
`language-libraries-wse3-tile-config-queue-flush`{.interpreted-text
role="ref"}).

Calling `@set_empty_queue_handler` for the same queue more than once is
not allowed and will result in an error.

If there is at least 1 call to `@set_empty_queue_handler` in the program
then no task is allowed to be bound to the *teardown task ID* and
vice-versa. The teardown task ID is the value returned by the CSL
standard library through the teardown API (see
`language-libraries-tile-config-teardown`{.interpreted-text
role="ref"}).
