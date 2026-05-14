*(Original URL: https://sdk.cerebras.net/csl/Language/Libraries_wse3.html)*

---

# Libraries for WSE-3 {#language-libraries-wse3}

This section documents CSL libraries that are only supported for WSE-3.

## \<message_passing\> {#language-libraries-message-passing}

The `message_passing` library provides functions for point-to-point
communication between PEs on the WSE-3 architecture.

By default, the library uses colors 14, 15, 16, 17, and 18 for message
passing. These colors may not be used for any other purpose while
message passing is enabled.

The library consists of one module, imported into the PE program. When
importing the library an input and output queue ID must be specified.
Additionally, an optional boolean `enable_asserts` parameter may be set
to `true` to enable runtime correctness checks on message legnth and
destination.

``` csl
const mp = @import_module("<message_passing>", .{ .enable_asserts = true,
                          .input_queue_id = 2, .output_queue_id = 3 });
```

The following functions and constants are provided by the library:

``` csl
// Input and output queues used by the library, specified by
// the input_queue_id and output_queue_id parameters
const mp_input_queue: input_queue;
const mp_output_queue: output_queue;

// Routable color into which a receiver receives messages
const mp_comm_color: color;

// Initialize the message passing library
fn init();

// Send message of length len from buf to PE at (dest_col, dest_row)
fn send_message(dest_col: u16, dest_row: u16, buf: anytype, len: u16,
                comptime on_term: comptime_struct);

// Receive message into buffer buf
fn recv_message(buf: anytype, comptime on_term: comptime_struct);
```

`init()` initializes the library. This function must be called before
sending or receiving any messages.

`send_message` launches a microthreaded operation that sends a message
of a fixed length to another PE, and upon completion, either activates
or unblocks a callback local task. It has the following arguments:

- `dest_col`: the column ID of the receiving PE within the program
  rectangle.
- `dest_row`: the row ID of the receiving PE within the program
  rectangle.
- `buf`: pointer to an array with a 16-bit or 32-bit element type, i.e.,
  `i16`, `u16`, `@fp16()`, `i32`, `u32`, or `f32`.
- `len`: number of elements to send. `buf` must point to at least `len`
  elements or the operation is undefined.
- `on_term`: a `comptime_struct` of the form `.{ .<ACTION> = <ID> }`,
  where `ACTION` is an `activate` or `unblock` performed at the end of
  the microthreaded send operation, and ID is a task or queue ID to be
  activated or unblocked. Passing `{}` indicates no termination behavior
  is desired.

`recv_message` launches a microthreaded operation that receives a
message sent from another PE, and upon completion, either activates or
unblocks a callback to a local task. It has the following arguments:

- `buf`: pointer to an array with a 16-bit or 32-bit element type, i.e.,
  `i16`, `u16`, `@fp16()`, `i32`, `u32`, or `f32`.
- `action`: specifies whether to unblock or activate the callback.
- `on_term`: a `comptime_struct` of the form `.{ .<ACTION> = <ID> }`,
  where `ACTION` is an `activate` or `unblock` performed at the end of
  the microthreaded receive operation, and ID is a task or queue ID to
  be activated or unblocked. Passing `{}` indicates no termination
  behavior is desired.

The length of the received message is determined by the message header
added to the message when it is sent. `buf` must point to an array large
enough to store the received message.

A message can also be received using a `data_task` bound to
`mp_input_queue`. The last wavelet in a message, which has a control bit
set, will NOT be received by the `data_task`. If the sender sends a
message of length `N`, the receiver\'s `data_task` will receive the
header, followed by `N-1` elements of the message. Thus, the sender must
pad the sent message with one additional wavelet at the end in this
case.

## \<tile_config\> {#language-libraries-wse3-tile-config}

The `tile_config` library contains APIs related to the hardware
configuration of a PE. This section documents the submodules of the
`tile_config` library that are only supported for WSE-3. For more
information about the `tile_config` library in general see
`language-libraries-tile-config`{.interpreted-text role="ref"}.

### `queue_flush` {#language-libraries-wse3-tile-config-queue-flush}

This submodule of `tile_config` contains APIs that allow us to read and
modify the queue flush status register. This register can tell us
whether the `@queue_flush` (see
`language-builtins-wse3-qflush`{.interpreted-text role="ref"}) builtin
operation has been executed and the respective queue has been flushed
(i.e., it is now empty).

``` csl
// Reset the status of the queue flush status register that corresponds
// to `qid`. Depending on the type of `qid` this could be the input or
// output queue flush status register.
fn exit(comptime qid : anytype) void

// Return the value of the "input queue flush status" register.
fn iq_status() reg_type

// Return the value of the "output queue flush status" register.
fn oq_status() reg_type

// Given a value that represents the "queue flush status" register, which
// has 1 bit per queue indicating the ones that have been flushed, return
// `true` iff the queue id `q` has been flushed.
fn is_flushed(value: reg_type, comptime qid: anytype) bool
```
