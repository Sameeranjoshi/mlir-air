*(Original URL: https://sdk.cerebras.net/csl/Language/Microthreads_wse3.html)*

---

# Microthread IDs {#language-microthreads-wse3}

On WSE-3 it is possible to explicitly specify the microthread ID of an
asynchronous DSD operation (i.e., a DSD operation with the `.async`
setting set to `true` as explained in
`language-dsds-async`{.interpreted-text role="ref"}).

## Constructor and Type

Microthread ID expressions have type `ut_id`. Values of `ut_id` type can
be constructed from integers using `@get_ut_id` (see
`language-builtins-wse3-get-ut-id`{.interpreted-text role="ref"}).

## Usage and Semantics

An asynchronous DSD operation can be assigned a microthread ID through
the `.ut_id` setting of the DSD operation\'s configuration struct or
through the `.ut_id` setting of the `@load_to_dsr`\'s configuration
struct of explicit DSR operands. The value of the `.ut_id` setting must
be comptime-known.

If multiple DSR operands have the `.ut_id` setting specified then the
hardware will pick one of them according to the following order:

- Destination operand
- First source operand
- Second source operand

The same order will be followed even if a DSD operation consists of a
mix of explicit DSR and DSD operands. In this case the DSD operands will
have the microthread ID specified by the `.ut_id` setting of the DSD
operation\'s configuration struct.

If the `.ut_id` setting is not specified in the DSD operation\'s
configuration struct or by any of the explicit DSR operands, then the
microthread ID will be the same as the queue ID of the first fabric
operand according to the operand ordering defined above.

## Example

``` csl
const ut0 = @get_ut_id(0);
const ut1 = @get_ut_id(1);
const ut2 = @get_ut_id(2);
const ut3 = @get_ut_id(3);
const ut4 = @get_ut_id(4);

// These asynchronous DSD operations use two different
// microthreads (i.e., ut0 and ut1) which means that they
// can be executed concurrently even if their queues
// are the same.
@mov16(out_dsd, in_dsd, .{.async = true, .ut_id = ut0});
@mov16(out_dsd, in_dsd, .{.async = true, .ut_id = ut1});

// In the case of explicit DSRs, the microthread ID can be
// specified as a setting to the @load_to_dsr calls.
@load_to_dsr(out_dsr, out_dsd, .{.async = true, .ut_id = ut2});
@load_to_dsr(in_dsr, in_dsd, .{.async = true, .ut_id = ut3});

// This operation will use the microthread ID specified by
// out_dsr according to the .ut_id setting of the respective
// @load_to_dsr call which is ut2. The microthread ID ut4
// will be ignored in this case.
@mov16(out_dsr, in_dsd, .{.async = true, .ut_id = ut4});

// This operation will use the microthread ID specified by
// the operations's .ut_id setting which is ut4. That's because
// out_dsd takes priority over in_dsr.
@mov16(out_dsd, in_dsr, .{.async = true, .ut_id = ut4});

// The .ut_id setting is not required. In this scenario, the
// microthread ID will be the same as the queue ID of the
// highest priority fabric operand (i.e., destination > first source >
// second source ... etc).
@mov16(out_dsd, in_dsd, .{.async = true});
```

## Blocking and Unblocking Microthreads

Microthreads can be blocked/unblocked from a top-level comptime block or
at runtime using the `@block` and `@unblock` builtins (see
`language-builtins-block`{.interpreted-text role="ref"} and
`language-builtins-unblock`{.interpreted-text role="ref"}). For example:

``` csl
const ut0 = @get_ut_id(0);
var ut: ut_id;

task main() void {
  // It is possible to block/unblock a microthread
  // at runtime. The microthread ID may be comptime-known
  // or not.
  @unblock(ut0);
  @block(ut);
}

comptime {
 // Set microthread 'ut0' as blocked when the program starts.
 @block(ut0);
}
```
