*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-06-routes-1/device-code.html)*

---

What do we need to modify in our layout to distribute our GEMV between
two PEs?

1.  We need to define several new parameters for our two PE programs.
    This includes a `pe_id`, used to differentiate between the left and
    right PEs, and a color, which will be used to route data between the
    PEs.
2.  We need to set the color configuration on both PEs for the color
    that will be used to send the left PE\'s `y` array to the right PE.

Let\'s take a look at our new `layout.csl`, included below.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-06-routes-1/layout.csl
:::

We have two `@set_tile_code` calls, one for the left PE (0, 0), and one
for the right PE (1, 0). Both PEs take a new parameter, `N_per_PE`,
equal to `N / 2`. This is the number of columns of `A` that each PE will
receive and operate on. Both PEs also receive as a parameter a `pe_id`:
the left PE has `pe_id` 0, and the right PE has `pe_id` 1. We\'ll see in
`pe_program.csl` how we use `pe_id` to parameterize the behavior of the
program.

We also have two `@set_color_config` calls, to set the configuration of
`send_color` on each PE:

``` csl
@set_color_config(0, 0, send_color, .{.routes = .{ .rx = .{RAMP}, .tx = .{EAST} }});
...
@set_color_config(1, 0, send_color, .{.routes = .{ .rx = .{WEST}, .tx = .{RAMP} }});
```

The router of each PE has five directions: `RAMP`, `NORTH`, `SOUTH`,
`EAST`, `WEST`. The cardinal directions refer to the routers of
neighboring PEs: `NORTH` is the PE directly above our PE, and so on.
`RAMP` refers to the connection between our PE\'s router and its compute
element (CE). When setting a route for a color on a given PE, the
receive `rx` and transmit `tx` fields are from the perspective of the
router. Thus, receiving form the `RAMP` means that our compute element
is sending data up to the fabric, where it can then be transmitted
across the fabric.

For the left PE (0, 0), `send_color` will send up the PE\'s `RAMP` to
the fabric, and then transmit data to the `EAST`. For the right PE (1,
0), `send_color` will receive data from the `WEST` on the fabric (i.e.,
from the left PE), and then transmit it down the `RAMP` to its compute
element.

Now let\'s take a look at our new `pe_program.csl`, included below.

::: {.literalinclude language="csl"}
../../code-examples/tutorials/gemv-06-routes-1/pe_program.csl
:::

In addition to our new parameters `N_per_PE`, `pe_id`, and `send_color`,
we also introduce `exit_task_id`, our first value of type
`local_task_id`. We\'ll talk about its use a bit later.

The `A` array now has size `M*N_per_PE` instead of `M*N`, since each PE
only stores half the columns. To make our data transfer easier, we also
now store `A` column-major instead of row-major. Notice that `A_dsd` now
accesses `M` contiguous elements, instead of `M` elements strided by the
row size, since we now store column-major.

Our `gemv` function operates almost identically to before, except we
only loop over `N_per_PE` columns instead of `N` columns. Since `A` is
now column-major, `@increment_dsd_offset` must increment by the length
of an entire column instead of by one element. Note that on the left PE,
`y` already contains the elements of `b` before `gemv` executes.

# Fabric DSDs and async operations

The `compute` function, which is called from the host, first calls
`gemv` to compute the local contribution to `y` on each PE. Then, the
left PE calls `send_right`, while the right PE calls `recv_left`.

`send_left` defines a `fabout_dsd`, which is used to send wavelets to
the fabric along the color `send_color`. Note that we give this
`fabout_dsd` the extent `M`, since we intend to send the `M` elements of
`y` along the fabric. The `@fmovs` operation copies the `M` elements
accessed by `y_dsd` into `out_dsd`. The `.async = true` field makes this
operation asynchronous. The `.activate` field specifies a
`local_task_id` to activate when this operation completes. When this
operation completes, `exit_task_id` will be activated.

`recv_right` defines a `fabin_dsd` to receive the wavelets sent along
`send_color`. The `@fadds` operation here increments the right PE\'s
`y_dsd` by the elements received in `in_dsd`. Thus, after this
operation, `y_dsd` contains our final GEMV result. This builtin also
executes asynchronously, and actives `exit_task_id` when complete.

:::: warning
::: title
Warning
:::

Whenever using fabric DSDs in builtin operations, always make these
operations execute asynchronously. Using fabric DSDs synchronously can
result in poor performance or deadlocks.
::::

# Tasks and activatable task IDs

Now, what does activating `exit_task_id` do? In the comptime block, the
`@bind_local_task` builtin binds `exit_task_id` to the task `exit_task`.
When `exit_task_id` is activated, `exit_task`, which unblocks the
`memcpy` command stream, executes. This task must execute on both PEs
before control is returned to the host.
