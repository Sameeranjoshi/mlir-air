*(Original URL: https://sdk.cerebras.net/csl/Language/Advanced-Features.html)*

---

# Advanced Hardware Features {#language-advanced-features}

## Color Swapping {#language-advanced-features-color-swapping}

The hardware supports color \"swapping,\" a feature allowing incoming
data on one color to be sent out on another color. Two colors are
swappable if they differ only in the low bit (i.e. `x ^ y == 1`). Color
swapping is set per-color East-West (horizontal) or North-South
(vertical).

Consider a color pair consisting of colors 2 and 3. If East-West color
swapping is enabled for color 3, but not color 2, then wavelets arriving
from either the East or West on color 2 will be received on both color 2
and color 3.

If both color 2 and color 3 have East-West color swapping enabled, then
wavelets arriving from either the East or West on color 2 will be
received on color 3, and wavelets arriving from either the East or West
on color 3 will be received by color 2.

Note that the transmission is still subject to which directions are
enabled in the receive and transmit fields for the color. The behavior
described above also applies to North-South color swapping.

If *both* East-West color swap and North-South color swap are enabled
for a given color, color swap is also enabled for wavelets arriving from
the CE (onramp).

## Compute Element (CE) Injection

:::: warning
::: title
Warning
:::

This feature is only available on WSE-2, and is not supported on WSE-3.
::::

CE inject mode is a per-color setting that can be enabled at
compile-time. A CE inject configured color is hard wired to the tile\'s
output queue corresponding to the color number divided by 4. E.g. color
4 in this mode is connected to output queue 1. This is an integer floor
division, so e.g. if color 5 is in this mode it is also connected to
output queue 1. Note that this means in practice only one of every four
colors can be used in CE inject mode at once.

Either low priority or high priority can be selected when configuring
this mode. When either the fabric buffers are empty or the relevant
output queue is empty, these modes behave identically. When both are
non-empty, low priority mode will preference the fabric buffers, whereas
high priority mode will preference the relevant output queue.

To explain this more concretely, in low priority mode, the switch for
this color flips to position 1 when the fabric buffers are empty and
begins transmitting from the relevant output queue. It switches back to
position 0 if the fabric buffers ever become non-empty. In high priority
mode, the switch will flip to position 1 whenever the relevant output
queue is non-empty and will only flip back to position 0 when the output
queue becomes empty. In either mode, control wavelets on this color have
no effect; the switch position is controlled entirely by buffer and
queue occupancy.
