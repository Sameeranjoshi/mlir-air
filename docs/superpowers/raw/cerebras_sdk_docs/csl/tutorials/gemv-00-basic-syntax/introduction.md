*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax/introduction.html)*

---

CSL is a language for writing programs that run on the Cerebras Wafer
Scale Engine (WSE). The WSE consists of hundreds of thousands of
independent processing elements (PEs). Each PE has room for a small
program and some data. CSL is designed to help you handle the multi-PE
nature of the WSE and the specific challenges of writing dataflow
programs for the Cerebras hardware.

The design of CSL is based heavily on [Zig](https://ziglang.org), a
general-purpose language with powerful compile-time programming
constructs. Zig was chosen as the basis for CSL since its compile-time
facilities make it possible to write maintainable yet highly performant
code for PEs. Note, however, that while CSL\'s syntax and semantics are
very similar to Zig, CSL is not 100% compatible with Zig. Some features
of Zig are not implemented in CSL, and CSL also includes some features
that are not present in Zig.

:::: note
::: title
Note
:::

The CSL language is not 100% compatible with Zig, and its compiler does
not share any code with the Zig compiler. Some of the CSL documentation
is derived from the Zig documentation. Any bug reports or other feedback
on CSL, including its documentation, should be directed to Cerebras, and
not to the maintainers of Zig or to Zig community forums.
::::

# Types

CSL includes some basic types such as:

- `bool` for boolean values
- `i16` and `i32` for 16- and 32-bit signed integers
- `u16` and `u32` for 16- and 32-bit unsigned integers
- `f16` and `f32` for 16- and 32-bit IEEE-754 floating point numbers

In addition to the above, CSL also supports array types and pointer
types.

# Functions

Functions are declared using the `fn` keyword. The compiler provides
special functions called *Builtins*, whose names start with `@` and
whose implementation is provided by the compiler. All CSL builtins are
described in `language-builtins`{.interpreted-text role="ref"}.

# Conditional Statements and Loops

CSL includes support for `if` statements and `while` and `for` loops.
