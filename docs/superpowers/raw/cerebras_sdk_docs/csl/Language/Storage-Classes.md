*(Original URL: https://sdk.cerebras.net/csl/Language/Storage-Classes.html)*

---

# Storage Classes {#language-syntax-storage-classes}

This document describes storage classes in CSL.

## Overview

By default, any variable declared in a CSL program is accessible only
within the program itself. This means that when linking two
separately-compiled objects `A` and `B`, variables declared in the
source code for `A` will not be visible in the source code for `B`, and
vice versa. It also means that tools manipulating symbols in the
compiled binary (such as `readelf` or the Cerebras `ELFLoader` class in
Python) will not necessarily be able to access the contents of the
symbol by name.

CSL features two storage classes for global variables and functions that
modify this default behavior: `extern` and `export`.

### extern

The `extern` storage class declares that a symbol for a variable or
function, with a certain type, is expected to be defined in an `export`
declaration elsewhere. That `export` declaration may either be in the
object file currently being compiled, or in another object file that
will be linked with the object being compiled. Declaring a variable with
`extern` makes that variable or function available for use within the
current source file, under the declared name.

A variable declaration with the `extern` storage class must *not* have
an initializer expression, and a function declaration with the `extern`
storage class must *not* have a body block.

Declarations with `extern` storage must have *export-compatible* types.
See
`language-syntax-storage-classes-type-restrictions`{.interpreted-text
role="ref"} for more details.

Currently, no calling convention for `extern` functions is formally
defined. Calling `extern` functions at comptime is not allowed.

Examples:

``` csl
extern var x: i16;
extern var y: *const fn(i16) i16;
extern const z: [*]f16;

extern fn f() void;
extern fn g(x: i16, y: [*]f16) i16;
```

### export

The `export` storage class defines a variable or function with a certain
name and type, and makes that variable or function available to other
object files that are linked with the object being compiled.

A variable declaration with the `export` storage class may have an
initializer expression, but it is not required. A function declaration
with the `export` storage class *must* have a body block.

Declarations with `export` storage must have *export-compatible* types.
See
`language-syntax-storage-classes-type-restrictions`{.interpreted-text
role="ref"} for more details.

Currently, no calling convention for `export` functions is formally
defined.

In addition to making symbols accessible by other objects, `export`
ensures that the symbol will be accessible via the Python `ELFLoader`
class (or other tools that operate on ELF files). If a symbol is
intended to be accessed either in the compiled binary or in a
post-execution core dump via `ELFLoader`, that symbol must be declared
`export`.

Examples:

``` csl
fn internal_func(x: i16) i16 { return x+1; }

export var x: i16 = 42;
export var y: *const fn(i16) i16 = &internal_func;
export const z: [*]f16;

export fn f() void { x += 1; }
export fn g(x: i16) i16 { return x+2; }
```

## Object file symbols for `extern` and `export` declarations

Unless a `linkname` is supplied (see below), the symbol name for an
`export` or `extern` declaration will be the same as its CSL variable
name, even if the variable is declared inside an imported module.
Additionally, `export` or `extern` symbols will be part of a single
global namespace shared with all other objects with which the compiled
object is linked. For example:

``` csl
///// Source file "root.csl"

// Symbol name in the compiled object file will be "x", and "x" will
// have global scope.
export var x: i16 = 32;

const m = @import_module("submodule.csl");


///// Source file "submodule.csl"

// Symbol name in the compiled object file will be "y", and "y" will
// have global scope.
export var y: i16 = 42;

// Symbol name in the compiled object file will be "z", and "z" will
// have global scope. Its actual definition is expected to be supplied by
// another object.
extern var z: i16;
```

## Interaction with `linkname`

Both `export` and `extern` declarations may be assigned a `linkname`. If
a `linkname` is present, that name is used for the symbol in the object
file. For example:

``` csl
// Symbol name in the compiled object will be "foo", and "foo" will have
// global scope, i.e., will be accessible by other objects.
export var x: i16 linkname("foo") = 99;

// Note that within the current program, the name "x" must still be used.
...
x += 1;    // valid
foo += 1;  // invalid
...
```

This can be useful when `export` declarations are present in code that
is intended to be imported with `@import_module`. Here, comptime
determination of the `linkname` can be used to avoid clashes among
symbol names. For example:

``` csl
///// Source file "root.csl"

const m1 = @import_module("submodule.csl", .{ .sym_name = "m1_foo" });
const m2 = @import_module("submodule.csl", .{ .sym_name = "m2_foo" });

...
m1.x  // corresponds to the object symbol "m1_foo"
m2.x  // corresponds to the object symbol "m2_foo"
...


///// Source file "submodule.csl"

param sym_name: comptime_string;

// Without the variable linkname, two imported copies of "submodule.csl"
// would always have symbol naming conflicts, since they would both
// attempt to define the object symbol "x".
export var x: i16 linkname(sym_name);
```

## Mixing `extern` and `export` declarations

It is legal to have multiple `extern` declarations with the same object
symbol name within a compiled program, and up to one `export`
declaration, as long as all declarations sharing a symbol name agree on
declaration kind (`var`, `const`, or `fn`) and have the same type.
Neither an `extern` nor an `export` declaration may share a symbol name
with a declaration that has no storage class (i.e., one which is neither
`export` nor `extern`).

For example:

``` csl
///// Source file "root.csl"

// The variable "x" will have a symbol name of "x", since no linkname is
// specified.
export var x: i16;

// The variable "y" will essentially be aliased to "x", since they share a
// symbol name.
extern var y: i16 linkname("x");

const m = @import_module("submodule.csl");


///// Source file "submodule.csl"

// The variable "x" will have a symbol name of "x", since no linkname is
// specified. Since it shares a symbol name with the variable "x" from
// root.csl, it will effectively be aliased to that variable.
extern var x: i16;

// The variable "z" will essentially be aliased to "x", since they share a
// symbol name.
extern var z: i16 linkname("x");

// Note that the following declarations would not be allowed.

// A declaration without a storage class cannot have the same symbol name
// as an 'export' or 'extern' declaration.
// var bad_var_decl: i16 linkname("x");

// All declarations sharing a symbol name must have the same type.
// extern var bad_type: i32 linkname("x");

// Variable declarations sharing a symbol name must match with respect to
// constness.
// extern const bad_constness: i16 linkname("x");
```

## Type restrictions {#language-syntax-storage-classes-type-restrictions}

Declarations with `export` or `extern` storage class must have a type
that is *export compatible*.

For variable declarations, export compatibility is defined as follows:

> - The built-in types `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`,
>   `u64`, `@fp16()`, `f32`, `bool`, and `color` are export compatible.
>   Note that `@fp16()` gives the type of the selected runtime FP16
>   format (see `language-builtins-fp16`{.interpreted-text role="ref"}).
> - An array type of the form `[N]T` is export compatible if its base
>   type `T` is export compatible.
> - A pointer type of the form `*T` or `*const T` is export compatible
>   if its base type is export compatible.
> - Function pointers are a special case: a pointer type of the form
>   `*const T`, where `T` is a function type, is export compatible if
>   `T` is export compatible for a function declaration (see below).
>   (The `const` qualifier is required.)
> - `enum` types are export compatible.

In particular, note that comptime-only types, `struct` types, DSD and
DSR types, the `direction` type, the `range` type, `void` (except in
function return type position\--see below), and function types (as
opposed to function [pointer]{#pointer}\_ types) are not export
compatible.

For function declarations, export compatibility is defined as follows:

> - A function type of the form `fn(t1, t2, ..., tn) t` is export
>   compatible if all of its argument types (`t1, t2, ..., tn`) are
>   export compatible for variable declarations (see above) *and* are
>   allowed for CSL function arguments, and its return type `t` either
>   is export compatible and allowed for CSL function return types, or
>   is `void`.

The restrictions here are motivated by ABI compatibility with C and
assembly code. Note, however, that there is currently no specific
calling convention defined for `extern` or `export` functions.
