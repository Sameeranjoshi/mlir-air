*(Original URL: https://sdk.cerebras.net/csl/Language/Types.html)*

---

# Type System in CSL {#language-types}

## `void` type

Expressions of type `void` have a single possible value. It describes
constructs that do not produce a result. For example, blocks which do
not break values have type `void` and `void` is the return type of
functions and builtins that do not return anything. Using a block as an
expression, the void value can be expressed with `{}`:

``` csl
const void_val = {}; // type is void.
const also_void_val : void = foo();

fn foo() void {}
```

`void` is allowed at runtime as a function return type. All other uses
of `void` values must be comptime-known; see
`language-comptime`{.interpreted-text role="ref"} for more information
on comptime.

## Numeric Types

These types describe numbers:

- signed integers (`i8`, `i16`, `i32`, `i64`)
- unsigned integers (`u8`, `u16`, `u32`, `u64`)
- arbitrary precision integers (`comptime_int`)
- floating point numbers (`f16`, `f32`, `bf16`, `cb16`,
  `comptime_float`)

### The `comptime_int` Type

Values of `comptime_int` type can hold arbitrarily large (or small)
integers. Integer literals have type `comptime_int`:

``` csl
const ten = 10; // type is comptime_int.
const also_ten : comptime_int = 10;
```

Character literals also have type `comptime_int`:

``` csl
const some_char = 'a';          // type is comptime_int, value is 97
                                // (ASCII value of 'a')
const some_other_char = '\x0a'; // type is comptime_int, value is 10
                                // (i.e., 0x0a, in hexadecimal)
const yet_another_char = '\n';  // type is comptime_int, value is 10
                                // (ASCII value of newline character)
```

Arithmetic between `comptime_int` values happen at compile time, produce
another `comptime_int` value, and never underflow or overflow:

``` csl
const thousand = 1000;
const trillion = thousand * thousand * thousand * thousand;
const one = trillion / trillion;
```

Operations between a value of type `comptime_int` and a value of
fixed-precision integer type cause the `comptime_int` value to be
converted to the fixed-precision type. An error is emitted upon overflow
or underflow:

``` csl
const thousand = 1000; // comptime_int.
const ten : i16 = 10;  // 10 is converted from comptime_int to i16.
const hundred : i16 = thousand / 10; // thousand is converted to i16.

const overflow : i16 = 10000000000000; // error.
```

The builtin `@as` can be used to force literals to have a specific
width, for example `@as(u16, 1) | 0xbeef`.

All values of type `comptime_int` must be comptime-known, see
`language-comptime`{.interpreted-text role="ref"} for more information
on comptime.

### The `comptime_float` Type

Values of `comptime_float` type can hold any IEEE double precision
floating point number. Float literals have type `comptime_float`:

``` csl
const ten = 10.0; // type is comptime_float.
const also_ten : comptime_float = 10.0;
```

Arithmetic between `comptime_float` values happen at compile time and
produce another `comptime_float` value. If an operation performs
division by zero or generates a `NaN` value, an error is emitted.

Operations between a value of type `comptime_float` and a value of
different float type cause the `comptime_float` value to be converted to
other float type. An error is emitted if this is not possible.

All values of type `comptime_float` must be comptime-known, see
`language-comptime`{.interpreted-text role="ref"} for more information
on comptime.

### FP16 Types

`f16`, `cb16`, and `bf16` are 16-bit floating point (\"FP16\") types of
different formats:

  Type     Description           Exponent                  Mantissa
  -------- --------------------- ------------------------- ----------
  `f16`    IEEE half-precision   5 bits                    10 bits
  `cb16`   Cerebras float16      6 bits, customized bias   9 bits
  `bf16`   Brain float16         8 bits                    7 bits

``` csl
const ieee_six: f16 = 6.0;
const cb_six: cb16 = 6.0;
const bf_six: bf16 = 6.0;

comptime {
   @comptime_assert(@bitcast(u16, ieee_six) == 0b0100011000000000);
   @comptime_assert(@bitcast(u16, cb_six)   == 0b0101100100000000);
   @comptime_assert(@bitcast(u16, bf_six)   == 0b0100000011000000);
}
```

CSL supports use of *a single FP16 type* within the runtime code of a
program. This type is chosen by the value of the `--fp16-format` command
line option, with a default of `f16`. All values of the other FP16 types
must be comptime-known:

``` csl
// With --fp16-format=f16 or --fp16-format absent, OK
// Otherwise, error: variable of type 'f16' must be comptime-known
var ieee_one: f16 = 1.0;

// With --fp16-format=cb16, OK
// Otherwise, error: variable of type 'cb16' must be comptime-known
var cb_one: cb16 = 1.0;

// With --fp16-format=bf16, OK
// Otherwise, error: variable of type 'bf16' must be comptime-known
var bf_one: bf16 = 1.0;
```

The `@fp16()` builtin (see `language-builtins-fp16`{.interpreted-text
role="ref"}) facilitates programming with respect to the selected FP16
format.

## The `type` Type

In CSL, the type `type` can be used to describe values that are
themselves types:

``` csl
const my_type = i16;
const same_type : type = i16;
```

These values can be used anywhere a type is expected.

``` csl
const my_type = i16;
const array = @zeros([10]my_type);

fn foo() my_type { ... }
```

All values of type `type` must be comptime-known, see
`language-comptime`{.interpreted-text role="ref"} for more information
on comptime.

## Function Types

Values of function type contain the name of a function and may be used
anywhere a function is expected. The type is written as
`fn(<arg types>) <return_type>`.

``` csl
fn foo(arg1 : i16, arg2 : f16) void {...}

const also_foo1 = foo;
const also_foo2 : fn(i16, f16) void = foo;

also_foo1(10, 10.0);
also_foo2(20, 20.0);
```

Copying a function value does not create a new function, it copies the
name of the function.

All values of function type must be comptime-known, see
`language-comptime`{.interpreted-text role="ref"} for more information
on comptime.

## Struct Types

There are three kinds of struct types in CSL: anonymous structs, named
structs, and `comptime_struct`.

### The Anonymous Struct Types

Anonymous struct types are defined by an optional *list* of field names
and a *list* of types. Two anonymous struct types are the same if they
have the same list of field names (or both lack field names) and the
same list of types.

A value of an anonymous struct type with named fields is created with
the syntax: `.{.field1 = value1, .field2 = value2, ...}`. A value of an
anonymous struct type with unnamed fields is created with the syntax:
`.{value1, value2, ...}`.

Anonymous struct types with nameless fields are also known as *tuple*
types. The elements of tuples may be accessed with the `[]` operator, as
long as the index is known at compile time.

``` csl
// Type: {a : comptime_int, b : comptime_float}
var struct1 = .{.a = 10, .b = 1.0};
var struct2 = .{.a = 20, .b = 2.0};
var struct1 = struct2; // ok, same type!


// Type: {a : comptime_float, b : comptime_float}
var struct3 = .{.a = 10.0, .b = 1.0};
struct3 = struct1;           // error: different types!
var some_float = struct3[1]; // error: struct3 is not a tuple type!

// Type: {comptime_float, comptime_float}
var struct4 = .{10.0, 1.0};
var some_float = struct4[0];       // ok!
var some_other_float = struct4[2]; // error: index 2 is out of bounds!
task t(i: u16) void {
    var yet_another_float = struct4[i]; // error: i is not known
                                        // at comptime!
}
```

Currently, it is not possible to spell out anonymous struct types using
CSL syntax.

### The Named Struct Types

Named struct types are similar to anonymous struct types, except that
two named struct types defined at different places in the source code
are considered to be different types, even if their field names and
types are the same.

A named struct type is expressed with the form
`struct { field1: type1, field2: type2, ... }`. Once a named struct type
has been defined, a value of that type can be created by giving the name
of the type, followed by a field initializer list of the form
`{ .field1 = value1, .field2 = value2, ... }`.

``` csl
const complex = struct {
  real_part: f16,
  imag_part: f16
};

const one = complex { .real_part = 1.0, .imag_part = 0.0 };
const zero = complex { .real_part = 0.0, .imag_part = 0.0 };
```

As mentioned above, two named struct types are considered equal if and
only if they have identical field names and types [and]{#and}\_ they
were both defined at the same point in the program (i.e., by the same
`struct` expression).

``` csl
const some_struct = struct {
  x: i16,
  y: i16
};

const some_other_struct = struct {
  x: i16,
  y: i16
};

comptime {
  @comptime_assert(@is_same_type(some_struct, some_struct));
  @comptime_assert(@is_same_type(some_other_struct, some_other_struct));

  // Although some_struct and some_other_struct have the same field names and
  // types, they are *not* the same type, because they were defined at
  // different source locations.
  @comptime_assert(!@is_same_type(some_struct, some_other_struct));
}
```

Named struct types can also be returned from functions. When combined
with comptime `type` arguments, this can be used to define parameterized
struct types, whose field types can be customized by the user.

``` csl
fn pair(comptime T1: type, comptime T2: type) type {
  return struct {
    first: T1,
    second: T2
  };
}

const my_pair = pair(i32, comptime_string) {
  .first = 42,
  .second = "this is a struct"
};

comptime {
  @comptime_assert(@is_same_type(@type_of(my_pair),
                                 pair(i32, comptime_string)));
  @comptime_assert(@is_same_type(@type_of(my_pair.first), i32));
  @comptime_assert(@is_same_type(@type_of(my_pair.second), comptime_string));
}
```

The fields of a struct can be mutated by assigning to them via `.`
syntax.

``` csl
const s = struct {
   x: i32,
   y: i32
};

comptime {
  var my_s = s {
    .x = 15,
    .y = 99
  };

  @comptime_assert(my_s.x == 15);
  @comptime_assert(my_s.y == 99);

  my_s.x = 0;
  my_s.y = 33;

  @comptime_assert(my_s.x == 0);
  @comptime_assert(my_s.y == 33);
}
```

### The `comptime_struct` Type

The `comptime_struct` type serves as a superset of all struct types: any
value of any struct type, whether anonymous or named, may be assigned to
a variable of type `comptime_struct`:

``` csl
comptime {
  // Type: {a : comptime_int, b : comptime_float}
  var some_struct = .{.a = 10, .b = 1.0};

  var any_struct : comptime_struct = some_struct; // ok!

  // Type: {name: comptime_string}
  const different_struct = .{.name = "Bob"};
  any_struct = different_struct; // ok!

  const some_named_struct_ty = struct { name: comptime_string, age: i32 };
  any_struct = some_named_struct_ty { .name = "Mary", .age = 61 }; // ok!
}
```

All values of type `comptime_struct` must be comptime-known, see
`language-comptime`{.interpreted-text role="ref"} for more information
on comptime.

## Enumeration Types

An enumeration type is a set of named elements, each of which is
represented by a unique integer value:

``` csl
const colors = enum(u16) { red, white, blue };
const favorite = colors.red;
```

The underlying integer type is specified by the type argument (`u16` in
the example above) and it can be any fixed-precision integer type (e.g.
`i16` or `u32`).

Any element can be assigned a comptime-known integer value:

``` csl
const colors = enum(u16) { red, white = 1, blue };
```

The values assigned must be unique within the type. Any element not
assigned a value will be assigned a value by the compiler. The compiler
assigns values from left to right, using consecutive integers starting
with zero. In the example above, the compiler will assign the value `0`
to `red` and the value `2` to `blue`. If `white` is instead assigned the
value `4`, then the compiler will assign the value `0` to `red` and `1`
to `blue`.

Enumeration type values can be cast to and from their underlying numeric
values using the `@as()` builtin, as the following assertions
demonstrate:

``` csl
@comptime_assert(@as(i16, colors.red) == 0);
@comptime_assert(@as(u16, colors.white) == 1);
@comptime_assert(@as(f32, colors.blue) == 2.0);
@comptime_assert(@as(colors, 1) == colors.white);
```

An enumeration type can be cast to and from any numeric type regardless
of its underlying integer type, subject only to the general
compatibility rules of type casts.

An enumeration value cannot be cast directly to a value of another
enumeration type, but the same effect can be achieved by casting via a
numeric type:

``` csl
const game = enum(i16) {rock, paper = -3, scissors};
// @as(color, game.rock)  <= Error!
const myred = @as(colors, @as(u16, game.rock)); // OK
@comptime_assert(@as(colors, as(i16, colors.white) + 1)) == colors.blue);
```

Two expressions of the same enumeration type can be compared using the
`==` and `!=` operators.

### Enumeration Type Equality

Two enumeration types are the same if and only if both of the following
conditions are true:

- they have the same structure, i.e., the same underlying integer type,
  and the same set of element values, each of which is assigned the same
  numeric value.
- their definitions originate at the same source code location.

``` csl
const e1 = enum(i16) {red, white};
const e2 = enum(i16) {red, white};

fn enum_type(base_type: type) type {
  return enum(base_type) {red, white};
};

const e3 = enum_type(i16);
const e4 = enum_type(i32);
const e5 = e1;
const myred = e1.red;

// different types, not originating from the same location:
@comptime_assert(!@is_same_type(e1, e2));
// different types, same originating location, but not same structure
@comptime_assert(!@is_same_type(e3, e4));
@comptime_assert(@is_same_type(e5, e1)); // same type
@comptime_assert(@is_same_type(@type_of(myred), e5); // same type
```

## Array Types

An array type is parameterized by an element type, describing a
collection of elements of the base type. An array type whose element
type is `T` can be written as `[size]T`.

The element type must not be another array type. Multidimensional arrays
are specified with a sequence of dimensions: `[size1, size2, size3]T`.

``` csl
var 1d_array = @zeros([10]u16);
var 2d_array = @zeros([10, 10]u16);
var another_array = [2]i16 {1,2}
```

## Pointer Types

A value of pointer type contains the memory address where a variable is
located.

Pointer types are described by an element type and an optional `const`
qualifier.

In CSL, pointers are *only* created by taking the address of a variable.
This provides the property that pointers always point to valid data when
they are created.

There are two kinds of pointers in CSL: pointers to a single element and
pointers to an unknown number of elements.

### Pointers to a Single Element

A pointer to a single value of type `T` is written as `*T`. For example:

- a pointer to a single `i16` is written as `*i16`
- a pointer to a single array of ten integers is written as `*[10]i16`.

Pointers to a single element are created with the address-of operator
(`&`):

``` csl
var array = @zeros([10]u16);
const ptr = &array;
const same_ptr:*[10]u16 = &array;
```

The only operation allowed on pointers to a single element is to
dereference them with the dereference `.*` operator:

``` csl
var array = @zeros([10]i32);
const ptr = &array;
ptr.* = @constants([10]i32, 42);
ptr.*[2] = 1;
```

Pointer types may be `const` qualified, indicating that this pointer may
not be used to modify the underlying memory:

``` csl
var array = @zeros([10]i32);
var ptr:*const[10]i32 = &array;
ptr.* = @constants([10]i32, 42); // Error: pointer type is const qualified.
ptr.*[2] = 1; // Error: pointer type is const qualified.
```

``` csl
const array = @zeros([10]i32);
var ptr = &array;  // Type of ptr is *const[10]i32
ptr.* = @constants([10]i32, 42); // Error: pointer type is const qualified.
ptr.*[2] = 1; // Error: pointer type is const qualified.
```

In the example above, `ptr` itself is mutable, but the memory it points
to is not.

### Pointers to Unknown Number of Elements

A pointer to an unknown number of elements of type `T` is written as
`[*]T`. For example, a pointer to an unknown number of `f16` elements is
written as `[*]f16`.

Pointers to an unknown number of elements are created through coercion
from pointers to a single element of array type (e.g. `*[2]i16`):

``` csl
fn foo(ptr : [*]i32) void {
 // ...
}

var array10 = @zeros([10]i32);
var array20 = @zeros([20]i32);
foo(&array10); // ok!
foo(&array20); // ok!

var array_float = @zeros([20]f16);
foo(&array_float); // Error: base type mismatch.

var ptr : [*]i32 = &array10;
ptr = &array20;
```

The original array must have rank one:

``` csl
var array = @zeros([3,3,3]i32);
var ptr : [*]i32 = &array; // Error.
```

Dereferencing a pointer to an unknown number of elements is not allowed.
The access operator `[]` must be used instead:

``` csl
fn foo(ptr : [*]i32) void {
  ptr.* = 10; // Error: dereferencing is not allowed.
  ptr[0] = 10;
}
```

It is illegal to access an element whose index is out-of-bounds on the
original array:

``` csl
fn foo(ptr : [*]i32) void {
  ptr[1000] = 10;
}

var array10 = @zeros([10]i32);
foo(&array10); // Bad: will create out-of-bounds access.
```

An error is emitted if the compiler is able to detect an out-of-bounds
access.

### Pointers and Configuration Memory

It is illegal to dereference or use the access operator `[]` on pointers
occurring in the selected target\'s configuration address range. An
error is emitted if the compiler is able to detect such an access.
Configuration memory should be accessed using the builtins `@get_config`
and `@set_config` instead (see
`language-builtins-get-config`{.interpreted-text role="ref"},
`language-builtins-set-config`{.interpreted-text role="ref"}).

## The `comptime_string` Type

:::: warning
::: title
Warning
:::

Support for compile-time strings is still experimental, and the set of
operations available on the type `comptime_string` is very limited.
::::

Values of `comptime_string` type hold immutable strings that can be
manipulated at compile time. All values of type `comptime_string` must
be comptime-known. See ref:\`language-comptime\` for more information on
comptime.

``` csl
const hello = "abc"; // type is comptime_string

fn bool_to_str(b: bool) comptime_string {
   if (b) {
     return "true";
   } else {
     return "false";
   }
}

comptime {
  const true_str = bool_to_str(true);
  @comptime_assert(true_str == "true");

  var s = "hello";
  @comptime_assert(s == "hello");
  s = "goodbye";
  @comptime_assert(s != "hello");
  @comptime_assert(s == "goodbye");
}
```

As in C and C++, strings in CSL are sequences of bytes. A Unicode
character may correspond to a sequence of more than one byte, and CSL
does not have a \"wide character\" type.

Like `std::string` in C++, but *unlike* `char *` strings in C, strings
in CSL are *not* null-terminated. This means that the NUL character can
occur anywhere in a string. For example, the following
`@comptime_assert`s will succeed:

``` csl
@comptime_assert("abc\x00xyz" != "abc")
@comptime_assert("abc\x00xyz" == "abc\x00xyz")
@comptime_assert(@strlen("abc\x00xyz") != 3)
@comptime_assert(@strlen("abc\x00xyz") == 7)
```

UTF-8 encoded text is allowed in string literals, but if the text
contains non-ASCII characters, the length of the string (as returned by
`@strlen`) will not necessarily match the number of characters in the
string. For example, the `@comptime_assert`s in the following code will
succeed:

``` csl
// Note: The UTF-8 encoding of the "thumbs up" emoji is F0 9F 91 8D.

const thumbs_up = "👍";
@comptime_assert(@strlen(thumbs_up) == 4);

var as_array: [4]u8 = @get_array(thumbs_up);
@comptime_assert(as_array[0] == 0xf0);
@comptime_assert(as_array[1] == 0x9f);
@comptime_assert(as_array[2] == 0x91);
@comptime_assert(as_array[3] == 0x8d);
```

## The `imported_module` Type

TODO.

## The `direction` Type

TODO.
