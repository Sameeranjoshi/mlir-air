*(Original URL: https://sdk.cerebras.net/csl/Language/Generics.html)*

---

# Generics {#language-generics}

## Overview

CSL\'s type system does not have a notion of generic types like that of
C++ or Java. Instead, generic programming is achieved through CSL\'s
comptime features. The basic idea is that `type` is a type, with values
such as `i16`, `i32`, `bool`, and so on, and one can perform computation
using `type`s at comptime.

A very simple generic function looks something like:

``` csl
fn identity(comptime T: type, x: T) T {
  return x;
}

task t() void {
  var arg: i16 = 1;
  var one = identity(i16, arg);
}
```

Since `type` is a comptime-only type just like `comptime_int`,
`comptime_float`, or `comptime_string`, `T` must be marked `comptime` in
order to appear as the type for `x` and as the function\'s return type.

CSL\'s generics resemble C++ templates in some ways. Generic functions
are monomorphized by the compiler. This means a generic function does
not exist at runtime; instead, a copy of the function is compiled using
each set of type arguments it is called with. The program will compile
as long as each such copy is well-formed. An implication is that, for
instance, a generic function that uses the unary `-` operator will not
compile if it is called with a type like `comptime_string` that cannot
be used with unary `-`:

``` csl
fn negate(comptime T: type, x: T) T {
  return -x;
}

task t() void {
  negate(f16, -3.14); // OK
  negate(i16, 42); // OK
  negate(comptime_string, "hello"); // error
}
```

### anytype

While explicitly passing a parameter of type `type` to a generic
function is a straightforward mechanism, it can be verbose. For the user
of a library that provides an `abs` function, there is not much benefit
to writing `abs(f32, 1.0 - x)` versus a non-generic equivalent of
`abs_f32(1.0 - x)`. The `anytype` keyword provides a solution.

`anytype` can only appear as the type of function parameters. It is
another way to write a generic function and it has the same effect of
creating a version of the function for each type that it is called with.
When declaring parameters with `anytype`, the `@type_of` builtin is
useful to relate the types of parameters and the return value to each
other:

``` csl
fn ignore_x(x: anytype, y: @type_of(x)) @type_of(x) {
  return y;
}

task t() void {
  var arg: i16 = 1;
  var two = ignore_x(arg, arg + 1);
}
```

Generic structs are also supported using the same notion of comptime
computation with `type`s as generic functions:

``` csl
fn Point(comptime T: type) type {
  return struct {
    x: T,
    y: T,
  };
}

const origin = Point(u16) { .x = 0, .y = 0 };
comptime {
  @comptime_print(origin); // {x = 0, y = 0}
}
```

The generic `Point` above is a function that takes a `type` parameter
and returns a struct parameterized by that type.

## Constraining Type Parameters

If a generic function is called with an invalid type, an error occurs
when the compiler discovers that the generic function\'s body is trying
to do something invalid with its argument. This is typically a
lower-level error than the actual mistake of calling the function with
an incorrect argument type:

``` csl
fn abs(x: anytype) @type_of(x) {
  if (x < 0.0) {
    return -x;
  }
  return x;
}

task t() void {
  abs("hello");
  // error: invalid comparison operation for type: 'comptime_string'
}
```

Here it is not so hard to piece together what went wrong, but if `abs`
were a more complicated function, the mistake will be less obvious.
Programmers who have used C++ templates may find this situation familar.

Instead, the `@is_same_type` builtin can test the provided type and fail
a comptime assertion if it is invalid:

``` csl
fn abs(x: anytype) @type_of(x) {
  const T = @type_of(x);
  @comptime_assert(
    @is_same_type(T, f16) or @is_same_type(T, f32),
    "x is not a float");
  if (x < 0.0) {
    return -x;
  }
  return x;
}

task t() void {
  abs("hello"); // error: comptime_assert failed: x is not a float
}
```

## Specializing Logic

A related scenario is writing a generic function where a portion of the
logic is only valid for some of the types over which one wants to define
the function. Consider the example of `sign` from the `<math>` library.
This function returns `-1` if its argument is negative, `1` if it is
positive, and `0` if it is zero. `sign` could naïvely be written like:

``` csl
fn sign(x : anytype) @type_of(x) {
  if (x < 0) {
    return -1;
  } else if (x > 0) {
    return 1;
  }
  return x;
}

comptime {
  var x: i16 = 12;
  @comptime_print(sign(x)); // 1
}
```

`math.sign` allows `x` to be an unsigned integer. While
`sign(<some u16>)` is not quite as interesting as `sign` of a float or
signed integer, it is perfectly valid to allow. However, the above code
would not compile if passed a `u16` because `-1` is not a valid `u16`.

To solve this problem, guard the `if (x < 0)` case with a check for the
argument type:

``` csl
fn is_signed(comptime T: type) bool {
  return @is_same_type(T, f16) or @is_same_type(T, f32)
    or @is_same_type(T, i8) or @is_same_type(T, i16)
    or @is_same_type(T, i32) or @is_same_type(T, i64);
}

fn sign(x : anytype) @type_of(x) {
  if (comptime is_signed(@type_of(x))) {
    if (x < 0) {
      return -1;
    }
  }
  if (x > 0) {
    return 1;
  }
  return x;
}

comptime {
  var x: i16 = -12;
  var y: u16 = 25;
  @comptime_print(sign(x), sign(y)); // -1, 1
}
```

Evaluating the `if` condition at comptime ensures that the `if (x < 0)`
case is only compiled at all if the type is correct.

There is one final change that needs to be added to properly support
floats:

``` csl
// using same is_signed() as above

fn sign(x : anytype) @type_of(x) {
  const T = @type_of(x);
  if (comptime is_signed(T)) {
    if (x < @as(T, 0)) {
      return @as(T, -1);
    }
  }
  if (x > @as(T, 0)) {
    return @as(T, 1);
  }
  return x;
}

comptime {
  var x: i16 = -12;
  var y: u16 = 25;
  var z: f16 = 0.0;
  @comptime_print(sign(x), sign(y), sign(z)); // -1, 1, 0
}
```

Since `1` and `0` are `comptime_int`s, they do not automatically convert
to floats.

## Computing With Types

As the previous use of `@type_of` alludes to, type specifiers can be any
expression that has type `type`:

``` csl
fn Point(comptime T: type) type {
  return struct {
    x: T,
    y: T,
  };
}

fn make_point(n: anytype) Point(@type_of(n)) {
  return Point(@type_of(n)) {
    .x = n,
    .y = n + 1,
  };
}

comptime {
  @comptime_print(make_point(3)); // {x = 3, y = 4}
}
```

A generic function can also abstract over properties of a type:

``` csl
fn size_of_int(comptime T: type) comptime_int {
  return if (@is_same_type(T, i8)) 1
    else if (@is_same_type(T, i16)) 2
    else if (@is_same_type(T, i32)) 4
    else if (@is_same_type(T, i64)) 8
    else @comptime_assert(false, "not an int");
}

comptime {
  const word_type = i16;
  @comptime_print(size_of_int(word_type)); // 2
}
```

For a slightly more complex example, we can combine these two techniques
to generically convert a float to its binary representation and extract
the mantissa. The `@comptime_assert`s in helper functions also take care
of validating that the type parameter is a float.

``` csl
fn bits_type(comptime T: type) type {
  return if (@is_same_type(T, f16)) u16
    else if (@is_same_type(T, f32)) u32
    else @comptime_assert(false, "not a float");
}

fn mantissa_len(comptime T: type) comptime_int {
  return if (@is_same_type(T, f16)) 10
    else if (@is_same_type(T, f32)) 23
    else @comptime_assert(false, "not a float");
}

fn mantissa_mask(comptime T: type) comptime_int {
  return comptime (1 << mantissa_len(T)) - 1;
}

fn get_mantissa(x: anytype) bits_type(@type_of(x)) {
  const T = @type_of(x);
  const bits = @bitcast(bits_type(T), x);
  return bits & mantissa_mask(T);
}

comptime {
  var x: f16 = 1.5;
  @comptime_print(get_mantissa(x)); // 512 (== 0x200)
  @comptime_print(get_mantissa(@as(f32, x))); // 4194304 (== 0x400000)
}
```

The `<math>` library internally uses this pattern to generically
implement IEEE floating point functions like `isNaN`, `isInf`, and even
`ceil` and `floor`.
