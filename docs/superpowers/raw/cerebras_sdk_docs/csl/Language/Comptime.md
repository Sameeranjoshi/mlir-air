*(Original URL: https://sdk.cerebras.net/csl/Language/Comptime.html)*

---

# Comptime {#language-comptime}

In CSL, it is possible to ensure that code is executed at compile-time
by using the `comptime` keyword, which guarantees that the code will
have no run-time footprint. An error is emitted if compile-time
evaluation is not possible.

## Comptime Variables

A `comptime` variable guarantees that all loads and stores to this
variable happen at compile-time. As such, this variable has no run-time
footprint and its address cannot be obtained. Unlike constants,
`comptime` variables can be modified, as shown below.

``` csl
task foo() void {
  comptime var bitmap: u16 = 0xffff;
  bitmap &= 0x0400;
  ...
}
```

`comptime` variables need to be declared inside a block or a function.
For global variables, using `const` is enough: the initializer of global
variables is always implicitly comptime, since CSL does not support
run-time initialization of global variables.

Since all loads and stores to a `comptime` variable must happen at
compile-time, the stored value must be comptime-known (explained below)
and any offsets (like array indices) must be comptime-known as well.
Similarly, stores to `comptime` variables must not depend on run-time
control flow.

## Comptime-Known Values

All values that are known to the compiler at compile-time are
comptime-known values. Formally, the compiler uses the following rules
to determine whether a value is comptime-known:

1.  All literals (e.g. `1.0`) are comptime-known values.
2.  All `const` variables with a comptime-known initializer (e.g.
    `const x = 1.0`) are comptime-known values.
3.  All uses of `comptime` or `param` variables are comptime-known
    values.
4.  Expressions comprised of two or more comptime-known values (e.g.
    `1.0 + 2.0 * x`) are comptime-known values. However, function calls
    are an exception to this rule.

The compiler ensures that, with the exception of function calls, all
compositions of comptime-known values are comptime-known values as well.
The next section describes how function calls can be explicitly marked
as comptime-known.

## Comptime Expressions {#language-comptime-expressions}

Expressions that depend on constants or other `comptime` variables can
be explicitly marked for evaluation at compile time by prefixing the
expression with the `comptime` keyword. For instance, the following
snippet ensures that the call to `foo()` is replaced with its return
value at compile time, so that we do not pay a run-time cost for the
function call.

``` csl
param bar: u16;

fn foo(arg: u16) u16 {
  return arg * 2;
}

task goo() void {
  var myVar = comptime foo(bar);
  ...
}
```

When evaluating a function call at compile-time, all expressions and
variables are implicitly comptime. The compiler will error out if such
function attempts to read a non comptime-known global variable or
attempts to store to a global variable.

Note that the evaluation of binary logical connectives (i.e., `and` and
`or` operators) will short-circuit, if possible, even at comptime. When
short-circuiting applies, semantic checks like type-checking and checks
for unbound identifiers will not be applied to the right-hand operand,
as shown in the example below:

``` csl
// The 'invalid' term of the following expression will
// not be evaluated because comptime evaluation will short-circuit
// on the 'false' term causing the whole expression to evaluate to
// 'false'.
const and_result = false and invalid;

// Similarly, the following expression will evaluate to 'true'
// without evaluating the 'invalid' term.
const or_result = true or invalid;
```

## Types Whose Values are Required to be Comptime

Certain types are only allowed to exist in expressions and variables
that are `comptime`. In general, these refer to types whose values:

1.  have no possible memory layout associated with them, or
2.  are required to be comptime to enable compiler analyses.

If any of these types is used as the type of a function parameter or the
type returned by a function, all calls to such function must be
`comptime`.

If any of these types is used as the type of a variable, these variables
must be `comptime var` or `const` with a `comptime` initializer.

Pointers to these types are not allowed.

Values of the following types must always be comptime-known:

- `comptime_int`
- `comptime_float`
- `type`
- `comptime_struct`
- `comptime_string`
- function type
- `imported_module`

If these types are used to create a new type, like an array, the new
type is subject to the same constraints.

See `language-types`{.interpreted-text role="ref"} for more information
on each of those types.

## Evaluation of Comptime-Known Control Flow

If the predicate of an `if` statement is a comptime-known value, the
`if` statement is replaced with the block corresponding to the branch
taken (if any), no run-time branches are created, and the block
corresponding to the branch not taken is not semantically checked, as
illustrated below:

``` csl
param mytype : type = f16;
fn foo() mytype {
  // Since the predicate of the `if` statement is comptime-known to be
  // `true`, the compiler will prune the `else` branch, making the code
  // semantically correct.
  if (@is_same_type(mytype, f16)) {
    return 1.0;
  } else {
    return 1;
  }
}
```

In `comptime` loops, all expressions and variables are implicitly
`comptime`. This includes the induction variables of `for` loops and the
continue expression of `while` loops.

``` csl
fn foo() void {
  comptime var sum_values: u16 = 0;
  comptime var sum_indices: u16 = 0;

  const three: u16 = 3;

  comptime {
    for ([3]u16 {1, 2, three})  |value, idx| {
       sum_values += value;
       sum_indices += idx;
    }
  };

  @comptime_assert(sum_values == 6);
  @comptime_assert(sum_indices == 3);
}
```

## Typical Uses of the `comptime` Keyword

`comptime` variables and operations enable powerful operations such as
non-trivial memory initialization or routing rules, without paying a
performance penalty at run-time. For instance, the following code
initializes a global array as an identity matrix at compile time.

``` csl
param size: u16;

// global initializers are implicitly comptime
const identity = createIdentityMatrix();

fn createIdentityMatrix() [size, size]f16 {
  var result = @zeros([size, size]f16);

  var i: u16 = 0;
  while (i < size) : (i += 1) {
    result[i,i] = 1.0;
  }

  return result;
}
```

The above program contains no run-time calls to
`createIdentityMatrix()`, since that function is called on the host
during program compilation.
