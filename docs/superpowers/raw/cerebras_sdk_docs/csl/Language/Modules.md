*(Original URL: https://sdk.cerebras.net/csl/Language/Modules.html)*

---

# Modules {#language-modules}

A CSL module consists of a *group of global symbols* defined in a file,
which can be imported into a CSL program through the `@import_module`
builtin.

## Builtin syntax

The `@import_module` builtin has two overloads:

``` csl
@import_module(filename)
@import_module(filename, param_binding)
```

Where:

- `filename` is a comptime string.
- `param_binding` is a comptime anonymous struct.

## Semantics

Unless `filename` is an absolute path, the compiler looks for a file
named `filename`, relative to the location of the importer. The file is
opened and its contents parsed. If no such file exists, a compilation
error is issued. If the filename is enclosed in angled brackets, then
the compiler imports the specified standard library. For instance,
`const math = @import_module("<math>")` makes all math functions
available under the `math` module variable.

For each name in `param_binding`, the imported module is searched for a
param/color with a matching name. The initial value of the param/color
is set to corresponding value in `param_binding`. Warnings are raised if
a name doesn\'t exist inside the module.

A value of `imported_module` type is returned. Such values must always
be assigned to a `const` global variable. This variable has no runtime
footprint.

Importing the same file twice results in two independent copies of the
module.

If the imported module contains a layout block, the block is ignored.

Values of `imported_module` type contain a name for the imported module.
Copying these values does not create a new copy of the module, creates a
new name for the same module.

The compiler must be able to resolve access to module members at compile
time, as such the module values themselves must be `comptime`.

## Using a module

To access the symbols contained in a module, the `.` operator can be
used.

## Example

``` csl
/// A/B/multiply.csl

param factor:i16;

fn multiply(arg:i16) i16 {
  return arg * factor;
}
```

``` csl
/// A/accumulate.csl

var accumulator:i16 = 0;
```

``` csl
/// A/main.csl

const multiply_module = @import_module("B/multiply.csl", .{.factor = 42});
const accumulate_module = @import_module("accumulate.csl");

fn foo() void {
  const x = multiply_module.multiply(10);
  accumulate_module.accumulator += x;
}
```

## Binary symbol names

To find an imported symbol in the compiled program binary, the module
name and `.` should be included in the name being searched for. Given a
module variable `basename = @import_module(...)`, containing a global
`importedname`, its symbol name in the resulting binary will be
`basename.importedname`.
