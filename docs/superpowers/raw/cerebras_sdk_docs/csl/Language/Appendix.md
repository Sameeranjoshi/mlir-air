*(Original URL: https://sdk.cerebras.net/csl/Language/Appendix.html)*

---

# Appendix {#language-appendix}

## SIMD Mode {#language-appendix-simd}

Many of the
`builtins for DSD operations<language-builtins-for-dsd-operations>`{.interpreted-text
role="ref"} have a SIMD (single instruction, multiple data) mode, in
which multiple operations can be performed in a single cycle. Under
appropriate conditions, these builtins will automatically execute in
SIMD mode when operating on DSDs, if possible.

In particular, builtins can only operate at their full SIMD width if no
bank conflicts occur when fetching the operands from memory. The 48 KB
of memory in a PE are laid out into 8 banks of 6 KB each. Each
successive 16 bits are located in successive banks. In a single cycle,
the PE can perform two 32-bit reads and one 32-bit write. However, the
reads must occur from separate banks. More specifically, if the 8 banks
are numbered 0 to 7, then the bank IDs `bank_1` and `bank_2` of the two
reads must be such that `bank_1 % 4 != bank_2 % 4`.

For best results in avoiding bank conflicts with SIMD operations, the
operand addresses should be 32-bit aligned, and
`(src0_addr % 8) == ((src1_addr + 4) % 8)`, where `src0_addr` and
`src1_addr` are the addresses of the operands. Dumping the ELF file\'s
symbol table with the `--sym` option of `cs_readelf` provides addresses
and banking information for all symbols in a compiled CSL program, and
can be useful for determining if bank conflicts may occur.

Additionally, if the DSD operands have non-contiguous strided accesses,
the SIMD width may be limited:

- Strides of 0 and 1 can operate at full SIMD width.
- Strides such that `stride % 8` is 2, 3, 5, or 6 can operate at full
  SIMD width.
- Unless stride is 1, strides such that `stride % 8` is 1 or 7 is
  limited to two operations per cycle, or a SIMD width of 2.
- Unless stride is 0, strides such that `stride % 8` is 0 is limited to
  one operation per cycle, or a SIMD width of 1.
- Strides such that `stride % 8` is 4 are limited to two operations per
  cycle, or a SIMD width of 4.

The maximum width of builtins which can operate in SIMD mode are given
in the table below.

  Builtin                                    WSE-2 SIMD Width   WSE-3 SIMD Width
  ------------------------------------------ ------------------ ------------------
  `@add16`{.interpreted-text role="ref"}     4                  8
  `@addc16`{.interpreted-text role="ref"}    1                  8
  `@and16`{.interpreted-text role="ref"}     4                  8
  `@fabsh`{.interpreted-text role="ref"}     4                  8
  `@fabss`{.interpreted-text role="ref"}     2                  4
  `@faddh`{.interpreted-text role="ref"}     4                  8
  `@faddhs`{.interpreted-text role="ref"}    2                  4
  `@fadds`{.interpreted-text role="ref"}     2                  4
  `@fnormh`{.interpreted-text role="ref"}    4                  8
  `@fnorms`{.interpreted-text role="ref"}    2                  4
  `@fh2s`{.interpreted-text role="ref"}      1                  4
  `@fh2xp16`{.interpreted-text role="ref"}   1                  8
  `@fmach`{.interpreted-text role="ref"}     4                  8
  `@fmachs`{.interpreted-text role="ref"}    2                  4
  `@fmaxh`{.interpreted-text role="ref"}     1                  8
  `@fmaxs`{.interpreted-text role="ref"}     1                  4
  `@fmovh`{.interpreted-text role="ref"}     4                  8
  `@fmovs`{.interpreted-text role="ref"}     2                  4
  `@fmulh`{.interpreted-text role="ref"}     4                  8
  `@fnegh`{.interpreted-text role="ref"}     4                  8
  `@fnegs`{.interpreted-text role="ref"}     2                  4
  `@fs2h`{.interpreted-text role="ref"}      1                  4
  `@fs2xp16`{.interpreted-text role="ref"}   1                  4
  `@fscaleh`{.interpreted-text role="ref"}   4                  8
  `@fscales`{.interpreted-text role="ref"}   2                  4
  `@fsubh`{.interpreted-text role="ref"}     4                  8
  `@fsubs`{.interpreted-text role="ref"}     2                  4
  `@mov16`{.interpreted-text role="ref"}     4                  8
  `@mov32`{.interpreted-text role="ref"}     2                  4
  `@or16`{.interpreted-text role="ref"}      4                  8
  `@sar16`{.interpreted-text role="ref"}     1                  4
  `@sll16`{.interpreted-text role="ref"}     1                  4
  `@slr16`{.interpreted-text role="ref"}     1                  4
  `@sub16`{.interpreted-text role="ref"}     4                  8
  `@xor16`{.interpreted-text role="ref"}     4                  8
  `@xp162fh`{.interpreted-text role="ref"}   1                  8
  `@xp162fs`{.interpreted-text role="ref"}   1                  4
