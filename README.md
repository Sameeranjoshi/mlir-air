# MLIR-AIR GPU Project

## Quick Start

Build and environment setup instructions are in [CLAUDE.md](CLAUDE.md).

### CSL Dialect Usage

For testing the clean two-level abstraction (CSL → CSL Runtime → Python):

```bash
# Full pipeline: CSL dialect IR → conversion → CSL Runtime → Python output
air-opt input.mlir -csl-to-csl-rt | air-translate --emit-csl-rt -o output.py
```

See [docs/more_docs/CSL_TESTING_GUIDE.md](docs/more_docs/CSL_TESTING_GUIDE.md) for detailed testing instructions, validation checklist, and troubleshooting.

### AIR to CSL Lowering

```bash
air-opt input.mlir -air-to-csl="output-dir=./output" --mlir-print-ir-after-all
```

## Documentation

- **[CSL_TESTING_GUIDE.md](docs/more_docs/CSL_TESTING_GUIDE.md)** - Complete testing guide for two-level abstraction
- **[CSL_Dialect_Reference.md](docs/more_docs/CSL_Dialect_Reference.md)** - CSL dialect op reference
- **[CSL_RUNTIME_DIALECT_SPEC.md](docs/more_docs/CSL_RUNTIME_DIALECT_SPEC.md)** - CSL Runtime specification
- **[design_air_to_csl.md](docs/more_docs/design_air_to_csl.md)** - AIR to CSL design documentation
- **[CLAUDE.md](CLAUDE.md)** - Build, environment setup, and testing

## Architecture

The project implements a clean two-level abstraction for the CSL dialect:

```
CSL Dialect (PE-level IR)
    ↓ (-csl-to-csl-rt conversion)
CSL Runtime Dialect (High-level SDK API)
    ↓ (--emit-csl-rt translation)
Python Output
```

This ensures:
- ✅ Clear separation between PE-level and SDK-level abstractions
- ✅ Modular, testable pipeline
- ✅ Single code path for emission (no duplication)