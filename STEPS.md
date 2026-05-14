# STEPS

```
  python3 test/csl/test_vecadd_e2e.py

  2. Manual pipeline (what the test actually does)

  cd /home/bricklib_dataflow/air-csl/mlir-air
  rm -rf /tmp/vecadd_e2e && mkdir -p /tmp/vecadd_e2e

  # Step 1: MLIR pipeline — AIR -> csl.* -> csl_rt.* -> text files
  OUT_DIR=/tmp/vecadd_e2e/
  install/bin/air-opt mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir -air-to-csl-dialect &> ${OUT_DIR}/vecadd_csld.mlir 
  install/bin/air-opt ${OUT_DIR}/vecadd_csld.mlir -csl-to-csl-rt &> ${OUT_DIR}/vecadd_cslrt.mlir 
  install/bin/air-translate ${OUT_DIR}/vecadd_cslrt.mlir --emit-csl-rt --csl-output-dir=${OUT_DIR} -o /dev/null
  # Now ${OUT_DIR} contains: layout.csl, vecadd_pe.csl, run.py

  # Compile
  cd $OUT_DIR && \
    cslc --arch=wse3 layout.csl \
         --fabric-dims=8,3 --fabric-offsets=4,1 \
         -o out --memcpy --channels 1
  # Run
  cs_python run.py --name out --check
  # Expected: PASS

```

# Introduction to Markdown

Markdown is a lightweight markup language that allows you to format text using a plain-text syntax. It is widely used for writing documentation, README files, and content for static sites like GitHub Pages.

## Basic Syntax

### Headings

Create headings by starting a line with one or more `#` symbols, followed by a space.

```markdown
# Heading 1
## Heading 2
### Heading 3
```

### Emphasis

- *Italic*: Use single asterisks or underscores (`*text`* or `_text_`)
- **Bold**: Use double asterisks or underscores (`**text`** or `__text__`)
- ***Bold and Italic***: Use three asterisks (`***text`***)

### Lists

**Unordered list** (use `-`, `+`, or `*`):

```markdown
- Item 1
- Item 2
```

**Ordered list**:

```markdown
1. First item
2. Second item
```

### Links and Images

- [Link text](https://www.example.com)
- Alt text for image

### Code

For inline code, use backticks: `code`  
For code blocks, use triple backticks:

```
```python
print("Hello, Markdown!")
```

```

### Blockquotes

Prefix with `>`:

```markdown
> This is a quote.
```

### Horizontal Rule

Use `---`, `***`, or `___`:

```markdown
---
```

## Example

Here's an example markdown snippet:

```markdown
# Sample Document

This is a **bold** word, and this is *italic*.

- List item 1
- List item 2

> A simple quote.
```

---

