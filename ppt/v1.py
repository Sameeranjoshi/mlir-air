"""Generate a presentation summarizing the CSL backend work in mlir-air.

Run:
    python out/build_csl_deck.py
Outputs:
    out/mlir_air_csl_deck.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------- theme ----------
BG = RGBColor(0x10, 0x16, 0x22)  # dark navy
PANEL = RGBColor(0x18, 0x20, 0x30)
ACCENT = RGBColor(0x4F, 0xC3, 0xF7)  # cyan
ACCENT2 = RGBColor(0xFF, 0xB7, 0x4D)  # amber
GOOD = RGBColor(0x81, 0xC7, 0x84)  # green
TEXT = RGBColor(0xE6, 0xEC, 0xF2)
DIM = RGBColor(0x9A, 0xA5, 0xB1)
CODE_BG = RGBColor(0x0B, 0x10, 0x18)
CODE_FG = RGBColor(0xC9, 0xD1, 0xD9)
KEYWORD = RGBColor(0xFF, 0x9D, 0x55)
COMMENT = RGBColor(0x70, 0x80, 0x90)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def add_bg(slide, color=BG):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    *,
    size=18,
    bold=False,
    color=TEXT,
    align=PP_ALIGN.LEFT,
    font="Calibri",
    anchor=MSO_ANCHOR.TOP,
):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return tb


def add_rect(slide, x, y, w, h, fill=PANEL, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s


def header(slide, title, subtitle=None):
    add_bg(slide)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(0.55), Inches(0.12), Inches(0.55)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    add_text(
        slide,
        Inches(0.65),
        Inches(0.45),
        Inches(12.2),
        Inches(0.7),
        title,
        size=30,
        bold=True,
        color=TEXT,
    )
    if subtitle:
        add_text(
            slide,
            Inches(0.65),
            Inches(1.05),
            Inches(12.2),
            Inches(0.4),
            subtitle,
            size=15,
            color=DIM,
        )
    add_rect(slide, Inches(0.4), Inches(7.05), Inches(12.55), Emu(9525), fill=PANEL)


def footer(slide, page):
    add_text(
        slide,
        Inches(0.4),
        Inches(7.12),
        Inches(8.0),
        Inches(0.3),
        "mlir-air  ·  CSL backend  ·  air-to-fire",
        size=10,
        color=DIM,
    )
    add_text(
        slide,
        Inches(11.6),
        Inches(7.12),
        Inches(1.4),
        Inches(0.3),
        f"{page} / {TOTAL}",
        size=10,
        color=DIM,
        align=PP_ALIGN.RIGHT,
    )


def code_block(slide, x, y, w, h, code, *, size=11, hl=()):
    """Code block. `hl` is iterable of substrings to color as keywords."""
    bg = add_rect(slide, x, y, w, h, fill=CODE_BG)
    bg.line.color.rgb = RGBColor(0x2A, 0x33, 0x42)
    bg.line.width = Pt(0.5)
    tb = slide.shapes.add_textbox(
        x + Emu(60000), y + Emu(60000), w - Emu(120000), h - Emu(120000)
    )
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    lines = code.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        stripped = line.lstrip(" ")
        indent = line[: len(line) - len(stripped)]
        if indent:
            r = p.add_run()
            r.text = indent
            r.font.name = "Consolas"
            r.font.size = Pt(size)
            r.font.color.rgb = CODE_FG
        if stripped.startswith("//") or stripped.startswith("#"):
            r = p.add_run()
            r.text = stripped
            r.font.name = "Consolas"
            r.font.size = Pt(size)
            r.font.color.rgb = COMMENT
            continue
        rest = stripped
        i = 0
        while i < len(rest):
            matched = None
            for kw in hl:
                if rest.startswith(kw, i):
                    if i > 0 and (
                        rest[i - 1].isalnum() or rest[i - 1] == "_" or rest[i - 1] == "."
                    ):
                        continue
                    matched = kw
                    break
            if matched:
                r = p.add_run()
                r.text = matched
                r.font.name = "Consolas"
                r.font.size = Pt(size)
                r.font.color.rgb = KEYWORD
                r.font.bold = True
                i += len(matched)
            else:
                j = i + 1
                while j < len(rest):
                    skip = False
                    for kw in hl:
                        if rest.startswith(kw, j):
                            if j > 0 and (
                                rest[j - 1].isalnum()
                                or rest[j - 1] == "_"
                                or rest[j - 1] == "."
                            ):
                                continue
                            skip = True
                            break
                    if skip:
                        break
                    j += 1
                r = p.add_run()
                r.text = rest[i:j]
                r.font.name = "Consolas"
                r.font.size = Pt(size)
                r.font.color.rgb = CODE_FG
                i = j


def bullets(slide, x, y, w, h, items, *, size=16, color=TEXT, line_spacing=1.25):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    for i, txt in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = "▸  "
        r.font.size = Pt(size)
        r.font.color.rgb = ACCENT
        r.font.bold = True
        r.font.name = "Calibri"
        r2 = p.add_run()
        r2.text = txt
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
        r2.font.name = "Calibri"


def label_chip(slide, x, y, w, h, text, *, fill=ACCENT, fg=BG):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    s.adjustments[0] = 0.5
    tf = s.text_frame
    tf.margin_left = Emu(40000)
    tf.margin_right = Emu(40000)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = fg
    r.font.name = "Calibri"


def arrow_right(slide, x, y, w, h, color=ACCENT):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False


MLIR_KW = (
    "module",
    "func.func",
    "return",
    "scf.for",
    "scf.if",
    "scf.index_switch",
    "scf.yield",
    "case",
    "default",
    "else",
    "memref.load",
    "memref.store",
    "memref.subview",
    "memref",
    "arith.constant",
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.cmpi",
    "arith.cmpf",
    "arith.ori",
    "arith.subi",
    "arith.addi",
    "arith.remsi",
    "arith.mulf",
    "air.launch",
    "air.segment",
    "air.herd",
    "air.launch_terminator",
    "air.segment_terminator",
    "air.herd_terminator",
    "csl.wafer",
    "csl.program",
    "csl.func",
    "csl.var",
    "csl.export",
    "csl.return",
    "csl.host",
    "csl.layout",
    "csl.get_mem_dsd",
    "csl.builtin_call",
    "csl.import_module",
    "csl_layout.place",
    "csl_host.memcpy_h2d",
    "csl_host.memcpy_d2h",
    "csl_host.launch",
    "f32",
    "i1",
    "index",
    "strided",
    "over",
    "at",
)

CSL_KW = (
    "fn",
    "var",
    "const",
    "comptime",
    "param",
    "while",
    "if",
    "else",
    "switch",
    "case",
    "void",
    "layout",
    "f32",
    "u16",
    "@import_module",
    "@get_dsd",
    "@set_rectangle",
    "@set_tile_code",
    "@fmacs",
    "@fadds",
    "@fmuls",
    "@fmovs",
    "@fnegs",
    "@fsubs",
    "@export_symbol",
    "@increment_dsd_offset",
    "mem1d_dsd",
    "mem4d_dsd",
)

PY_KW = (
    "import",
    "from",
    "def",
    "if",
    "for",
    "while",
    "return",
    "True",
    "False",
    "None",
    "as",
    "in",
    "not",
    "and",
    "or",
)


slides = []


def slide_title():
    s = prs.slides.add_slide(BLANK)
    add_bg(s)
    for i in range(1, 8):
        x = Inches(0.5 + i * 1.6)
        ln = s.shapes.add_connector(1, x, Inches(0), x, SH)
        ln.line.color.rgb = RGBColor(0x1A, 0x22, 0x33)
        ln.line.width = Pt(0.5)
    add_rect(s, Inches(0.7), Inches(1.6), Inches(0.18), Inches(2.6), fill=ACCENT)
    add_text(
        s,
        Inches(1.0),
        Inches(1.55),
        Inches(11.5),
        Inches(0.6),
        "MLIR · AIR → CSL",
        size=18,
        bold=True,
        color=ACCENT2,
        font="Consolas",
    )
    add_text(
        s,
        Inches(1.0),
        Inches(2.05),
        Inches(11.5),
        Inches(1.6),
        "A Compiler Path from AIR to the\nCerebras Wafer-Scale Engine",
        size=44,
        bold=True,
        color=TEXT,
    )
    add_text(
        s,
        Inches(1.0),
        Inches(4.4),
        Inches(11.5),
        Inches(0.5),
        "CSL dialect · -air-to-csl · -csl-auto-vectorize · --emit-csl",
        size=16,
        color=DIM,
        font="Consolas",
    )
    add_text(
        s,
        Inches(1.0),
        Inches(5.0),
        Inches(11.5),
        Inches(0.5),
        "Branch  air-to-fire   ·   simulator-validated end-to-end",
        size=14,
        color=DIM,
    )
    add_text(
        s,
        Inches(1.0),
        Inches(6.4),
        Inches(11.5),
        Inches(0.4),
        "Project: mlir-air     CSL backend overview",
        size=12,
        color=DIM,
        font="Consolas",
    )


slides.append(slide_title)


def slide_pipeline():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "The Pipeline at a Glance",
        "AIR program → structured CSL IR → three text artifacts the Cerebras toolchain ingests",
    )
    y = Inches(2.0)
    boxes = [
        ("AIR\nMLIR", "air.launch / segment / herd", ACCENT),
        ("-air-to-csl", "lower to csl.wafer container", ACCENT2),
        ("-csl-auto-vectorize", "scf.for → @fadds/@fmuls/@fmacs", ACCENT2),
        ("-csl-infer-exports", "host I/O ↔ device symbols", ACCENT2),
        ("--emit-csl", "layout.csl + pe.csl + run.py", GOOD),
        ("cslc + sim", "Cerebras SDK runtime", RGBColor(0xCE, 0x93, 0xD8)),
    ]
    bw = Inches(1.85)
    bh = Inches(1.35)
    gap = Inches(0.18)
    x = Inches(0.5)
    for title, sub, col in boxes:
        add_rect(s, x, y, bw, bh, fill=PANEL, line=RGBColor(0x33, 0x40, 0x55))
        add_rect(s, x, y, bw, Inches(0.14), fill=col)
        add_text(
            s,
            x + Inches(0.1),
            y + Inches(0.22),
            bw - Inches(0.2),
            Inches(0.55),
            title,
            size=14,
            bold=True,
            color=TEXT,
            font="Consolas",
        )
        add_text(
            s,
            x + Inches(0.1),
            y + Inches(0.78),
            bw - Inches(0.2),
            Inches(0.5),
            sub,
            size=10,
            color=DIM,
        )
        x += bw + gap
    ay = y + bh + Inches(0.18)
    ax = Inches(0.5) + bw
    for _ in range(5):
        arrow_right(s, ax - Inches(0.12), ay, Inches(0.32), Inches(0.18), color=ACCENT)
        ax += bw + gap

    add_text(
        s,
        Inches(0.5),
        Inches(4.5),
        Inches(12.3),
        Inches(0.5),
        "Two MLIR tools do the heavy lifting:",
        size=16,
        bold=True,
        color=TEXT,
    )
    code_block(
        s,
        Inches(0.5),
        Inches(5.0),
        Inches(12.3),
        Inches(1.7),
        "$ air-opt vecadd.mlir -air-to-csl -csl-auto-vectorize -csl-infer-exports \\\n"
        "    | air-translate --emit-csl --output-dir=out/\n"
        "\n"
        "$ ls out/vecadd/\n"
        "  pe.csl       layout.csl       run.py       commands_wse{2,3}.sh",
        size=13,
        hl=(),
    )


slides.append(slide_pipeline)


def slide_air_primer():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "AIR Primer  ·  three-level spatial hierarchy",
        "AIR is the core abstraction in mlir-air — borrowed from the AIE backend, now retargeted to CSL",
    )
    rows = [
        ("Scope", "AIR op", "Maps to (CSL)"),
        ("Host", "air.launch", "csl.host  +  csl.wafer"),
        ("L2 buf", "air.segment", "(elided — wafer fabric is flat)"),
        ("Compute", "air.herd", "csl.program @pe placed by csl_layout.place"),
    ]
    x0, y0 = Inches(0.6), Inches(2.0)
    cw = [Inches(2.2), Inches(3.6), Inches(6.2)]
    rh = Inches(0.55)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x = x0 + sum(cw[:c], Inches(0))
            fill = ACCENT if r == 0 else (PANEL if r % 2 else CODE_BG)
            color = BG if r == 0 else TEXT
            add_rect(s, x, y0 + r * rh, cw[c], rh, fill=fill)
            add_text(
                s,
                x + Inches(0.15),
                y0 + r * rh + Inches(0.12),
                cw[c] - Inches(0.3),
                rh - Inches(0.2),
                cell,
                size=14 if r == 0 else 13,
                bold=(r == 0 or c == 1),
                color=color,
                font="Consolas" if c == 1 else "Calibri",
            )

    add_text(s, Inches(0.6), Inches(4.5), Inches(12.0), Inches(0.4), "Memory spaces", size=16, bold=True, color=ACCENT2)
    bullets(
        s,
        Inches(0.6),
        Inches(4.85),
        Inches(12.0),
        Inches(1.5),
        [
            "memref<...>     —  L3 / DDR  (host-visible)",
            "memref<..., 1>  —  L2 segment buffer",
            "memref<..., 2>  —  L1 / per-tile  (CSL: per-PE)",
        ],
        size=14,
    )
    add_text(
        s,
        Inches(0.6),
        Inches(6.5),
        Inches(12.0),
        Inches(0.4),
        "Data movement: air.dma_memcpy_nd / air.channel — direct cross-level loads/stores are illegal.",
        size=12,
        color=DIM,
    )


slides.append(slide_air_primer)


def slide_dialect_overview():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "The CSL Dialect  ·  six op groups, one container",
        "All ops live under csl.wafer — the per-program scope. Verifiers reject patterns the Cerebras SDK can't express.",
    )
    groups = [
        ("Layout", "csl.layout · csl_layout.place {at | over [a:b, c:d]}", "spatial grid + tile→code mapping"),
        ("Kernel", "csl.program · csl.func · csl.var · csl.export", "per-PE program; what host sees as symbols"),
        ("Data movement", "csl.get_mem_dsd · csl.builtin_call \"@fmacs\"…", "DSDs over memref, generic builtin op"),
        ("Runtime", "csl.import_module · csl.host", "user-level imports + host I/O block"),
        ("Host ops", "csl_host.memcpy_h2d · memcpy_d2h · launch", "structured replacement for ad-hoc run.py"),
        ("Top level", "csl.wafer { arch = \"wse2\" | \"wse3\" }", "container; one .pptx output dir per wafer"),
    ]
    x0, y0 = Inches(0.5), Inches(1.95)
    bw, bh = Inches(6.15), Inches(1.55)
    for i, (name, ops, desc) in enumerate(groups):
        col = i % 2
        row = i // 2
        x = x0 + col * (bw + Inches(0.2))
        y = y0 + row * (bh + Inches(0.2))
        add_rect(s, x, y, bw, bh, fill=PANEL, line=RGBColor(0x2A, 0x33, 0x42))
        add_rect(s, x, y, Inches(0.15), bh, fill=ACCENT)
        add_text(s, x + Inches(0.3), y + Inches(0.15), bw - Inches(0.4), Inches(0.4), name, size=15, bold=True, color=TEXT)
        add_text(
            s,
            x + Inches(0.3),
            y + Inches(0.55),
            bw - Inches(0.4),
            Inches(0.5),
            ops,
            size=12,
            color=ACCENT2,
            font="Consolas",
        )
        add_text(s, x + Inches(0.3), y + Inches(1.0), bw - Inches(0.4), Inches(0.5), desc, size=11, color=DIM)


slides.append(slide_dialect_overview)


def slide_air_vecadd():
    s = prs.slides.add_slide(BLANK)
    header(s, "Example  ·  vecadd in AIR", "Plain three-level AIR program — same shape as any AIE/GPU AIR test")
    code_block(
        s,
        Inches(0.5),
        Inches(1.85),
        Inches(12.3),
        Inches(5.0),
        """func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>,
                  %c: memref<256xf32>) {
  %one = arith.constant 1 : index
  air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b, %lc=%c)
      : memref<256xf32>, memref<256xf32>, memref<256xf32> {
    air.segment @seg0 args(%sa=%la, %sb=%lb, %sc=%lc) ... {
      %one2 = arith.constant 1 : index
      air.herd @h tile(%htx, %hty) in (%hsx=%one2, %hsy=%one2)
          args(%ha=%sa, %hb=%sb, %hc=%sc) ... {
        %lo = arith.constant 0   : index
        %hi = arith.constant 256 : index
        %st = arith.constant 1   : index
        scf.for %i = %lo to %hi step %st {
          %va = memref.load  %ha[%i] : memref<256xf32>
          %vb = memref.load  %hb[%i] : memref<256xf32>
          %vc = arith.addf   %va, %vb : f32
          memref.store %vc, %hc[%i]   : memref<256xf32>
        }
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}""",
        size=12,
        hl=MLIR_KW,
    )


slides.append(slide_air_vecadd)


def slide_csl_wafer():
    s = prs.slides.add_slide(BLANK)
    header(s, "After -air-to-csl  ·  structured csl.wafer IR", "The herd body is preserved verbatim; placement and host I/O are made explicit")
    code_block(
        s,
        Inches(0.5),
        Inches(1.85),
        Inches(12.3),
        Inches(5.0),
        """csl.wafer @vecadd {arch = "wse3"} {
  csl.program @pe {
    %a = csl.var @a : memref<256xf32>
    %b = csl.var @b : memref<256xf32>
    %c = csl.var @c : memref<256xf32>
    csl.func @compute {
      // ... unchanged scf.for body from AIR herd ...
      csl.return
    }
    csl.export @a {alias = "a"}     // -csl-infer-exports fills these in
    csl.export @b {alias = "b"}
    csl.export @c {alias = "c"}
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 1, height = 1} @main_layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main(%a_in: ..., %b_in: ..., %c_out: ...) {layout=@main_layout} {
    csl_host.memcpy_h2d %a_in to @main_layout::@a {px=0, py=0, ...}
    csl_host.memcpy_h2d %b_in to @main_layout::@b ...
    csl_host.launch     @main_layout::@compute
    csl_host.memcpy_d2h @main_layout::@c to %c_out ...
  }
}""",
        size=12,
        hl=MLIR_KW,
    )


slides.append(slide_csl_wafer)


def slide_emitted():
    s = prs.slides.add_slide(BLANK)
    header(s, "What --emit-csl produces", "Three text artifacts, byte-for-byte runnable on the Cerebras simulator")
    add_text(s, Inches(0.5), Inches(1.8), Inches(4.0), Inches(0.4), "pe.csl  (per-PE kernel)", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(0.5),
        Inches(2.15),
        Inches(4.0),
        Inches(4.5),
        """param memcpy_params: comptime_struct;
const sys_mod = @import_module(
   "<memcpy/memcpy>", memcpy_params);

var arg0: [256]f32;
var arg1: [256]f32;
var arg2: [256]f32;
var arg0_ptr: [*]f32 = &arg0;
var arg1_ptr: [*]f32 = &arg1;
const arg2_ptr: [*]f32 = &arg2;

fn compute() void {
  var i: u16 = 0;
  while (i < 256) : (i += 1) {
    var va = arg0[i];
    var vb = arg1[i];
    var vc: f32 = va + vb;
    arg2[i] = vc;
  }
  sys_mod.unblock_cmd_stream();
}
comptime {
  @export_symbol(arg0_ptr, "arg0");
  @export_symbol(arg1_ptr, "arg1");
  @export_symbol(arg2_ptr, "arg2");
  @export_symbol(compute);
}""",
        size=10,
        hl=CSL_KW,
    )
    add_text(s, Inches(4.7), Inches(1.8), Inches(4.0), Inches(0.4), "layout.csl  (host-side wrapper)", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(4.7),
        Inches(2.15),
        Inches(4.0),
        Inches(4.5),
        """const memcpy = @import_module(
   "<memcpy/get_params>", .{
   .width  = 1,
   .height = 1,
});

layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "h.csl", .{
    .memcpy_params = memcpy.get_params(0)
  });
}""",
        size=10,
        hl=CSL_KW,
    )
    add_text(s, Inches(8.9), Inches(1.8), Inches(4.0), Inches(0.4), "run.py  (host driver + numpy check)", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(8.9),
        Inches(2.15),
        Inches(4.0),
        Inches(4.5),
        """import argparse, numpy as np
from cerebras.sdk.runtime\\
    .sdkruntimepybind import (
        SdkRuntime, MemcpyOrder,
        MemcpyDataType,
)
N = 256
arg0 = np.arange(N, dtype=np.float32)
arg1 = np.arange(N, dtype=np.float32)*2
arg2 = np.zeros(N, dtype=np.float32)

runner = SdkRuntime(args.name,
   simfab_numthreads=32)
runner.load(); runner.run()
runner.memcpy_h2d(
   runner.get_id("arg0"), arg0,
   0, 0, 1, 1, 256, ...)
runner.memcpy_h2d(...)
runner.launch("compute",
              nonblock=False)
runner.memcpy_d2h(arg2, ...)
runner.stop()
print("SUCCESS!")""",
        size=10,
        hl=PY_KW,
    )


slides.append(slide_emitted)


def slide_dsds():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "DSDs  ·  Cerebras's data-structure-descriptor model",
        "csl.get_mem_dsd builds a runtime-time DSD from a memref; csl.builtin_call dispatches the SDK intrinsic",
    )
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "MLIR  —  saxpy via DSDs", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(0.5),
        Inches(2.2),
        Inches(6.0),
        Inches(4.5),
        """csl.func @compute {
  %c0 = arith.constant 0   : index
  %a  = memref.load %alpha[%c0]
        : memref<1xf32>
  %xd = csl.get_mem_dsd %x
        : memref<128xf32> -> !csl.dsd
  %yd = csl.get_mem_dsd %y
        : memref<128xf32> -> !csl.dsd
  csl.builtin_call "fmacs"
      (%yd, %yd, %xd, %a)
      : (!csl.dsd, !csl.dsd,
         !csl.dsd, f32) -> ()
  csl.return
}""",
        size=12,
        hl=MLIR_KW,
    )
    add_text(s, Inches(6.7), Inches(1.85), Inches(6.0), Inches(0.4), "Emitted CSL", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(6.7),
        Inches(2.2),
        Inches(6.0),
        Inches(4.5),
        """fn compute() void {
  const xd = @get_dsd(mem1d_dsd, .{
      .base_address = &x,
      .extent = 128,
  });
  const yd = @get_dsd(mem1d_dsd, .{
      .base_address = &y,
      .extent = 128,
  });
  @fmacs(yd, yd, xd, alpha[0]);
}""",
        size=12,
        hl=CSL_KW,
    )


slides.append(slide_dsds)


def slide_strided():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "Strided DSDs  ·  driven by memref.subview",
        "Default stride/offset collapse to bare @get_dsd; non-defaults emit .stride or @increment_dsd_offset",
    )
    rows = [
        ("default", "no subview", "@get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 128 });"),
        ("offset only", "subview %x[8] [64] [1]", "@get_dsd(...) .extent=64;\n@increment_dsd_offset(d, 8, f32);"),
        ("stride only", "subview %x[0] [32] [4]", "@get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 32, .stride = 4 });"),
        ("both", "subview %x[3] [32] [4]", "@get_dsd(... .extent=32, .stride=4);\n@increment_dsd_offset(d, 3, f32);"),
    ]
    x0 = Inches(0.5)
    y0 = Inches(2.0)
    cw = [Inches(1.6), Inches(3.4), Inches(7.4)]
    add_rect(s, x0, y0, sum(cw, Inches(0)), Inches(0.5), fill=ACCENT)
    for c, h in enumerate(("Pattern", "memref.subview args", "Emitted CSL")):
        add_text(s, x0 + sum(cw[:c], Inches(0)) + Inches(0.15), y0 + Inches(0.13), cw[c] - Inches(0.3), Inches(0.4), h, size=14, bold=True, color=BG)
    rh = Inches(1.05)
    for i, (name, sv, csl) in enumerate(rows):
        y = y0 + Inches(0.5) + i * rh
        fill = PANEL if i % 2 == 0 else CODE_BG
        add_rect(s, x0, y, sum(cw, Inches(0)), rh, fill=fill)
        add_text(s, x0 + Inches(0.15), y + Inches(0.32), cw[0] - Inches(0.3), Inches(0.6), name, size=14, bold=True, color=ACCENT2, font="Consolas")
        add_text(s, x0 + cw[0] + Inches(0.15), y + Inches(0.32), cw[1] - Inches(0.3), Inches(0.6), sv, size=12, color=TEXT, font="Consolas")
        tb = s.shapes.add_textbox(x0 + cw[0] + cw[1] + Inches(0.15), y + Inches(0.12), cw[2] - Inches(0.3), rh - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_top = Emu(0)
        for li, line in enumerate(csl.split("\n")):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            r = p.add_run()
            r.text = line
            r.font.name = "Consolas"
            r.font.size = Pt(11)
            r.font.color.rgb = CODE_FG


slides.append(slide_strided)


def slide_subgrid():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "Subgrid Placement  ·  one program, many PEs",
        "csl_layout.place @pe over [0:4, 0:2]  →  same kernel runs on 8 PEs, host shards 256 elems → 32/PE",
    )
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "MLIR — N-PE SIMD saxpy", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(0.5),
        Inches(2.2),
        Inches(6.0),
        Inches(4.5),
        """csl.wafer @simd {arch = "wse3"} {
  csl.program @pe {
    %A = csl.var @A : memref<32xf32>
    %y = csl.var @y : memref<32xf32>
    csl.func @compute {
      %a  = arith.constant 2.0 : f32
      %Ad = csl.get_mem_dsd %A
            : memref<32xf32> -> !csl.dsd
      %yd = csl.get_mem_dsd %y
            : memref<32xf32> -> !csl.dsd
      csl.builtin_call "fmacs"
          (%yd, %yd, %Ad, %a) ...
      csl.return
    }
  }
  csl.layout {width=4, height=2} @layout {
    csl_layout.place @pe over [0:4, 0:2]
  }
  csl.host @main(...) {
    csl_host.memcpy_h2d ...
        {px=0, py=0, width=4, height=2}
        : memref<8x32xf32>
  }
}""",
        size=11,
        hl=MLIR_KW,
    )
    add_text(s, Inches(6.7), Inches(1.85), Inches(6.0), Inches(0.4), "Emitted layout.csl", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(6.7),
        Inches(2.2),
        Inches(6.0),
        Inches(2.0),
        """layout {
  @set_rectangle(4, 2);
  var px: u16 = 0;
  while (px < 4) : (px += 1) {
    var py: u16 = 0;
    while (py < 2) : (py += 1) {
      @set_tile_code(px, py, "pe.csl", .{ ... });
    }
  }
}""",
        size=11,
        hl=CSL_KW,
    )
    gx = Inches(6.7)
    gy = Inches(4.5)
    cell = Inches(0.7)
    pad = Inches(0.1)
    for px in range(4):
        for py in range(2):
            cx = gx + px * (cell + pad)
            cy = gy + py * (cell + pad)
            add_rect(s, cx, cy, cell, cell, fill=PANEL, line=ACCENT)
            add_text(s, cx, cy + Inches(0.18), cell, Inches(0.4), f"{px},{py}", size=12, bold=True, color=ACCENT, align=PP_ALIGN.CENTER, font="Consolas")
    add_text(s, gx + Inches(3.4), gy + Inches(0.4), Inches(2.4), Inches(1.0), "8 PEs\n32 f32 / PE\n= 256 total", size=12, color=DIM, font="Consolas")


slides.append(slide_subgrid)


def slide_autovec_overview():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "-csl-auto-vectorize  ·  scf.for → DSD intrinsic",
        "Greedy rewriter that recognizes element-wise loop idioms and rewrites them as a single SDK builtin call",
    )
    rows = [
        ("@fadds", "c[i] = a[i] + b[i]", "FaddsPattern · Rank2FaddsPattern"),
        ("@fsubs", "c[i] = a[i] - b[i]", "FsubsPattern"),
        ("@fmuls", "c[i] = a[i] * b[i]", "FmulsPattern · FmulsScalarPattern  (α*a[i])"),
        ("@fmacs", "y[i] = α*a[i] + y[i]", "FmacsScalarPattern · Rank2FmacsPattern"),
        ("@fmovs", "c[i] = a[i]", "FmovsPattern"),
        ("@fnegs", "c[i] = -a[i]", "FnegsPattern"),
    ]
    x0 = Inches(0.6)
    y0 = Inches(2.0)
    cw = [Inches(2.0), Inches(4.4), Inches(5.6)]
    add_rect(s, x0, y0, sum(cw, Inches(0)), Inches(0.5), fill=ACCENT)
    for c, h in enumerate(("CSL builtin", "Recognized idiom (f32)", "Pattern")):
        add_text(s, x0 + sum(cw[:c], Inches(0)) + Inches(0.15), y0 + Inches(0.13), cw[c] - Inches(0.3), Inches(0.4), h, size=14, bold=True, color=BG)
    for i, row in enumerate(rows):
        y = y0 + Inches(0.5) + i * Inches(0.55)
        fill = PANEL if i % 2 == 0 else CODE_BG
        add_rect(s, x0, y, sum(cw, Inches(0)), Inches(0.55), fill=fill)
        for c, val in enumerate(row):
            add_text(
                s,
                x0 + sum(cw[:c], Inches(0)) + Inches(0.18),
                y + Inches(0.13),
                cw[c] - Inches(0.3),
                Inches(0.45),
                val,
                size=13,
                color=ACCENT2 if c == 0 else TEXT,
                font="Consolas",
                bold=(c == 0),
            )
    add_text(s, Inches(0.6), Inches(5.85), Inches(12.0), Inches(0.4), "Loop predicate (13 rules)", size=15, bold=True, color=ACCENT2)
    add_text(
        s,
        Inches(0.6),
        Inches(6.2),
        Inches(12.0),
        Inches(0.9),
        "shape 1-5 (lb=0, step=1, ub=N const, single induction var, body terminated by yield)  ·  purity 6-9 (no side-effects beyond stores under analysis)  ·  access 10-12 (canonical affine, equal extents, unit stride)  ·  rule 13 (no stride-0 store target).",
        size=12,
        color=DIM,
    )


slides.append(slide_autovec_overview)


def slide_autovec_example():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "-csl-auto-vectorize  ·  fadds in action",
        "Drop-in replacement for the scf.for; same input MLIR, completely different emitted CSL",
    )
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "Input MLIR (scf.for over 1024 f32)", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(0.5),
        Inches(2.2),
        Inches(6.0),
        Inches(4.5),
        """csl.func @compute {
  %c0 = arith.constant 0    : index
  %n  = arith.constant 1024 : index
  %c1 = arith.constant 1    : index
  scf.for %i = %c0 to %n step %c1 {
    %va = memref.load %a[%i]
          : memref<1024xf32>
    %vb = memref.load %b[%i]
          : memref<1024xf32>
    %vc = arith.addf %va, %vb : f32
    memref.store %vc, %c[%i]
          : memref<1024xf32>
  }
  csl.return
}""",
        size=12,
        hl=MLIR_KW,
    )
    add_text(s, Inches(6.7), Inches(1.85), Inches(6.0), Inches(0.4), "After -csl-auto-vectorize  →  emitted CSL", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(6.7),
        Inches(2.2),
        Inches(6.0),
        Inches(4.5),
        """fn compute() void {
  const da = @get_dsd(mem1d_dsd, .{
     .base_address = &a, .extent = 1024
  });
  const db = @get_dsd(mem1d_dsd, .{
     .base_address = &b, .extent = 1024
  });
  const dc = @get_dsd(mem1d_dsd, .{
     .base_address = &c, .extent = 1024
  });
  @fadds(dc, da, db);
}""",
        size=12,
        hl=CSL_KW,
    )


slides.append(slide_autovec_example)


def slide_stencil_autovec():
    s = prs.slides.add_slide(BLANK)
    header(
        s,
        "-csl-auto-vectorize  ·  stencil with subview-offset",
        "c[i] = a[i-1] + a[i] for i ∈ [1, N-1)   →   two subviews of `a` + one of `c` + @fadds",
    )
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "Input MLIR", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(0.5),
        Inches(2.2),
        Inches(6.0),
        Inches(4.7),
        """csl.func @compute {
  %c1  = arith.constant 1   : index
  %Nm1 = arith.constant 127 : index
  scf.for %i = %c1 to %Nm1 step %c1 {
    %im1 = arith.subi %i, %c1 : index
    %vl  = memref.load %a[%im1]
           : memref<128xf32>
    %vc  = memref.load %a[%i]
           : memref<128xf32>
    %s   = arith.addf %vl, %vc : f32
    memref.store %s, %c[%i]
           : memref<128xf32>
  }
  csl.return
}""",
        size=12,
        hl=MLIR_KW,
    )
    add_text(s, Inches(6.7), Inches(1.85), Inches(6.0), Inches(0.4), "Emitted CSL", size=14, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(6.7),
        Inches(2.2),
        Inches(6.0),
        Inches(4.7),
        """fn compute() void {
  const da  = @get_dsd(mem1d_dsd, .{
    .base_address = &a, .extent = 128
  });
  const dc  = @get_dsd(mem1d_dsd, .{
    .base_address = &c, .extent = 128
  });
  // a[i-1]  →  base + 0
  const da0 = @increment_dsd_offset(da, 0, f32);
  // a[i]    →  base + 1
  const da1 = @increment_dsd_offset(da, 1, f32);
  // c[i]    →  base + 1
  const dc1 = @increment_dsd_offset(dc, 1, f32);
  @fadds(dc1, da0, da1);
}""",
        size=12,
        hl=CSL_KW,
    )


slides.append(slide_stencil_autovec)


def slide_control_flow():
    s = prs.slides.add_slide(BLANK)
    header(s, "Control Flow  ·  scf maps cleanly to CSL", "scf.for → while loop, scf.if → if/else, scf.index_switch → switch")
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "1-D 3-pt stencil  (scf.for + scf.if)", size=13, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(0.5),
        Inches(2.2),
        Inches(6.0),
        Inches(4.7),
        """scf.for %i = %c0 to %cN step %c1 {
  %is_b = arith.ori %ii0, %iiN : i1
  scf.if %is_b {
    %v = memref.load %x[%i] : memref<128xf32>
    memref.store %v, %y[%i] : ...
  } else {
    %vm = memref.load %x[%i-1] : ...
    %vi = memref.load %x[%i]   : ...
    %vp = memref.load %x[%i+1] : ...
    %s2 = ... 0.5*vm + vi + 0.5*vp ...
    memref.store %s2, %y[%i] : ...
  }
}""",
        size=11,
        hl=MLIR_KW,
    )
    add_text(s, Inches(6.7), Inches(1.85), Inches(6.0), Inches(0.4), "Branch on i % 3  (scf.index_switch)", size=13, bold=True, color=ACCENT)
    code_block(
        s,
        Inches(6.7),
        Inches(2.2),
        Inches(6.0),
        Inches(4.7),
        """scf.for %i = %c0 to %n step %c1 {
  %mod = arith.remsi %i, %c3 : index
  scf.index_switch %mod
  case 0 {
    %s = arith.addf %va, %vb : f32   // emits switch (mod) {
    memref.store %s, %c[%i] : ...    //   0 => { c[i] = a[i] + b[i]; }
    scf.yield                        //   1 => { c[i] = a[i] - b[i]; }
  }                                  //   else => { c[i] = a[i] * b[i]; }
  case 1 { ... }                     // }
  default { ... arith.mulf ... }
}""",
        size=11,
        hl=MLIR_KW,
    )


slides.append(slide_control_flow)


def slide_air_to_csl_e2e():
    s = prs.slides.add_slide(BLANK)
    header(s, "End-to-End  ·  AIR vecadd → emitted CSL", "One air-opt invocation; one air-translate invocation; produces a runnable directory")
    code_block(
        s,
        Inches(0.5),
        Inches(1.85),
        Inches(12.3),
        Inches(1.0),
        """$ air-opt vecadd_e2e.mlir -air-to-csl -csl-infer-exports \\
    | air-translate --emit-csl --output-dir=out/

$ ls out/vecadd/
  pe.csl       layout.csl       run.py       commands_wse2.sh    commands_wse3.sh""",
        size=13,
        hl=(),
    )
    panel_y = Inches(3.1)
    add_rect(s, Inches(0.5), panel_y, Inches(12.3), Inches(3.7), fill=PANEL)
    add_text(s, Inches(0.7), panel_y + Inches(0.18), Inches(12.0), Inches(0.4), "What -air-to-csl actually does", size=16, bold=True, color=ACCENT2)
    bullets(
        s,
        Inches(0.7),
        panel_y + Inches(0.65),
        Inches(12.0),
        Inches(3.0),
        [
            "air.launch  →  csl.host  (drives memcpy_h2d / launch / memcpy_d2h)",
            "air.segment  →  elided  (no L2 buffer concept on the wafer)",
            "air.herd over [0:N, 0]  →  csl_layout.place @pe over [0:N, 0]   (N-PE SIMD)",
            "herd body  →  csl.func @compute  (preserved verbatim, including scf.for/if)",
            "memref args  →  csl.var + csl.export   (so host can memcpy by name)",
            "verifier rejects: async tokens, channel ops, dma in body, dynamic memrefs, multiple herds, non-supported eltypes  (—  see Conversion/AIRToCSL/reject_*.mlir)",
        ],
        size=14,
    )


slides.append(slide_air_to_csl_e2e)


def slide_tests_ci():
    s = prs.slides.add_slide(BLANK)
    header(s, "Test Corpus & CI", "Lit + FileCheck for every IR transition; pre-push runs the full simulator suite")
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "Test corpus", size=15, bold=True, color=ACCENT)
    rows = [
        ("Dialect/CSL", "21 tests", "round-trip + verifier (datamovement, kernel, layout, runtime)"),
        ("Dialect/CSL/Transforms", "auto-vec", "noop, rank-2 fadds/fmacs, fall-through"),
        ("Conversion/AIRToCSL", "13 tests", "vecadd e2e, simd_herd, infer_exports, 7× reject_*"),
        ("Targets/CSLEmit/e2e", "35 tests", "elementwise, sizes, layouts, multifunc, multi_wafer, …"),
        ("CSLEmit/e2e/scientific", "8 kernels", "saxpy, dot, stencil_1d, norm_sq, even_saxpy, diag, broadcast, matrix_row_col"),
        ("CSLEmit/e2e/auto-vectorize", "8 patterns", "fadds, fsubs, fmuls, fmovs, fnegs, fmuls_scalar, fmacs_scalar, stencil_fadds"),
        ("CSLEmit/e2e/switch", "1 kernel", "switch_basic — runs on simulator"),
    ]
    x0 = Inches(0.5)
    y0 = Inches(2.25)
    cw = [Inches(3.4), Inches(1.5), Inches(7.4)]
    add_rect(s, x0, y0, sum(cw, Inches(0)), Inches(0.45), fill=ACCENT)
    for c, h in enumerate(("Dir", "Count", "What it covers")):
        add_text(s, x0 + sum(cw[:c], Inches(0)) + Inches(0.15), y0 + Inches(0.1), cw[c] - Inches(0.3), Inches(0.4), h, size=13, bold=True, color=BG)
    for i, row in enumerate(rows):
        y = y0 + Inches(0.45) + i * Inches(0.45)
        add_rect(s, x0, y, sum(cw, Inches(0)), Inches(0.45), fill=PANEL if i % 2 == 0 else CODE_BG)
        for c, val in enumerate(row):
            add_text(
                s,
                x0 + sum(cw[:c], Inches(0)) + Inches(0.15),
                y + Inches(0.08),
                cw[c] - Inches(0.3),
                Inches(0.4),
                val,
                size=12,
                color=ACCENT2 if c == 0 else TEXT,
                font="Consolas" if c < 2 else "Calibri",
                bold=(c < 2),
            )

    add_text(s, Inches(0.5), Inches(5.9), Inches(12.3), Inches(0.4), "CI / pre-push", size=15, bold=True, color=ACCENT)
    bullets(
        s,
        Inches(0.5),
        Inches(6.25),
        Inches(12.3),
        Inches(1.0),
        [
            "GitHub Actions: lit-only on stock runners (slim LLVM build, cached aggressively)",
            "Pre-push hook: full CSL suite — lit + parallel simulator runs via run_csl_ci.sh",
            "Branch policy: non-CSL workflows disabled on air-to-fire so failures stay focused",
        ],
        size=13,
    )


slides.append(slide_tests_ci)


def slide_summary():
    s = prs.slides.add_slide(BLANK)
    header(s, "Recap & What's Next", "Where we are vs. what's still on the bench")
    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4), "Recap  —  shipped on this branch", size=16, bold=True, color=GOOD)
    bullets(
        s,
        Inches(0.5),
        Inches(2.3),
        Inches(6.0),
        Inches(4.0),
        [
            "CSL dialect v2 → v5  (six op groups, csl.wafer container)",
            "-air-to-csl  lowering with verifier rejecting unsupported AIR forms",
            "--emit-csl  → pe.csl + layout.csl + run.py + numpy reference check",
            "Subgrid  csl_layout.place over [a:b, c:d]  +  N-PE SIMD via air.herd",
            "DSDs:  @get_dsd, strided memref.subview, @increment_dsd_offset",
            "-csl-auto-vectorize  with 8 patterns (fadds/fsubs/fmuls/fmovs/fnegs + scalar + rank-2)",
            "Control flow:  scf.for/if/while/index_switch  →  while/if/switch",
            "CI: lit on GitHub, full simulator on pre-push",
        ],
        size=13,
    )
    add_text(s, Inches(6.8), Inches(1.85), Inches(6.0), Inches(0.4), "What's next", size=16, bold=True, color=ACCENT2)
    bullets(
        s,
        Inches(6.8),
        Inches(2.3),
        Inches(6.0),
        Inches(4.0),
        [
            "Auto-vectorize: integer dtypes (i16/i32) — locked f32-only for T1",
            "Multi-PE communication: csl.routing + air.channel lowering",
            "AIR async tokens → csl tasks (today rejected by verifier)",
            "GEMM / tiled kernels — needs sharded multi-axis subview support",
            "Translation validation (SMT sketch in docs/superpowers/futureideas/)",
            "SPADA gap analysis driving next round of dialect ops",
        ],
        size=13,
    )
    add_text(
        s,
        Inches(0.5),
        Inches(6.6),
        Inches(12.3),
        Inches(0.4),
        "Specs:  docs/superpowers/specs/{csl-redesign, csl-dialect-v2, csl-v3-ir-refinements, csl-v4-subgrid, csl-auto-vectorize}.md",
        size=11,
        color=DIM,
        font="Consolas",
    )


slides.append(slide_summary)


def slide_qa():
    s = prs.slides.add_slide(BLANK)
    add_bg(s)
    add_rect(s, Inches(0.7), Inches(2.7), Inches(0.18), Inches(2.1), fill=ACCENT)
    add_text(s, Inches(1.0), Inches(2.7), Inches(11.5), Inches(1.0), "Questions?", size=60, bold=True, color=TEXT)
    add_text(
        s,
        Inches(1.0),
        Inches(3.9),
        Inches(11.5),
        Inches(0.6),
        "Demo:  air-opt … -air-to-csl  |  air-translate --emit-csl",
        size=18,
        color=ACCENT2,
        font="Consolas",
    )
    add_text(s, Inches(1.0), Inches(4.6), Inches(11.5), Inches(0.5), "Branch  air-to-fire  ·  35 e2e tests green on simulator", size=14, color=DIM)


slides.append(slide_qa)


TOTAL = len(slides)
for i, fn in enumerate(slides, start=1):
    fn()
    last = prs.slides[-1]
    if i not in (1, TOTAL):
        footer(last, i)

out = "/home/bricklib_dataflow/air-csl/mlir-air/out/mlir_air_csl_deck.pptx"
prs.save(out)
print(f"Wrote {out}  ({TOTAL} slides)")
