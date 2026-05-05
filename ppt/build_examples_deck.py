"""CSL Examples v3 — short focused before/after snippets.
Each slide: one concept, max 12 lines per panel, full-width columns.

Run:
    /home/bricklib_dataflow/sdk/sdk_venv/bin/python ppt/build_examples_deck.py
Output:
    out/mlir_air_csl_examples_v2.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ── theme ─────────────────────────────────────────────────────────────────────
BG      = RGBColor(0x10, 0x16, 0x22)
PANEL   = RGBColor(0x18, 0x20, 0x30)
ACCENT  = RGBColor(0x4F, 0xC3, 0xF7)
ACCENT2 = RGBColor(0xFF, 0xB7, 0x4D)
GOOD    = RGBColor(0x81, 0xC7, 0x84)
WARN    = RGBColor(0xFF, 0x80, 0x80)
TEXT    = RGBColor(0xE6, 0xEC, 0xF2)
DIM     = RGBColor(0x9A, 0xA5, 0xB1)
CODE_BG = RGBColor(0x0B, 0x10, 0x18)
CODE_FG = RGBColor(0xC9, 0xD1, 0xD9)
KEYWORD = RGBColor(0xFF, 0x9D, 0x55)
COMMENT = RGBColor(0x70, 0x80, 0x90)
SECT_BG = RGBColor(0x08, 0x0D, 0x16)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK  = prs.slide_layouts[6]

# column geometry — full-width use
LX = Inches(0.22)   # left col x
LW = Inches(6.40)   # left col width  (ends 6.62")
RX = Inches(6.75)   # right col x
RW = Inches(6.40)   # right col width (ends 13.15")
CY = Inches(2.18)   # code top y
CH = Inches(4.62)   # code height (bottom at 6.80")

MLIR_KW = (
    "csl.wafer","csl.program","csl.func","csl.var","csl.export","csl.return",
    "csl.host","csl.layout","csl.get_mem_dsd","csl.builtin_call",
    "csl.dataflow.put","csl.dataflow.get","csl.color",
    "csl_layout.place","csl_layout.dataflow","csl_layout.set_color_config",
    "csl_host.memcpy_h2d","csl_host.memcpy_d2h","csl_host.launch",
    "air.launch","air.segment","air.herd",
    "air.herd_terminator","air.segment_terminator","air.launch_terminator",
    "func.func","func.return","func.call",
    "memref.load","memref.store","memref.subview","memref",
    "arith.constant","arith.addf","arith.subf","arith.mulf","arith.divf",
    "arith.negf","arith.maximumf","arith.minimumf","arith.xori",
    "arith.addi","arith.subi","arith.muli","arith.remsi",
    "arith.cmpf","arith.cmpi","arith.index_cast",
    "scf.for","scf.if","scf.yield","scf.index_switch",
    "case","default","else","over","at","!csl.dsd",
    "f32","i32","i1","index","strided",
)

CSL_KW = (
    "fn","var","const","comptime","while","if","else","switch","return",
    "void","f32","u16","i16","i32","bool",
    "@import_module","@get_dsd","@set_rectangle","@set_tile_code",
    "@export_symbol","@get_color",
    "@fmacs","@fadds","@fmuls","@fmovs","@fnegs","@fsubs",
    "@increment_dsd_offset","@as",
    "mem1d_dsd","fabout_dsd","fabin_dsd",
)

PY_KW = ("import","from","def","if","for","while","True","False","None",
         "assert","return","as","in","not","and","or")


# ── primitives ────────────────────────────────────────────────────────────────

def add_bg(slide, color=BG):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    s.fill.solid(); s.fill.fore_color.rgb = color
    s.line.fill.background(); s.shadow.inherit = False

def add_rect(slide, x, y, w, h, fill=PANEL, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(0.75)
    s.shadow.inherit = False
    return s

def add_text(slide, x, y, w, h, text, *, size=18, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top   = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run()
    r.text = text; r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.name = font

def bullets(slide, x, y, w, h, items, *, size=13, color=TEXT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    for i, txt in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT; p.line_spacing = 1.25
        r = p.add_run(); r.text = "▸  "; r.font.size = Pt(size)
        r.font.color.rgb = ACCENT; r.font.bold = True; r.font.name = "Calibri"
        r2 = p.add_run(); r2.text = txt; r2.font.size = Pt(size)
        r2.font.color.rgb = color; r2.font.name = "Calibri"

def code_block(slide, x, y, w, h, code, *, size=11, hl=()):
    bg = add_rect(slide, x, y, w, h, fill=CODE_BG)
    bg.line.color.rgb = RGBColor(0x2A, 0x33, 0x42); bg.line.width = Pt(0.5)
    PAD = Emu(50000)
    tb = slide.shapes.add_textbox(x+PAD, y+PAD, w-2*PAD, h-2*PAD)
    tf = tb.text_frame; tf.word_wrap = False
    bodyPr = tf._txBody.find(qn('a:bodyPr'))
    if bodyPr is not None:
        bodyPr.set('wrap', 'none')
        for tag in (qn('a:noAutofit'), qn('a:spAutoFit'), qn('a:normAutofit')):
            for el in bodyPr.findall(tag): bodyPr.remove(el)
        etree.SubElement(bodyPr, qn('a:noAutofit'))
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top   = tf.margin_bottom = Emu(0)
    for li, line in enumerate(code.split("\n")):
        p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        stripped = line.lstrip(" ")
        indent   = line[:len(line)-len(stripped)]
        if indent:
            r = p.add_run(); r.text = indent
            r.font.name = "Consolas"; r.font.size = Pt(size)
            r.font.color.rgb = CODE_FG
        if stripped.startswith("//") or stripped.startswith("#"):
            r = p.add_run(); r.text = stripped
            r.font.name = "Consolas"; r.font.size = Pt(size)
            r.font.color.rgb = COMMENT; continue
        rest = stripped; i = 0
        while i < len(rest):
            matched = None
            for kw in hl:
                if rest.startswith(kw, i):
                    if i > 0 and (rest[i-1].isalnum() or rest[i-1] in "_."):
                        continue
                    matched = kw; break
            if matched:
                r = p.add_run(); r.text = matched
                r.font.name = "Consolas"; r.font.size = Pt(size)
                r.font.color.rgb = KEYWORD; r.font.bold = True
                i += len(matched)
            else:
                j = i+1
                while j < len(rest):
                    skip = False
                    for kw in hl:
                        if rest.startswith(kw, j):
                            if j>0 and (rest[j-1].isalnum() or rest[j-1] in "_."):
                                continue
                            skip = True; break
                    if skip: break
                    j += 1
                r = p.add_run(); r.text = rest[i:j]
                r.font.name = "Consolas"; r.font.size = Pt(size)
                r.font.color.rgb = CODE_FG; i = j

def header(slide, title, subtitle=None):
    add_bg(slide)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0.4), Inches(0.55),
                                  Inches(0.12), Inches(0.55))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    add_text(slide, Inches(0.65), Inches(0.45), Inches(12.2), Inches(0.7),
             title, size=26, bold=True, color=TEXT)
    if subtitle:
        add_text(slide, Inches(0.65), Inches(1.1), Inches(12.2), Inches(0.38),
                 subtitle, size=12, color=DIM)
    add_rect(slide, Inches(0.4), Inches(7.05), Inches(12.55), Emu(9525), fill=PANEL)

def side_labels(slide, left, right):
    add_text(slide, LX, Inches(1.82), LW, Inches(0.32),
             left,  size=11, bold=True, color=ACCENT)
    add_text(slide, RX, Inches(1.82), RW, Inches(0.32),
             right, size=11, bold=True, color=ACCENT)

def divider(slide):
    ln = slide.shapes.add_connector(
        1, Inches(6.63), Inches(1.78), Inches(6.63), Inches(7.0))
    ln.line.color.rgb = RGBColor(0x2A, 0x33, 0x42); ln.line.width = Pt(0.75)

def arrow(slide, y=Inches(4.44)):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                Inches(6.35), y, Inches(0.45), Inches(0.32))
    s.fill.solid(); s.fill.fore_color.rgb = ACCENT2
    s.line.fill.background(); s.shadow.inherit = False

def label_chip(slide, x, y, w, h, text, *, fill=ACCENT, fg=BG):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background(); s.shadow.inherit = False
    s.adjustments[0] = 0.5
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Emu(40000)
    tf.margin_top  = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(11); r.font.bold = True
    r.font.color.rgb = fg; r.font.name = "Calibri"

def section_break(slide, num, name, desc):
    add_bg(slide, SECT_BG)
    for i in range(1, 8):
        x = Inches(0.5 + i * 1.6)
        ln = slide.shapes.add_connector(1, x, Inches(0), x, SH)
        ln.line.color.rgb = RGBColor(0x14, 0x1C, 0x28); ln.line.width = Pt(0.5)
    add_rect(slide, Inches(0.7), Inches(2.2), Inches(0.22), Inches(2.8), fill=ACCENT)
    add_text(slide, Inches(1.1), Inches(2.1), Inches(11.5), Inches(0.55),
             f"Section {num}", size=16, bold=True, color=ACCENT2, font="Consolas")
    add_text(slide, Inches(1.1), Inches(2.6), Inches(11.0), Inches(1.4),
             name, size=52, bold=True, color=TEXT)
    add_text(slide, Inches(1.1), Inches(4.2), Inches(11.0), Inches(0.6),
             desc, size=16, color=DIM)

def src_tag(slide, tag):
    add_text(slide, LX, Inches(6.9), Inches(12.5), Inches(0.3),
             f"src: {tag}", size=9, color=DIM, font="Consolas")

slides = []

# ─────────────────────────────────────────────────────────────────────────────
# 01 TITLE
# ─────────────────────────────────────────────────────────────────────────────
def slide_title():
    s = prs.slides.add_slide(BLANK)
    add_bg(s)
    for i in range(1, 8):
        x = Inches(0.5 + i * 1.6)
        ln = s.shapes.add_connector(1, x, Inches(0), x, SH)
        ln.line.color.rgb = RGBColor(0x1A, 0x22, 0x33); ln.line.width = Pt(0.5)
    add_rect(s, Inches(0.7), Inches(1.6), Inches(0.18), Inches(2.6), fill=ACCENT)
    add_text(s, Inches(1.0), Inches(1.55), Inches(11.5), Inches(0.6),
             "MLIR · CSL  ·  examples-v3", size=16, bold=True, color=ACCENT2, font="Consolas")
    add_text(s, Inches(1.0), Inches(2.05), Inches(11.5), Inches(1.6),
             "Lowering AIR to CSL:\nBefore & After Each Pass",
             size=42, bold=True, color=TEXT)
    add_text(s, Inches(1.0), Inches(4.35), Inches(11.5), Inches(0.5),
             "Left: MLIR dialect input   ·   Right: emitted CSL output",
             size=15, color=DIM, font="Consolas")
    for i, (lbl, col) in enumerate([
        ("A · Passes", ACCENT), ("B · Constructs", ACCENT2),
        ("C · DSDs", GOOD), ("D · Multi-PE", RGBColor(0xCE,0x93,0xD8))
    ]):
        label_chip(s, Inches(1.0 + i*2.95), Inches(5.15), Inches(2.6), Inches(0.42),
                   lbl, fill=col, fg=BG)
    add_text(s, Inches(1.0), Inches(6.5), Inches(11.5), Inches(0.4),
             "branch: air-to-fire  ·  mlir-air  ·  Cerebras WSE-3",
             size=11, color=DIM, font="Consolas")
slides.append(slide_title)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A
# ─────────────────────────────────────────────────────────────────────────────
def slide_sec_a():
    s = prs.slides.add_slide(BLANK)
    section_break(s, "A", "The Passes",
                  "-air-to-csl  ·  -csl-auto-vectorize  ·  -csl-infer-exports  ·  --emit-csl")
slides.append(slide_sec_a)

# A1: Pipeline
def slide_pipeline():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass Pipeline  ·  Overview",
           "Two tools, four passes — transforms AIR dialect to runnable CSL text")
    steps = [
        ("AIR MLIR",         "air.herd\nair.dma_memcpy_nd",          BG,    ACCENT),
        ("-air-to-csl",       "herd body → csl.program\nlayout + host scaffolded", PANEL, ACCENT2),
        ("-csl-auto-\nvectorize","scf.for idioms → DSD builtins\n@fadds / @fmacs …", PANEL, GOOD),
        ("-csl-infer-\nexports","insert csl.export per memcpy\nalias / kind / direction", PANEL, ACCENT2),
        ("--emit-csl",        "pe.csl + layout.csl\nrun.py + commands_wse3.sh", PANEL, GOOD),
    ]
    bw, bh, gap = Inches(2.25), Inches(1.48), Inches(0.14)
    y0, x0 = Inches(2.05), Inches(0.3)
    for i, (title, sub, bg_col, bar_col) in enumerate(steps):
        x = x0 + i*(bw+gap)
        add_rect(s, x, y0, bw, bh, fill=bg_col, line=RGBColor(0x33,0x40,0x55))
        add_rect(s, x, y0, bw, Inches(0.13), fill=bar_col)
        add_text(s, x+Inches(0.1), y0+Inches(0.2), bw-Inches(0.2), Inches(0.52),
                 title, size=12, bold=True, color=TEXT, font="Consolas")
        add_text(s, x+Inches(0.1), y0+Inches(0.76), bw-Inches(0.2), Inches(0.65),
                 sub, size=10, color=DIM)
        if i < len(steps)-1:
            ax = x + bw + Inches(0.02); ay = y0 + bh/2 - Inches(0.1)
            sv = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, ax, ay, Inches(0.18), Inches(0.2))
            sv.fill.solid(); sv.fill.fore_color.rgb = ACCENT
            sv.line.fill.background(); sv.shadow.inherit = False
    add_text(s, Inches(0.3), Inches(3.78), Inches(12.7), Inches(0.34),
             "Command:", size=13, bold=True, color=ACCENT2)
    code_block(s, Inches(0.3), Inches(4.12), Inches(12.7), Inches(0.76),
        "air-opt input.mlir -air-to-csl -csl-auto-vectorize -csl-infer-exports \\\n"
        "  | air-translate --emit-csl --output-dir=out/<wafer>/", size=12)
    add_text(s, Inches(0.3), Inches(5.08), Inches(12.7), Inches(0.34),
             "Emitter outputs per wafer:", size=13, bold=True, color=ACCENT2)
    code_block(s, Inches(0.3), Inches(5.42), Inches(12.7), Inches(1.26),
        "out/<wafer-name>/\n"
        "  pe.csl            ← one file per csl.program\n"
        "  layout.csl        ← grid rectangle + tile placement\n"
        "  run.py            ← SdkRuntime host + numpy golden check\n"
        "  commands_wse3.sh  ← cslc compile + cs_python simulator launch",
        size=11)
slides.append(slide_pipeline)

# A2: -air-to-csl  (from air_to_csl_vecadd.mlir)
def slide_air_to_csl():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass: -air-to-csl",
           "air.launch/segment/herd → csl.wafer/program/func  ·  herd args become csl.var")
    side_labels(s, "MLIR input  —  air.herd vecadd body",
                   "CSL IR output  —  csl.program @h")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """air.herd @h tile(%x,%y) in (%sx=%one,%sy=%one)
    args(%ha=%a, %hb=%b, %hc=%c)
    : memref<256xf32>, memref<256xf32>,
      memref<256xf32> {
  %lo = arith.constant 0   : index
  %hi = arith.constant 256 : index
  %c1 = arith.constant 1   : index
  scf.for %i = %lo to %hi step %c1 {
    %va = memref.load %ha[%i] : memref<256xf32>
    %vb = memref.load %hb[%i] : memref<256xf32>
    %vc = arith.addf %va, %vb : f32
    memref.store %vc, %hc[%i] : memref<256xf32>
  }
  air.herd_terminator
}""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """csl.program @h {
  // herd args → module-level vars
  %a = csl.var @arg0 : memref<256xf32>
  %b = csl.var @arg1 : memref<256xf32>
  %c = csl.var @arg2 : memref<256xf32>
  csl.func @compute {
    %lo = arith.constant 0   : index
    %hi = arith.constant 256 : index
    %c1 = arith.constant 1   : index
    scf.for %i = %lo to %hi step %c1 {
      %va = memref.load %a[%i] : memref<256xf32>
      %vb = memref.load %b[%i] : memref<256xf32>
      %vc = arith.addf %va, %vb : f32
      memref.store %vc, %c[%i] : memref<256xf32>
    }
    csl.return
  }
}""", size=11, hl=MLIR_KW)
    src_tag(s, "Conversion/AIRToCSL/air_to_csl_vecadd.mlir")
slides.append(slide_air_to_csl)

# A3: -csl-auto-vectorize @fadds  (from auto-vectorize/fadds.mlir)
def slide_fadds():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass: -csl-auto-vectorize  ·  @fadds",
           "Element-wise c[i]=a[i]+b[i] loop pattern → single SIMD call  (src: e2e/auto-vectorize/fadds.mlir)")
    side_labels(s, "Before  —  scf.for + arith.addf",
                   "After  —  DSD get + @fadds call")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(2.9),
        """csl.func @compute {
  %c0 = arith.constant 0    : index
  %n  = arith.constant 1024 : index
  %c1 = arith.constant 1    : index
  scf.for %i = %c0 to %n step %c1 {
    %va = memref.load %a[%i] : memref<1024xf32>
    %vb = memref.load %b[%i] : memref<1024xf32>
    %vc = arith.addf %va, %vb : f32
    memref.store %vc, %c[%i] : memref<1024xf32>
  }
  csl.return
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(2.9),
        """fn compute() void {
  const da = @get_dsd(mem1d_dsd, .{
    .base_address = &a, .extent = 1024 });
  const db = @get_dsd(mem1d_dsd, .{
    .base_address = &b, .extent = 1024 });
  const dc = @get_dsd(mem1d_dsd, .{
    .base_address = &c, .extent = 1024 });
  @fadds(dc, da, db);
  sys_mod.unblock_cmd_stream();
}""", size=12, hl=CSL_KW)
    # pattern table
    add_text(s, LX, Inches(5.35), Inches(12.5), Inches(0.34),
             "All six element-wise patterns recognised:", size=13, bold=True, color=ACCENT2)
    rows = [
        ("@fadds","c[i] = a[i] + b[i]"),("@fsubs","c[i] = a[i] - b[i]"),
        ("@fmuls","c[i] = a[i] * b[i]"),("@fmacs","c[i] = α*a[i] + c[i]  (loop-invariant α)"),
        ("@fmovs","c[i] = a[i]"),       ("@fnegs","c[i] = -a[i]"),
    ]
    x0, y0 = LX, Inches(5.72)
    cw = [Inches(1.55), Inches(10.7)]
    for i, (bi, pat) in enumerate(rows):
        y = y0 + i*Inches(0.185)
        add_rect(s, x0, y, sum(cw), Inches(0.19), fill=PANEL if i%2==0 else CODE_BG)
        add_text(s, x0+Inches(0.1), y+Inches(0.02), cw[0]-Inches(0.15),
                 Inches(0.18), bi, size=11, bold=True, color=KEYWORD, font="Consolas")
        add_text(s, x0+cw[0]+Inches(0.1), y+Inches(0.02), cw[1],
                 Inches(0.18), pat, size=11, color=CODE_FG, font="Consolas")
slides.append(slide_fadds)

# A4: @fmacs SAXPY  (from auto-vectorize/fmacs_scalar.mlir)
def slide_fmacs():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass: -csl-auto-vectorize  ·  @fmacs  (SAXPY)",
           "y[i] = α*A[i]+y[i] with loop-invariant scalar → @fmacs  (src: auto-vectorize/fmacs_scalar.mlir)")
    side_labels(s, "Before  —  mulf + addf inside loop",
                   "After  —  scalar α hoisted, @fmacs")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(3.3),
        """csl.func @compute {
  %alpha = arith.constant 2.0 : f32
  %c0 = arith.constant 0   : index
  %n  = arith.constant 128 : index
  %c1 = arith.constant 1   : index
  scf.for %i = %c0 to %n step %c1 {
    %va = memref.load %A[%i] : memref<128xf32>
    %vy = memref.load %y[%i] : memref<128xf32>
    %m  = arith.mulf %va, %alpha : f32
    %s  = arith.addf %m, %vy : f32
    memref.store %s, %y[%i] : memref<128xf32>
  }
  csl.return
}""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(3.3),
        """fn compute() void {
  // loop-invariant scalar hoisted out
  const alpha: f32 = 2.0;
  const dA = @get_dsd(mem1d_dsd, .{
    .base_address = &A, .extent = 128 });
  const dy = @get_dsd(mem1d_dsd, .{
    .base_address = &y, .extent = 128 });
  // y[i] = alpha * A[i] + y[i]
  @fmacs(dy, dy, dA, alpha);
  sys_mod.unblock_cmd_stream();
}""", size=11, hl=CSL_KW)
    bullets(s, LX, Inches(5.72), Inches(12.5), Inches(1.0), [
        "lb=0 step=1 ub=compile-time const  ·  one store target  ·  no other side-effects",
        "Scalar multiplier must be loop-invariant  ·  stencil_fadds.mlir shows 2D variant",
    ], size=12)
slides.append(slide_fmacs)

# A5: -csl-infer-exports  (from Conversion/AIRToCSL/infer_exports.mlir)
def slide_infer_exports():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass: -csl-infer-exports",
           "Reads host memcpy ops → inserts csl.export with alias / kind / direction  (src: infer_exports.mlir)")
    side_labels(s, "Before  —  no exports, host ops only",
                   "After  —  csl.export injected")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(2.85),
        """csl.program @pe {
  %a = csl.var @a : memref<256xf32>
  %b = csl.var @b : memref<256xf32>
  %c = csl.var @c : memref<256xf32>
  csl.func @compute { ... csl.return }
  // no csl.export ops yet
}
// host uses memcpy to give hints:
csl_host.memcpy_h2d %a_in to @layout::@a ...
csl_host.launch @layout::@compute
csl_host.memcpy_d2h @layout::@c to %c_out ...""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(2.85),
        """csl.program @pe {
  %a = csl.var @a : memref<256xf32>
  %b = csl.var @b : memref<256xf32>
  %c = csl.var @c : memref<256xf32>
  csl.func @compute { ... csl.return }
  // inserted by -csl-infer-exports:
  csl.export @a {alias="a", direction="in"}
  csl.export @b {alias="b", direction="in"}
  csl.export @c {alias="c", direction="out"}
  csl.export @compute {kind="func"}
}""", size=11, hl=MLIR_KW)
    add_text(s, LX, Inches(5.28), Inches(12.5), Inches(0.34),
             "--emit-csl translates exports to comptime block:", size=13, bold=True, color=ACCENT2)
    code_block(s, LX, Inches(5.62), Inches(12.5), Inches(1.05),
        """comptime {
  @export_symbol(a_ptr, "a");    // direction=in  → var ptr (host writes)
  @export_symbol(c_ptr, "c");    // direction=out → const ptr (host reads)
  @export_symbol(compute);       // kind=func
}""", size=11, hl=CSL_KW)
    src_tag(s, "Conversion/AIRToCSL/infer_exports.mlir  ·  derive_exports.mlir")
slides.append(slide_infer_exports)

# A6: --emit-csl output files
def slide_emit_csl():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass: --emit-csl  ·  Three Output Files",
           "Translates CSL dialect IR to text — one pe.csl per program, one layout.csl, one run.py")
    side_labels(s, "layout.csl  —  grid setup + tile placement",
                   "run.py  —  SDK host driver")
    divider(s)
    code_block(s, LX, CY, LW, CH,
        """// layout.csl (emitted for a 4x2 SIMD grid)
const memcpy = @import_module(
  "<memcpy/get_params>",
  .{ .width = 4, .height = 2 });

layout {
  @set_rectangle(4, 2);
  // csl_layout.place over [0:4, 0:2]
  // → while loop over all tiles
  var px: u16 = 0;
  while (px < 4) : (px += 1) {
    var py: u16 = 0;
    while (py < 2) : (py += 1) {
      @set_tile_code(px, py, "pe.csl", .{
        .memcpy_params =
            memcpy.get_params(px + py * 4),
      });
    }
  }
}""", size=11, hl=CSL_KW)
    code_block(s, RX, CY, RW, CH,
        """# run.py  (emitted host driver)
import numpy as np
from cerebras.sdk.runtime.sdkruntimepb2 import *

runner = SdkRuntime("simd",
    simfab_numthreads=32)
runner.load()
runner.run()

# memcpy_h2d: width=4, height=2 → 8 PEs
runner.memcpy_h2d(
    runner.get_id("A"), A_data,
    0, 0, 4, 2, 32, streaming=False,
    data_type=memcpy_dtype_t.MEMCPY_32BIT)
runner.launch("compute", nonblock=False)
runner.memcpy_d2h(y_out,
    runner.get_id("y"), 0, 0, 4, 2, 32, ...)
runner.stop()
assert np.allclose(y_out, ref)
print("SUCCESS!")""", size=11, hl=PY_KW)
    src_tag(s, "e2e/simd_dsd.mlir  ·  e2e/tutorials/gemv-05-multiple-pes.mlir")
slides.append(slide_emit_csl)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — LANGUAGE CONSTRUCTS
# ─────────────────────────────────────────────────────────────────────────────
def slide_sec_b():
    s = prs.slides.add_slide(BLANK)
    section_break(s, "B", "Language Constructs",
                  "Types · loops · branches · switch · arithmetic · memory · helpers")
slides.append(slide_sec_b)

# B1: Type table
def slide_types():
    s = prs.slides.add_slide(BLANK)
    header(s, "Type Lowering  ·  MLIR → CSL",
           "The emitter maps each MLIR type to its CSL equivalent at point of use")
    rows = [
        ("index",                 "u16",         "Loop vars, array indices (WSE ≤65535)"),
        ("f32",                   "f32",          "Direct map"),
        ("i32",                   "i32",          "Direct map"),
        ("i1",                    "bool",         "arith.cmpf / arith.cmpi results"),
        ("memref<N×f32>  (var)",  "[N]f32",       "Module-level csl.var — PE array storage"),
        ("memref strided<[S],?> ","DSD + stride", "subview with runtime offset → @increment_dsd_offset"),
        ("!csl.dsd",              "@get_dsd(…)",  "Runtime DSD — used by @fadds/@fmacs; never stored"),
        ("arith.constant N:index","u16 = N",      "Folded to immediate; drives while-loop bound"),
        ("arith.constant X:f32",  "f32 = X",      "Folded to float literal; loop-invariant scalars hoisted"),
    ]
    x0, y0 = Inches(0.3), Inches(1.88)
    cw = [Inches(2.85), Inches(2.05), Inches(7.55)]
    rh = Inches(0.52)
    add_rect(s, x0, y0, sum(cw), Inches(0.44), fill=ACCENT)
    for c, h in enumerate(("MLIR type / value","CSL type","Notes")):
        add_text(s, x0+sum(cw[:c])+Inches(0.1), y0+Inches(0.09),
                 cw[c]-Inches(0.18), Inches(0.36), h, size=13, bold=True, color=BG)
    for i, (mlir, csl, note) in enumerate(rows):
        y = y0 + Inches(0.44) + i*rh
        add_rect(s, x0, y, sum(cw), rh, fill=PANEL if i%2==0 else CODE_BG)
        add_text(s, x0+Inches(0.1), y+Inches(0.11), cw[0]-Inches(0.18),
                 rh-Inches(0.14), mlir, size=11, color=ACCENT2, font="Consolas")
        add_text(s, x0+cw[0]+Inches(0.1), y+Inches(0.11), cw[1]-Inches(0.18),
                 rh-Inches(0.14), csl, size=11, color=GOOD, font="Consolas")
        add_text(s, x0+cw[0]+cw[1]+Inches(0.1), y+Inches(0.11), cw[2]-Inches(0.18),
                 rh-Inches(0.14), note, size=11, color=DIM)
slides.append(slide_types)

# B2: scf.for → while  (from e2e/control_flow.mlir)
def slide_loop():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: scf.for  →  while",
           "lb=0, step=1, ub=const — induction var becomes u16  (src: e2e/control_flow.mlir)")
    side_labels(s, "MLIR  —  scf.for copy loop",
                   "Emitted CSL  —  while + u16 index")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(2.45),
        """// Simple loop: copy a[i] to b[i]
%c0 = arith.constant 0  : index
%n  = arith.constant 64 : index
%c1 = arith.constant 1  : index
scf.for %i = %c0 to %n step %c1 {
  %v = memref.load %a[%i] : memref<64xf32>
  memref.store %v, %b[%i] : memref<64xf32>
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(2.45),
        """// index → u16; lb=0 as initialiser
var i0: u16 = 0;
while (i0 < 64) : (i0 += 1) {
  var v: f32 = a[i0];
  b[i0] = v;
}""", size=12, hl=CSL_KW)
    # nested loop
    add_text(s, LX, Inches(4.9), Inches(12.5), Inches(0.32),
             "Nested scf.for — each loop gets a fresh u16 variable:", size=12, bold=True, color=ACCENT2)
    code_block(s, LX, Inches(5.25), LW, Inches(1.55),
        """scf.for %i = %c0 to %c8 step %c1 {
  scf.for %j = %c0 to %c8 step %c1 {
    %v = memref.load %m[%j]: memref<64xf32>
    memref.store %v, %m[%j]: memref<64xf32>
  }
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, Inches(5.25), RW, Inches(1.55),
        """var i1: u16 = 0;
while (i1 < 8) : (i1 += 1) {
  var j0: u16 = 0;
  while (j0 < 8) : (j0 += 1) {
    var v: f32 = m[j0]; m[j0] = v;
  }
}""", size=12, hl=CSL_KW)
slides.append(slide_loop)

# B3: scf.if → if / if-else  (from e2e/control_flow_if.mlir)
def slide_if():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: scf.if  →  if / if-else",
           "arith.cmpf/cmpi → bool; scf.if → if; else region → else  (src: e2e/control_flow_if.mlir)")
    side_labels(s, "MLIR  —  clamp negative (no else)",
                   "Emitted CSL")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(2.5),
        """%zero = arith.constant 0.0 : f32
scf.for %i = %c0 to %c8 step %c1 {
  %v  = memref.load %x[%i] : memref<8xf32>
  // arith.cmpf olt → bool
  %lt = arith.cmpf olt, %v, %zero : f32
  scf.if %lt {
    memref.store %zero, %x[%i] : memref<8xf32>
  }
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(2.5),
        """var i0: u16 = 0;
while (i0 < 8) : (i0 += 1) {
  var v: f32 = x[i0];
  const lt: bool = v < 0.0;
  if (lt) {
    x[i0] = 0.0;
  }
}""", size=12, hl=CSL_KW)
    add_text(s, LX, Inches(4.92), Inches(12.5), Inches(0.32),
             "With else — endpoint vs interior:", size=12, bold=True, color=ACCENT2)
    code_block(s, LX, Inches(5.27), LW, Inches(1.55),
        """%is_zero = arith.cmpi eq, %vi, %zero_i32 : i32
scf.if %is_zero {
  memref.store %v, %y[%i] : memref<8xf32>
} else {
  %d = arith.mulf %v, %two : f32
  memref.store %d, %y[%i] : memref<8xf32>
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, Inches(5.27), RW, Inches(1.55),
        """const is_zero: bool = vi == 0;
if (is_zero) {
  y[i1] = v;
} else {
  var d: f32 = v * 2.0;
  y[i1] = d;
}""", size=12, hl=CSL_KW)
slides.append(slide_if)

# B4: scf.index_switch → switch  (from e2e/switch/switch_basic.mlir)
def slide_switch():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: scf.index_switch  →  switch",
           "Each case region → CSL arm; default → else  (src: e2e/switch/switch_basic.mlir)")
    side_labels(s, "MLIR  —  index_switch on (i % 3)",
                   "Emitted CSL  —  switch statement")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """%c3  = arith.constant 3 : index
%mod = arith.remsi %i, %c3 : index

scf.index_switch %mod
case 0 {
  %s = arith.addf %va, %vb : f32
  memref.store %s, %c[%i] : memref<64xf32>
  scf.yield
}
case 1 {
  %s = arith.subf %va, %vb : f32
  memref.store %s, %c[%i] : memref<64xf32>
  scf.yield
}
default {
  %s = arith.mulf %va, %vb : f32
  memref.store %s, %c[%i] : memref<64xf32>
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """var mod: u16 = i0 % 3;

switch (mod) {
  0 => {
    var s: f32 = va + vb;
    c[i0] = s;
  },
  1 => {
    var s: f32 = va - vb;
    c[i0] = s;
  },
  else => {
    var s: f32 = va * vb;
    c[i0] = s;
  },
}""", size=12, hl=CSL_KW)
slides.append(slide_switch)

# B5: Arithmetic ops table
def slide_arith():
    s = prs.slides.add_slide(BLANK)
    header(s, "Arithmetic Ops  ·  Quick Reference",
           "All arith.* ops the emitter handles; each SSA def → var or const in CSL")
    rows = [
        ("arith.addf %a,%b : f32",      "a + b",         "f32",  ""),
        ("arith.subf %a,%b : f32",      "a - b",         "f32",  ""),
        ("arith.mulf %a,%b : f32",      "a * b",         "f32",  ""),
        ("arith.divf %a,%b : f32",      "a / b",         "f32",  "arith_extra.mlir"),
        ("arith.negf %a : f32",         "-a",            "f32",  "arith_extra.mlir"),
        ("arith.maximumf %a,%b : f32",  "if a>b  a else b", "f32","ReLU; if-expr"),
        ("arith.minimumf %a,%b : f32",  "if a<b  a else b", "f32","clamp; if-expr"),
        ("arith.addi %a,%b : index",    "a + b",         "u16",  ""),
        ("arith.muli %a,%b : index",    "a * b",         "u16",  "2-D index"),
        ("arith.remsi %a,%b : index",   "a % b",         "u16",  "switch discriminant"),
        ("arith.cmpf olt/oeq …",        "a < b / a==b",  "bool", "drives scf.if"),
        ("arith.cmpi eq/ne …",          "a==b / a!=b",   "bool", "drives scf.if"),
        ("arith.xori %a,%b : i1",       "a != b",        "bool", "logical XOR on bool"),
        ("arith.index_cast %i : index to i32", "@as(i32,i)", "i32","for cmpi"),
    ]
    x0, y0 = Inches(0.3), Inches(1.85)
    cw = [Inches(4.25), Inches(2.35), Inches(0.82), Inches(4.25)]
    rh = Inches(0.35)
    add_rect(s, x0, y0, sum(cw), Inches(0.42), fill=ACCENT)
    for c, h in enumerate(("MLIR op","CSL output","Type","Notes")):
        add_text(s, x0+sum(cw[:c])+Inches(0.08), y0+Inches(0.08),
                 cw[c]-Inches(0.15), Inches(0.36), h, size=12, bold=True, color=BG)
    for i, row in enumerate(rows):
        y = y0 + Inches(0.42) + i*rh
        add_rect(s, x0, y, sum(cw), rh, fill=PANEL if i%2==0 else CODE_BG)
        cols_color = [ACCENT2, CODE_FG, GOOD, DIM]
        for c, (val, col) in enumerate(zip(row, cols_color)):
            add_text(s, x0+sum(cw[:c])+Inches(0.08), y+Inches(0.04),
                     cw[c]-Inches(0.15), rh-Inches(0.07),
                     val, size=10, color=col, font="Consolas")
slides.append(slide_arith)

# B6: Memory load/store  (from e2e/kernels.mlir dot)
def slide_memory():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: memref.load / store  ·  2-D index",
           "Array reads/writes map to direct CSL indexing  (src: e2e/kernels.mlir dot wafer)")
    side_labels(s, "MLIR  —  load, store, 2-D index",
                   "Emitted CSL")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """// load → scalar read
%va = memref.load %a[%i] : memref<128xf32>

// store → scalar write
memref.store %va, %b[%i] : memref<128xf32>

// 2-D linear index: row i, col j of A[M×N]
%iN   = arith.muli %i, %n   : index
%iNj  = arith.addi %iN, %j  : index
%aij  = memref.load %A[%iNj]: memref<24xf32>
%xj   = memref.load %x[%j]  : memref<6xf32>
%prod = arith.mulf %aij, %xj : f32
%acc  = memref.load %y[%i]  : memref<4xf32>
%s    = arith.addf %acc, %prod : f32
memref.store %s, %y[%i]    : memref<4xf32>""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// load → inline read
var va: f32 = a[i0];

// store → inline write
b[i0] = va;

// 2-D linear index
var iN: u16  = i0 * 6;
var iNj: u16 = iN + j0;
var aij: f32 = A[iNj];
var xj: f32  = x[j0];
var prod: f32 = aij * xj;
var acc: f32  = y[i0];
var s: f32   = acc + prod;
y[i0] = s;""", size=11, hl=CSL_KW)
    src_tag(s, "e2e/kernels.mlir  ·  e2e/tutorials/gemv-01-complete-program.mlir")
slides.append(slide_memory)

# B7: memref.subview → @get_dsd variants  (from e2e/subview_patterns.mlir)
def slide_subview():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: memref.subview  →  @get_dsd  variants",
           "Four subview patterns; defaults suppressed; src: e2e/subview_patterns.mlir")
    side_labels(s, "MLIR  —  four subview patterns",
                   "Emitted CSL  —  @get_dsd + @increment variants")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """// 1) stride=1, offset=0 (both defaults)
%d1 = csl.get_mem_dsd %x : memref<128xf32> -> !csl.dsd

// 2) offset=8, stride=1 (non-default offset only)
%v2 = memref.subview %x[8][64][1]
      : memref<128xf32>
      to memref<64xf32, strided<[1], offset: 8>>
%d2 = csl.get_mem_dsd %v2 : ... -> !csl.dsd

// 3) stride=4, offset=0 (non-default stride only)
%v3 = memref.subview %x[0][32][4]
      : memref<128xf32>
      to memref<32xf32, strided<[4], offset: 0>>
%d3 = csl.get_mem_dsd %v3 : ... -> !csl.dsd

// 4) stride=4, offset=3 (both non-default)
%v4 = memref.subview %x[3][32][4]
      : memref<128xf32>
      to memref<32xf32, strided<[4], offset: ?>>
%d4 = csl.get_mem_dsd %v4 : ... -> !csl.dsd""", size=10, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// 1) both defaults collapsed → minimal form
const d1 = @get_dsd(mem1d_dsd, .{
  .base_address = &x, .extent = 128 });

// 2) non-default offset → @increment (literal)
const d2 = @get_dsd(mem1d_dsd, .{
  .base_address = &x, .extent = 64 });
const d2b = @increment_dsd_offset(d2, 8, f32);

// 3) non-default stride → .stride field only
const d3 = @get_dsd(mem1d_dsd, .{
  .base_address = &x, .extent = 32, .stride = 4 });

// 4) both → .stride + @increment
const d4 = @get_dsd(mem1d_dsd, .{
  .base_address = &x, .extent = 32, .stride = 4 });
// runtime offset → @as(i16, …)
const d4b = @increment_dsd_offset(
  d4, @as(i16, offset), f32);""", size=10, hl=CSL_KW)
slides.append(slide_subview)

# B8: func.func private helper  (from e2e/multifunc.mlir)
def slide_helper():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: func.func private  →  CSL helper fn",
           "Private helpers emit as CSL fn; func.call → direct call  (src: e2e/multifunc.mlir)")
    side_labels(s, "MLIR  —  private helper + func.call",
                   "Emitted CSL  pe.csl")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """csl.program @pe {
  %a = csl.var @a : memref<256xf32>
  %b = csl.var @b : memref<256xf32>
  %c = csl.var @c : memref<256xf32>

  // private helper: x*s + y
  func.func private @scaled_add(
      %x: f32, %y: f32, %s: f32) -> f32 {
    %m = arith.mulf %x, %s : f32
    %r = arith.addf %m, %y : f32
    func.return %r : f32
  }

  csl.func @compute {
    %scale = arith.constant 2.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %va = memref.load %a[%i] : memref<256xf32>
      %vb = memref.load %b[%i] : memref<256xf32>
      %vr = func.call @scaled_add(%va,%vb,%scale)
             : (f32,f32,f32) -> f32
      memref.store %vr, %c[%i] : memref<256xf32>
    }
    csl.return
  }
}""", size=10, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// helper emitted before compute fn
fn scaled_add(a0: f32, a1: f32, a2: f32) f32 {
  var t0: f32 = a0 * a2;
  var t1: f32 = t0 + a1;
  return t1;
}

fn compute() void {
  const scale: f32 = 2.0;
  var i: u16 = 0;
  while (i < 256) : (i += 1) {
    var va: f32 = a[i];
    var vb: f32 = b[i];
    // func.call → direct fn call
    var vr: f32 = scaled_add(va, vb, scale);
    c[i] = vr;
  }
  sys_mod.unblock_cmd_stream();
}""", size=10, hl=CSL_KW)
    src_tag(s, "e2e/multifunc.mlir  (helper_add wafer)")
slides.append(slide_helper)

# B9: arith extras — divf / minimumf / negf / xori  (from e2e/arith_extra.mlir)
def slide_arith_extra():
    s = prs.slides.add_slide(BLANK)
    header(s, "Construct: arith extras  ·  divf / minimumf / negf / xori",
           "Four patterns not in the core set  (src: e2e/arith_extra.mlir)")
    side_labels(s, "MLIR  —  four kernel bodies",
                   "Emitted CSL  —  direct mapping")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """// arith.divf → /
scf.for %i = %c0 to %n step %c1 {
  %vx = memref.load %x[%i] : memref<128xf32>
  %r  = arith.divf %vx, %two : f32
  memref.store %r, %y[%i]   : memref<128xf32>
}

// arith.minimumf → if-expr
%r = arith.minimumf %vx, %thresh : f32
// emits: if (vx < 0.5) vx else 0.5

// arith.negf → unary minus
%r = arith.negf %vx : f32
// emits: -vx

// arith.xori on i1 → != (logical xor)
%r = arith.xori %ba, %bb : i1
// emits: ba != bb""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// arith.divf → /
var i: u16 = 0;
while (i < 128) : (i += 1) {
  var vx: f32 = x[i];
  var r: f32  = vx / 2.0;
  y[i] = r;
}

// arith.minimumf → inline ternary
var r: f32 = if (vx < 0.5) vx else 0.5;

// arith.negf → -
var r: f32 = -vx;

// arith.xori i1 → !=
// (bitwise ^ would be wrong for bool)
var r: bool = ba != bb;""", size=11, hl=CSL_KW)
slides.append(slide_arith_extra)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — DSDs
# ─────────────────────────────────────────────────────────────────────────────
def slide_sec_c():
    s = prs.slides.add_slide(BLANK)
    section_break(s, "C", "DSDs",
                  "Data Structure Descriptors — Cerebras SIMD abstraction over memory and fabric")
slides.append(slide_sec_c)

# C1: Simple mem1d_dsd  (from e2e/simd_dsd.mlir)
def slide_dsd_simple():
    s = prs.slides.add_slide(BLANK)
    header(s, "DSDs  ·  mem1d_dsd  (contiguous)",
           "csl.get_mem_dsd over flat memref → @get_dsd(mem1d_dsd, base+extent) — no stride, no offset")
    side_labels(s, "MLIR  —  saxpy via csl.builtin_call",
                   "Emitted CSL  pe.csl  (src: e2e/simd_dsd.mlir)")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(3.5),
        """// module-level vars (csl.var)
// %A = csl.var @A : memref<128xf32>
// %y = csl.var @y : memref<128xf32>

csl.func @compute {
  %n    = arith.constant 128 : index
  %scal = arith.constant 2.0 : f32
  // contiguous DSDs — no subview needed
  %Ad = csl.get_mem_dsd %A
        : memref<128xf32> -> !csl.dsd
  %yd = csl.get_mem_dsd %y
        : memref<128xf32> -> !csl.dsd
  // y = 2.0*A + y  (SAXPY)
  csl.builtin_call "fmacs"(%yd,%yd,%Ad,%scal)
      : (!csl.dsd,!csl.dsd,!csl.dsd,f32)->()
  csl.return
}""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(3.5),
        """var A: [128]f32;
var y: [128]f32;

fn compute() void {
  const alpha: f32 = 2.0;
  const dA = @get_dsd(mem1d_dsd, .{
    .base_address = &A, .extent = 128 });
  const dy = @get_dsd(mem1d_dsd, .{
    .base_address = &y, .extent = 128 });
  // single call replaces 128 scalar ops
  @fmacs(dy, dy, dA, alpha);
  sys_mod.unblock_cmd_stream();
}""", size=11, hl=CSL_KW)
    add_text(s, LX, Inches(5.94), Inches(12.5), Inches(0.34),
             "@get_dsd fields:  .base_address = &var   .extent = N   (.stride = S  for strided subviews)",
             size=11, color=DIM, font="Consolas")
slides.append(slide_dsd_simple)

# C2: Strided DSD + @increment  (from tutorials/gemv-02-memory-dsds.mlir)
def slide_dsd_strided():
    s = prs.slides.add_slide(BLANK)
    header(s, "DSDs  ·  Strided mem1d_dsd  (column-major access)",
           "subview stride=[N] + dynamic offset → @get_dsd .stride + @increment_dsd_offset(@as(i16,j))")
    side_labels(s, "MLIR  —  column j of row-major A",
                   "Emitted CSL  (src: gemv-02-memory-dsds.mlir)")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(3.7),
        """// y_dsd contiguous (no subview needed)
%y_dsd = csl.get_mem_dsd %y
         : memref<4xf32> -> !csl.dsd

scf.for %j = %c0 to %n step %c1 {
  // A[:,j]: stride=6, offset=j (dynamic)
  %col = memref.subview %A[%j] [4] [6]
      : memref<24xf32>
      to memref<4xf32, strided<[6], offset: ?>>
  %A_col = csl.get_mem_dsd %col
      : memref<4xf32, strided<[6], offset: ?>>
      -> !csl.dsd
  %xj = memref.load %x[%j] : memref<6xf32>
  csl.builtin_call "fmacs"
      (%y_dsd,%y_dsd,%A_col,%xj)
      : (!csl.dsd,!csl.dsd,!csl.dsd,f32)->()
}""", size=10, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(3.7),
        """const y_dsd = @get_dsd(mem1d_dsd, .{
  .base_address = &y, .extent = 4 });

var j: u16 = 0;
while (j < 6) : (j += 1) {
  // stride=6 from subview stride attr
  const col_base = @get_dsd(mem1d_dsd, .{
    .base_address = &A,
    .extent = 4, .stride = 6 });
  // offset=j is u16 → must cast to i16
  const A_col = @increment_dsd_offset(
    col_base, @as(i16, j), f32);
  const xj: f32 = x[j];
  @fmacs(y_dsd, y_dsd, A_col, xj);
}""", size=10, hl=CSL_KW)
    add_text(s, LX, Inches(6.07), Inches(12.5), Inches(0.34),
             "Key: runtime (SSA) offset → @as(i16, …)  ·  compile-time offset → integer literal (no @as)",
             size=11, color=DIM, font="Consolas")
slides.append(slide_dsd_strided)

# C3: DSD builtins table
def slide_dsd_builtins():
    s = prs.slides.add_slide(BLANK)
    header(s, "DSDs  ·  Builtin Operations",
           "csl.builtin_call in MLIR → @builtin(…) in CSL; -csl-auto-vectorize generates these from scf.for")
    rows = [
        ("@fadds","c_dsd, a_dsd, b_dsd",      "c[i] = a[i] + b[i]",        "Vector add"),
        ("@fsubs","c_dsd, a_dsd, b_dsd",      "c[i] = a[i] - b[i]",        "Vector sub"),
        ("@fmuls","c_dsd, a_dsd, b_dsd",      "c[i] = a[i] * b[i]",        "Vector mul"),
        ("@fmacs","c_dsd, c_dsd, a_dsd, α",  "c[i] = α*a[i] + c[i]",      "SAXPY; α loop-invariant"),
        ("@fmovs","c_dsd, a_dsd",             "c[i] = a[i]",               "Vector copy / fabric send"),
        ("@fnegs","c_dsd, a_dsd",             "c[i] = -a[i]",              "Vector negate"),
        ("@increment_dsd_offset",
                  "dsd, offset:i16, T",        "advance base pointer",      "offset=@as(i16,u16) for runtime"),
    ]
    x0, y0 = Inches(0.3), Inches(1.9)
    cw = [Inches(2.35), Inches(3.25), Inches(3.45), Inches(3.45)]
    rh = Inches(0.55)
    add_rect(s, x0, y0, sum(cw), Inches(0.46), fill=ACCENT)
    for c, h in enumerate(("Builtin","Signature (abridged)","Semantics","Notes")):
        add_text(s, x0+sum(cw[:c])+Inches(0.1), y0+Inches(0.09),
                 cw[c]-Inches(0.18), Inches(0.38), h, size=13, bold=True, color=BG)
    for i, row in enumerate(rows):
        y = y0 + Inches(0.46) + i*rh
        add_rect(s, x0, y, sum(cw), rh, fill=PANEL if i%2==0 else CODE_BG)
        colors = [KEYWORD, CODE_FG, ACCENT2, DIM]
        for c, (val, col) in enumerate(zip(row, colors)):
            add_text(s, x0+sum(cw[:c])+Inches(0.1), y+Inches(0.09),
                     cw[c]-Inches(0.18), rh-Inches(0.14),
                     val, size=11, color=col,
                     font="Consolas" if c < 2 else "Calibri")
    add_text(s, Inches(0.3), Inches(6.15), Inches(12.7), Inches(0.32),
             "csl.builtin_call in MLIR:", size=12, bold=True, color=ACCENT2)
    code_block(s, Inches(0.3), Inches(6.47), Inches(12.7), Inches(0.72),
        """csl.builtin_call "fadds"(%dc,%da,%db) : (!csl.dsd,!csl.dsd,!csl.dsd)->()  →  @fadds(dc, da, db);
csl.builtin_call "fmacs"(%dy,%dy,%dA,%α) : (!csl.dsd,!csl.dsd,!csl.dsd,f32)->()  →  @fmacs(dy,dy,dA,α);""",
        size=10)
slides.append(slide_dsd_builtins)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — MULTI-PE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
def slide_sec_d():
    s = prs.slides.add_slide(BLANK)
    section_break(s, "D", "Multi-PE Patterns",
                  "1-PE · N-PE SPMD · 2-PE fabric routing — real test files end-to-end")
slides.append(slide_sec_d)

# D1: Wafer / Program / Layout structure  (from e2e/simd_dsd.mlir)
def slide_wafer_structure():
    s = prs.slides.add_slide(BLANK)
    header(s, "Structure: csl.wafer / program / layout / host",
           "Four top-level containers; every CSL IR file follows this shape  (src: e2e/simd_dsd.mlir)")
    side_labels(s, "MLIR  —  full wafer structure",
                   "Concept map")
    divider(s)
    code_block(s, LX, CY, LW, CH,
        """csl.wafer @simd {arch = "wse3"} {

  // PE program (one per distinct tile type)
  csl.program @pe {
    %A = csl.var @A : memref<32xf32>
    %y = csl.var @y : memref<32xf32>
    csl.func @compute {
      %a = arith.constant 2.0 : f32
      %Ad = csl.get_mem_dsd %A : ... -> !csl.dsd
      %yd = csl.get_mem_dsd %y : ... -> !csl.dsd
      csl.builtin_call "fmacs"(%yd,%yd,%Ad,%a)
          : (!csl.dsd,!csl.dsd,!csl.dsd,f32)->()
      csl.return
    }
    csl.export @A {alias="A"}
    csl.export @compute {kind="func"}
  }

  // Placement grid
  csl.layout {width=4,height=2} @layout {
    csl_layout.place @pe over [0:4, 0:2]
  }

  // Host transfers + launch
  csl.host @main(...) {layout=@layout} {
    csl_host.memcpy_h2d ... to @layout::@A ...
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@y to ...
  }
}""", size=10, hl=MLIR_KW)
    # concept map on right
    boxes = [
        (Inches(0.15), Inches(0.5), "csl.wafer @name {arch}", ACCENT,
         "Top-level; one per MLIR module"),
        (Inches(0.55), Inches(1.35), "csl.program @pe {…}", ACCENT2,
         "PE kernel; vars + funcs; one per tile type"),
        (Inches(0.95), Inches(2.2), "csl.var / csl.func", GOOD,
         "Module-level arrays; entry fn(s)"),
        (Inches(0.55), Inches(3.05), "csl.layout {w,h} @L {…}", ACCENT2,
         "Grid size; place ops map PEs to programs"),
        (Inches(0.55), Inches(3.9), "csl.host @main(…) {…}", ACCENT,
         "h2d transfers; launch; d2h; numpy assert"),
    ]
    for (dx, dy, title, col, desc) in boxes:
        bx = RX + dx; by = CY + dy
        add_rect(s, bx, by, Inches(5.8), Inches(0.72), fill=CODE_BG,
                 line=col)
        add_text(s, bx+Inches(0.1), by+Inches(0.06), Inches(5.6), Inches(0.34),
                 title, size=12, bold=True, color=col, font="Consolas")
        add_text(s, bx+Inches(0.1), by+Inches(0.4), Inches(5.6), Inches(0.28),
                 desc, size=10, color=DIM)
slides.append(slide_wafer_structure)

# D2: 1-PE SAXPY  (from e2e/simd_dsd.mlir — single PE)
def slide_1pe():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pattern: 1-PE  ·  SAXPY via DSDs",
           "Single PE, 1×1 grid — host sends A and y, PE runs SAXPY, host reads y back")
    side_labels(s, "MLIR  —  1×1 placement + host",
                   "Emitted CSL  (pe.csl + layout.csl)")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """// 1-PE: single program at (0,0)
csl.wafer @saxpy {arch="wse3"} {
  csl.program @pe {
    %A = csl.var @A : memref<128xf32>
    %y = csl.var @y : memref<128xf32>
    csl.func @compute {
      %a  = arith.constant 2.0 : f32
      %Ad = csl.get_mem_dsd %A
            : memref<128xf32> -> !csl.dsd
      %yd = csl.get_mem_dsd %y
            : memref<128xf32> -> !csl.dsd
      csl.builtin_call "fmacs"
          (%yd,%yd,%Ad,%a)
          : (!csl.dsd,!csl.dsd,!csl.dsd,f32)->()
      csl.return
    }
    csl.export @A {alias="A"}
    csl.export @y {alias="y"}
    csl.export @compute {kind="func"}
  }
  csl.layout {width=1,height=1} @layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main(%A_in:memref<128xf32>,
                  %y_io:memref<128xf32>)
      {layout=@layout} {
    csl_host.memcpy_h2d %A_in to @layout::@A
        {px=0,py=0,width=1,height=1}
        : memref<128xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@y to %y_io
        {px=0,py=0,width=1,height=1}
        : memref<128xf32>
  }
}""", size=9, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// pe.csl
var A: [128]f32;
var y: [128]f32;

fn compute() void {
  const alpha: f32 = 2.0;
  const dA = @get_dsd(mem1d_dsd, .{
    .base_address = &A, .extent = 128 });
  const dy = @get_dsd(mem1d_dsd, .{
    .base_address = &y, .extent = 128 });
  @fmacs(dy, dy, dA, alpha);
  sys_mod.unblock_cmd_stream();
}
comptime { @export_symbol(A_ptr,"A");
           @export_symbol(y_ptr,"y");
           @export_symbol(compute); }

// layout.csl
layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "pe.csl", .{
    .memcpy_params = memcpy.get_params(0) });
}""", size=10, hl=CSL_KW)
    src_tag(s, "e2e/simd_dsd.mlir  ·  e2e/tutorials/gemv-01-complete-program.mlir")
slides.append(slide_1pe)

# D3: N-PE SPMD — air.herd 8×1 → over [0:8,0]  (from Conversion/AIRToCSL/simd_herd.mlir)
def slide_npe_herd():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pattern: N-PE SPMD  ·  air.herd  →  csl_layout.place over",
           "-air-to-csl maps herd shape to subgrid range form  (src: Conversion/AIRToCSL/simd_herd.mlir)")
    side_labels(s, "MLIR  —  air.herd with N tiles",
                   "CSL IR output  —  over [0:W, 0:H]")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(3.7),
        """// 8x1 herd → width=8 height=1
air.herd @pe
    tile(%htx,%hty)
    in (%hsx=%N8, %hsy=%one)
    args(%ha=%sa, %hb=%sb)
    : memref<256xf32>, memref<256xf32> {
  // herd body ...
  air.herd_terminator
}
// N=8 in (%hsx=%N8) →
//   layout width=8, height=1
//   place over [0:8, 0]
//   host memcpy width=8, height=1""", size=11, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(3.7),
        """// csl.layout {width=8, height=1}
csl.layout {width=8, height=1} @layout {
  csl_layout.place @pe over [0:8, 0]
}
// host: width=8 → one op → 8 PEs
csl_host.memcpy_h2d %a_in
    to @layout::@arg0
    {px=0, py=0, width=8, height=1}
    : memref<8x32xf32>
csl_host.launch @layout::@compute
csl_host.memcpy_d2h @layout::@arg1 ...""", size=11, hl=MLIR_KW)
    add_text(s, LX, Inches(6.1), Inches(12.5), Inches(0.34),
             "Shape table:", size=12, bold=True, color=ACCENT2)
    rows = [("8×1 herd","over [0:8, 0]","8","1"),
            ("4×4 herd","over [0:4, 0:4]","4","4"),
            ("1×4 herd","over [0, 0:4]","1","4")]
    x0, y0 = LX, Inches(6.45)
    cw = [Inches(2.5), Inches(3.5), Inches(1.5), Inches(1.5)]
    for i, (herd, over, w, h) in enumerate(rows):
        bx = x0 + sum(cw[:0])
        by = y0 + i*Inches(0.24)
        add_rect(s, x0, by, sum(cw), Inches(0.23),
                 fill=PANEL if i%2==0 else CODE_BG)
        for c, val in enumerate([herd, over, f"width={w}", f"height={h}"]):
            add_text(s, x0+sum(cw[:c])+Inches(0.08), by+Inches(0.03),
                     cw[c]-Inches(0.14), Inches(0.2),
                     val, size=11, color=CODE_FG, font="Consolas")
slides.append(slide_npe_herd)

# D4: 2-PE dataflow put/get  (from e2e/multi_pe/ping_2pe.mlir)
def slide_dataflow():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pattern: 2-PE Routing  ·  csl.dataflow.put / .get",
           "Left sends buffer east via fabric; right receives  (src: e2e/multi_pe/ping_2pe.mlir)")
    side_labels(s, "MLIR  —  high-level dataflow ops",
                   "After --csl-dataflow-to-csl  (lowered CSL)")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, CH,
        """// Left PE: send 128 floats east
csl.program @left_pe {
  %buf = csl.var @buf : memref<128xf32>
  csl.func @compute {
    %n = arith.constant 128 : index
    csl.dataflow.put @send_ch
        source(%buf) extent(%n : index)
        : memref<128xf32>
    csl.return
  }
  csl.export @buf  {alias="buf_left",direction="in"}
  csl.export @compute {kind="func",direction="internal"}
}

// Right PE: receive into buf
csl.program @right_pe {
  %buf = csl.var @buf : memref<128xf32>
  csl.func @compute {
    %n = arith.constant 128 : index
    csl.dataflow.get @send_ch
        target(%buf) extent(%n : index)
        : memref<128xf32>
    csl.return
  }
  csl.export @buf  {alias="buf_right",direction="out"}
  csl.export @compute {kind="func",direction="internal"}
}

// Channel declared in layout
csl.layout {width=2,height=1} @layout {
  csl_layout.dataflow @send_ch from(0,0) to(1,0)
  csl_layout.place @left_pe  at (0, 0)
  csl_layout.place @right_pe at (1, 0)
}""", size=9, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// left_pe.csl  (after lowering)
const send_color: color = @get_color(0);
const out_dsd = @get_dsd(fabout_dsd, .{
  .fabric_color = send_color,
  .extent = 128 });
const y_mem = @get_dsd(mem1d_dsd, .{
  .base_address = &buf, .extent = 128 });
@fmovs(out_dsd, y_mem);
sys_mod.unblock_cmd_stream();

// right_pe.csl  (after lowering)
const recv_color: color = @get_color(0);
const in_dsd = @get_dsd(fabin_dsd, .{
  .fabric_color = recv_color,
  .extent = 128 });
const b_mem = @get_dsd(mem1d_dsd, .{
  .base_address = &buf, .extent = 128 });
// accumulate: buf += received
@fadds(b_mem, b_mem, in_dsd);
sys_mod.unblock_cmd_stream();""", size=10, hl=CSL_KW)
    src_tag(s, "e2e/multi_pe/ping_2pe.mlir  ·  tutorials/gemv-06-routes-1.mlir")
slides.append(slide_dataflow)

# D5: Routing transforms  (from Dialect/CSL/Transforms/)
def slide_routing_transforms():
    s = prs.slides.add_slide(BLANK)
    header(s, "Transforms: Color Allocation  +  Routing Lowering",
           "--csl-allocate-color-ids assigns IDs; --csl-lower-dataflow-routing materialises set_color_config")
    side_labels(s, "Before  —  abstract color + dataflow",
                   "After  —  concrete IDs + tx/rx config")
    divider(s); arrow(s)
    code_block(s, LX, CY, LW, Inches(3.0),
        """// Before --csl-allocate-color-ids:
csl.layout {width=2,height=1} @layout {
  csl.color @c0         // no id yet
  csl_layout.dataflow @ch
      from(0,0) to(1,0) {color = @c0}
  csl_layout.place @p at (0, 0)
  csl_layout.place @p at (1, 0)
}""", size=12, hl=MLIR_KW)
    code_block(s, RX, CY, RW, Inches(3.0),
        """// After --csl-allocate-color-ids:
csl.layout {width=2,height=1} @layout {
  csl.color @c0 {id = 0 : i32}
  csl_layout.dataflow @ch
      from(0,0) to(1,0) {color = @c0}
  // After --csl-lower-dataflow-routing:
  csl_layout.set_color_config @c0
      at(0,0) rx(RAMP) tx(EAST)
  csl_layout.set_color_config @c0
      at(1,0) rx(WEST) tx(RAMP)
  csl_layout.place @p at (0, 0)
  csl_layout.place @p at (1, 0)
}""", size=12, hl=MLIR_KW)
    add_text(s, LX, Inches(5.42), Inches(12.5), Inches(0.32),
             "Direction table (from Transforms/lower-routing/):", size=12, bold=True, color=ACCENT2)
    dirs = [("east.mlir","from(0,0)→(1,0)","tx(EAST) / rx(WEST)"),
            ("west.mlir","from(1,0)→(0,0)","tx(WEST) / rx(EAST)"),
            ("north.mlir","from(0,1)→(0,0)","tx(NORTH) / rx(SOUTH)"),
            ("south.mlir","from(0,0)→(0,1)","tx(SOUTH) / rx(NORTH)"),
            ("multi_hop_east.mlir","from(0,0)→(2,0)","relay through (1,0): RAMP→EAST→RAMP")]
    x0, y0 = LX, Inches(5.78)
    cw = [Inches(2.4), Inches(3.2), Inches(6.65)]
    for i, (src, direction, config) in enumerate(dirs):
        by = y0 + i*Inches(0.22)
        add_rect(s, x0, by, sum(cw), Inches(0.21), fill=PANEL if i%2==0 else CODE_BG)
        for c, val in enumerate([src, direction, config]):
            add_text(s, x0+sum(cw[:c])+Inches(0.08), by+Inches(0.02),
                     cw[c]-Inches(0.14), Inches(0.2),
                     val, size=10, color=[ACCENT2, CODE_FG, GOOD][c], font="Consolas")
slides.append(slide_routing_transforms)

# D6: Scientific kernels  (from e2e/kernels.mlir)
def slide_kernels():
    s = prs.slides.add_slide(BLANK)
    header(s, "Scientific Kernels  ·  dot / reduce / relu",
           "Four real-shape kernels; focus on function body  (src: e2e/kernels.mlir)")
    side_labels(s, "dot  —  sum of element-wise product",
                   "reduce  +  relu  (arith.maximumf)")
    divider(s)
    code_block(s, LX, CY, LW, CH,
        """// dot: out[0] = sum(a[i] * b[i])
csl.func @compute {
  %c0 = arith.constant 0 : index
  %n  = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %va   = memref.load %a[%i]   : memref<128xf32>
    %vb   = memref.load %b[%i]   : memref<128xf32>
    %prod = arith.mulf %va, %vb   : f32
    %acc  = memref.load %out[%c0] : memref<1xf32>
    %s    = arith.addf %acc, %prod : f32
    memref.store %s, %out[%c0]   : memref<1xf32>
  }
  csl.return
}
// emits:
// while (i<128) : (i+=1) {
//   var prod: f32 = a[i] * b[i];
//   out[0] = out[0] + prod; }""", size=10, hl=MLIR_KW)
    code_block(s, RX, CY, RW, CH,
        """// reduce: out[0] = sum(a[i])
csl.func @compute {
  scf.for %i = %c0 to %n step %c1 {
    %va  = memref.load %a[%i]   : memref<128xf32>
    %acc = memref.load %out[%c0]: memref<1xf32>
    %s   = arith.addf %acc, %va : f32
    memref.store %s, %out[%c0]  : memref<1xf32>
  }
  csl.return
}

// relu: y[i] = max(x[i], 0.0)
csl.func @compute {
  scf.for %i = %c0 to %n step %c1 {
    %v = memref.load %x[%i]   : memref<128xf32>
    %r = arith.maximumf %v, %zero : f32
    memref.store %r, %y[%i]   : memref<128xf32>
  }
  csl.return
}
// relu emits:
// var r: f32 = if (v > 0.0) v else 0.0;""", size=10, hl=MLIR_KW)
    src_tag(s, "e2e/kernels.mlir  ·  e2e/scientific/dot.mlir  ·  e2e/scientific/saxpy.mlir")
slides.append(slide_kernels)

# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE MAP
# ─────────────────────────────────────────────────────────────────────────────
def slide_coverage():
    s = prs.slides.add_slide(BLANK)
    header(s, "Test Coverage at a Glance",
           "Green = passing  ·  Amber = partial/workaround  ·  Red = missing dialect feature")
    rows = [
        ("gemv-01","Scalar GEMV self-init",  "nested scf.for, load/store",             GOOD,  "Supported"),
        ("gemv-02","DSD strided GEMV",       "@fmacs, subview stride, @increment",      GOOD,  "Supported"),
        ("gemv-03","Host-provided arrays",   "memcpy_h2d/d2h, 4 vars",                 GOOD,  "Supported"),
        ("gemv-04","Compile-time params",    "arith.constant replaces CSL param",       ACCENT2,"Partial"),
        ("gemv-05","SPMD 1×2 grid",          "place over [0:2,0], wide shards",         GOOD,  "Supported"),
        ("gemv-06","2-PE fabric routing",    "dataflow.put/get, dataflow-to-csl",       GOOD,  "Supported"),
        ("gemv-07","Data tasks",             "@bind_data_task, task fn(payload) void",  WARN,  "Blocked"),
        ("gemv-08","NxM + coord",            "get_x/y_coord(), 3-color routing",        WARN,  "Blocked"),
        ("ping_2pe","2-PE passthrough",      "csl.dataflow.put/get e2e simulator",      GOOD,  "Supported"),
        ("simd_dsd","4×2 SIMD SAXPY",        "4×2 grid, @fmacs, sharded host",          GOOD,  "Supported"),
        ("kernels", "dot/reduce/saxpy/relu", "mulf, addf, maximumf, accum loop",        GOOD,  "Supported"),
        ("multifunc","Private helpers",       "func.func private + func.call",           GOOD,  "Supported"),
        ("arith_extra","divf/negf/xori",     "arith.divf, minimumf, negf, xori i1",    GOOD,  "Supported"),
        ("subview_patterns","DSD variants",  "4 subview forms → @get_dsd shapes",       GOOD,  "Supported"),
    ]
    x0, y0 = Inches(0.3), Inches(1.92)
    cw = [Inches(1.75), Inches(2.3), Inches(5.85), Inches(1.7)]
    rh = Inches(0.345)
    add_rect(s, x0, y0, sum(cw), Inches(0.4), fill=ACCENT)
    for c, h in enumerate(("Test","Theme","Key constructs / blocker","Status")):
        add_text(s, x0+sum(cw[:c])+Inches(0.1), y0+Inches(0.07),
                 cw[c]-Inches(0.18), Inches(0.34), h, size=12, bold=True, color=BG)
    for i, (tut, theme, key, col, status) in enumerate(rows):
        y = y0 + Inches(0.4) + i*rh
        add_rect(s, x0, y, sum(cw), rh, fill=PANEL if i%2==0 else CODE_BG)
        add_text(s, x0+Inches(0.1), y+Inches(0.07), cw[0]-Inches(0.18), rh-Inches(0.1),
                 tut, size=11, bold=True, color=ACCENT2, font="Consolas")
        add_text(s, x0+cw[0]+Inches(0.08), y+Inches(0.07), cw[1]-Inches(0.15), rh-Inches(0.1),
                 theme, size=11, color=TEXT)
        add_text(s, x0+cw[0]+cw[1]+Inches(0.08), y+Inches(0.07), cw[2]-Inches(0.15), rh-Inches(0.1),
                 key, size=10, color=DIM, font="Consolas")
        label_chip(s, x0+sum(cw[:3])+Inches(0.1), y+Inches(0.04),
                   cw[3]-Inches(0.2), rh-Inches(0.08),
                   status, fill=col, fg=BG)
slides.append(slide_coverage)

# ─────────────────────────────────────────────────────────────────────────────
# Q&A
# ─────────────────────────────────────────────────────────────────────────────
def slide_qa():
    s = prs.slides.add_slide(BLANK)
    add_bg(s)
    add_rect(s, Inches(0.7), Inches(2.7), Inches(0.18), Inches(2.1), fill=ACCENT)
    add_text(s, Inches(1.0), Inches(2.7), Inches(11.5), Inches(1.0),
             "Questions?", size=60, bold=True, color=TEXT)
    add_text(s, Inches(1.0), Inches(3.9), Inches(11.5), Inches(0.6),
             "A · Passes   B · Constructs   C · DSDs   D · Multi-PE",
             size=18, color=ACCENT2, font="Consolas")
slides.append(slide_qa)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────
TOTAL = len(slides)

for fn in slides:
    fn()

import pathlib, os
out = pathlib.Path("out")
out.mkdir(exist_ok=True)
path = out / "mlir_air_csl_examples_v2.pptx"
prs.save(str(path))
print(f"Wrote {path}  ({TOTAL} slides)")
