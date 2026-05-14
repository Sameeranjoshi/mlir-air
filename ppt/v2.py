"""v2 deck — story-arc walkthrough of the air-to-fire branch.

Story:
  1. Why a new dialect (W-questions, AIR fit, pros/cons)
  2. CSL dialect v2 (op groups + key ops table)
  3. CSL text emitter (three sub-emitters + outputs)
  4. Pass pipeline overview (vertical), then per-pass before/after
  5. Multi-PE dataflow: 4-pass pipeline overview (vertical), then per-pass
  6. End-to-end multi-PE milestones (birds-eye + one slide per family)
  7. Simple stats table

Run:
    python ppt/v2.py
Output:
    ppt/out/mlir_air_csl_v2.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ── theme  ·  "deep plum" palette ────────────────────────────────────────────
BG      = RGBColor(0x18, 0x14, 0x26)   # deep plum
PANEL   = RGBColor(0x26, 0x21, 0x3E)   # lighter plum slate
ACCENT  = RGBColor(0x7D, 0xD3, 0xFC)   # sky blue
ACCENT2 = RGBColor(0xFB, 0xBF, 0x24)   # warm gold
GOOD    = RGBColor(0x86, 0xEF, 0xAC)   # mint green
WARN    = RGBColor(0xFD, 0xA4, 0xAF)   # soft coral
TEXT    = RGBColor(0xFA, 0xF5, 0xFF)   # off-white
DIM     = RGBColor(0xA8, 0xA1, 0xB8)   # muted lavender
CODE_BG = RGBColor(0x0B, 0x08, 0x14)   # near-black plum
CODE_FG = RGBColor(0xE9, 0xE3, 0xF5)
KEYWORD = RGBColor(0xC4, 0xB5, 0xFD)   # soft lavender for code keywords
COMMENT = RGBColor(0x71, 0x6B, 0x82)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK  = prs.slide_layouts[6]


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
    tf.margin_top  = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run()
    r.text = text; r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, *, size=14, color=TEXT, line_spacing=1.25,
            bullet_color=ACCENT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    for i, txt in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT; p.line_spacing = line_spacing
        r = p.add_run(); r.text = "▸  "
        r.font.size = Pt(size); r.font.color.rgb = bullet_color
        r.font.bold = True; r.font.name = "Calibri"
        r2 = p.add_run(); r2.text = txt
        r2.font.size = Pt(size); r2.font.color.rgb = color
        r2.font.name = "Calibri"


def code_block(slide, x, y, w, h, code, *, size=11, hl=()):
    bg = add_rect(slide, x, y, w, h, fill=CODE_BG)
    bg.line.color.rgb = RGBColor(0x2A, 0x33, 0x42); bg.line.width = Pt(0.5)
    tb = slide.shapes.add_textbox(x+Emu(60000), y+Emu(60000),
                                  w-Emu(120000), h-Emu(120000))
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top  = tf.margin_bottom = Emu(0)
    for li, line in enumerate(code.split("\n")):
        p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        stripped = line.lstrip(" ")
        indent = line[:len(line)-len(stripped)]
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
                            if j > 0 and (rest[j-1].isalnum() or rest[j-1] in "_."):
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
        add_text(slide, Inches(0.65), Inches(1.05), Inches(12.2), Inches(0.4),
                 subtitle, size=13, color=DIM)
    add_rect(slide, Inches(0.4), Inches(7.05), Inches(12.55), Emu(9525), fill=PANEL)


def footer(slide, page):
    add_text(slide, Inches(0.4), Inches(7.12), Inches(8.0), Inches(0.3),
             "mlir-air  ·  CSL backend  ·  branch air-to-fire",
             size=10, color=DIM)
    add_text(slide, Inches(11.6), Inches(7.12), Inches(1.4), Inches(0.3),
             f"{page} / {TOTAL}", size=10, color=DIM, align=PP_ALIGN.RIGHT)


def arrow_down(slide, x, y, w, h, color=ACCENT):
    s = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = color
    s.line.fill.background(); s.shadow.inherit = False


def arrow_right(slide, x, y, w, h, color=ACCENT):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = color
    s.line.fill.background(); s.shadow.inherit = False


def pill(slide, x, y, w, h, text, *, fill=ACCENT, fg=BG, size=12):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background(); s.shadow.inherit = False
    s.adjustments[0] = 0.5
    tf = s.text_frame
    tf.margin_left = Emu(60000); tf.margin_right = Emu(60000)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = fg; r.font.name = "Calibri"


def table(slide, x, y, col_widths, header_row, rows, *,
          header_fill=ACCENT, header_fg=BG, row_h=Inches(0.42),
          body_size=12, header_size=13, mono_cols=()):
    total_w = sum(col_widths, Inches(0))
    add_rect(slide, x, y, total_w, row_h, fill=header_fill)
    for c, h in enumerate(header_row):
        cx = x + sum(col_widths[:c], Inches(0))
        add_text(slide, cx + Inches(0.12), y + Inches(0.08),
                 col_widths[c] - Inches(0.24), row_h - Inches(0.16),
                 h, size=header_size, bold=True, color=header_fg)
    for ri, row in enumerate(rows):
        ry = y + row_h + ri * row_h
        fill = PANEL if ri % 2 == 0 else CODE_BG
        add_rect(slide, x, ry, total_w, row_h, fill=fill)
        for c, val in enumerate(row):
            cx = x + sum(col_widths[:c], Inches(0))
            font = "Consolas" if c in mono_cols else "Calibri"
            color = ACCENT2 if c == 0 else TEXT
            add_text(slide, cx + Inches(0.12), ry + Inches(0.08),
                     col_widths[c] - Inches(0.24), row_h - Inches(0.12),
                     val, size=body_size, color=color, font=font, bold=(c == 0))


def stage_box(slide, x, y, w, h, *, label, sub, accent=ACCENT):
    add_rect(slide, x, y, w, h, fill=PANEL, line=RGBColor(0x33, 0x40, 0x55))
    add_rect(slide, x, y, Inches(0.14), h, fill=accent)
    add_text(slide, x+Inches(0.25), y+Inches(0.12), w-Inches(0.35), Inches(0.45),
             label, size=15, bold=True, color=TEXT, font="Consolas")
    add_text(slide, x+Inches(0.25), y+Inches(0.55), w-Inches(0.35), h-Inches(0.65),
             sub, size=11, color=DIM)


MLIR_KW = (
    "csl.wafer","csl.program","csl.func","csl.var","csl.task","csl.export","csl.return",
    "csl.host","csl.layout","csl.color","csl.get_mem_dsd","csl.get_fab_dsd",
    "csl.builtin_call","csl.import_module",
    "csl.dataflow.put","csl.dataflow.get","csl.dataflow.send_wavelet",
    "csl.get_x_coord","csl.get_y_coord",
    "csl_layout.place","csl_layout.dataflow","csl_layout.export","csl_layout.set_color_config",
    "csl_host.memcpy_h2d","csl_host.memcpy_d2h","csl_host.launch",
    "air.launch","air.segment","air.herd",
    "air.herd_terminator","air.segment_terminator","air.launch_terminator",
    "func.func","return","scf.for","scf.if","scf.yield","scf.index_switch","scf.while",
    "memref.load","memref.store","memref.subview","memref",
    "arith.constant","arith.addf","arith.subf","arith.mulf","arith.cmpi","arith.cmpf",
    "arith.select","arith.remsi","arith.index_cast",
    "f32","i1","i16","index","strided","over","at","case","default","else",
    "trigger_kind","data_task","local_task_id","color","direction","route",
)
CSL_KW = (
    "fn","var","const","comptime","while","if","else","switch","case","void","layout",
    "f32","u16","i16","@import_module","@get_dsd","@set_rectangle","@set_tile_code",
    "@fmacs","@fadds","@fmuls","@fmovs","@fnegs","@fsubs","@export_symbol",
    "@increment_dsd_offset","@bind_data_task","@get_x_coord","@get_y_coord",
    "mem1d_dsd","fabin_dsd","fabout_dsd",
)
PY_KW = ("import","from","def","if","for","while","return","assert","True","False","as","in","not","and","or")


slides_fns = []


# ─────────────────────────────────────────────────────────────────────────────
# 1. title
# ─────────────────────────────────────────────────────────────────────────────

def slide_title():
    s = prs.slides.add_slide(BLANK)
    add_bg(s)
    for i in range(1, 8):
        x = Inches(0.5 + i * 1.6)
        ln = s.shapes.add_connector(1, x, Inches(0), x, SH)
        ln.line.color.rgb = RGBColor(0x1A, 0x22, 0x33); ln.line.width = Pt(0.5)
    add_rect(s, Inches(0.7), Inches(1.55), Inches(0.18), Inches(2.7), fill=ACCENT)
    add_text(s, Inches(1.0), Inches(1.5), Inches(11.5), Inches(0.5),
             "branch  air-to-fire", size=16, bold=True,
             color=ACCENT2, font="Consolas")
    add_text(s, Inches(1.0), Inches(2.0), Inches(11.5), Inches(1.8),
             "AIR → CSL\nA Compiler Path to the Cerebras Wafer",
             size=42, bold=True, color=TEXT)
    add_text(s, Inches(1.0), Inches(4.6), Inches(11.5), Inches(0.5),
             "188 commits  ·  single-PE → multi-PE  ·  validated on WSE-3 simulator",
             size=16, color=DIM)
    add_text(s, Inches(1.0), Inches(5.2), Inches(11.5), Inches(0.5),
             "What shipped, with one example per layer.",
             size=15, color=ACCENT, font="Consolas")
    add_text(s, Inches(1.0), Inches(6.5), Inches(11.5), Inches(0.4),
             "mlir-air  ·  CSL backend overview",
             size=12, color=DIM, font="Consolas")


slides_fns.append(slide_title)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vertical pipeline overview  +  side "why each box"
# ─────────────────────────────────────────────────────────────────────────────

def slide_vertical_pipeline():
    s = prs.slides.add_slide(BLANK)
    header(s, "The Pipeline  ·  top-down flow",
           "User program  →  AIR  →  CSL IR  →  CSL language  →  WSE-3 simulator")

    # Left column: 5 vertical stages
    lx = Inches(0.55); lw = Inches(4.4)
    by = Inches(2.0); bh = Inches(0.78); gap = Inches(0.18)
    stages = [
        ("Frontend",     "user .mlir (or AIE / GPU lowerings)", ACCENT),
        ("AIR dialect",  "air.launch / segment / herd",         ACCENT2),
        ("CSL dialect",  "csl.wafer · program · layout · host", ACCENT2),
        ("CSL language", "pe_program.csl + layout.csl + run.py", GOOD),
        ("Simulator",    "WSE-2 / WSE-3 (cslc + sdkruntime)",   RGBColor(0xCE, 0x93, 0xD8)),
    ]
    cy = by
    for label, sub, col in stages:
        stage_box(s, lx, cy, lw, bh, label=label, sub=sub, accent=col)
        cy += bh
        if cy < by + 5 * (bh + gap):
            arrow_down(s, lx + lw/2 - Inches(0.18), cy - Inches(0.02),
                       Inches(0.36), Inches(0.22), color=ACCENT)
            cy += gap

    # Right column: side "why" cloud
    rx = Inches(5.4); rw = Inches(7.5)
    ry = Inches(1.95); rh = Inches(5.0)
    add_rect(s, rx, ry, rw, rh, fill=PANEL, line=RGBColor(0x33, 0x40, 0x55))
    add_text(s, rx+Inches(0.3), ry+Inches(0.18), rw-Inches(0.6), Inches(0.45),
             "Why each layer?", size=18, bold=True, color=ACCENT2)
    rows = [
        ("Frontend",     "Whatever produces AIR — same entry-point as the AIE/GPU backends."),
        ("AIR",          "Three-level spatial hierarchy: launch (host) · segment (L2) · herd (compute)."),
        ("CSL dialect",  "Typed MLIR — verifiers reject malformed CSL early; passes do real work."),
        ("CSL language", "Cerebras-native text the SDK ingests; trivial to inspect and tweak by hand."),
        ("Simulator",    "Closes the loop — every multi-PE test on this branch prints SUCCESS!"),
    ]
    rry = ry + Inches(0.75); col_w = rw - Inches(0.6)
    for label, why in rows:
        add_text(s, rx+Inches(0.3), rry, Inches(1.7), Inches(0.4),
                 label, size=13, bold=True, color=ACCENT, font="Consolas")
        add_text(s, rx+Inches(2.05), rry, col_w-Inches(1.75), Inches(0.7),
                 why, size=12, color=TEXT)
        rry += Inches(0.78)

    # Two commands at the bottom
    code_block(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.45),
               "$ air-opt input.mlir -air-to-csl ... | air-translate --emit-csl --output-dir=out/",
               size=12)


slides_fns.append(slide_vertical_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Why a new dialect?  — W-questions table
# ─────────────────────────────────────────────────────────────────────────────

def slide_w_questions():
    s = prs.slides.add_slide(BLANK)
    header(s, "Why a new dialect?  ·  the questions a Cerebras program must answer",
           "Each W-question maps directly to an op group in csl.*")

    table(s, Inches(0.5), Inches(1.95),
          [Inches(1.7), Inches(2.2), Inches(8.6)],
          ("Question", "Dialect concern", "What it covers"),
          [
              ("What",  "Kernel",         "loops · arithmetic · DSD vector ops · tasks · functions"),
              ("Where", "Layout",         "spatial properties · relative PE positions · problem-to-wafer mapping"),
              ("How",   "Data movement",  "memory ↔ registers inside a PE  ·  inter-PE moves"),
              ("Where (2)", "Routing",    "paths across PEs · sketch the wafer · network-traffic management"),
              ("Which", "Access pattern", "scalar vs vector (DSD) · access patterns over memref"),
              ("When",  "Async / sync",   "async tasks · completion callbacks · synchronization"),
              ("Why",   "Optimization",   "is it necessary? · can we skip work? (forwarding, DSD layouts)"),
          ],
          mono_cols=(1,), row_h=Inches(0.55), body_size=13)

    add_text(s, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.4),
             "Take-away: existing MLIR dialects don't capture spatial placement, "
             "fabric routing, or wavelet semantics natively — that's the gap we fill.",
             size=12, color=DIM)


slides_fns.append(slide_w_questions)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AIR fit — properties, pros, cons, opportunities
# ─────────────────────────────────────────────────────────────────────────────

def slide_air_fit():
    s = prs.slides.add_slide(BLANK)
    header(s, "Can AIR be the right choice?  ·  fit assessment",
           "AIR captures spatial + dataflow + placement.  We extend it with CSL ops for what it can't express.")

    # Top: property checklist
    add_text(s, Inches(0.5), Inches(1.85), Inches(12.3), Inches(0.4),
             "Properties checked against AIR + CSL extension", size=14, bold=True, color=ACCENT)
    props = [
        ("Spatial constructs",   "✓"),
        ("Dataflow graphs",      "✓"),
        ("Placement",            "✓"),
        ("Routing (fabric)",     "✓ (new in csl.*)"),
        ("Layout & placement",   "✓"),
        ("Communication",        "✓ (csl.dataflow.*)"),
        ("Scheduling / control", "✓ (scf reused)"),
        ("H2D / D2H",            "✓ (csl_host.*)"),
        ("Optimizations easy",   "✓ (typed IR)"),
        ("Tooling",              "✓ (lit + sim)"),
    ]
    px = Inches(0.5); py = Inches(2.3); pw = Inches(3.8); ph = Inches(0.38)
    for i, (k, v) in enumerate(props):
        col = i // 5; row = i % 5
        x = px + col * (pw + Inches(0.3))
        y = py + row * (ph + Inches(0.08))
        add_rect(s, x, y, pw, ph, fill=PANEL)
        add_text(s, x+Inches(0.15), y+Inches(0.08), pw-Inches(1.0), ph-Inches(0.14),
                 k, size=12, color=TEXT)
        add_text(s, x+pw-Inches(1.1), y+Inches(0.08), Inches(0.95), ph-Inches(0.14),
                 v, size=12, color=GOOD, bold=True, font="Consolas")

    # Right column: pros / cons / opportunities
    rx = Inches(8.5); rw = Inches(4.4)
    add_text(s, rx, Inches(2.3), rw, Inches(0.4),
             "Pros", size=14, bold=True, color=GOOD)
    bullets(s, rx, Inches(2.7), rw, Inches(1.6),
            [
                "optimizations possible on typed IR",
                "lowering = a normal conversion pass",
                "output codegen is manageable text",
                "captures semantics AIE / GPU dialects don't",
            ], size=11, bullet_color=GOOD)
    add_text(s, rx, Inches(4.4), rw, Inches(0.4),
             "Cons", size=14, bold=True, color=WARN)
    bullets(s, rx, Inches(4.8), rw, Inches(0.8),
            [
                "time to build + maintain a new dialect",
            ], size=11, bullet_color=WARN)
    add_text(s, rx, Inches(5.4), rw, Inches(0.4),
             "Opportunities", size=14, bold=True, color=ACCENT2)
    bullets(s, rx, Inches(5.8), rw, Inches(1.4),
            [
                "PE-grid forwarding opt (routing + fabric as 1st class)",
                "map-schedule-on-grid pass",
                "DSD-layout optimizations",
            ], size=11, bullet_color=ACCENT2)


slides_fns.append(slide_air_fit)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CSL Dialect v2 — op groups + key ops
# ─────────────────────────────────────────────────────────────────────────────

def slide_dialect():
    s = prs.slides.add_slide(BLANK)
    header(s, "1.  CSL Dialect v2  ·  op groups + key ops",
           "Six TableGen op groups under mlir/include/air/Dialect/CSL/ — all under one csl.wafer container")

    table(s, Inches(0.5), Inches(1.95),
          [Inches(1.7), Inches(4.0), Inches(6.6)],
          ("Op group", "TableGen file", "Key ops"),
          [
              ("Layout",       "CSLLayoutOps.td",        "csl.wafer · csl.layout"),
              ("Placement",    "CSLPlacementOps.td",     "csl_layout.place {at | over [a:b,c:d]}"),
              ("Kernel",       "CSLKernelOps.td",        "csl.program · csl.func · csl.var · csl.task · csl.return"),
              ("Routing",      "CSLRoutingOps.td",       "csl.color · csl_layout.set_color_config"),
              ("DataMovement", "CSLDataMovementOps.td",  "csl.get_mem_dsd · csl.get_fab_dsd · csl.builtin_call"),
              ("Runtime",      "CSLRuntimeOps.td",       "csl.import_module · csl.export · csl.host"),
          ],
          mono_cols=(1, 2), row_h=Inches(0.55), body_size=12)

    add_text(s, Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.4),
             "Why this layout?", size=14, bold=True, color=ACCENT2)
    bullets(s, Inches(0.5), Inches(6.25), Inches(12.3), Inches(1.0),
            [
                "One file per concern — verifiers stay focused, tests are scoped, op IDs stay stable.",
                "Old Phase-1 raw-text emitter, csl_rt dialect, spatial_placement ops fully removed.",
            ], size=12)


slides_fns.append(slide_dialect)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CSL text emitter — three sub-emitters + features
# ─────────────────────────────────────────────────────────────────────────────

def slide_emitter_overview():
    s = prs.slides.add_slide(BLANK)
    header(s, "2.  --emit-csl  ·  three sub-emitters, one runnable directory",
           "air-translate --emit-csl writes a directory the Cerebras simulator runs as-is")

    sub = [
        ("Program",  "pe_program.csl", "per-PE kernel — fn compute, DSDs, comptime exports",   ACCENT),
        ("Layout",   "layout.csl",     "@set_rectangle + @set_tile_code; subgrid placement loop", ACCENT2),
        ("Host",     "run.py",         "SdkRuntime: memcpy_h2d / launch / memcpy_d2h + np check", GOOD),
    ]
    by = Inches(2.0); bh = Inches(1.0); gap = Inches(0.15); x = Inches(0.5)
    bw = Inches(4.05)
    for label, file, desc, col in sub:
        cx = x
        add_rect(s, cx, by, bw, bh, fill=PANEL, line=RGBColor(0x33, 0x40, 0x55))
        add_rect(s, cx, by, Inches(0.14), bh, fill=col)
        add_text(s, cx+Inches(0.25), by+Inches(0.1), bw-Inches(0.35), Inches(0.4),
                 label, size=15, bold=True, color=TEXT)
        add_text(s, cx+Inches(0.25), by+Inches(0.45), bw-Inches(0.35), Inches(0.35),
                 file, size=13, color=col, font="Consolas")
        add_text(s, cx+Inches(0.25), by+Inches(0.78), bw-Inches(0.35), Inches(0.4),
                 desc, size=11, color=DIM)
        x += bw + gap

    add_text(s, Inches(0.5), Inches(3.3), Inches(6.0), Inches(0.4),
             "Supports today", size=14, bold=True, color=ACCENT)
    bullets(s, Inches(0.5), Inches(3.75), Inches(6.0), Inches(3.5),
            [
                "func.func / func.call / func.return",
                "arith: add / sub / mul / div / max / min / neg",
                "scf.for · scf.while · scf.if · scf.index_switch",
                "arith.cmpf / cmpi / index_cast / select",
                "memref.subview  →  strided DSDs",
                "csl.import_module · csl.builtin_call (bare + module forms)",
                "auto numpy reference check in run.py",
                "multi-program: one .csl per csl.program",
                "WSE-2 and WSE-3 commands.sh in same dir",
            ], size=12)

    add_text(s, Inches(6.8), Inches(3.3), Inches(6.0), Inches(0.4),
             "Strided DSDs from memref.subview", size=14, bold=True, color=ACCENT2)
    code_block(s, Inches(6.8), Inches(3.75), Inches(6.0), Inches(3.3),
               "// MLIR\n"
               "%v = memref.subview %a[3] [32] [4]\n"
               "         : memref<128xf32> to\n"
               "           memref<32xf32, strided<[4], offset:3>>\n"
               "%d = csl.get_mem_dsd %v\n"
               "         : memref<32xf32, strided<[4], offset:3>>\n"
               "\n"
               "// Emitted CSL\n"
               "const d = @get_dsd(mem1d_dsd, .{\n"
               "  .base_address = &a,\n"
               "  .extent = 32, .stride = 4,\n"
               "});\n"
               "@increment_dsd_offset(d, 3, f32);",
               size=11, hl=MLIR_KW + CSL_KW)


slides_fns.append(slide_emitter_overview)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Emitter outputs — side-by-side three artifacts
# ─────────────────────────────────────────────────────────────────────────────

def slide_emitter_outputs():
    s = prs.slides.add_slide(BLANK)
    header(s, "2b.  --emit-csl output  ·  pe_program.csl  ·  layout.csl  ·  run.py",
           "All three files end up under  out/<wafer>/   along with  commands_wse{2,3}.sh")

    add_text(s, Inches(0.5), Inches(1.85), Inches(4.0), Inches(0.4),
             "pe_program.csl", size=14, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(4.0), Inches(4.6),
               "param memcpy_params: comptime_struct;\n"
               "const sys = @import_module(\n"
               "  \"<memcpy/memcpy>\", memcpy_params);\n"
               "\n"
               "var a: [256]f32;\n"
               "var b: [256]f32;\n"
               "var c: [256]f32;\n"
               "\n"
               "fn compute() void {\n"
               "  const da = @get_dsd(mem1d_dsd,\n"
               "    .{.base_address=&a, .extent=256});\n"
               "  const db = @get_dsd(mem1d_dsd, ...);\n"
               "  const dc = @get_dsd(mem1d_dsd, ...);\n"
               "  @fadds(dc, da, db);\n"
               "  sys.unblock_cmd_stream();\n"
               "}\n"
               "comptime {\n"
               "  @export_symbol(&a, \"a\");\n"
               "  @export_symbol(compute);\n"
               "}",
               size=11, hl=CSL_KW)

    add_text(s, Inches(4.7), Inches(1.85), Inches(4.0), Inches(0.4),
             "layout.csl", size=14, bold=True, color=ACCENT2)
    code_block(s, Inches(4.7), Inches(2.2), Inches(4.0), Inches(4.6),
               "const memcpy = @import_module(\n"
               "  \"<memcpy/get_params>\", .{\n"
               "    .width = 1, .height = 1,\n"
               "  });\n"
               "\n"
               "layout {\n"
               "  @set_rectangle(1, 1);\n"
               "  @set_tile_code(0, 0,\n"
               "    \"pe_program.csl\",\n"
               "    .{.memcpy_params =\n"
               "        memcpy.get_params(0)});\n"
               "}",
               size=11, hl=CSL_KW)

    add_text(s, Inches(8.9), Inches(1.85), Inches(4.0), Inches(0.4),
             "run.py", size=14, bold=True, color=GOOD)
    code_block(s, Inches(8.9), Inches(2.2), Inches(4.0), Inches(4.6),
               "import numpy as np\n"
               "from cerebras.sdk.runtime import (\n"
               "  SdkRuntime, MemcpyOrder)\n"
               "\n"
               "a = np.arange(256, dtype=np.float32)\n"
               "b = np.ones(256, dtype=np.float32)\n"
               "c = np.zeros(256, dtype=np.float32)\n"
               "\n"
               "r = SdkRuntime(name,\n"
               "   simfab_numthreads=32)\n"
               "r.load(); r.run()\n"
               "r.memcpy_h2d(r.get_id(\"a\"), a, ...)\n"
               "r.memcpy_h2d(r.get_id(\"b\"), b, ...)\n"
               "r.launch(\"compute\", nonblock=False)\n"
               "r.memcpy_d2h(c, r.get_id(\"c\"), ...)\n"
               "r.stop()\n"
               "assert np.allclose(c, a + b)\n"
               "print(\"SUCCESS!\")",
               size=11, hl=PY_KW)


slides_fns.append(slide_emitter_outputs)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pass pipeline overview  ·  vertical
# ─────────────────────────────────────────────────────────────────────────────

def slide_pass_pipeline():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass pipeline  ·  birds-eye view",
           "What input MLIR enters, what IR each pass produces, what comes out at the end")

    # Left: vertical pass list
    lx = Inches(0.55); lw = Inches(5.5)
    by = Inches(1.95); bh = Inches(0.62); gap = Inches(0.1)
    stages = [
        ("input.mlir",              "AIR program (or csl.wafer source)", DIM),
        ("-air-to-csl",             "AIR herd → csl.program + csl.host",  ACCENT),
        ("-csl-verify-params",      "validate program param shapes",      ACCENT),
        ("-csl-infer-exports",      "auto-fill csl.export from host ops", ACCENT),
        ("-csl-auto-vectorize",     "scf.for idioms → DSD builtins",      ACCENT),
        ("--csl-dataflow-to-csl",   "multi-PE: 4 sub-passes (next slide)", ACCENT2),
        ("air-translate --emit-csl","csl.wafer → pe/layout/run.py",       GOOD),
        ("out/<wafer>/",            "runnable directory + commands.sh",   GOOD),
    ]
    cy = by
    for label, sub, col in stages:
        add_rect(s, lx, cy, lw, bh, fill=PANEL, line=RGBColor(0x33, 0x40, 0x55))
        add_rect(s, lx, cy, Inches(0.14), bh, fill=col)
        add_text(s, lx+Inches(0.25), cy+Inches(0.05), Inches(2.6), Inches(0.5),
                 label, size=13, bold=True, color=col, font="Consolas")
        add_text(s, lx+Inches(2.85), cy+Inches(0.05), lw-Inches(2.95), Inches(0.55),
                 sub, size=11, color=TEXT)
        cy += bh
        if cy < by + len(stages) * (bh + gap) - bh:
            arrow_down(s, lx + lw/2 - Inches(0.12), cy - Inches(0.03),
                       Inches(0.24), Inches(0.13), color=ACCENT)
            cy += gap

    # Right: key
    rx = Inches(6.6); rw = Inches(6.3)
    add_text(s, rx, Inches(1.95), rw, Inches(0.4),
             "What changes at each step", size=15, bold=True, color=ACCENT2)
    bullets(s, rx, Inches(2.4), rw, Inches(4.8),
            [
                "input  →  AIR three-level scaffold (launch/segment/herd)",
                "after -air-to-csl  →  csl.wafer container; AIR ops gone",
                "after -csl-verify-params  →  same IR, programs verified",
                "after -csl-infer-exports  →  csl.export ops materialized",
                "after -csl-auto-vectorize  →  scf.for replaced by @fadds/@fmacs/...",
                "after --csl-dataflow-to-csl  →  fabric colors + routing + tasks",
                "after --emit-csl  →  no MLIR left; just text files for cslc",
            ], size=13)
    add_text(s, rx, Inches(6.55), rw, Inches(0.4),
             "Each pass is independently testable — one FileCheck file per transition.",
             size=11, color=DIM)


slides_fns.append(slide_pass_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# 9. -air-to-csl  ·  before / after
# ─────────────────────────────────────────────────────────────────────────────

def slide_air_to_csl_ba():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass:  -air-to-csl  ·  before / after",
           "Lowers AIR's three-level hierarchy to the csl.* dialect (today: 1-PE and N-PE SIMD)")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input  ·  AIR vecadd", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "func.func @vecadd(%a, %b, %c) {\n"
               "  %one = arith.constant 1 : index\n"
               "  air.launch (%tx) in (%sx=%one) args(...) {\n"
               "    air.segment @seg args(...) {\n"
               "      air.herd @h tile(%i, %j)\n"
               "          in (%hx=%one, %hy=%one)\n"
               "          args(%ha=..., %hb=..., %hc=...) {\n"
               "        scf.for %i = %c0 to %c256 step %c1 {\n"
               "          %va = memref.load %ha[%i]\n"
               "          %vb = memref.load %hb[%i]\n"
               "          %vc = arith.addf %va, %vb : f32\n"
               "          memref.store %vc, %hc[%i]\n"
               "        }\n"
               "        air.herd_terminator\n"
               "      }\n"
               "    }\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Output  ·  csl.wafer", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "csl.wafer @vecadd {arch = \"wse3\"} {\n"
               "  csl.program @pe {\n"
               "    %a = csl.var @a : memref<256xf32>\n"
               "    %b = csl.var @b : memref<256xf32>\n"
               "    %c = csl.var @c : memref<256xf32>\n"
               "    csl.func @compute { /* same scf.for body */ }\n"
               "    csl.export @a {alias=\"a\"}\n"
               "    csl.export @b {alias=\"b\"}\n"
               "    csl.export @c {alias=\"c\"}\n"
               "    csl.export @compute {kind=\"func\"}\n"
               "  }\n"
               "  csl.layout {width=1, height=1} @main {\n"
               "    csl_layout.place @pe at (0, 0)\n"
               "  }\n"
               "  csl.host @h(...) {layout = @main} {\n"
               "    csl_host.memcpy_h2d ... ; csl_host.launch ...\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)


slides_fns.append(slide_air_to_csl_ba)


# ─────────────────────────────────────────────────────────────────────────────
# 10. -csl-auto-vectorize  ·  before / after
# ─────────────────────────────────────────────────────────────────────────────

def slide_autovec_ba():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass:  -csl-auto-vectorize  ·  before / after",
           "scf.for over memrefs  →  single DSD builtin call  (8 patterns, 13 loop rules)")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input  ·  scf.for add", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(2.9),
               "scf.for %i = %c0 to %n step %c1 {\n"
               "  %va = memref.load %a[%i] : memref<1024xf32>\n"
               "  %vb = memref.load %b[%i] : memref<1024xf32>\n"
               "  %vc = arith.addf %va, %vb : f32\n"
               "  memref.store %vc, %c[%i] : memref<1024xf32>\n"
               "}",
               size=12, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Output  ·  one @fadds intrinsic", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(2.9),
               "%da = csl.get_mem_dsd %a : memref<1024xf32>\n"
               "%db = csl.get_mem_dsd %b : memref<1024xf32>\n"
               "%dc = csl.get_mem_dsd %c : memref<1024xf32>\n"
               "csl.builtin_call \"fadds\" (%dc, %da, %db)\n"
               "    : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()",
               size=12, hl=MLIR_KW)

    table(s, Inches(0.5), Inches(5.3),
          [Inches(2.0), Inches(4.5), Inches(5.8)],
          ("Builtin", "Idiom (f32)", "Pattern"),
          [
              ("@fadds",        "c[i] = a[i] + b[i]",        "FaddsPattern · Rank2FaddsPattern"),
              ("@fsubs/@fmuls", "c[i] = a[i] ± / × b[i]",    "FsubsPattern · FmulsPattern"),
              ("@fmacs",        "y[i] = α·a[i] + y[i]",      "FmacsScalarPattern · Rank2FmacsPattern"),
              ("@fmovs/@fnegs", "c[i] = a[i]  ·  c[i] = -a[i]", "FmovsPattern · FnegsPattern"),
          ],
          mono_cols=(0, 1), row_h=Inches(0.4), body_size=11)


slides_fns.append(slide_autovec_ba)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Multi-PE dataflow pipeline overview  ·  vertical
# ─────────────────────────────────────────────────────────────────────────────

def slide_dataflow_pipeline():
    s = prs.slides.add_slide(BLANK)
    header(s, "4.  Multi-PE dataflow  ·  4-pass pipeline overview",
           "--csl-dataflow-to-csl  =  4 sub-passes  (renamed from 'stream' → 'dataflow')")

    # Left: vertical pass list
    lx = Inches(0.55); lw = Inches(6.2)
    by = Inches(1.95); bh = Inches(0.72); gap = Inches(0.13)
    passes = [
        ("input",                              "csl.dataflow.put/get + csl_layout.dataflow", DIM),
        ("-csl-materialize-dataflow-colors",   "synthesize csl.color symbol per stream",     ACCENT),
        ("-csl-allocate-color-ids",            "assign integer color IDs (0, 1, 2 …)",       ACCENT),
        ("-csl-lower-dataflow-routing",        "emit set_color_config per PE endpoint",      ACCENT2),
        ("-csl-lower-dataflow-data",           "expand put/get → fabric DSDs + tasks",       ACCENT2),
        ("output",                             "csl.builtin_call + tasks; ready for --emit-csl", GOOD),
    ]
    cy = by
    for label, sub, col in passes:
        add_rect(s, lx, cy, lw, bh, fill=PANEL, line=RGBColor(0x33, 0x40, 0x55))
        add_rect(s, lx, cy, Inches(0.14), bh, fill=col)
        add_text(s, lx+Inches(0.25), cy+Inches(0.08), lw-Inches(0.35), Inches(0.4),
                 label, size=13, bold=True, color=col, font="Consolas")
        add_text(s, lx+Inches(0.25), cy+Inches(0.42), lw-Inches(0.35), Inches(0.3),
                 sub, size=11, color=TEXT)
        cy += bh
        if cy < by + len(passes) * (bh + gap) - bh:
            arrow_down(s, lx + lw/2 - Inches(0.14), cy - Inches(0.02),
                       Inches(0.28), Inches(0.16), color=ACCENT)
            cy += gap

    # Right side: what the passes accomplish
    rx = Inches(7.0); rw = Inches(5.9)
    add_text(s, rx, Inches(1.95), rw, Inches(0.4),
             "What gets resolved", size=15, bold=True, color=ACCENT2)
    bullets(s, rx, Inches(2.4), rw, Inches(4.5),
            [
                "Colors:  semantic name  →  csl.color symbol  →  numeric ID",
                "Routing: per-PE config (direction in/out, route N/S/E/W)",
                "Data:  high-level put/get  →  fabric DSDs + completion tasks",
                "Each invariant is one pass — easy to test in isolation",
                "Each pass has its own FileCheck file under mlir/test/",
            ], size=12)

    code_block(s, Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.45),
               "$ air-opt --csl-dataflow-to-csl  input.mlir  | air-translate --emit-csl",
               size=12)


slides_fns.append(slide_dataflow_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# 12-15. Per-pass before/after (4 slides)
# ─────────────────────────────────────────────────────────────────────────────

def slide_pass_materialize():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass 1/4:  -csl-materialize-dataflow-colors",
           "Synthesizes a csl.color symbol for each csl_layout.dataflow edge")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Before", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.4),
               "csl.layout {width=2, height=1} @layout {\n"
               "  csl_layout.dataflow @send_ch\n"
               "      from(0, 0) to(1, 0)\n"
               "\n"
               "  csl_layout.place @left  at (0, 0)\n"
               "  csl_layout.place @right at (1, 0)\n"
               "}",
               size=12, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "After", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.4),
               "csl.layout {width=2, height=1} @layout {\n"
               "  csl.color @send_ch       // ← synthesized\n"
               "  csl_layout.dataflow @send_ch\n"
               "      from(0, 0) to(1, 0)\n"
               "\n"
               "  csl_layout.place @left  at (0, 0)\n"
               "  csl_layout.place @right at (1, 0)\n"
               "}",
               size=12, hl=MLIR_KW)


slides_fns.append(slide_pass_materialize)


def slide_pass_allocate():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass 2/4:  -csl-allocate-color-ids",
           "Assigns a monotonic integer ID to every csl.color (0, 1, 2, …)")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Before", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.4),
               "csl.layout @layout {\n"
               "  csl.color @ch01\n"
               "  csl.color @ch12\n"
               "  csl.color @ch23\n"
               "  ...\n"
               "}",
               size=12, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "After", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.4),
               "csl.layout @layout {\n"
               "  csl.color @ch01 { id = 0 : i16 }\n"
               "  csl.color @ch12 { id = 1 : i16 }\n"
               "  csl.color @ch23 { id = 2 : i16 }\n"
               "  ...\n"
               "}",
               size=12, hl=MLIR_KW)


slides_fns.append(slide_pass_allocate)


def slide_pass_routing():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass 3/4:  -csl-lower-dataflow-routing",
           "Per-PE: emits csl_layout.set_color_config for each endpoint of every dataflow edge")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Before  ·  edge-level dataflow", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.4),
               "csl.layout @layout {\n"
               "  csl.color @send_ch { id = 0 }\n"
               "  csl_layout.dataflow @send_ch\n"
               "      from(0, 0) to(1, 0)\n"
               "  csl_layout.place @left  at (0, 0)\n"
               "  csl_layout.place @right at (1, 0)\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "After  ·  per-PE routing config", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.4),
               "csl.layout @layout {\n"
               "  csl.color @send_ch { id = 0 }\n"
               "  csl_layout.set_color_config\n"
               "      @left::@send_ch\n"
               "      { direction = \"out\", route = \"east\" }\n"
               "  csl_layout.set_color_config\n"
               "      @right::@send_ch\n"
               "      { direction = \"in\",  route = \"west\" }\n"
               "  csl_layout.place @left  at (0, 0)\n"
               "  csl_layout.place @right at (1, 0)\n"
               "}",
               size=11, hl=MLIR_KW)


slides_fns.append(slide_pass_routing)


def slide_pass_data():
    s = prs.slides.add_slide(BLANK)
    header(s, "Pass 4/4:  -csl-lower-dataflow-data",
           "Replaces dataflow.put/get with fabric DSDs + async @fmovs + completion task")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Before  ·  high-level stream put", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.4),
               "csl.func @compute {\n"
               "  %n = arith.constant 128 : index\n"
               "  csl.dataflow.put @send_ch\n"
               "      source(%buf) extent(%n : index)\n"
               "      : memref<128xf32>\n"
               "  csl.return\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "After  ·  fabric DSD + async builtin", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.4),
               "csl.func @compute {\n"
               "  %fd = csl.get_fab_dsd @send_ch\n"
               "      direction = \"out\" extent = 128\n"
               "  %md = csl.get_mem_dsd %buf\n"
               "      : memref<128xf32> -> !csl.dsd\n"
               "  csl.builtin_call \"fmovs\"\n"
               "      (%fd, %md) async\n"
               "      on_complete = @done\n"
               "  csl.return\n"
               "}\n"
               "csl.task @done { ... }",
               size=11, hl=MLIR_KW)


slides_fns.append(slide_pass_data)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Multi-PE milestones overview (table)
# ─────────────────────────────────────────────────────────────────────────────

def slide_milestones_overview():
    s = prs.slides.add_slide(BLANK)
    header(s, "5.  Multi-PE milestones  ·  birds-eye",
           "All green on the WSE-3 simulator — tests live in mlir/test/Targets/CSLEmit/e2e/multi_pe/")

    table(s, Inches(0.5), Inches(1.95),
          [Inches(2.6), Inches(1.0), Inches(4.4), Inches(4.3)],
          ("Milestone", "PEs", "Pattern", "Test file"),
          [
              ("2-PE ping",       "2",  "1-hop fabric stream east",        "ping_2pe.mlir"),
              ("ping W / N / S",  "2",  "directional + two-stream variant","ping_2pe_west / north / south"),
              ("3-PE chain",      "3",  "multi-hop pass-through",          "ping_3pe_chain.mlir"),
              ("4-PE chain",      "4",  "east-direction 4-hop",            "chain_4pe_east.mlir"),
              ("2-row / 2-col",   "≥4", "east-west 2-row · N-S 2-col",     "east_west_2row · north_south_2col"),
              ("Fan-in",          "4",  "3 sources → 1 sink",              "fanin_3src.mlir"),
              ("Relay (DMA)",     "4",  "local_task_id completion task",   "reduce_chain_4pe.mlir"),
              ("Relay (wavelet)", "4",  "@bind_data_task per wavelet",     "reduce_chain_4pe_wavelet.mlir"),
              ("X-parity branch", "4",  "get_x_coord + arith.select",      "reduce_chain_4pe_xparity.mlir"),
              ("GEMV tutorial",   "1-N","6 tutorial stages",               "gemv-01 … gemv-06"),
          ],
          mono_cols=(1, 3), row_h=Inches(0.4), body_size=11)

    add_text(s, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.4),
             "Each milestone unlocked a new op or verifier path:  "
             "csl.task data_task · csl.dataflow.send_wavelet · csl.get_x_coord / get_y_coord · arith.select.",
             size=11, color=DIM)


slides_fns.append(slide_milestones_overview)


# ─────────────────────────────────────────────────────────────────────────────
# 17. 2-PE ping example
# ─────────────────────────────────────────────────────────────────────────────

def slide_ex_ping_2pe():
    s = prs.slides.add_slide(BLANK)
    header(s, "5a.  Example  ·  ping_2pe  (first inter-PE comm)",
           "(0,0) sends 128 f32 east to (1,0)  ·  host asserts arg0 == arg1  ·  SUCCESS!")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.wafer @ping_2pe {arch = \"wse3\"} {\n"
               "  csl.program @left_pe {\n"
               "    %buf = csl.var @buf : memref<128xf32>\n"
               "    csl.func @compute {\n"
               "      %n = arith.constant 128 : index\n"
               "      csl.dataflow.put @send_ch\n"
               "         source(%buf) extent(%n : index)\n"
               "    }\n"
               "  }\n"
               "  csl.program @right_pe {\n"
               "    csl.func @compute {\n"
               "      csl.dataflow.get @send_ch\n"
               "         target(%buf) extent(%n : index)\n"
               "    }\n"
               "  }\n"
               "  csl.layout {width=2, height=1} @layout {\n"
               "    csl_layout.dataflow @send_ch\n"
               "         from(0, 0) to(1, 0)\n"
               "    csl_layout.place @left_pe  at (0, 0)\n"
               "    csl_layout.place @right_pe at (1, 0)\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "// left_pe.csl  (sender)\n"
               "var buf: [128]f32;\n"
               "const ch = @get_color(0);\n"
               "\n"
               "fn compute() void {\n"
               "  const fd = @get_dsd(fabout_dsd,\n"
               "    .{.extent=128, .fabric_color=ch});\n"
               "  const md = @get_dsd(mem1d_dsd,\n"
               "    .{.base_address=&buf, .extent=128});\n"
               "  @fmovs(fd, md, .{.async=true});\n"
               "}\n"
               "\n"
               "// layout.csl\n"
               "layout {\n"
               "  @set_rectangle(2, 1);\n"
               "  @set_color_config(0, 0, ch,\n"
               "    .{.routes={.tx=EAST,  .rx=RAMP}});\n"
               "  @set_color_config(1, 0, ch,\n"
               "    .{.routes={.tx=RAMP,  .rx=WEST}});\n"
               "  @set_tile_code(0, 0, \"left_pe.csl\", ...);\n"
               "  @set_tile_code(1, 0, \"right_pe.csl\", ...);\n"
               "}",
               size=11, hl=CSL_KW)


slides_fns.append(slide_ex_ping_2pe)


# ─────────────────────────────────────────────────────────────────────────────
# 18. multi-hop chain example
# ─────────────────────────────────────────────────────────────────────────────

def slide_ex_chain():
    s = prs.slides.add_slide(BLANK)
    header(s, "5b.  Example  ·  ping_3pe_chain  (multi-hop fabric routing)",
           "P0 sends; P1 forwards at fabric level; P2 receives.  Three programs, one color.")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.layout {width=3, height=1} @layout {\n"
               "  csl_layout.dataflow @ch from(0,0) to(2,0)\n"
               "  csl_layout.place @p0 at (0, 0)\n"
               "  csl_layout.place @p1 at (1, 0)  // pass-through\n"
               "  csl_layout.place @p2 at (2, 0)\n"
               "}\n"
               "\n"
               "csl.program @p0 {\n"
               "  csl.func @compute {\n"
               "    csl.dataflow.put @ch\n"
               "       source(%buf) extent(%n : index)\n"
               "  }\n"
               "}\n"
               "\n"
               "// P2 receives — P1 has no dataflow code\n"
               "csl.program @p2 {\n"
               "  csl.func @compute {\n"
               "    csl.dataflow.get @ch\n"
               "       target(%buf) extent(%n : index)\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "// layout.csl  — routing handles the hop\n"
               "layout {\n"
               "  @set_rectangle(3, 1);\n"
               "  @set_color_config(0, 0, ch,\n"
               "    .{.routes={.tx=EAST}});\n"
               "  @set_color_config(1, 0, ch,\n"
               "    .{.routes={.rx=WEST, .tx=EAST}});\n"
               "  @set_color_config(2, 0, ch,\n"
               "    .{.routes={.rx=WEST, .tx=RAMP}});\n"
               "  @set_tile_code(0, 0, \"p0.csl\", ...);\n"
               "  @set_tile_code(1, 0, \"p1.csl\", ...);\n"
               "  @set_tile_code(2, 0, \"p2.csl\", ...);\n"
               "}\n"
               "\n"
               "// p1.csl  — relay PE\n"
               "// no compute body; fabric forwards\n"
               "// because .rx=WEST, .tx=EAST.\n"
               "fn compute() void {\n"
               "  sys.unblock_cmd_stream();\n"
               "}",
               size=11, hl=CSL_KW)


slides_fns.append(slide_ex_chain)


# ─────────────────────────────────────────────────────────────────────────────
# 19. fanin
# ─────────────────────────────────────────────────────────────────────────────

def slide_ex_fanin():
    s = prs.slides.add_slide(BLANK)
    header(s, "5c.  Example  ·  fanin_3src  (3 sources → 1 sink)",
           "Three separate dataflow streams all terminate at the same sink PE — three colors, three fabric DSDs")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.layout {width=4, height=1} @layout {\n"
               "  csl_layout.dataflow @ch_a from(0,0) to(3,0)\n"
               "  csl_layout.dataflow @ch_b from(1,0) to(3,0)\n"
               "  csl_layout.dataflow @ch_c from(2,0) to(3,0)\n"
               "  csl_layout.place @src_a at (0, 0)\n"
               "  csl_layout.place @src_b at (1, 0)\n"
               "  csl_layout.place @src_c at (2, 0)\n"
               "  csl_layout.place @sink  at (3, 0)\n"
               "}\n"
               "\n"
               "csl.program @sink {\n"
               "  csl.func @compute {\n"
               "    csl.dataflow.get @ch_a\n"
               "       target(%buf_a) extent(%n : index)\n"
               "    csl.dataflow.get @ch_b\n"
               "       target(%buf_b) extent(%n : index)\n"
               "    csl.dataflow.get @ch_c\n"
               "       target(%buf_c) extent(%n : index)\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (sink PE)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "// sink.csl  (PE 3, 0)\n"
               "var buf_a: [N]f32;\n"
               "var buf_b: [N]f32;\n"
               "var buf_c: [N]f32;\n"
               "const ch_a = @get_color(0);\n"
               "const ch_b = @get_color(1);\n"
               "const ch_c = @get_color(2);\n"
               "\n"
               "fn compute() void {\n"
               "  const fda = @get_dsd(fabin_dsd,\n"
               "    .{.fabric_color=ch_a, .extent=N});\n"
               "  @fmovs(da, fda, .{.async=true});\n"
               "  const fdb = @get_dsd(fabin_dsd,\n"
               "    .{.fabric_color=ch_b, .extent=N});\n"
               "  @fmovs(db, fdb, .{.async=true});\n"
               "  const fdc = @get_dsd(fabin_dsd,\n"
               "    .{.fabric_color=ch_c, .extent=N});\n"
               "  @fmovs(dc, fdc, .{.async=true});\n"
               "}",
               size=11, hl=CSL_KW)


slides_fns.append(slide_ex_fanin)


# ─────────────────────────────────────────────────────────────────────────────
# 20. relay DMA
# ─────────────────────────────────────────────────────────────────────────────

def slide_ex_relay_dma():
    s = prs.slides.add_slide(BLANK)
    header(s, "5d.  Example  ·  reduce_chain_4pe  (DMA relay)",
           "Middle PEs receive whole buffer, compute, then PUT — wake-up via local_task_id completion")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (relay PE)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "// P1: receive 64 f32, subtract 100,\n"
               "//     forward to P2.\n"
               "csl.program @p1 {\n"
               "  %buf = csl.var @buf : memref<64xf32>\n"
               "  csl.func @compute {\n"
               "    csl.dataflow.get @ch01\n"
               "      target(%buf) extent(%n : index)\n"
               "      on_complete = @step2\n"
               "  }\n"
               "  csl.task @step2 attributes\n"
               "    {trigger_kind=\"local_task_id\",\n"
               "     id = 1 : i16} {\n"
               "    scf.for %i = %c0 to %c64 step %c1 {\n"
               "      %v  = memref.load %buf[%i]\n"
               "      %nv = arith.subf %v, %hundred : f32\n"
               "      memref.store %nv, %buf[%i]\n"
               "    }\n"
               "    csl.dataflow.put @ch12\n"
               "      source(%buf) extent(%n : index)\n"
               "    csl.return\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (p1.csl)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "var buf: [64]f32;\n"
               "const ch01 = @get_color(0);\n"
               "const ch12 = @get_color(1);\n"
               "\n"
               "fn compute() void {\n"
               "  // GET → step2 fires on completion\n"
               "  const fd = @get_dsd(fabin_dsd,\n"
               "    .{.fabric_color=ch01, .extent=64});\n"
               "  const md = @get_dsd(mem1d_dsd,\n"
               "    .{.base_address=&buf, .extent=64});\n"
               "  @fmovs(md, fd,\n"
               "    .{.async=true, .activate=step2});\n"
               "}\n"
               "\n"
               "task step2() void {\n"
               "  for (i in 0..64) buf[i] -= 100.0;\n"
               "  const fd2 = @get_dsd(fabout_dsd,\n"
               "    .{.fabric_color=ch12, .extent=64});\n"
               "  @fmovs(fd2, md, .{.async=true});\n"
               "}\n"
               "comptime {\n"
               "  @bind_local_task(step2,\n"
               "    @get_local_task_id(1));\n"
               "}",
               size=11, hl=CSL_KW)


slides_fns.append(slide_ex_relay_dma)


# ─────────────────────────────────────────────────────────────────────────────
# 21. wavelet data-task
# ─────────────────────────────────────────────────────────────────────────────

def slide_ex_wavelet():
    s = prs.slides.add_slide(BLANK)
    header(s, "5e.  Example  ·  reduce_chain_4pe_wavelet  (per-wavelet data task)",
           "Native CSL data task fires once per arriving wavelet — simpler and cheaper than DMA relay")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "// P1: fires per wavelet on @ch01\n"
               "csl.program @p1 {\n"
               "  csl.task @relay attributes\n"
               "      {trigger_kind=\"data_task\",\n"
               "       color = @ch01} {\n"
               "  ^bb0(%val: f32):\n"
               "    %delta = arith.constant -100.0 : f32\n"
               "    %nv = arith.addf %val, %delta : f32\n"
               "    csl.dataflow.send_wavelet @ch12\n"
               "       value(%nv) : f32\n"
               "    csl.return\n"
               "  }\n"
               "}\n"
               "\n"
               "csl.layout {width=4, height=1} @layout {\n"
               "  csl_layout.dataflow @ch01 from(0,0) to(1,0)\n"
               "  csl_layout.dataflow @ch12 from(1,0) to(2,0)\n"
               "  csl_layout.dataflow @ch23 from(2,0) to(3,0)\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (p1.csl)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "const ch01 = @get_color(0);\n"
               "const ch12 = @get_color(1);\n"
               "\n"
               "// fires once per arriving wavelet\n"
               "task relay(val: f32) void {\n"
               "  const nv = val + (-100.0);\n"
               "  const fd = @get_dsd(fabout_dsd,\n"
               "    .{.fabric_color=ch12, .extent=1});\n"
               "  @fmovs(fd, nv, .{.async=true});\n"
               "}\n"
               "\n"
               "comptime {\n"
               "  @bind_data_task(relay, ch01);\n"
               "}\n"
               "\n"
               "// no fn compute() needed —\n"
               "// the data task fires on its own.",
               size=11, hl=CSL_KW)


slides_fns.append(slide_ex_wavelet)


# ─────────────────────────────────────────────────────────────────────────────
# 5g. GEMV tutorial  ·  overview + one slide per tutorial
# ─────────────────────────────────────────────────────────────────────────────

def slide_gemv_overview():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f.  GEMV tutorial  ·  six progressive kernels",
           "y = A·x + b  ·  same problem, each stage adds one capability — mirrors the official Cerebras SDK GEMV tutorial")

    table(s, Inches(0.5), Inches(1.95),
          [Inches(1.5), Inches(4.4), Inches(6.5)],
          ("Stage", "File", "Adds"),
          [
              ("gemv-01", "gemv-01-complete-program.mlir", "1-PE scalar GEMV  ·  PE self-initializes A, x, b"),
              ("gemv-02", "gemv-02-memory-dsds.mlir",      "memory DSDs  ·  strided A col  ·  @fmacs / @fadds"),
              ("gemv-03", "gemv-03-memcpy.mlir",           "host owns the data  ·  memcpy_h2d for A / x / b"),
              ("gemv-04", "gemv-04-params.mlir",           "compile-time M / N (param) — captured as constants today"),
              ("gemv-05", "gemv-05-multiple-pes.mlir",     "1×2 PE grid  ·  column partition (N_per_PE = 3)"),
              ("gemv-06", "gemv-06-routes-1.mlir",         "inter-PE fabric  ·  csl.dataflow.put/get + add"),
          ],
          mono_cols=(1,), row_h=Inches(0.48), body_size=12)

    add_text(s, Inches(0.5), Inches(5.95), Inches(12.3), Inches(0.4),
             "Why GEMV as the tutorial?", size=14, bold=True, color=ACCENT2)
    bullets(s, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.8),
            [
                "Smallest non-trivial dense kernel that exercises every layer of the stack.",
                "Each stage is one FileCheck file under  mlir/test/Targets/CSLEmit/e2e/tutorials/.",
            ], size=12)


slides_fns.append(slide_gemv_overview)


def slide_gemv_01():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f-1.  gemv-01  ·  complete program (scalar, 1 PE)",
           "y[i] = Σⱼ A[i·N + j] · x[j] + b[i]   (M = 4, N = 6)  ·  PE self-initializes all arrays")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (nested scf.for)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.func @init_and_compute {\n"
               "  // y[i] = sum_j A[i*N + j] * x[j]\n"
               "  scf.for %i = %c0 to %m step %c1 {\n"
               "    scf.for %j = %c0 to %n step %c1 {\n"
               "      %iN  = arith.muli %i, %n   : index\n"
               "      %iNj = arith.addi %iN, %j  : index\n"
               "      %aij = memref.load %A[%iNj]\n"
               "      %xj  = memref.load %x[%j]\n"
               "      %p   = arith.mulf %aij, %xj : f32\n"
               "      %yi  = memref.load %y[%i]\n"
               "      %s   = arith.addf %yi, %p : f32\n"
               "      memref.store %s, %y[%i]\n"
               "    }\n"
               "    // y[i] += b[i]\n"
               "    ...\n"
               "  }\n"
               "  csl.return\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (pe.csl)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "var A: [24]f32;\n"
               "var x: [6]f32;\n"
               "var b: [4]f32;\n"
               "var y: [4]f32;\n"
               "\n"
               "fn init_and_compute() void {\n"
               "  // ... init A=1, x=2, b=3, y=0 ...\n"
               "  var i: u16 = 0;\n"
               "  while (i < 4) : (i += 1) {\n"
               "    var j: u16 = 0;\n"
               "    while (j < 6) : (j += 1) {\n"
               "      y[i] = y[i] + A[i*6 + j] * x[j];\n"
               "    }\n"
               "    y[i] = y[i] + b[i];\n"
               "  }\n"
               "  sys.unblock_cmd_stream();\n"
               "}",
               size=11, hl=CSL_KW)


slides_fns.append(slide_gemv_01)


def slide_gemv_02():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f-2.  gemv-02  ·  memory DSDs (column-major A)",
           "Same GEMV, now vectorized: contiguous y / b DSDs + strided A column DSD (stride = N)")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (DSDs + @fmacs / @fadds)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.func @compute {\n"
               "  %yd = csl.get_mem_dsd %y : memref<4xf32>\n"
               "  %bd = csl.get_mem_dsd %b : memref<4xf32>\n"
               "  %Aj0 = memref.subview %A[0] [4] [6]\n"
               "       : memref<24xf32> to\n"
               "         memref<4xf32, strided<[6]>>\n"
               "  %Ad = csl.get_mem_dsd %Aj0 : !csl.dsd\n"
               "\n"
               "  scf.for %j = %c0 to %n step %c1 {\n"
               "    %xj = memref.load %x[%j]\n"
               "    csl.builtin_call \"fmacs\"\n"
               "       (%yd, %yd, %Ad, %xj)\n"
               "    %Ad = csl.builtin_call\n"
               "       \"increment_dsd_offset\" (%Ad, %c1)\n"
               "  }\n"
               "  csl.builtin_call \"fadds\" (%yd, %yd, %bd)\n"
               "  csl.return\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (pe.csl)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "fn compute() void {\n"
               "  const y_dsd = @get_dsd(mem1d_dsd,\n"
               "    .{.base_address=&y, .extent=4});\n"
               "  const b_dsd = @get_dsd(mem1d_dsd,\n"
               "    .{.base_address=&b, .extent=4});\n"
               "  var A_dsd = @get_dsd(mem1d_dsd,\n"
               "    .{.base_address=&A, .extent=4,\n"
               "      .stride=6});\n"
               "\n"
               "  var j: u16 = 0;\n"
               "  while (j < 6) : (j += 1) {\n"
               "    @fmacs(y_dsd, y_dsd, A_dsd, x[j]);\n"
               "    A_dsd = @increment_dsd_offset(\n"
               "       A_dsd, 1, f32);\n"
               "  }\n"
               "  @fadds(y_dsd, y_dsd, b_dsd);\n"
               "}",
               size=11, hl=CSL_KW)


slides_fns.append(slide_gemv_02)


def slide_gemv_03():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f-3.  gemv-03  ·  host-side memcpy (data lives on host)",
           "Same compute as gemv-02; all four arrays now arrive via memcpy_h2d  ·  host owns the data")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (csl.host block)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.host @main(\n"
               "    %Ah : memref<24xf32>,\n"
               "    %xh : memref<6xf32>,\n"
               "    %bh : memref<4xf32>,\n"
               "    %yh : memref<4xf32>)\n"
               "  {layout = @main_layout} {\n"
               "  csl_host.memcpy_h2d %Ah\n"
               "    to @main_layout::@A\n"
               "    {px=0, py=0, width=1, height=1}\n"
               "  csl_host.memcpy_h2d %xh\n"
               "    to @main_layout::@x {...}\n"
               "  csl_host.memcpy_h2d %bh\n"
               "    to @main_layout::@b {...}\n"
               "  csl_host.launch @main_layout::@compute\n"
               "  csl_host.memcpy_d2h @main_layout::@y\n"
               "    to %yh {...}\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated run.py", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "A = np.random.rand(24).astype(np.float32)\n"
               "x = np.random.rand(6).astype(np.float32)\n"
               "b = np.random.rand(4).astype(np.float32)\n"
               "y = np.zeros(4, dtype=np.float32)\n"
               "\n"
               "r = SdkRuntime(name, simfab_numthreads=32)\n"
               "r.load(); r.run()\n"
               "r.memcpy_h2d(r.get_id(\"A\"), A,\n"
               "    0,0, 1,1, 24, ...)\n"
               "r.memcpy_h2d(r.get_id(\"x\"), x, ...)\n"
               "r.memcpy_h2d(r.get_id(\"b\"), b, ...)\n"
               "r.launch(\"compute\", nonblock=False)\n"
               "r.memcpy_d2h(y, r.get_id(\"y\"), ...)\n"
               "r.stop()\n"
               "ref = A.reshape(4,6) @ x + b\n"
               "assert np.allclose(y, ref)\n"
               "print(\"SUCCESS!\")",
               size=11, hl=PY_KW)


slides_fns.append(slide_gemv_03)


def slide_gemv_04():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f-4.  gemv-04  ·  compile-time parameters",
           "SDK uses  param M : i16  and  param N : i16  — MLIR represents them as constants today")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (params → arith.constant today)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.program @pe {\n"
               "  // future: csl.comptime_param @M : i16\n"
               "  csl.func @compute {\n"
               "    %m = arith.constant 4 : index   // M\n"
               "    %n = arith.constant 6 : index   // N\n"
               "    %yd = csl.get_mem_dsd %y : memref<4xf32>\n"
               "    %bd = csl.get_mem_dsd %b : memref<4xf32>\n"
               "    %Ad = csl.get_mem_dsd %A_col : ...\n"
               "    scf.for %j = %c0 to %n step %c1 {\n"
               "      csl.builtin_call \"fmacs\" (%yd, %yd, %Ad, %xj)\n"
               "    }\n"
               "    csl.builtin_call \"fadds\" (%yd, %yd, %bd)\n"
               "  }\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (true params)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "param M: i16;\n"
               "param N: i16;\n"
               "\n"
               "var y: [M]f32;\n"
               "var b: [M]f32;\n"
               "var x: [N]f32;\n"
               "var A: [M*N]f32;\n"
               "\n"
               "fn compute() void {\n"
               "  for (j in 0..N) {\n"
               "    @fmacs(y_dsd, y_dsd, A_dsd, x[j]);\n"
               "  }\n"
               "  @fadds(y_dsd, y_dsd, b_dsd);\n"
               "}",
               size=11, hl=CSL_KW)

    add_text(s, Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.3),
             "Gap noted: a future csl.comptime_param op could close this without changing the emitter.",
             size=10, color=DIM)


slides_fns.append(slide_gemv_04)


def slide_gemv_05():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f-5.  gemv-05  ·  multiple PEs (1×2 column partition)",
           "Same program on both PEs (SPMD)  ·  N=6 split into N_per_PE=3  ·  host broadcasts shards")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (layout + host shard)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.layout {width=2, height=1} @main {\n"
               "  csl_layout.place @pe over [0:2, 0]\n"
               "}\n"
               "\n"
               "csl.host @main(\n"
               "    %Ah : memref<2x12xf32>,\n"
               "    %xh : memref<2x3xf32>,\n"
               "    %bh : memref<4xf32>,\n"
               "    %yh : memref<2x4xf32>)\n"
               "  {layout = @main} {\n"
               "  csl_host.memcpy_h2d %Ah to @main::@A\n"
               "    {px=0, py=0, width=2, height=1}\n"
               "  csl_host.memcpy_h2d %xh to @main::@x\n"
               "    {px=0, py=0, width=2, height=1}\n"
               "  csl_host.launch @main::@compute\n"
               "  csl_host.memcpy_d2h @main::@y to %yh\n"
               "    {px=0, py=0, width=2, height=1}\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL  (layout.csl)", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "const memcpy = @import_module(\n"
               "  \"<memcpy/get_params>\", .{\n"
               "    .width = 2, .height = 1,\n"
               "  });\n"
               "\n"
               "layout {\n"
               "  @set_rectangle(2, 1);\n"
               "  var px: u16 = 0;\n"
               "  while (px < 2) : (px += 1) {\n"
               "    @set_tile_code(px, 0, \"pe.csl\",\n"
               "      .{.memcpy_params =\n"
               "          memcpy.get_params(px)});\n"
               "  }\n"
               "}\n"
               "\n"
               "// pe.csl  (same for both PEs)\n"
               "// each holds 4×3 A shard + 3 x-shard\n"
               "// + 4-elem b + writes 4-elem y",
               size=11, hl=CSL_KW)


slides_fns.append(slide_gemv_05)


def slide_gemv_06():
    s = prs.slides.add_slide(BLANK)
    header(s, "5f-6.  gemv-06  ·  inter-PE routing (final stage)",
           "Distinct left / right programs  ·  left sends partial y east  ·  right adds + returns result")

    add_text(s, Inches(0.5), Inches(1.85), Inches(6.0), Inches(0.4),
             "Input MLIR  (two programs + edge)", size=13, bold=True, color=ACCENT)
    code_block(s, Inches(0.5), Inches(2.2), Inches(6.0), Inches(4.7),
               "csl.program @left_pe {\n"
               "  csl.func @compute {\n"
               "    // y_L = A_L · x_L\n"
               "    scf.for ... { ... fmacs ... }\n"
               "    csl.dataflow.put @send_ch\n"
               "      source(%y) extent(%m : index)\n"
               "  }\n"
               "}\n"
               "csl.program @right_pe {\n"
               "  csl.func @compute {\n"
               "    // y_R = A_R · x_R\n"
               "    scf.for ... { ... fmacs ... }\n"
               "    csl.dataflow.get @send_ch\n"
               "      target(%y_L) extent(%m : index)\n"
               "    csl.builtin_call \"fadds\"\n"
               "      (%y_dsd, %y_dsd, %y_L_dsd)\n"
               "  }\n"
               "}\n"
               "csl.layout {width=2, height=1} @layout {\n"
               "  csl_layout.dataflow @send_ch\n"
               "    from(0, 0) to(1, 0)\n"
               "}",
               size=11, hl=MLIR_KW)

    add_text(s, Inches(6.7), Inches(1.85), Inches(6.1), Inches(0.4),
             "Generated CSL", size=13, bold=True, color=GOOD)
    code_block(s, Inches(6.7), Inches(2.2), Inches(6.1), Inches(4.7),
               "// left_pe.csl\n"
               "const send = @get_color(0);\n"
               "fn compute() void {\n"
               "  // ... gemv-02 style fmacs loop ...\n"
               "  const fd = @get_dsd(fabout_dsd,\n"
               "    .{.fabric_color=send, .extent=4});\n"
               "  @fmovs(fd, y_dsd, .{.async=true});\n"
               "}\n"
               "\n"
               "// right_pe.csl\n"
               "const recv = @get_color(0);\n"
               "var y_L: [4]f32;\n"
               "fn compute() void {\n"
               "  // ... gemv-02 style fmacs loop ...\n"
               "  const fd = @get_dsd(fabin_dsd,\n"
               "    .{.fabric_color=recv, .extent=4});\n"
               "  @fmovs(y_L_dsd, fd, .{.async=true});\n"
               "  @fadds(y_dsd, y_dsd, y_L_dsd);\n"
               "}",
               size=11, hl=CSL_KW)
    add_text(s, Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.3),
             "All six tutorial stages: air-opt ... | air-translate --emit-csl → bash commands_wse3.sh → SUCCESS!",
             size=10, color=DIM)


slides_fns.append(slide_gemv_06)


# ─────────────────────────────────────────────────────────────────────────────
# 24. Stats (simple table — no fancy boxes)
# ─────────────────────────────────────────────────────────────────────────────

def slide_stats():
    s = prs.slides.add_slide(BLANK)
    header(s, "Stats  ·  what landed on  air-to-fire",
           "Numbers, no fancy boxes")

    table(s, Inches(0.5), Inches(1.95),
          [Inches(5.2), Inches(2.0), Inches(5.1)],
          ("Item", "Count", "Notes"),
          [
              ("Commits ahead of main",          "188",  "since branch creation"),
              ("New air-opt passes",             "4",    "verify-params · infer-exports · auto-vectorize · dataflow-to-csl"),
              ("Multi-PE dataflow sub-passes",   "4",    "materialize · allocate · routing · data"),
              ("Auto-vectorize patterns",        "8",    "fadds · fsubs · fmuls · fmovs · fnegs · fmacs · scalar · rank-2"),
              ("Loop-recognition rules",         "13",   "shape · purity · access predicates"),
              ("CSL dialect TableGen files",     "6",    "Layout · Placement · Kernel · Routing · DataMovement · Runtime"),
              ("CSL Dialect lit tests",          "21",   "round-trip + verifier"),
              ("Conversion/AIRToCSL tests",      "13",   "vecadd e2e + 7× reject_*"),
              ("Targets/CSLEmit/e2e tests",      "35+",  "elementwise · control-flow · multi-program · scientific"),
              ("Multi-PE e2e tests",             "16",   "ping · chain · fanin · relay · wavelet · xparity"),
              ("GEMV tutorial stages",           "6",    "gemv-01 … gemv-06"),
              ("Simulator target",               "WSE-3","cslc + sdkruntime · also emits WSE-2 commands.sh"),
          ],
          mono_cols=(1,), row_h=Inches(0.4), body_size=12)


slides_fns.append(slide_stats)


# ─────────────────────────────────────────────────────────────────────────────
# 25. Q&A
# ─────────────────────────────────────────────────────────────────────────────

def slide_qa():
    s = prs.slides.add_slide(BLANK)
    add_bg(s)
    add_rect(s, Inches(0.7), Inches(2.6), Inches(0.18), Inches(2.2), fill=ACCENT)
    add_text(s, Inches(1.0), Inches(2.6), Inches(11.5), Inches(1.2),
             "Questions?", size=60, bold=True, color=TEXT)
    add_text(s, Inches(1.0), Inches(3.9), Inches(11.5), Inches(0.6),
             "Demo:  air-opt ... --csl-dataflow-to-csl  |  air-translate --emit-csl",
             size=18, color=ACCENT2, font="Consolas")
    add_text(s, Inches(1.0), Inches(4.6), Inches(11.5), Inches(0.5),
             "Branch  air-to-fire  ·  188 commits  ·  16 multi-PE e2e green on WSE-3",
             size=14, color=DIM)


slides_fns.append(slide_qa)


# ── render ───────────────────────────────────────────────────────────────────

TOTAL = len(slides_fns)
for i, fn in enumerate(slides_fns, start=1):
    fn()
    last = prs.slides[-1]
    if i not in (1, TOTAL):
        footer(last, i)

import os
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "mlir_air_csl_v2.pptx")
prs.save(out)
print(f"Wrote {out}  ({TOTAL} slides)")
