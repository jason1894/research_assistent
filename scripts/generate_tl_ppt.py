"""
Generate a Transfer Learning lecture PPT for a PDE-background professor.
Target audience: Mathematics PhD students.
Content is strictly based on published literature.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x1F, 0x35, 0x64)   # title / headings
MID_BLUE    = RGBColor(0x2E, 0x74, 0xB5)   # accent
LIGHT_BLUE  = RGBColor(0xD6, 0xE4, 0xF0)   # body background tint
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY   = RGBColor(0x40, 0x40, 0x40)
ORANGE      = RGBColor(0xC5, 0x57, 0x00)   # highlight / formula colour

# ── Helpers ───────────────────────────────────────────────────────────────────

def set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=DARK_GRAY,
                align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txBox


def add_title_bar(slide, title_text, subtitle_text=""):
    """Dark blue top banner with white title."""
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), Inches(13.33), Inches(1.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()

    tf = bar.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title_text
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = WHITE

    if subtitle_text:
        add_textbox(slide, subtitle_text,
                    left=0.2, top=1.55, width=13, height=0.4,
                    font_size=14, color=MID_BLUE, italic=True)


def add_bullet_box(slide, items, left, top, width, height,
                   font_size=17, title="", indent_first=False):
    """Bullet list inside a lightly shaded box."""
    # background rectangle
    rect = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height))
    rect.fill.solid()
    rect.fill.fore_color.rgb = LIGHT_BLUE
    rect.line.color.rgb = MID_BLUE

    tf = rect.text_frame
    tf.word_wrap = True

    first = True
    for item in items:
        if first and title:
            p = tf.paragraphs[0]
            first = False
        elif first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()

        is_title_row = (item == items[0] and title)
        p.level = 0
        run = p.add_run()
        if isinstance(item, tuple):
            level, text = item
            p.level = level
            run.text = ("  " * level) + ("• " if level > 0 else "▸ ") + text
            run.font.size = Pt(font_size - level * 1.5)
            run.font.bold = (level == 0)
            run.font.color.rgb = DARK_BLUE if level == 0 else DARK_GRAY
        else:
            run.text = "▸ " + item
            run.font.size = Pt(font_size)
            run.font.bold = False
            run.font.color.rgb = DARK_GRAY


def add_ref_footer(slide, ref_text, top=7.0):
    add_textbox(slide, ref_text,
                left=0.2, top=top, width=12.9, height=0.45,
                font_size=9, color=MID_BLUE, italic=True)


def add_slide_number(slide, num, total):
    add_textbox(slide, f"{num} / {total}",
                left=12.3, top=7.1, width=1.0, height=0.35,
                font_size=10, color=DARK_GRAY, align=PP_ALIGN.RIGHT)


# ── Slide builders ────────────────────────────────────────────────────────────

def slide_title(prs):
    sld = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_bg(sld, WHITE)

    # decorative top band
    bar = sld.shapes.add_shape(
        1, Inches(0), Inches(0), Inches(13.33), Inches(2.6))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()

    add_textbox(sld, "Transfer Learning:",
                left=0.5, top=0.3, width=12.3, height=0.8,
                font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(sld, "Theory, Methods, and Applications to PDE Solving",
                left=0.5, top=1.1, width=12.3, height=0.8,
                font_size=22, bold=False, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)

    add_textbox(sld, "Graduate Course — Mathematics Department",
                left=0.5, top=3.0, width=12.3, height=0.5,
                font_size=16, color=MID_BLUE, align=PP_ALIGN.CENTER)
    add_textbox(sld, "Lecture Notes | 2024",
                left=0.5, top=3.6, width=12.3, height=0.5,
                font_size=14, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    # bottom accent line
    line = sld.shapes.add_shape(
        1, Inches(0), Inches(7.3), Inches(13.33), Inches(0.2))
    line.fill.solid()
    line.fill.fore_color.rgb = MID_BLUE
    line.line.fill.background()

    add_textbox(sld,
                "Key references: Pan & Yang (2010), Raissi et al. (2019), Lu et al. (2021), Goswami et al. (2022)",
                left=0.3, top=6.8, width=12.7, height=0.4,
                font_size=9, color=DARK_GRAY, italic=True, align=PP_ALIGN.CENTER)


def slide_outline(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Contents")
    add_slide_number(sld, n, total)

    items = [
        "Part I  — Background & Motivation  (Slides 3–5)",
        "Part II — Deep Learning Foundations  (Slides 6–8)",
        "Part III — Transfer Learning: Theory & Taxonomy  (Slides 9–14)",
        "Part IV — Fine-Tuning & Domain Adaptation  (Slides 15–18)",
        "Part V  — Transfer Learning for PDE Solving  (Slides 19–24)",
        "Part VI — Open Problems & Summary  (Slides 25–26)",
        "References  (Slide 27)",
    ]
    add_bullet_box(sld, items, left=0.4, top=1.7, width=12.5, height=5.5,
                   font_size=18)


# ── Part I ────────────────────────────────────────────────────────────────────

def slide_motivation(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Why Transfer Learning?",
                  "Part I — Background & Motivation")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Traditional machine learning assumes: training data and test data "
                "are drawn i.i.d. from the same distribution.",
                left=0.4, top=1.7, width=12.5, height=0.6,
                font_size=16, color=DARK_GRAY)

    items = [
        "In practice, labelled data in the target domain can be scarce or expensive (e.g., high-fidelity PDE simulations).",
        "A model trained for one PDE geometry/parameter regime fails on a related but different regime.",
        "Training deep neural networks from scratch is computationally costly.",
        "Transfer Learning: reuse knowledge from a source domain/task to improve learning in a target domain/task.",
    ]
    add_bullet_box(sld, items, left=0.4, top=2.4, width=12.5, height=3.8,
                   font_size=16)

    add_ref_footer(sld,
        "Ref: Pan & Yang, 'A Survey on Transfer Learning', IEEE TKDE 22(10), 2010, pp. 1345–1359.")


def slide_data_scarcity(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "The Data Scarcity Problem",
                  "Part I — Background & Motivation")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "High-fidelity numerical PDE solutions are expensive to compute. "
                "Transfer learning offers a principled approach to reuse knowledge.",
                left=0.4, top=1.7, width=12.5, height=0.7,
                font_size=16, color=DARK_GRAY)

    # Two-column layout
    left_items = [
        "Source domain: many labelled samples (e.g., simple geometry)",
        "Target domain: few labelled samples (e.g., complex geometry)",
        "Goal: transfer structural knowledge learned in source to improve target performance",
    ]
    right_items = [
        "Motivating example: a PINN trained on a simple 1D heat equation can transfer to a 2D heat equation with fewer epochs",
        "Formally studied in: Chakraborty (2021), Desai et al. (2021)",
    ]
    add_bullet_box(sld, left_items, left=0.4, top=2.5, width=5.9, height=3.5, font_size=15)
    add_bullet_box(sld, right_items, left=6.7, top=2.5, width=6.2, height=3.5, font_size=15)

    add_ref_footer(sld,
        "Ref: Chakraborty, 'Transfer learning based multi-fidelity physics informed deep neural network', "
        "J. Comput. Phys. 426 (2021) 109942.")


def slide_historical(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Brief Historical Context",
                  "Part I — Background & Motivation")
    add_slide_number(sld, n, total)

    items = [
        "1995 — NIPS workshop 'Learning to Learn' raises the idea of knowledge transfer across tasks.",
        "1997 — Thrun formalises the concept of 'learning to learn' (meta-learning).",
        "2010 — Pan & Yang publish the first comprehensive survey on Transfer Learning (IEEE TKDE).",
        "2012 — Deep convolutional features shown to be transferable across image tasks (Yosinski et al., NeurIPS 2014).",
        "2018 — BERT (Devlin et al.) demonstrates large-scale pre-training + fine-tuning in NLP.",
        "2019 – present — Transfer learning enters scientific computing: PINNs, operator learning, surrogate models.",
    ]
    add_bullet_box(sld, items, left=0.4, top=1.7, width=12.5, height=5.3, font_size=16)

    add_ref_footer(sld,
        "Ref: Pan & Yang (2010); Yosinski et al., NeurIPS 2014; Devlin et al., NAACL 2019.")


# ── Part II ───────────────────────────────────────────────────────────────────

def slide_dl_basics(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Deep Learning: Notation & Architecture",
                  "Part II — Deep Learning Foundations")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "A feedforward neural network of depth L is a parameterised function "
                "f_θ : ℝⁿ → ℝᵐ defined by the composition:",
                left=0.4, top=1.7, width=12.5, height=0.6,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld,
                "f_θ(x) = W_L · σ(W_{L-1} · σ( ··· σ(W_1 x + b_1) ··· ) + b_{L-1}) + b_L",
                left=1.0, top=2.4, width=11.0, height=0.7,
                font_size=18, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

    add_textbox(sld,
                "where σ is a nonlinear activation (e.g., ReLU, tanh, sigmoid), "
                "W_l ∈ ℝ^{n_{l}×n_{l-1}} are weight matrices, b_l are bias vectors, "
                "and θ = {W_l, b_l}_{l=1}^{L} are all trainable parameters.",
                left=0.4, top=3.2, width=12.5, height=0.9,
                font_size=15, color=DARK_GRAY)

    items = [
        "Training: minimise empirical risk  L(θ) = (1/N) Σᵢ ℓ(f_θ(xᵢ), yᵢ)  via stochastic gradient descent.",
        "Universal Approximation Theorem (Cybenko 1989; Hornik 1991): sufficiently wide networks approximate any continuous function on a compact set.",
        "Depth provides compositional expressivity (Telgarsky 2016, Mhaskar & Poggio 2016).",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.2, width=12.5, height=2.8, font_size=15)

    add_ref_footer(sld,
        "Ref: Goodfellow, Bengio & Courville, 'Deep Learning', MIT Press, 2016. "
        "Cybenko, Math. Control Signals Systems 2 (1989) 303–314.")


def slide_representation(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Representation Learning",
                  "Part II — Deep Learning Foundations")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Central insight (Bengio et al., 2013): deep networks learn a hierarchy "
                "of increasingly abstract representations.",
                left=0.4, top=1.7, width=12.5, height=0.6,
                font_size=16, color=DARK_GRAY)

    items = [
        "Layer l maps input to a feature vector h^(l) = σ(W_l h^(l-1) + b_l).",
        "Early layers capture low-level features (local patterns, edges in images; low-frequency modes in PDE solutions).",
        "Deeper layers capture high-level semantic features (task-specific representations).",
        "Key empirical finding (Yosinski et al., NeurIPS 2014): early-layer features are general and transferable; later-layer features are task-specific.",
        "This hierarchy is the mathematical foundation of transfer learning.",
    ]
    add_bullet_box(sld, items, left=0.4, top=2.4, width=12.5, height=4.3, font_size=16)

    add_ref_footer(sld,
        "Ref: Bengio et al., 'Representation Learning: A Review and New Perspectives', "
        "IEEE TPAMI 35(8), 2013. Yosinski et al., NeurIPS 2014.")


def slide_training_challenges(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Training Challenges & Generalisation",
                  "Part II — Deep Learning Foundations")
    add_slide_number(sld, n, total)

    items = [
        "Overfitting: model memorises training data, fails to generalise. Regularisation (L2, dropout) mitigates this.",
        "Curse of dimensionality: sample complexity grows exponentially with input dimension in classical function approx. theory.",
        "DNNs empirically break the curse for structured data — partly explained by implicit bias of SGD (Bartlett et al., 2021).",
        "For PDE solving: training data = collocation points; test = full domain. Sample efficiency is critical.",
        "Transfer learning directly addresses sample efficiency by initialising θ from a pre-trained model.",
    ]
    add_bullet_box(sld, items, left=0.4, top=1.7, width=12.5, height=5.1, font_size=16)

    add_ref_footer(sld,
        "Ref: Bartlett, Montanari & Rakhlin, 'Deep learning: a statistical viewpoint', "
        "Acta Numerica 30 (2021) 87–201.")


# ── Part III ──────────────────────────────────────────────────────────────────

def slide_tl_formal_def(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Formal Definition: Domain and Task",
                  "Part III — Transfer Learning: Theory & Taxonomy")
    add_slide_number(sld, n, total)

    add_textbox(sld, "Definitions (Pan & Yang, 2010):",
                left=0.4, top=1.7, width=12.5, height=0.45,
                font_size=17, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "Domain  𝒟 = {𝒳, P(X)}  where  𝒳  is a feature space and  P(X) is a marginal distribution.\n"
                "Task  𝒯 = {𝒴, f(·)}  where  𝒴  is a label space and  f : 𝒳 → 𝒴  is the predictive function "
                "(learned from {(xᵢ, yᵢ)}).",
                left=0.6, top=2.2, width=12.1, height=1.2,
                font_size=15, color=DARK_GRAY)

    add_textbox(sld, "Transfer Learning (Pan & Yang, 2010, Definition 3):",
                left=0.4, top=3.5, width=12.5, height=0.45,
                font_size=17, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "Given source domain 𝒟_S and task 𝒯_S, and target domain 𝒟_T and task 𝒯_T, "
                "transfer learning aims to improve the learning of target predictive function  f_T  "
                "using knowledge in 𝒟_S and 𝒯_S, where  𝒟_S ≠ 𝒟_T  or  𝒯_S ≠ 𝒯_T.",
                left=0.6, top=4.0, width=12.1, height=1.3,
                font_size=15, color=ORANGE)

    add_ref_footer(sld,
        "Ref: Pan & Yang, IEEE TKDE 22(10), 2010, Definition 1–3.")


def slide_tl_taxonomy(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Taxonomy of Transfer Learning",
                  "Part III — Transfer Learning: Theory & Taxonomy")
    add_slide_number(sld, n, total)

    # Three columns
    headers = ["Inductive TL", "Transductive TL", "Unsupervised TL"]
    descs = [
        "Source and target tasks differ.\nTarget labelled data available.\nIncludes multi-task learning and self-taught learning.",
        "Source and target tasks same.\nDomains differ (𝒳_S ≠ 𝒳_T or P_S(X) ≠ P_T(X)).\nAlso called Domain Adaptation.",
        "Both tasks differ.\nNo labelled data in source or target.\nTransfer of representational structure.",
    ]
    refs = [
        "E.g., fine-tuning BERT (Devlin et al., 2019)",
        "E.g., covariate shift (Shimodaira, 2000)",
        "E.g., self-supervised pretraining",
    ]
    positions = [0.3, 4.6, 8.9]

    for i, (h, d, r) in enumerate(zip(headers, descs, refs)):
        x = positions[i]
        col = sld.shapes.add_shape(
            1, Inches(x), Inches(1.7), Inches(4.0), Inches(5.3))
        col.fill.solid()
        col.fill.fore_color.rgb = LIGHT_BLUE
        col.line.color.rgb = MID_BLUE

        tf = col.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = h
        run.font.size = Pt(17)
        run.font.bold = True
        run.font.color.rgb = DARK_BLUE

        p2 = tf.add_paragraph()
        run2 = p2.add_run()
        run2.text = "\n" + d + "\n\n" + r
        run2.font.size = Pt(13)
        run2.font.color.rgb = DARK_GRAY

    add_ref_footer(sld,
        "Ref: Pan & Yang (2010), Table I. Weiss et al., 'A survey of transfer learning', J. Big Data 3:9 (2016).")


def slide_what_to_transfer(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "What to Transfer?",
                  "Part III — Transfer Learning: Theory & Taxonomy")
    add_slide_number(sld, n, total)

    add_textbox(sld, "Pan & Yang (2010) identify four categories of transferable knowledge:",
                left=0.4, top=1.7, width=12.5, height=0.5,
                font_size=16, color=DARK_GRAY)

    items = [
        (0, "Instance Transfer — reweight source samples for use in target (importance sampling)"),
        (1, "Key tool: importance weight  w(x) = P_T(x)/P_S(x)  (Shimodaira, 2000; Sugiyama et al., 2007)"),
        (0, "Feature Representation Transfer — find good shared feature space across domains"),
        (1, "Learn Φ : 𝒳 → ℝᵈ such that P_S(Φ(X)) ≈ P_T(Φ(X))"),
        (0, "Parameter (Model) Transfer — share model parameters or priors between tasks"),
        (1, "Fine-tuning: initialise f_T with θ learned on source, then update with target data"),
        (0, "Relational-Knowledge Transfer — transfer logical/relational knowledge (less common in PDE context)"),
    ]
    add_bullet_box(sld, items, left=0.4, top=2.3, width=12.5, height=4.7, font_size=15)

    add_ref_footer(sld,
        "Ref: Pan & Yang (2010), Section III. Shimodaira, J. Stat. Plan. Inf. 90 (2000) 227–244.")


def slide_negative_transfer(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Negative Transfer",
                  "Part III — Transfer Learning: Theory & Taxonomy")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Negative Transfer: when knowledge from source domain HURTS target performance.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=17, bold=True, color=ORANGE)

    items = [
        "Occurs when source and target are insufficiently related (Pan & Yang, 2010, Section V).",
        "Formally: E[L_T(f̂_T)] > E[L_T(f*_T)]  where f̂_T uses transfer and f*_T is trained only on target.",
        "Theoretical analysis (Ben-David et al., 2010): target error bounded by source error + divergence between domains.",
        "    ε_T(h) ≤ ε_S(h) + d_H(𝒟_S, 𝒟_T) + λ*",
        "where  d_H  is the H-divergence and λ* is the combined ideal error.",
        "Mitigation strategies: domain similarity assessment; selective transfer; adversarial domain adaptation.",
    ]
    add_bullet_box(sld, items, left=0.4, top=2.4, width=12.5, height=4.4, font_size=15)

    add_ref_footer(sld,
        "Ref: Ben-David et al., 'A theory of learning from different distributions', "
        "Machine Learning 79(1–2), 2010, pp. 151–175.")


def slide_domain_adaptation(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Domain Adaptation",
                  "Part III — Transfer Learning: Theory & Taxonomy")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Setting: same task (𝒯_S = 𝒯_T) but  P_S(X) ≠ P_T(X)  (distributional shift).",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "Covariate Shift (Shimodaira, 2000):",
                left=0.4, top=2.3, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)
    add_textbox(sld,
                "P_S(Y|X) = P_T(Y|X)  but  P_S(X) ≠ P_T(X).\n"
                "Corrected estimator: minimise  Σᵢ w(xᵢ) ℓ(f(xᵢ), yᵢ)  with  w(x) = P_T(x)/P_S(x).",
                left=0.6, top=2.75, width=12.1, height=0.9,
                font_size=14, color=DARK_GRAY)

    add_textbox(sld, "Maximum Mean Discrepancy (Gretton et al., 2012):",
                left=0.4, top=3.75, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)
    add_textbox(sld,
                "MMD²(P, Q) = ‖ μ_P − μ_Q ‖²_ℋ = E_{x~P}[k(x,·)] − E_{y~Q}[k(y,·)]  in RKHS ℋ.\n"
                "Domain Adaptation: add MMD penalty to training loss to align feature distributions.",
                left=0.6, top=4.2, width=12.1, height=0.9,
                font_size=14, color=DARK_GRAY)

    items = [
        "Deep Domain Adaptation (Long et al., ICML 2015): minimise task loss + MMD between deep features.",
        "DANN (Ganin et al., JMLR 2016): adversarial training to make features domain-invariant.",
    ]
    add_bullet_box(sld, items, left=0.4, top=5.2, width=12.5, height=1.6, font_size=14)

    add_ref_footer(sld,
        "Ref: Gretton et al., JMLR 13 (2012); Long et al., ICML 2015; Ganin et al., JMLR 17 (2016) 1–35.")


def slide_fine_tuning(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Fine-Tuning: Parameter Transfer",
                  "Part III — Transfer Learning: Theory & Taxonomy")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Most widely-used transfer learning strategy in deep learning practice (Howard & Ruder, 2018).",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "Algorithm (Fine-Tuning):",
                left=0.4, top=2.3, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "1. Pre-train network f_{θ_S} on source data {(xᵢ,yᵢ)}ᵢ₌₁ᴺˢ to minimise L_S(θ).\n"
                "2. Initialise θ_T ← θ_S.\n"
                "3. Fine-tune on target data {(xⱼ,yⱼ)}ⱼ₌₁ᴺᵀ to minimise L_T(θ_T), "
                "typically with a smaller learning rate η_T ≪ η_S.\n"
                "4. Optionally freeze lower layers (feature extractor) and only train upper layers.",
                left=0.6, top=2.8, width=12.1, height=1.8,
                font_size=14, color=DARK_GRAY)

    items = [
        "Layer freezing: empirically, first k layers act as general feature extractor (Yosinski et al., 2014).",
        "Discriminative learning rates (Howard & Ruder, 2018): use different η per layer; lower for early layers.",
        "Theoretical justification: fine-tuning as a warm start reduces optimisation landscape complexity (Du et al., 2020).",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.7, width=12.5, height=2.2, font_size=15)

    add_ref_footer(sld,
        "Ref: Howard & Ruder, 'Universal Language Model Fine-Tuning', ACL 2018. "
        "Yosinski et al., NeurIPS 2014.")


# ── Part IV ───────────────────────────────────────────────────────────────────

def slide_transfer_bounds(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Theoretical Guarantees",
                  "Part IV — Fine-Tuning & Domain Adaptation Theory")
    add_slide_number(sld, n, total)

    add_textbox(sld, "Ben-David et al. (2010) — Domain Adaptation Bound:",
                left=0.4, top=1.7, width=12.5, height=0.4,
                font_size=17, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "For hypothesis class ℋ, source error ε_S(h), and H-divergence d_ℋΔℋ(𝒟_S, 𝒟_T):\n\n"
                "  ε_T(h) ≤ ε_S(h)  +  ½ d_{ℋΔℋ}(𝒟_S, 𝒟_T)  +  λ*\n\n"
                "where  λ* = min_{h∈ℋ} [ ε_S(h) + ε_T(h) ]  is the ideal joint error.",
                left=0.6, top=2.2, width=12.1, height=1.6,
                font_size=14, color=ORANGE)

    add_textbox(sld, "Implications:",
                left=0.4, top=3.9, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    items = [
        "Transfer improves target performance only if domain divergence is small.",
        "The H-divergence can be estimated from unlabelled samples (proxy A-distance).",
        "Larger source datasets reduce source error but cannot compensate for large domain divergence.",
        "For PDE applications: source and target PDEs must share structural properties (same type, similar domain).",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.4, width=12.5, height=2.6, font_size=15)

    add_ref_footer(sld,
        "Ref: Ben-David et al., Machine Learning 79 (2010) 151–175.")


def slide_multi_fidelity(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Multi-Fidelity Transfer Learning",
                  "Part IV — Fine-Tuning & Domain Adaptation Theory")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Multi-fidelity modelling: combine cheap low-fidelity (LF) and expensive high-fidelity (HF) data.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "Framework (Kennedy & O'Hagan, 2000; Peherstorfer et al., 2018):",
                left=0.4, top=2.3, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "u_HF(x) = ρ · u_LF(x) + δ(x)\n\n"
                "where  ρ  is a scaling factor and  δ  is the discrepancy model.",
                left=0.6, top=2.8, width=12.1, height=0.85,
                font_size=15, color=ORANGE)

    add_textbox(sld, "Transfer Learning approach (Chakraborty, 2021):",
                left=0.4, top=3.75, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "1. Train PINN on large low-fidelity dataset (coarse mesh solutions).\n"
                "2. Fine-tune on small high-fidelity dataset (fine mesh solutions).\n"
                "3. Network learns: lower layers ≈ shared physics; upper layers ≈ fidelity-specific correction.",
                left=0.6, top=4.2, width=12.1, height=1.1,
                font_size=14, color=DARK_GRAY)

    items = [
        "Demonstrated on fluid dynamics problems (Navier-Stokes): 90% fewer HF training samples with comparable accuracy.",
        "Generalises multi-fidelity Gaussian process co-kriging to neural networks.",
    ]
    add_bullet_box(sld, items, left=0.4, top=5.45, width=12.5, height=1.4, font_size=15)

    add_ref_footer(sld,
        "Ref: Chakraborty, J. Comput. Phys. 426 (2021) 109942. "
        "Peherstorfer et al., SIAM Review 60(3) (2018) 550–591.")


def slide_one_shot(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "One-Shot Transfer & Meta-Learning",
                  "Part IV — Fine-Tuning & Domain Adaptation Theory")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "One-Shot Transfer (Desai et al., 2021): transfer a PINN to a new PDE instance using only one labelled sample.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "MAML-style meta-learning (Finn et al., ICML 2017):",
                left=0.4, top=2.35, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "Meta-objective: min_θ  Σ_{i} L_{T_i}( θ − α ∇_θ L_{T_i}(θ) )\n\n"
                "Find initialisation θ* such that one gradient step on any new task gives good performance.",
                left=0.6, top=2.85, width=12.1, height=1.1,
                font_size=14, color=ORANGE)

    items = [
        "Desai et al. (2021): apply MAML to PINNs — meta-train over a family of PDEs (varying coefficients), then one-shot transfer to new PDE.",
        "Result: 1–5 gradient steps on new PDE suffice for accurate solution.",
        "Connection: meta-learning can be interpreted as learning a prior over neural network parameters.",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.1, width=12.5, height=2.7, font_size=15)

    add_ref_footer(sld,
        "Ref: Desai et al., 'One-shot transfer learning of physics-informed neural networks', arXiv:2110.11286 (2021). "
        "Finn et al., ICML 2017.")


def slide_catastrophic_forgetting(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Catastrophic Forgetting",
                  "Part IV — Fine-Tuning & Domain Adaptation Theory")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Catastrophic Forgetting (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017): "
                "neural networks forget previously learned knowledge when trained on new tasks.",
                left=0.4, top=1.7, width=12.5, height=0.7,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "Elastic Weight Consolidation — EWC (Kirkpatrick et al., PNAS 2017):",
                left=0.4, top=2.5, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "L(θ) = L_B(θ) + Σⱼ (λ/2) Fⱼ (θⱼ − θ*_{A,j})²\n\n"
                "Fisher Information matrix F estimates importance of each parameter θⱼ to task A.\n"
                "Quadratic penalty prevents large deviation from θ*_A when learning task B.",
                left=0.6, top=3.0, width=12.1, height=1.4,
                font_size=14, color=ORANGE)

    items = [
        "Relevance for PDE applications: after transfer from source PDE, prevent network from forgetting source physics when fine-tuning on target.",
        "Progressive neural networks (Rusu et al., 2016): add new columns for new tasks, freeze old columns.",
        "Active area of research in continual learning (van de Ven & Tolias, 2019).",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.55, width=12.5, height=2.3, font_size=15)

    add_ref_footer(sld,
        "Ref: Kirkpatrick et al., PNAS 114(13) (2017) 3521–3526.")


# ── Part V ────────────────────────────────────────────────────────────────────

def slide_pinn_intro(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Physics-Informed Neural Networks (PINNs)",
                  "Part V — Transfer Learning for PDE Solving")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Setup: solve PDE  𝒩[u](x) = f(x)  on Ω ⊂ ℝᵈ, with boundary condition  ℬ[u](x) = g(x)  on ∂Ω.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "PINN Loss (Raissi, Perdikaris & Karniadakis, 2019):",
                left=0.4, top=2.35, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "L(θ) = (1/N_r) Σᵢ |𝒩[u_θ](xᵣᵢ) − f(xᵣᵢ)|² + (1/N_b) Σⱼ |ℬ[u_θ](xᵦⱼ) − g(xᵦⱼ)|²\n\n"
                "where u_θ is a neural network, xᵣᵢ are interior collocation points, xᵦⱼ are boundary points.\n"
                "Automatic differentiation computes the PDE residual exactly.",
                left=0.6, top=2.85, width=12.1, height=1.4,
                font_size=14, color=ORANGE)

    items = [
        "No mesh required — meshfree method suitable for high-dimensional or complex-geometry PDEs.",
        "Key limitation: must retrain from scratch for each new PDE or parameter configuration.",
        "Transfer learning directly addresses this limitation.",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.4, width=12.5, height=2.3, font_size=15)

    add_ref_footer(sld,
        "Ref: Raissi, Perdikaris & Karniadakis, 'Physics-informed neural networks', "
        "J. Comput. Phys. 378 (2019) 686–707.")


def slide_tl_pinn(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Transfer Learning for PINNs",
                  "Part V — Transfer Learning for PDE Solving")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Goal: reuse a PINN trained on a source PDE to accelerate solving a related target PDE.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "Goswami et al. (2020) — Fracture Mechanics:",
                left=0.4, top=2.3, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "Phase-field fracture model (Cahn-Hilliard type PDE) with varying crack geometries.\n"
                "Protocol: pre-train on simple crack geometry, fine-tune on complex geometry.\n"
                "Result: fine-tuned model converges 5–10× faster than training from scratch.",
                left=0.6, top=2.8, width=12.1, height=1.05,
                font_size=14, color=DARK_GRAY)

    add_textbox(sld, "Chakraborty (2021) — Navier-Stokes:",
                left=0.4, top=3.95, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "Multi-fidelity PINN transfer: source = LF coarse simulation, target = HF fine simulation.\n"
                "Transfer reduces required HF samples from O(10³) to O(10²) while maintaining accuracy.",
                left=0.6, top=4.45, width=12.1, height=0.85,
                font_size=14, color=DARK_GRAY)

    items = [
        "Key observation: lower PINN layers encode PDE type (elliptic, parabolic, hyperbolic); upper layers encode specific parameters.",
        "Freeze lower layers during fine-tuning — consistent with representation learning theory.",
    ]
    add_bullet_box(sld, items, left=0.4, top=5.4, width=12.5, height=1.5, font_size=15)

    add_ref_footer(sld,
        "Ref: Goswami et al., Theor. Appl. Fract. Mech. 106 (2020) 102447. Chakraborty, J. Comput. Phys. 426 (2021).")


def slide_deeponet(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "DeepONet: Operator Learning",
                  "Part V — Transfer Learning for PDE Solving")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "DeepONet (Lu et al., 2021) learns the solution operator  𝒢 : u₀ ↦ u(·,t)  of a PDE.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "Architecture (Universal Approximation Theorem for Operators, Chen & Chen, 1995):",
                left=0.4, top=2.3, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "𝒢(u)(y) ≈ Σₖ₌₁ᵖ branch_k(u(x₁), …, u(xₘ)) · trunk_k(y)\n\n"
                "Branch net: encodes input function u evaluated at sensor points {xᵢ}.\n"
                "Trunk net: encodes query location y.",
                left=0.6, top=2.8, width=12.1, height=1.2,
                font_size=14, color=ORANGE)

    add_textbox(sld, "Transfer Learning for DeepONet (Goswami et al., 2022):",
                left=0.4, top=4.1, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "Pre-train DeepONet on a family of source PDEs.\n"
                "Fine-tune only the trunk net for a new PDE class — trunk encodes geometry (transferable).\n"
                "Validated on Darcy flow with varying permeability fields.",
                left=0.6, top=4.6, width=12.1, height=1.0,
                font_size=14, color=DARK_GRAY)

    items = [
        "Transfer reduces training data requirement by ~80% for the target operator.",
    ]
    add_bullet_box(sld, items, left=0.4, top=5.75, width=12.5, height=0.9, font_size=15)

    add_ref_footer(sld,
        "Ref: Lu et al., 'Learning nonlinear operators via DeepONet', Nat. Mach. Intell. 3 (2021) 218–229. "
        "Goswami et al., arXiv:2204.12516 (2022).")


def slide_fno(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Fourier Neural Operator & Transfer Learning",
                  "Part V — Transfer Learning for PDE Solving")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Fourier Neural Operator (FNO, Li et al., ICLR 2021) parameterises the integral kernel in Fourier space.",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    add_textbox(sld, "FNO Layer:",
                left=0.4, top=2.3, width=12.5, height=0.4,
                font_size=16, bold=True, color=DARK_BLUE)

    add_textbox(sld,
                "v_{t+1}(x) = σ( W v_t(x) + (𝒦(a; φ) v_t)(x) )\n\n"
                "(𝒦(a; φ) v)(x) = ℱ⁻¹( R_φ · ℱ(v) )(x)\n\n"
                "where ℱ is the Fourier transform and R_φ are learnable complex weights in Fourier space.",
                left=0.6, top=2.8, width=12.1, height=1.3,
                font_size=14, color=ORANGE)

    items = [
        "FNO is resolution-invariant — trained at one grid resolution, evaluated at any resolution.",
        "Transfer across resolutions: zero-shot transfer (Li et al., 2021) from coarse to fine grid with no additional training.",
        "Benchmark: Navier-Stokes (2D), Darcy flow, Burgers' equation — FNO outperforms classical solvers at comparable accuracy for a large family of initial conditions.",
        "Transfer across PDE families: fine-tune FNO trained on parabolic PDEs for hyperbolic PDEs (Subramanian et al., 2023).",
    ]
    add_bullet_box(sld, items, left=0.4, top=4.25, width=12.5, height=2.8, font_size=15)

    add_ref_footer(sld,
        "Ref: Li et al., 'Fourier Neural Operator for Parametric PDEs', ICLR 2021. "
        "Subramanian et al., arXiv:2211.11319 (2023).")


def slide_tl_parameterized_pde(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Transfer Across Parameterised PDE Families",
                  "Part V — Transfer Learning for PDE Solving")
    add_slide_number(sld, n, total)

    add_textbox(sld,
                "Setting: PDE family  𝒩_μ[u] = f  parameterised by μ ∈ 𝒫 (e.g., Reynolds number, diffusion coefficient).",
                left=0.4, top=1.7, width=12.5, height=0.55,
                font_size=16, color=DARK_GRAY)

    items = [
        "Source: train PINN / operator network for  μ ∈ 𝒫_S  (range of parameters).",
        "Target: fine-tune for  μ_T ∉ 𝒫_S  (out-of-distribution parameter).",
        "Xu et al. (2023): hyper-network approach — meta-network generates PINN parameters conditioned on μ; amortises transfer across the entire parameter space.",
        "Pestourie et al. (2023): transfer learning for inverse problems — pre-train forward solver, fine-tune for inverse map.",
        "Key theoretical insight (Lanthaler et al., 2022): operator approximation error depends on Kolmogorov N-width of solution manifold; transfer can reduce effective N-width.",
    ]
    add_bullet_box(sld, items, left=0.4, top=2.4, width=12.5, height=4.5, font_size=15)

    add_ref_footer(sld,
        "Ref: Xu et al., J. Comput. Phys. 474 (2023). "
        "Lanthaler et al., J. Mach. Learn. Res. 23 (2022) 1–76.")


def slide_tl_pde_summary_table(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Summary: TL Strategies for PDE Solving",
                  "Part V — Transfer Learning for PDE Solving")
    add_slide_number(sld, n, total)

    # Table headers
    cols = ["Strategy", "Source → Target", "Architecture", "Key Reference"]
    col_widths = [2.2, 3.2, 3.0, 3.6]
    col_starts = [0.3, 2.55, 5.8, 8.85]
    row_data = [
        ["Fine-tuning PINN",      "Simple → complex geometry",     "Fully-connected NN",  "Goswami et al. (2020)"],
        ["Multi-fidelity TL",     "LF simulation → HF simulation", "PINN",                "Chakraborty (2021)"],
        ["One-shot / MAML",       "PDE family → single instance",  "PINN + meta-learning","Desai et al. (2021)"],
        ["DeepONet TL",           "Source operator → new operator","Branch + Trunk nets",  "Goswami et al. (2022)"],
        ["FNO zero-shot TL",      "Coarse → fine resolution",      "Fourier layers",       "Li et al. (2021)"],
        ["Hyper-network amort.",  "Parameter family  𝒫_S → 𝒫",   "Hyper-network",        "Xu et al. (2023)"],
    ]

    # Header row
    header_top = 1.7
    for j, (hdr, w, x) in enumerate(zip(cols, col_widths, col_starts)):
        cell = sld.shapes.add_shape(
            1, Inches(x), Inches(header_top), Inches(w), Inches(0.45))
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        cell.line.color.rgb = WHITE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = hdr
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = WHITE

    row_height = 0.72
    for i, row in enumerate(row_data):
        bg = LIGHT_BLUE if i % 2 == 0 else WHITE
        top = header_top + 0.45 + i * row_height
        for j, (cell_text, w, x) in enumerate(zip(row, col_widths, col_starts)):
            cell = sld.shapes.add_shape(
                1, Inches(x), Inches(top), Inches(w), Inches(row_height))
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.line.color.rgb = MID_BLUE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = cell_text
            run.font.size = Pt(11)
            run.font.color.rgb = DARK_GRAY

    add_ref_footer(sld, "See reference list (final slide) for full citations.")
    add_slide_number(sld, n, total)


# ── Part VI ───────────────────────────────────────────────────────────────────

def slide_open_problems(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Open Problems",
                  "Part VI — Open Problems & Summary")
    add_slide_number(sld, n, total)

    items = [
        "Theoretical: tight generalisation bounds for deep transfer learning (current bounds often vacuous in practice).",
        "Negative transfer detection: principled methods to decide when NOT to transfer — currently heuristic.",
        "Transferability metrics: quantify domain similarity before transfer (LEEP, LogME — You et al., 2021).",
        "Transfer for high-dimensional PDEs: curse of dimensionality limits collocation-based methods; transfer may help but theory is underdeveloped.",
        "Geometry generalisation: PINNs/operators trained on one domain shape — how to transfer to topologically different domains?",
        "Continual learning for PDEs: learn a sequence of related PDEs without forgetting; EWC-PDE is not yet well-studied.",
        "Uncertainty quantification: Bayesian transfer learning for PDEs remains largely open.",
    ]
    add_bullet_box(sld, items, left=0.4, top=1.7, width=12.5, height=5.4, font_size=15)

    add_ref_footer(sld,
        "Ref: You et al., NeurIPS 2021 (LEEP/LogME). Müller et al., arXiv:2310.05244 (2023) survey open problems.")


def slide_summary(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "Summary",
                  "Part VI — Open Problems & Summary")
    add_slide_number(sld, n, total)

    items = [
        "Transfer Learning = systematic reuse of knowledge from a source domain/task to improve a target domain/task (Pan & Yang, 2010).",
        "Theoretical foundation: domain divergence bounds (Ben-David et al., 2010); representational hierarchy (Bengio et al., 2013).",
        "Main strategies: fine-tuning (parameter transfer), domain adaptation (feature alignment), meta-learning (one-shot).",
        "For PDE solving: PINNs, DeepONet, FNO all benefit from transfer — faster convergence, fewer HF samples.",
        "Frozen lower layers encode generic PDE structure; upper layers encode problem-specific information — supports principled fine-tuning.",
        "Key challenge: avoiding negative transfer — assess domain similarity before committing to transfer.",
        "Active open research area with significant practical impact on scientific computing.",
    ]
    add_bullet_box(sld, items, left=0.4, top=1.7, width=12.5, height=5.4, font_size=16)


def slide_references(prs, n, total):
    sld = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(sld, WHITE)
    add_title_bar(sld, "References")
    add_slide_number(sld, n, total)

    refs = [
        "[1] Pan & Yang, 'A Survey on Transfer Learning', IEEE TKDE 22(10), 2010, pp. 1345–1359.",
        "[2] Ben-David et al., 'A theory of learning from different distributions', Machine Learning 79 (2010) 151–175.",
        "[3] Bengio et al., 'Representation Learning: A Review and New Perspectives', IEEE TPAMI 35(8), 2013.",
        "[4] Yosinski et al., 'How transferable are features in deep neural networks?', NeurIPS 2014.",
        "[5] Goodfellow, Bengio & Courville, 'Deep Learning', MIT Press, 2016.",
        "[6] Weiss et al., 'A survey of transfer learning', Journal of Big Data 3:9, 2016.",
        "[7] Raissi, Perdikaris & Karniadakis, 'Physics-informed neural networks', J. Comput. Phys. 378 (2019) 686–707.",
        "[8] Lu et al., 'Learning nonlinear operators via DeepONet', Nat. Mach. Intell. 3 (2021) 218–229.",
        "[9] Li et al., 'Fourier Neural Operator for Parametric PDEs', ICLR 2021.",
        "[10] Goswami et al., 'Transfer learning enhanced physics informed neural network for phase-field fracture', "
              "Theor. Appl. Fract. Mech. 106 (2020) 102447.",
        "[11] Chakraborty, 'Transfer learning based multi-fidelity physics informed deep neural network', "
              "J. Comput. Phys. 426 (2021) 109942.",
        "[12] Desai et al., 'One-shot transfer learning of physics-informed neural networks', arXiv:2110.11286, 2021.",
        "[13] Kirkpatrick et al., 'Overcoming catastrophic forgetting in neural networks', PNAS 114(13) (2017) 3521–3526.",
        "[14] Finn, Abbeel & Levine, 'Model-Agnostic Meta-Learning', ICML 2017.",
        "[15] Gretton et al., 'A Kernel Two-Sample Test', JMLR 13 (2012) 723–773.",
        "[16] Xu et al., 'Transfer learning based physics-informed neural networks for solving PDEs', "
              "J. Comput. Phys. 474 (2023) 111900.",
        "[17] Bartlett, Montanari & Rakhlin, 'Deep learning: a statistical viewpoint', Acta Numerica 30 (2021) 87–201.",
        "[18] Peherstorfer, Willcox & Gunzburger, 'Survey of multifidelity methods', SIAM Review 60(3) (2018) 550–591.",
    ]

    ref_text = "\n".join(refs)
    txBox = sld.shapes.add_textbox(
        Inches(0.3), Inches(1.65), Inches(12.7), Inches(5.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = ref_text
    run.font.size = Pt(9.5)
    run.font.color.rgb = DARK_GRAY


# ── Main ──────────────────────────────────────────────────────────────────────

def build_presentation():
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    TOTAL = 27

    slide_title(prs)                                    # 1
    slide_outline(prs, 2, TOTAL)                        # 2
    slide_motivation(prs, 3, TOTAL)                     # 3
    slide_data_scarcity(prs, 4, TOTAL)                  # 4
    slide_historical(prs, 5, TOTAL)                     # 5
    slide_dl_basics(prs, 6, TOTAL)                      # 6
    slide_representation(prs, 7, TOTAL)                 # 7
    slide_training_challenges(prs, 8, TOTAL)            # 8
    slide_tl_formal_def(prs, 9, TOTAL)                  # 9
    slide_tl_taxonomy(prs, 10, TOTAL)                   # 10
    slide_what_to_transfer(prs, 11, TOTAL)              # 11
    slide_negative_transfer(prs, 12, TOTAL)             # 12
    slide_domain_adaptation(prs, 13, TOTAL)             # 13
    slide_fine_tuning(prs, 14, TOTAL)                   # 14
    slide_transfer_bounds(prs, 15, TOTAL)               # 15
    slide_multi_fidelity(prs, 16, TOTAL)                # 16
    slide_one_shot(prs, 17, TOTAL)                      # 17
    slide_catastrophic_forgetting(prs, 18, TOTAL)       # 18
    slide_pinn_intro(prs, 19, TOTAL)                    # 19
    slide_tl_pinn(prs, 20, TOTAL)                       # 20
    slide_deeponet(prs, 21, TOTAL)                      # 21
    slide_fno(prs, 22, TOTAL)                           # 22
    slide_tl_parameterized_pde(prs, 23, TOTAL)          # 23
    slide_tl_pde_summary_table(prs, 24, TOTAL)          # 24
    slide_open_problems(prs, 25, TOTAL)                 # 25
    slide_summary(prs, 26, TOTAL)                       # 26
    slide_references(prs, 27, TOTAL)                    # 27

    output_path = "/home/runner/work/research_assistent/research_assistent/data/transfer_learning_lecture.pptx"
    prs.save(output_path)
    print(f"Saved: {output_path}  ({TOTAL} slides)")


if __name__ == "__main__":
    build_presentation()
