from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# Photo-to-Coloring-Book Studio
# Local-first Streamlit application. Uploaded photos are processed in memory.
# ---------------------------------------------------------------------------

APP_NAME = "Keepsake Coloring Studio"
MAX_PHOTOS = 10
MAX_UPLOAD_MB = 20
A4_SIZE = (2480, 3508)       # 300 DPI
LETTER_SIZE = (2550, 3300)   # 300 DPI


def safe_filename(value: str, fallback: str = "coloring-book") -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", (value or "").strip()).strip("-_")
    return (value[:60] or fallback).lower()


def load_image(uploaded_file) -> Image.Image:
    """Decode an upload safely, apply EXIF orientation, and return RGB."""
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"{uploaded_file.name} is larger than {MAX_UPLOAD_MB} MB.")
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 250 or image.height < 250:
        raise ValueError("Image is too small. Please use an image at least 250 × 250 pixels.")
    return image


def fit_for_processing(image: Image.Image, max_side: int = 1800) -> Image.Image:
    """Limit processing size while preserving aspect ratio."""
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _normalize_gray(gray: np.ndarray, contrast: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=max(1.0, contrast), tileGridSize=(8, 8))
    return clahe.apply(gray)


def _clean_binary(binary: np.ndarray, speckle_removal: int = 1) -> np.ndarray:
    """Remove tiny noise while keeping line-art edges crisp."""
    if speckle_removal > 0:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=speckle_removal)
    return binary


def make_coloring_page(
    image: Image.Image,
    style: str = "Balanced",
    line_strength: int = 55,
    detail: int = 55,
    smoothing: int = 65,
    contrast: float = 2.0,
    invert: bool = False,
) -> Image.Image:
    """
    Convert a photograph into printable black outlines on a white background.

    This is a classical computer-vision conversion, not generative AI. It works
    best with sharp, well-lit photos and a clear subject/background separation.
    """
    image = fit_for_processing(image, 1800)
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Preserve useful texture while reducing camera noise.
    diameter = 5 if smoothing >= 50 else 3
    sigma_color = int(20 + smoothing * 1.1)
    sigma_space = int(20 + smoothing * 1.1)
    smooth = cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)
    normalized = _normalize_gray(smooth, contrast)

    # Adaptive threshold and pencil-dodge rendering capture useful boundaries.
    # The final conversion deliberately favors white paper over large dark fills.
    block = 31 if detail >= 50 else 41
    if block % 2 == 0:
        block += 1
    c = int(3 + (100 - detail) * 0.08)
    adaptive = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, c
    )

    # Pencil-dodge image: dark strokes on a mostly white page.
    inverted = 255 - normalized
    blur = cv2.GaussianBlur(inverted, (0, 0), sigmaX=9 + (100 - detail) * 0.10)
    dodge = cv2.divide(normalized, 255 - blur, scale=256)
    pencil_threshold = int(247 - detail * 0.42)
    pencil = np.where(dodge < pencil_threshold, 0, 255).astype(np.uint8)

    # Structural edges add definition around faces, bodies and objects.
    low = max(18, int(75 - detail * 0.45))
    high = min(230, low + 95)
    edges = cv2.Canny(normalized, low, high)
    thickness = 1 if line_strength < 45 else 2
    edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8), iterations=1)
    edge_lines = 255 - edges

    if style == "Bold & Simple":
        # Simplify texture and retain only the strongest structural contours.
        simple_gray = cv2.bilateralFilter(gray, 9, 100, 100)
        simple_gray = cv2.GaussianBlur(simple_gray, (7, 7), 0)
        simple_edges = cv2.Canny(simple_gray, 55, 145)
        simple_edges = cv2.dilate(simple_edges, np.ones((2, 2), np.uint8), iterations=1)
        result = 255 - simple_edges
    elif style == "Detailed":
        # Keep more pencil texture, but threshold it to avoid gray print areas.
        detail_lines = np.minimum(pencil, edge_lines)
        result = detail_lines
    else:
        # Balanced uses restrained pencil detail plus clear subject boundaries.
        result = np.minimum(pencil, edge_lines)

    result = _clean_binary(result, 1)
    if invert:
        result = 255 - result

    # Keep output monochrome and clean.
    result = _clean_binary(result, 1)
    if invert:
        result = 255 - result

    # Ensure black outlines on white by default.
    if np.mean(result) < 127 and not invert:
        result = 255 - result

    # Slightly reinforce near-black outlines without creating gray shading.
    result = np.where(result < 190, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L").convert("RGB")


def make_page_canvas(
    artwork: Image.Image,
    page_size: Tuple[int, int],
    margin_mm: int = 12,
    border: bool = True,
) -> Image.Image:
    """Place artwork on a high-resolution white printable page."""
    page = Image.new("RGB", page_size, "white")
    margin = int(page_size[0] * (margin_mm / (210 if page_size == A4_SIZE else 216)))
    max_w = page_size[0] - 2 * margin
    max_h = page_size[1] - 2 * margin
    art = artwork.copy()
    art.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (page_size[0] - art.width) // 2
    y = (page_size[1] - art.height) // 2
    page.paste(art, (x, y))
    if border:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(page)
        inset = max(12, margin // 2)
        draw.rectangle((inset, inset, page_size[0] - inset, page_size[1] - inset),
                       outline=(225, 225, 225), width=max(2, page_size[0] // 1200))
    return page


def create_cover(
    title: str,
    subtitle: str,
    recipient: str,
    hero: Image.Image,
    page_size: Tuple[int, int],
) -> Image.Image:
    """Create a clean cover with a preview illustration and personalized wording."""
    from PIL import ImageDraw, ImageFont

    page = Image.new("RGB", page_size, (255, 255, 255))
    draw = ImageDraw.Draw(page)
    w, h = page_size

    def font(size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        return ImageFont.load_default()

    title_font = font(int(w * 0.052), True)
    subtitle_font = font(int(w * 0.021))
    small_font = font(int(w * 0.018))
    accent = (52, 70, 82)

    def centered(text, y, selected_font, fill=accent):
        box = draw.textbbox((0, 0), text, font=selected_font)
        draw.text(((w - (box[2] - box[0])) / 2, y), text, font=selected_font, fill=fill)

    centered(title[:48], int(h * 0.09), title_font)
    centered(subtitle[:75], int(h * 0.17), subtitle_font, (95, 105, 112))
    if recipient.strip():
        centered(f"Made especially for {recipient.strip()[:40]}",
                 int(h * 0.22), small_font, (115, 115, 115))

    hero = hero.convert("RGB")
    hero.thumbnail((int(w * 0.72), int(h * 0.52)), Image.Resampling.LANCZOS)
    # Add a subtle inset frame around the featured artwork.
    x = (w - hero.width) // 2
    y = int(h * 0.30)
    page.paste(hero, (x, y))
    draw.rectangle((x - 8, y - 8, x + hero.width + 8, y + hero.height + 8),
                   outline=(205, 210, 212), width=max(2, w // 1000))
    centered("10 PERSONALIZED COLORING PAGES", int(h * 0.88), small_font, accent)
    centered("A keepsake made from your favorite photographs",
             int(h * 0.92), small_font, (125, 125, 125))
    return page


def build_pdf(pages: List[Image.Image], output: io.BytesIO, paper: str = "A4"):
    size = A4 if paper == "A4" else LETTER
    pdf = canvas.Canvas(output, pagesize=size, pageCompression=1)
    width, height = size
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        pdf.drawImage(ImageReader(buf), 0, 0, width=width, height=height,
                      preserveAspectRatio=False, mask="auto")
        pdf.showPage()
    pdf.setTitle("Personalized Photo Coloring Book")
    pdf.setAuthor("Keepsake Coloring Studio")
    pdf.save()
    output.seek(0)
    return output


def build_zip(pages: List[Image.Image], names: List[str], cover: Image.Image | None = None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        if cover is not None:
            buf = io.BytesIO()
            cover.save(buf, format="PNG", dpi=(300, 300))
            archive.writestr("00-cover.png", buf.getvalue())
        for index, (page, name) in enumerate(zip(pages, names), start=1):
            buf = io.BytesIO()
            page.save(buf, format="PNG", dpi=(300, 300), optimize=True)
            archive.writestr(f"{index:02d}-{safe_filename(Path(name).stem)}.png", buf.getvalue())
        archive.writestr("PRINTING-NOTES.txt",
                         "Print at Actual Size / 100% on A4 or US Letter paper. "
                         "Use white paper and disable fit-to-page scaling if possible.\n")
    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# User interface — premium, warm editorial studio
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep Streamlit's native controls, but give the workspace a cohesive product UI.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

:root {
  --ink: #263641;
  --muted: #73808A;
  --paper: #F7F6F2;
  --white: #FFFFFF;
  --line: #E7E5DF;
  --blue: #607B8C;
  --blue-dark: #405A6B;
  --sand: #EDE5D9;
  --sage: #DCE6DF;
  --success: #47745D;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
.block-container { max-width: 1420px; padding-top: 1.35rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1.5rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
h1 { font-family: 'Playfair Display', Georgia, serif !important; }
h2, h3 { font-weight: 700 !important; }
p, label, .stCaption { color: var(--muted); }
div[data-testid="stTabs"] button { font-weight: 650; }
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--blue-dark); border-bottom-color: var(--blue-dark);
}
div[data-testid="stTabs"] [role="tablist"] {
  gap: 1.2rem; border-bottom: 1px solid var(--line);
}
div[data-testid="stFileUploader"] section {
  background: #FCFCFA; border: 1.5px dashed #BAC6C9; border-radius: 18px;
  padding: 1.15rem;
}
div[data-testid="stFileUploader"] section:hover { border-color: var(--blue); }
.stButton>button, .stDownloadButton>button {
  border-radius: 12px; min-height: 2.8rem; font-weight: 650;
  border: 1px solid #D6DEE0; transition: all .18s ease;
}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {
  background: var(--blue-dark); color: white; border-color: var(--blue-dark);
}
.stButton>button:hover, .stDownloadButton>button:hover {
  border-color: var(--blue); transform: translateY(-1px);
}
div[data-testid="stMetric"] {
  background: white; border: 1px solid var(--line); border-radius: 15px; padding: 14px 16px;
}
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 700; }
div[data-testid="stExpander"] {
  background: white; border: 1px solid var(--line); border-radius: 14px;
}
div[data-testid="stImage"] img { border-radius: 12px; }
div[data-testid="stAlert"] { border-radius: 13px; }
input, textarea { border-radius: 10px !important; }
hr { border-color: var(--line); }
.brandbar {
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding: 7px 0 22px; border-bottom:1px solid var(--line); margin-bottom:25px;
}
.brandmark { display:flex; align-items:center; gap:11px; }
.brandicon {
  display:grid; place-items:center; width:42px; height:42px; border-radius:14px;
  color:#fff; background:var(--blue-dark); font-size:22px;
}
.brandname { font-weight:800; font-size:16px; letter-spacing:-.03em; color:var(--ink); }
.brandtag { font-size:11px; color:var(--muted); letter-spacing:.12em; text-transform:uppercase; }
.topnote {
  padding:8px 13px; border-radius:99px; background:#E9EFEA; color:#557260;
  font-size:12px; font-weight:700; white-space:nowrap;
}
.hero {
  background: linear-gradient(112deg, #E8EEF0 0%, #F3EEE6 58%, #F6F1E9 100%);
  border:1px solid #E0E4E1; border-radius:24px; padding:30px 34px;
  margin-bottom:22px; position:relative; overflow:hidden;
}
.hero:after {
  content:"✳"; position:absolute; right:5%; top:-34px; font-size:190px;
  color:rgba(96,123,140,.09); line-height:1;
}
.eyebrow { color:var(--blue-dark); font-size:11px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { color:var(--ink); font-size:clamp(2rem,4vw,3.15rem); line-height:1.08; margin:9px 0 10px; }
.hero p { max-width:660px; color:#65737A; font-size:15px; line-height:1.65; margin:0; }
.hero-foot { display:flex; flex-wrap:wrap; gap:9px; margin-top:19px; }
.pill {
  display:inline-flex; align-items:center; gap:6px; padding:7px 11px;
  border-radius:99px; background:rgba(255,255,255,.72); color:#53656F;
  border:1px solid rgba(255,255,255,.8); font-size:11px; font-weight:700;
}
.stepstrip {
  display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:20px 0 23px;
}
.step {
  display:flex; align-items:center; gap:11px; background:#fff; border:1px solid var(--line);
  border-radius:15px; padding:12px 15px;
}
.stepnum {
  width:30px; height:30px; border-radius:50%; display:grid; place-items:center;
  background:#EEF1F0; color:#61717A; font-size:12px; font-weight:800; flex-shrink:0;
}
.stepactive { border-color:#AABBC1; background:#F1F5F4; }
.stepactive .stepnum { background:var(--blue-dark); color:white; }
.steptitle { font-size:12px; font-weight:800; color:var(--ink); }
.stepsub { font-size:10px; color:var(--muted); margin-top:2px; }
.panel {
  background:white; border:1px solid var(--line); border-radius:19px; padding:20px 22px;
  margin:9px 0 16px;
}
.panelhead { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:13px; }
.paneltitle { font-weight:800; color:var(--ink); font-size:16px; }
.panelsub { color:var(--muted); font-size:12px; margin-top:4px; line-height:1.5; }
.counter {
  border-radius:10px; padding:6px 10px; background:#F0F3F1; color:#536A61;
  font-size:11px; font-weight:800; white-space:nowrap;
}
.mutednote { color:var(--muted); font-size:11px; line-height:1.6; }
.previewframe {
  background:#F1F0EB; border:1px solid var(--line); border-radius:17px; padding:10px;
}
.footerline { color:#8A9295; font-size:11px; text-align:center; padding:22px 0 4px; }
@media (max-width: 700px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .hero { padding:24px 21px; }
  .stepstrip { grid-template-columns:1fr; }
  .brandtag { display:none; }
  .topnote { font-size:10px; }
}
</style>
""", unsafe_allow_html=True)

# State initialization
for key, default in {
    "processed_pages": [],
    "source_names": [],
    "cover_image": None,
    "pdf_bytes": None,
    "zip_bytes": None,
    "export_name": "coloring-book",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<div class="brandbar">
  <div class="brandmark">
    <div class="brandicon">✳</div>
    <div>
      <div class="brandname">Keepsake Studio</div>
      <div class="brandtag">Photo to coloring book</div>
    </div>
  </div>
  <div class="topnote">PRIVATE · CREATIVE · PRINT-READY</div>
</div>
<div class="hero">
  <div class="eyebrow">Your memories, made creative</div>
  <h1>A coloring book made<br>from your favorite moments.</h1>
  <p>Bring ten meaningful photographs together in a personalized coloring book.
  Create the pages, make a cover that feels special, and export a keepsake ready to print.</p>
  <div class="hero-foot">
    <span class="pill">✦ 10 custom pages</span>
    <span class="pill">▤ A4 or US Letter</span>
    <span class="pill">↓ Print-ready PDF</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Main settings are placed in a compact expandable control panel instead of a
# permanently open sidebar, giving the workspace more room for photo previews.
with st.expander("⚙️ Coloring style & print settings", expanded=False):
    settings_left, settings_right = st.columns([1.1, 1])
    with settings_left:
        paper = st.selectbox("Page size", ["A4", "US Letter"], index=0)
        style = st.selectbox(
            "Illustration finish",
            ["Balanced", "Bold & Simple", "Detailed"],
            help="Balanced is a good starting point. Bold & Simple reduces texture; Detailed keeps more photographic detail."
        )
        show_border = st.checkbox("Add a subtle page border", value=False)
    with settings_right:
        line_strength = st.slider("Outline weight", 20, 90, 58)
        detail = st.slider("Detail level", 15, 90, 55)
        smoothing = st.slider("Texture smoothing", 10, 95, 68)
        contrast = st.slider("Photo contrast", 1.0, 4.0, 2.0, 0.1)

pages_ready = len(st.session_state.processed_pages)
st.markdown(f"""
<div class="stepstrip">
  <div class="step stepactive"><div class="stepnum">1</div><div><div class="steptitle">Add your photos</div><div class="stepsub">Choose your 10 memories</div></div></div>
  <div class="step {'stepactive' if pages_ready else ''}"><div class="stepnum">2</div><div><div class="steptitle">Create & review</div><div class="stepsub">Fine-tune your coloring pages</div></div></div>
  <div class="step {'stepactive' if pages_ready else ''}"><div class="stepnum">3</div><div><div class="steptitle">Personalize & export</div><div class="stepsub">Make your printable book</div></div></div>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_design, tab_export = st.tabs(
    ["01  ·  Upload photos", "02  ·  Review pages", "03  ·  Finish your book"]
)

with tab_upload:
    st.markdown("""
    <div class="panel">
      <div class="panelhead">
        <div><div class="paneltitle">Start with ten photographs</div>
        <div class="panelsub">Family, children, pets, celebrations, travel — choose the moments you want to turn into art.</div></div>
        <div class="counter">10 PHOTOS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Drop your photographs here, or browse files",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        max_upload_size=MAX_UPLOAD_MB,
        help=f"Select exactly 10 images. JPG, PNG and WEBP · Up to {MAX_UPLOAD_MB} MB each."
    )

    if uploads:
        if len(uploads) > MAX_PHOTOS:
            st.error(f"You selected {len(uploads)} files. Please choose no more than 10.")
        else:
            st.progress(len(uploads) / MAX_PHOTOS)
            status_col, count_col = st.columns([3, 1])
            with status_col:
                st.markdown(f"**{len(uploads)} of 10 photographs selected**")
                st.caption("Check the previews below before creating your pages.")
            with count_col:
                st.metric("Photos added", f"{len(uploads)}/10")

            preview_cols = st.columns(5)
            valid_uploads = True
            for i, file in enumerate(uploads):
                try:
                    img = load_image(file)
                    with preview_cols[i % 5]:
                        st.image(img, use_container_width=True)
                        st.caption(f"**{i+1:02d}.** {Path(file.name).stem[:17]}")
                        st.caption(f"{img.width} × {img.height}")
                except Exception as exc:
                    valid_uploads = False
                    st.error(f"{file.name}: {exc}")

            if len(uploads) != MAX_PHOTOS:
                st.info(f"{MAX_PHOTOS - len(uploads)} more photo(s) needed before you can create the book.")
            elif valid_uploads:
                st.markdown('<div class="mutednote">Tip: clear, bright photos with one main subject usually create the cleanest coloring outlines.</div>', unsafe_allow_html=True)
                if st.button("✦ Create my 10 coloring pages", type="primary", use_container_width=True):
                    new_pages, new_names, errors = [], [], []
                    progress = st.progress(0, text="Preparing your photographs…")
                    target_size = A4_SIZE if paper == "A4" else LETTER_SIZE
                    for index, file in enumerate(uploads):
                        try:
                            original = load_image(file)
                            art = make_coloring_page(
                                original, style, line_strength, detail, smoothing, contrast
                            )
                            page = make_page_canvas(art, target_size, border=show_border)
                            new_pages.append(page)
                            new_names.append(file.name)
                        except Exception as exc:
                            errors.append(f"{file.name}: {exc}")
                        progress.progress((index + 1) / MAX_PHOTOS,
                                          text=f"Creating page {index+1} of 10…")
                    progress.empty()
                    if errors:
                        for error in errors:
                            st.error(error)
                    if len(new_pages) == MAX_PHOTOS:
                        st.session_state.processed_pages = new_pages
                        st.session_state.source_names = new_names
                        st.session_state.cover_image = new_pages[0]
                        st.session_state.pdf_bytes = None
                        st.session_state.zip_bytes = None
                        st.success("Your ten coloring pages are ready. Open Review pages to inspect them.")
            else:
                st.warning("Please replace any image that could not be opened.")
    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:26px 18px;">
          <div style="font-size:34px;margin-bottom:8px;">▧</div>
          <div class="paneltitle">Your photo collection starts here</div>
          <div class="panelsub">Add 10 photographs to begin. Your selected images will appear as previews above.</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Privacy: in this local version, photo processing runs on the computer hosting the app. If you publish it online, images will pass through your hosting server.")

with tab_design:
    pages = st.session_state.processed_pages
    if not pages:
        st.markdown("""
        <div class="panel">
          <div class="paneltitle">Your coloring pages will appear here</div>
          <div class="panelsub">Once your ten photos are converted, review each page before exporting your book.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Pages created", "10 / 10")
        metric2.metric("Print resolution", "300 DPI")
        metric3.metric("Paper format", paper)
        st.markdown("### Your coloring page gallery")
        st.caption("Look for clear, open spaces that are comfortable to color. You can download any page individually.")
        gallery_cols = st.columns(2, gap="large")
        for i, page in enumerate(pages):
            with gallery_cols[i % 2]:
                st.markdown('<div class="previewframe">', unsafe_allow_html=True)
                st.image(page, use_container_width=True)
                st.markdown(f"**PAGE {i+1:02d}**  ·  {Path(st.session_state.source_names[i]).stem[:36]}")
                st.markdown('</div>', unsafe_allow_html=True)
                with st.expander(f"Page {i+1} options"):
                    page_buffer = io.BytesIO()
                    page.save(page_buffer, format="PNG", dpi=(300, 300))
                    st.download_button(
                        f"Download page {i+1} as PNG",
                        data=page_buffer.getvalue(),
                        file_name=f"coloring-page-{i+1:02d}.png",
                        mime="image/png",
                        key=f"single_page_{i}",
                        use_container_width=True
                    )

with tab_export:
    pages = st.session_state.processed_pages
    if not pages:
        st.markdown("""
        <div class="panel">
          <div class="paneltitle">One last touch: make it yours</div>
          <div class="panelsub">After creating your pages, add a title and recipient name, preview the cover, and download the finished book.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### Design your book cover")
        st.caption("A personal title and a recipient name turn the pages into a thoughtful gift.")
        cover_left, cover_right = st.columns([0.9, 1.1], gap="large")
        with cover_left:
            with st.container(border=True):
                book_title = st.text_input("Book title", value="My Special Coloring Book", max_chars=48)
                subtitle = st.text_input("Cover subtitle", value="A collection of favorite memories", max_chars=75)
                recipient = st.text_input("Made especially for", value="", max_chars=40,
                                          placeholder="e.g. Emma, Grandma, The Smith Family")
                cover_choice = st.selectbox(
                    "Choose the cover artwork",
                    options=list(range(1, len(pages) + 1)),
                    format_func=lambda x: f"Coloring page {x}"
                )
                add_cover = st.checkbox("Include this cover in my PDF and ZIP", value=True)
        with cover_right:
            cover = create_cover(book_title, subtitle, recipient, pages[cover_choice - 1],
                                 A4_SIZE if paper == "A4" else LETTER_SIZE)
            st.image(cover, caption="Live cover preview", use_container_width=True)

        st.markdown("### Export your keepsake")
        st.markdown("""
        <div class="panel">
          <div class="paneltitle">Your book includes</div>
          <div class="panelsub">10 personalized black-and-white pages, plus your custom cover when selected.
          Download a ready-to-print PDF or individual high-resolution PNG files in one ZIP.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✦ Build my printable coloring book", type="primary", use_container_width=True):
            with st.spinner("Laying out your pages and preparing downloads…"):
                export_pages = ([cover] if add_cover else []) + pages
                pdf_buffer = build_pdf(export_pages, io.BytesIO(), paper)
                zip_buffer = build_zip(pages, st.session_state.source_names, cover if add_cover else None)
                st.session_state.pdf_bytes = pdf_buffer.getvalue()
                st.session_state.zip_bytes = zip_buffer.getvalue()
                st.session_state.export_name = safe_filename(book_title)
            st.success("Your coloring book is ready.")

        if st.session_state.get("pdf_bytes"):
            base = st.session_state.get("export_name", "coloring-book")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "↓  Download print-ready PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"{base}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            with dl2:
                st.download_button(
                    "↓  Download PNG pages (ZIP)",
                    data=st.session_state.zip_bytes,
                    file_name=f"{base}-png-pages.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            st.caption("Printing tip: select the correct paper size and print at Actual Size / 100% on white paper.")

st.markdown("""
<div class="footerline">
  KEEPSAKE STUDIO &nbsp;·&nbsp; A little creativity, made from your memories.
  <br>Review every page before printing. Complex photos may need different conversion settings.
</div>
""", unsafe_allow_html=True)
