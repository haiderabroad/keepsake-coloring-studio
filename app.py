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
# User interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🖍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f7f4; }
.block-container { padding-top: 2rem; max-width: 1440px; }
.hero { background: linear-gradient(120deg,#e8edf0,#f5eee5); border:1px solid #dce2e4;
        border-radius:22px; padding:30px 34px; margin-bottom:22px; }
.hero h1 { font-family:'Playfair Display',serif; color:#344652; font-size:2.45rem; margin:0; }
.hero p { color:#58666e; font-size:1.03rem; margin:.55rem 0 0; }
.section-card { background:white; border:1px solid #e5e7e4; border-radius:18px;
                padding:18px 20px; margin:10px 0; }
.muted { color:#68757b; font-size:.92rem; }
div[data-testid="stMetric"] { background:white; padding:14px; border-radius:14px;
                              border:1px solid #e5e7e4; }
.stButton>button, .stDownloadButton>button { border-radius:12px; min-height:2.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Make a Coloring Book From Your Memories</h1>
  <p>Turn 10 favorite photographs into clean, printable coloring pages — then personalize the cover and export your keepsake book.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🖍️ Studio settings")
    paper = st.selectbox("Book size", ["A4", "US Letter"], index=0)
    style = st.selectbox("Illustration style", ["Balanced", "Bold & Simple", "Detailed"],
                         help="Bold & Simple reduces detail; Detailed keeps more texture and contours.")
    line_strength = st.slider("Outline weight", 20, 90, 58)
    detail = st.slider("Detail level", 15, 90, 55)
    smoothing = st.slider("Background / texture smoothing", 10, 95, 68)
    contrast = st.slider("Photo contrast enhancement", 1.0, 4.0, 2.0, 0.1)
    show_border = st.checkbox("Add a subtle printable page border", value=False)
    st.caption("Tip: sharp, bright photos with visible subjects usually produce the clearest outlines.")

tab_upload, tab_design, tab_export = st.tabs(
    ["① Upload photos", "② Design & preview", "③ Create your book"]
)

if "processed_pages" not in st.session_state:
    st.session_state.processed_pages = []
if "source_names" not in st.session_state:
    st.session_state.source_names = []
if "cover_image" not in st.session_state:
    st.session_state.cover_image = None

with tab_upload:
    st.subheader("Upload your 10 photographs")
    st.write("Choose exactly 10 photos. JPG, JPEG, PNG and WEBP are supported.")
    uploads = st.file_uploader(
        "Select up to 10 photographs",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        max_upload_size=MAX_UPLOAD_MB,
        help=f"Maximum {MAX_PHOTOS} photos; each file can be up to {MAX_UPLOAD_MB} MB."
    )
    if uploads:
        if len(uploads) > MAX_PHOTOS:
            st.error(f"You selected {len(uploads)} files. Please select no more than {MAX_PHOTOS}.")
        else:
            st.success(f"{len(uploads)} of {MAX_PHOTOS} photographs selected.")
            preview_cols = st.columns(5)
            for i, file in enumerate(uploads):
                try:
                    img = load_image(file)
                    with preview_cols[i % 5]:
                        st.image(img, caption=f"{i+1}. {file.name}", use_container_width=True)
                        st.caption(f"{img.width} × {img.height}px")
                except Exception as exc:
                    st.error(f"{file.name}: {exc}")

            if len(uploads) != MAX_PHOTOS:
                st.info(f"Add {MAX_PHOTOS - len(uploads)} more photo(s) to enable book creation.")
            else:
                if st.button("✨ Convert all 10 photos into coloring pages", type="primary", use_container_width=True):
                    new_pages, new_names = [], []
                    progress = st.progress(0, text="Preparing your photographs...")
                    errors = []
                    for index, file in enumerate(uploads):
                        try:
                            original = load_image(file)
                            art = make_coloring_page(
                                original, style, line_strength, detail, smoothing, contrast
                            )
                            target_size = A4_SIZE if paper == "A4" else LETTER_SIZE
                            page = make_page_canvas(art, target_size, border=show_border)
                            new_pages.append(page)
                            new_names.append(file.name)
                        except Exception as exc:
                            errors.append(f"{file.name}: {exc}")
                        progress.progress((index + 1) / MAX_PHOTOS,
                                          text=f"Converting photo {index+1} of {MAX_PHOTOS}...")
                    progress.empty()
                    if errors:
                        for error in errors:
                            st.error(error)
                    if len(new_pages) == MAX_PHOTOS:
                        st.session_state.processed_pages = new_pages
                        st.session_state.source_names = new_names
                        st.session_state.cover_image = new_pages[0]
                        st.success("All 10 coloring pages are ready. Open Design & preview to review them.")
                    else:
                        st.error("The book was not created because one or more photos could not be processed.")
    else:
        st.info("When run locally, your photos are processed on your own computer. If you deploy the app online, uploads will pass through your hosting server.")

with tab_design:
    st.subheader("Review your coloring pages")
    pages = st.session_state.processed_pages
    if not pages:
        st.info("Upload 10 photographs and convert them first.")
    else:
        st.markdown(f"**{len(pages)} pages created** · {paper} · 300 DPI export")
        st.caption("Review each result. If a page looks too dark or too detailed, adjust the studio settings and convert the photos again.")
        cols = st.columns(2)
        for i, page in enumerate(pages):
            with cols[i % 2]:
                st.image(page, caption=f"Coloring page {i+1}: {st.session_state.source_names[i]}",
                         use_container_width=True)
                with st.expander(f"Page {i+1} details"):
                    st.write("Use the original photo as the reference when deciding whether the outline is clear enough.")
                    page_buffer = io.BytesIO()
                    page.save(page_buffer, format="PNG", dpi=(300, 300))
                    st.download_button(
                        f"Download page {i+1} (PNG)",
                        data=page_buffer.getvalue(),
                        file_name=f"coloring-page-{i+1:02d}.png",
                        mime="image/png",
                        key=f"single_page_{i}"
                    )

with tab_export:
    st.subheader("Personalize and export your coloring book")
    pages = st.session_state.processed_pages
    if not pages:
        st.info("Create your 10 coloring pages first.")
    else:
        left, right = st.columns([1, 1])
        with left:
            book_title = st.text_input("Book title", value="My Special Coloring Book", max_chars=48)
            subtitle = st.text_input("Subtitle", value="A collection of favorite memories", max_chars=75)
            recipient = st.text_input("Made especially for", value="", max_chars=40,
                                      placeholder="e.g. Emma, Grandma, The Smith Family")
            cover_choice = st.selectbox(
                "Cover illustration",
                options=list(range(1, len(pages) + 1)),
                format_func=lambda x: f"Use coloring page {x}"
            )
            add_cover = st.checkbox("Include personalized cover in PDF and ZIP", value=True)
            include_notes = st.caption("The PDF contains the cover (if selected) followed by the 10 coloring pages.")
        with right:
            cover = create_cover(book_title, subtitle, recipient, pages[cover_choice - 1],
                                 A4_SIZE if paper == "A4" else LETTER_SIZE)
            st.image(cover, caption="Cover preview", use_container_width=True)

        if st.button("📚 Build my printable coloring book", type="primary", use_container_width=True):
            export_pages = []
            if add_cover:
                export_pages.append(cover)
            export_pages.extend(pages)
            pdf_buffer = build_pdf(export_pages, io.BytesIO(), paper)
            zip_buffer = build_zip(pages, st.session_state.source_names, cover if add_cover else None)
            st.session_state.pdf_bytes = pdf_buffer.getvalue()
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.export_name = safe_filename(book_title)
            st.success("Your coloring book is ready to download.")

        if st.session_state.get("pdf_bytes"):
            base = st.session_state.get("export_name", "coloring-book")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ Download print-ready PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"{base}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            with d2:
                st.download_button(
                    "⬇️ Download PNG pages ZIP",
                    data=st.session_state.zip_bytes,
                    file_name=f"{base}-png-pages.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            st.caption("Print tip: choose Actual Size / 100%, use white paper, and check that your printer is set to the selected paper size.")

st.divider()
st.markdown("""
<div class="muted">
<b>Quality note:</b> This local prototype uses computer vision to extract printable outlines.
It does not understand the scene like a human illustrator or an image-generation model. Busy backgrounds,
low light, motion blur, and overlapping subjects can produce imperfect results. Always review every page
before printing or selling the finished book.
</div>
""", unsafe_allow_html=True)
