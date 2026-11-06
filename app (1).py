"""
Keepsake Coloring Studio — AI Edition
-------------------------------------
AI image-to-image coloring-page generator using the OpenAI Images API.

Install:
    pip install -r requirements.txt

Run:
    streamlit run app.py

Secrets:
    Set OPENAI_API_KEY in your environment, or add it to
    .streamlit/secrets.toml as:
        OPENAI_API_KEY = "your-api-key"

Each preview or newly processed image makes a billable Images API request.
A preview is reused during batch processing when its source and settings match.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance


# ----------------------------- App configuration ----------------------------- #

st.set_page_config(
    page_title="Keepsake Coloring Studio",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://api.openai.com/v1/images/edits"
DEFAULT_MODEL = "gpt-image-2.5-sunburst"
MAX_UPLOADS = 10
MAX_FILE_MB = 25
REQUEST_TIMEOUT_SECONDS = 300


# ----------------------------- Styling ----------------------------- #

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f7f8fc; }
      [data-testid="stSidebar"] { background: #111827; }
      [data-testid="stSidebar"] * { color: #f8fafc !important; }
      [data-testid="stSidebar"] input { color: #111827 !important; }
      .hero {
        padding: 28px 30px; border-radius: 22px; color: white;
        background: linear-gradient(120deg, #18233d 0%, #344b77 58%, #6d7fb0 100%);
        margin-bottom: 22px; box-shadow: 0 12px 34px rgba(28,43,75,.16);
      }
      .hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.15rem; font-weight: 800; margin: 0 0 8px 0; color: white; }
      .hero p { color: #e5eaf5; font-size: 1rem; margin: 0; max-width: 780px; }
      .step-card {
        background: white; border: 1px solid #e6eaf2; border-radius: 16px;
        padding: 15px 17px; min-height: 90px; box-shadow: 0 4px 15px rgba(25,39,66,.04);
      }
      .step-number { color: #6679b7; font-size: .78rem; font-weight: 800; letter-spacing: .08em; }
      .step-title { color: #172033; font-size: 1rem; font-weight: 800; margin-top: 5px; }
      .step-copy { color: #667085; font-size: .84rem; margin-top: 3px; }
      .small-note { color: #667085; font-size: .83rem; }
      div.stButton > button[kind="primary"] {
        border: 0; border-radius: 12px; font-weight: 800;
        background: linear-gradient(90deg, #445d9d, #687fbd);
        min-height: 3rem;
      }
      div.stDownloadButton > button { border-radius: 10px; font-weight: 700; }
      [data-testid="stFileUploader"] {
        border: 1.5px dashed #aab6d0; border-radius: 15px; padding: 8px;
        background: #ffffff;
      }
      .section-heading { color: #172033; font-family: 'Manrope', sans-serif; font-weight: 800; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Utility functions ----------------------------- #

def get_secret_api_key() -> str:
    """Read the key from Streamlit secrets or the process environment."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (stem or "coloring_page") + "_coloring_page.png"


def image_mime_from_name(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".webp":
        return "image/webp"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def prepare_input_image(raw_bytes: bytes, max_edge: int = 2048) -> tuple[bytes, str]:
    """
    Apply EXIF orientation, flatten transparency onto white, and resize only
    when the uploaded photo is larger than max_edge. JPEG keeps API uploads light.
    """
    with Image.open(io.BytesIO(raw_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    if max(image.size) > max_edge:
        ratio = max_edge / max(image.size)
        new_size = (max(1, round(image.width * ratio)),
                    max(1, round(image.height * ratio)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=94, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def make_prompt(
    detail_level: str,
    background_level: str,
    line_style: str,
    extra_instruction: str,
) -> str:
    detail_map = {
        "Simple": (
            "Use simplified, uncluttered interior details and larger open coloring areas. "
            "Keep only the most important facial, clothing, and object details."
        ),
        "Balanced": (
            "Use a balanced amount of detail: clear facial features, hair shapes, clothing seams, "
            "a few natural folds, and recognizable important background objects."
        ),
        "Detailed": (
            "Keep more meaningful small details in hair, clothing patterns, accessories, and "
            "important background objects, but do not create photographic texture or visual clutter."
        ),
    }
    background_map = {
        "Minimal": "Simplify the background strongly into a few clean, relevant outline shapes.",
        "Balanced": "Retain recognizable background elements but simplify them into clean, sparse line art.",
        "Keep more": "Retain more of the original background layout while converting it into clean, readable outlines.",
    }
    line_map = {
        "Fine": "Use fine, elegant, smooth black outlines with clear separation between nearby lines.",
        "Balanced": "Use medium-weight, smooth, consistent black outer contours and slightly finer interior details.",
        "Bold": "Use slightly bolder, confident outer contours with lighter, thinner interior details.",
    }

    prompt = f"""
Transform the supplied photograph into ONE polished, professional, printable
black-and-white coloring-book illustration. The photograph is the strict visual
reference. This must look like carefully hand-inked children's coloring-book
line art, NOT a photo filter, edge map, pencil sketch, charcoal drawing, or
photorealistic grayscale conversion.

SUBJECT AND COMPOSITION PRESERVATION:
- Preserve the exact number of people, pets, and important objects.
- Keep each person recognizably faithful to the source: preserve their individual
  facial structure, expression, hairstyle, hairline, age appearance, and distinguishing
  visible features as closely as possible.
- Preserve pose, body proportions, hand placement, clothing, accessories, relative
  positions, camera angle, and overall framing.
- Do not merge faces, duplicate people, invent extra fingers, remove important objects,
  or add decorative objects that are not in the photo.
- Keep the full subjects inside the image. Do not crop heads, hands, feet, or important objects.

COLORING-BOOK LINE ART:
- Redraw shapes as clean, continuous, smooth black ink outlines.
- Use readable outer contours and controlled interior lines. Avoid overlapping,
  tangled, broken, scratchy, or excessively dense strokes.
- Draw eyes, eyebrows, noses, mouths, ears, and smiles clearly but naturally.
  Keep expressions recognizable; do not exaggerate facial features.
- Group hair into flowing, well-defined locks and strands instead of noisy texture.
- Convert clothing into clean contours with a few meaningful seams, folds, buttons,
  and patterns. Leave large areas open for coloring.
- Keep outlines separated so a child can color individual regions easily.
- {line_map[line_style]}
- {detail_map[detail_level]}
- BACKGROUND: {background_map[background_level]}

STRICT PRINT REQUIREMENTS:
- Pure white paper background.
- Black outlines only. No color.
- No gray shading, gradients, shadows, hatching, cross-hatching, stippling,
  halftone, photographic texture, or large filled black regions.
- Avoid heavy black patches in hair, clothing, faces, and background. Use outline
  shapes and open white areas instead.
- No title, letters, captions, labels, watermark, page mockup, book, phone,
  frame, border, or extra layout elements.
- Output only the finished coloring-page artwork, filling the canvas naturally.
- Preserve the original photo's orientation and approximate aspect ratio.

The final result should resemble a premium family-photo coloring-book page:
beautiful, clean, friendly, recognizable, uncluttered, and genuinely enjoyable to color.
""".strip()

    if extra_instruction.strip():
        prompt += "\n\nADDITIONAL USER DIRECTION (follow only if consistent with preserving the photo):\n"
        prompt += extra_instruction.strip()[:700]

    return prompt


def make_signature(
    raw_bytes: bytes,
    model: str,
    quality: str,
    detail_level: str,
    background_level: str,
    line_style: str,
    extra_instruction: str,
    output_width: int,
    threshold: int,
) -> str:
    h = hashlib.sha256()
    h.update(raw_bytes)
    settings = "|".join([
        model, quality, detail_level, background_level, line_style,
        extra_instruction.strip(), str(output_width), str(threshold)
    ])
    h.update(settings.encode("utf-8"))
    return h.hexdigest()


def postprocess_generated_image(
    image_bytes: bytes,
    output_width: int,
    threshold: int,
) -> bytes:
    """
    Clean residual gray shading into white while retaining ink lines, then
    upscale proportionally and write 300-DPI PNG metadata.
    """
    with Image.open(io.BytesIO(image_bytes)) as generated:
        image = ImageOps.exif_transpose(generated).convert("RGB")

    gray = ImageOps.grayscale(image)
    # Mild contrast helps distinguish ink from paper before binarization.
    gray = ImageEnhance.Contrast(gray).enhance(1.18)
    pixels = gray.load()
    width, height = gray.size

    # Threshold removes gray wash/shading. The slider lets the user preserve
    # lighter fine strokes by lowering the threshold.
    import numpy as np
    array = np.asarray(gray, dtype=np.uint8)
    cleaned = np.where(array < threshold, 0, 255).astype(np.uint8)
    result = Image.fromarray(cleaned, mode="L")

    if output_width and result.width != output_width:
        target_height = max(1, round(result.height * (output_width / result.width)))
        result = result.resize(
            (output_width, target_height),
            Image.Resampling.LANCZOS,
        )
        # Reassert a clean white page after interpolation.
        arr = np.asarray(result, dtype=np.uint8)
        result = Image.fromarray(np.where(arr < 220, 0, 255).astype(np.uint8), mode="L")

    output = io.BytesIO()
    result.save(output, format="PNG", dpi=(300, 300), optimize=True)
    return output.getvalue()


def call_openai_image_edit(
    api_key: str,
    raw_bytes: bytes,
    filename: str,
    model: str,
    quality: str,
    prompt: str,
) -> bytes:
    """Send one source photo to the OpenAI Images edit endpoint."""
    prepared_bytes, mime_type = prepare_input_image(raw_bytes)
    upload_name = Path(filename).stem + ".jpg"

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "size": "auto",
        "output_format": "png",
        "n": "1",
    }
    files = {
        "image": (upload_name, prepared_bytes, mime_type),
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            "The image request timed out. It may have taken too long to finish. "
            "Check your API usage before trying again."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not connect to the image service: {exc}") from exc

    if not response.ok:
        try:
            body = response.json()
            error = body.get("error", {})
            message = error.get("message") or response.text[:1000]
            code = error.get("code")
        except Exception:
            message, code = response.text[:1000], None

        if response.status_code == 401:
            raise RuntimeError("The API key was rejected. Check that your OpenAI API key is correct.")
        if response.status_code == 429:
            raise RuntimeError(
                "The API rate limit or usage limit was reached. Check API billing, available credits, "
                "and rate limits, then try again."
            )
        if code == "moderation_blocked":
            raise RuntimeError(
                "This image or request was blocked by the image service's safety checks. "
                "Try a different image or remove the additional prompt."
            )
        raise RuntimeError(f"Image API error ({response.status_code}): {message}")

    try:
        payload = response.json()
        encoded = payload["data"][0]["b64_json"]
        return base64.b64decode(encoded)
    except Exception as exc:
        raise RuntimeError("The image service returned an unexpected response with no image data.") from exc


def generate_coloring_page(
    api_key: str,
    raw_bytes: bytes,
    filename: str,
    model: str,
    quality: str,
    detail_level: str,
    background_level: str,
    line_style: str,
    extra_instruction: str,
    output_width: int,
    threshold: int,
) -> bytes:
    prompt = make_prompt(
        detail_level, background_level, line_style, extra_instruction
    )
    generated = call_openai_image_edit(
        api_key=api_key,
        raw_bytes=raw_bytes,
        filename=filename,
        model=model,
        quality=quality,
        prompt=prompt,
    )
    return postprocess_generated_image(generated, output_width, threshold)


def get_upload_records(uploads: list[Any]) -> list[dict[str, Any]]:
    records = []
    for upload in uploads:
        raw = upload.getvalue()
        records.append({
            "name": upload.name,
            "bytes": raw,
            "digest": hashlib.sha256(raw).hexdigest(),
        })
    return records


# ----------------------------- Header ----------------------------- #

st.markdown(
    """
    <div class="hero">
      <div style="font-size:.78rem;font-weight:800;letter-spacing:.16em;color:#cbd6f2;margin-bottom:8px;">
        AI-POWERED PHOTO TRANSFORMATION
      </div>
      <h1>Turn your favorite photos into coloring-book art</h1>
      <p>Preserve the people and moments you love, then transform each photo into clean,
      printable black-outline artwork. Preview one image before processing a batch of up to ten.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

step_cols = st.columns(3)
steps = [
    ("01", "Upload photos", "Choose 1–10 JPG, PNG, or WEBP images."),
    ("02", "Preview the artwork", "Generate one page and review the result."),
    ("03", "Process and download", "Reuse the preview and download pages or a ZIP."),
]
for col, (number, title, copy) in zip(step_cols, steps):
    with col:
        st.markdown(
            f'<div class="step-card"><div class="step-number">STEP {number}</div>'
            f'<div class="step-title">{title}</div><div class="step-copy">{copy}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")


# ----------------------------- Sidebar settings ----------------------------- #

with st.sidebar:
    st.markdown("## Studio settings")
    st.caption("AI conversion controls")

    secret_key = get_secret_api_key()
    api_key = st.text_input(
        "OpenAI API key",
        value=secret_key,
        type="password",
        placeholder="sk-...",
        help="Use an API key from your OpenAI developer account. It is used only by this app session.",
    )

    st.divider()
    model = st.selectbox(
        "Image model",
        [DEFAULT_MODEL, "gpt-image-2.5-flare"],
        index=0,
        help="The selected model performs the image-to-image redraw.",
    )
    quality = st.select_slider(
        "AI generation quality",
        options=["low", "medium", "high", "xhigh", "max"],
        value="high",
        help="Higher quality can take longer and may cost more. Start with high for a preview.",
    )

    st.divider()
    detail_level = st.select_slider(
        "Detail level",
        options=["Simple", "Balanced", "Detailed"],
        value="Balanced",
        help="Controls how much meaningful detail the AI should keep.",
    )
    line_style = st.select_slider(
        "Outline style",
        options=["Fine", "Balanced", "Bold"],
        value="Balanced",
    )
    background_level = st.select_slider(
        "Background detail",
        options=["Minimal", "Balanced", "Keep more"],
        value="Balanced",
    )
    output_width = st.selectbox(
        "PNG output width",
        [1800, 2400, 3000, 3600],
        index=2,
        help="The image is resized proportionally. 300 DPI is embedded in the PNG metadata.",
    )
    threshold = st.slider(
        "Remove gray shading",
        min_value=150,
        max_value=235,
        value=205,
        step=5,
        help="Lower values remove more light gray shading but can also remove faint lines. If thin lines disappear, lower this value.",
    )
    extra_instruction = st.text_area(
        "Optional extra direction",
        placeholder="Example: Keep the garden background simple and make the children's sweater patterns easy to color.",
        height=90,
        max_chars=700,
    )

    st.divider()
    st.markdown(
        """
        <div style="font-size:.82rem;line-height:1.5;color:#dbe3f5;">
        <b>API usage notice</b><br>
        Each new preview or image conversion uses a billable image-edit API request.
        A matching preview is reused during batch processing to avoid generating it twice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------- Upload area ----------------------------- #

st.markdown('<h3 class="section-heading">1. Upload your photos</h3>', unsafe_allow_html=True)
uploads = st.file_uploader(
    "Drag and drop photos here, or browse your device",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help=f"Upload up to {MAX_UPLOADS} photos. Maximum file size: {MAX_FILE_MB} MB per photo.",
    key="ai_coloring_uploads",
)

if not uploads:
    st.info("Upload one photo to test the result. You can add up to nine more for batch processing.")
    st.stop()

if len(uploads) > MAX_UPLOADS:
    st.error(f"You uploaded {len(uploads)} photos. The maximum allowed is {MAX_UPLOADS}. Remove the extra files.")
    st.stop()

valid_uploads = []
for upload in uploads:
    if upload.size > MAX_FILE_MB * 1024 * 1024:
        st.warning(f"{upload.name} is larger than {MAX_FILE_MB} MB and was skipped.")
    else:
        valid_uploads.append(upload)

if not valid_uploads:
    st.error("No valid photos are available. Upload an image smaller than the file-size limit.")
    st.stop()

records = get_upload_records(valid_uploads)
st.caption(f"{len(records)} valid photo(s) ready. Maximum batch size: {MAX_UPLOADS}.")


# ----------------------------- Session cache ----------------------------- #

if "ai_coloring_cache" not in st.session_state:
    st.session_state.ai_coloring_cache = {}
if "ai_coloring_batch" not in st.session_state:
    st.session_state.ai_coloring_batch = []


def current_signature(record: dict[str, Any]) -> str:
    return make_signature(
        record["bytes"], model, quality, detail_level, background_level,
        line_style, extra_instruction, output_width, threshold
    )


# ----------------------------- Preview and batch tabs ----------------------------- #

preview_tab, batch_tab = st.tabs(["2. Single-photo preview", "3. Batch processing"])

with preview_tab:
    st.markdown('<h3 class="section-heading">Test one photo first</h3>', unsafe_allow_html=True)
    selected_name = st.selectbox(
        "Choose which photo to preview",
        options=[record["name"] for record in records],
        key="ai_coloring_selected_photo",
    )
    selected_record = next(record for record in records if record["name"] == selected_name)
    selected_signature = current_signature(selected_record)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Original photo**")
        st.image(selected_record["bytes"], use_container_width=True)

    with right:
        st.markdown("**AI coloring-page result**")
        cached_preview = st.session_state.ai_coloring_cache.get(selected_signature)
        if cached_preview:
            st.image(cached_preview, use_container_width=True)
            st.success("Preview ready. This result will be reused in batch processing if settings stay the same.")
        else:
            st.info("Generate a preview to inspect the AI illustration here.")

    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar before generating an image.")
    preview_clicked = st.button(
        "Generate AI coloring-page preview",
        type="primary",
        use_container_width=True,
        disabled=not bool(api_key),
        key="ai_coloring_preview_button",
    )

    if preview_clicked:
        try:
            with st.spinner("Creating your coloring-book illustration. This can take up to a couple of minutes..."):
                result_bytes = generate_coloring_page(
                    api_key=api_key,
                    raw_bytes=selected_record["bytes"],
                    filename=selected_record["name"],
                    model=model,
                    quality=quality,
                    detail_level=detail_level,
                    background_level=background_level,
                    line_style=line_style,
                    extra_instruction=extra_instruction,
                    output_width=output_width,
                    threshold=threshold,
                )
            st.session_state.ai_coloring_cache[selected_signature] = result_bytes
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    cached_preview = st.session_state.ai_coloring_cache.get(selected_signature)
    if cached_preview:
        st.download_button(
            "Download preview PNG",
            data=cached_preview,
            file_name=safe_filename(selected_record["name"]),
            mime="image/png",
            use_container_width=True,
            key="download_single_ai_preview",
        )

with batch_tab:
    st.markdown('<h3 class="section-heading">Convert up to 10 photos</h3>', unsafe_allow_html=True)
    st.write(
        "The app will use the same AI settings for every photo. Any preview that matches "
        "the current image and settings will be reused instead of charged again."
    )

    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar before batch processing.")

    st.markdown("**Batch queue**")
    for index, record in enumerate(records, start=1):
        signature = current_signature(record)
        status = "Preview cached — will be reused" if signature in st.session_state.ai_coloring_cache else "Ready to generate"
        st.write(f"{index}. **{record['name']}** · {status}")

    confirm_batch = st.checkbox(
        "I understand that each photo without a matching preview will use an API image-generation request.",
        value=False,
        key="ai_coloring_confirm_batch",
    )
    run_batch = st.button(
        f"Process all {len(records)} photo(s)",
        type="primary",
        use_container_width=True,
        disabled=(not bool(api_key) or not confirm_batch),
        key="ai_coloring_batch_button",
    )

    if run_batch:
        completed = []
        progress = st.progress(0)
        status_box = st.empty()
        errors = []

        for index, record in enumerate(records, start=1):
            signature = current_signature(record)
            status_box.info(f"Processing {index}/{len(records)}: {record['name']}")
            try:
                cached = st.session_state.ai_coloring_cache.get(signature)
                if cached:
                    result_bytes = cached
                else:
                    result_bytes = generate_coloring_page(
                        api_key=api_key,
                        raw_bytes=record["bytes"],
                        filename=record["name"],
                        model=model,
                        quality=quality,
                        detail_level=detail_level,
                        background_level=background_level,
                        line_style=line_style,
                        extra_instruction=extra_instruction,
                        output_width=output_width,
                        threshold=threshold,
                    )
                    st.session_state.ai_coloring_cache[signature] = result_bytes

                completed.append({
                    "name": record["name"],
                    "filename": safe_filename(record["name"]),
                    "bytes": result_bytes,
                })
            except Exception as exc:
                errors.append((record["name"], str(exc)))
            progress.progress(index / len(records))

        st.session_state.ai_coloring_batch = completed
        if errors:
            for name, message in errors:
                st.error(f"{name}: {message}")
        if completed:
            status_box.success(f"Finished: {len(completed)} of {len(records)} photo(s) converted.")
        else:
            status_box.error("No photos were converted successfully.")

    batch_results = st.session_state.ai_coloring_batch
    if batch_results:
        st.divider()
        st.markdown("**Completed coloring pages**")
        for index, result in enumerate(batch_results):
            with st.expander(result["filename"], expanded=(index == 0)):
                st.image(result["bytes"], use_container_width=True)
                st.download_button(
                    f"Download {result['filename']}",
                    data=result["bytes"],
                    file_name=result["filename"],
                    mime="image/png",
                    key=f"ai_batch_download_{index}_{hashlib.md5(result['bytes']).hexdigest()[:8]}",
                )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for result in batch_results:
                archive.writestr(result["filename"], result["bytes"])

        st.download_button(
            "Download all completed pages as ZIP",
            data=zip_buffer.getvalue(),
            file_name="keepsake_coloring_pages.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )


# ----------------------------- Footer ----------------------------- #

st.divider()
st.markdown(
    """
    <div class="small-note">
    <b>Quality note:</b> AI image editing is much better suited to this illustration task than
    traditional edge detection, but no model can guarantee a perfect likeness or flawless line
    art on every photograph. Always inspect each page before printing or selling it.
    </div>
    """,
    unsafe_allow_html=True,
)
