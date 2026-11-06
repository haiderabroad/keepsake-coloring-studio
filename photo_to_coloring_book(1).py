"""
Photo to Coloring Book Studio
-----------------------------
A local Streamlit app that converts 1–10 photos into clean black-and-white
coloring pages. It includes a one-photo preview before batch processing.

Install:
    pip install streamlit opencv-python-headless pillow numpy

Run:
    streamlit run photo_to_coloring_book.py
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# --------------------------- Image processing --------------------------- #

def load_rgb(uploaded_file) -> np.ndarray:
    """Read an upload safely, correcting phone-camera orientation."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image)


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove isolated specks while retaining connected outline detail."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary == 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.full(binary.shape, 255, dtype=np.uint8)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 0
    return cleaned


def make_coloring_page(
    rgb: np.ndarray,
    detail: int = 65,
    line_strength: int = 50,
    smoothness: int = 60,
    output_width: int = 2400,
    remove_specks: bool = True,
) -> Image.Image:
    """
    Multi-stage photo-to-line-art conversion.

    detail:       0–100, higher keeps more small features and texture.
    line_strength: 0–100, higher makes contours darker/thicker.
    smoothness:   0–100, higher suppresses photo texture.
    output_width: target output width in pixels; aspect ratio is preserved.
    """
    if rgb is None or rgb.size == 0:
        raise ValueError("The uploaded image is empty.")

    h, w = rgb.shape[:2]
    if min(h, w) < 80:
        raise ValueError("Image is too small. Please use an image at least 80 px wide and tall.")

    # Upscale small inputs before extracting outlines.
    scale = min(2.0, max(1.0, 1200.0 / max(h, w)))
    if scale > 1:
        rgb = cv2.resize(
            rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Edge-preserving cartoon smoothing removes photographic texture while
    # keeping major shapes. OpenCV's stylizer is optional; fallback is bilateral.
    smooth_amount = smoothness / 100.0
    sigma_s = int(20 + smooth_amount * 70)
    sigma_r = 0.20 + smooth_amount * 0.35

    try:
        stylized = cv2.stylization(
            bgr, sigma_s=sigma_s, sigma_r=float(sigma_r)
        )
    except Exception:
        stylized = cv2.bilateralFilter(
            bgr, d=9, sigmaColor=45 + int(100 * smooth_amount),
            sigmaSpace=45 + int(100 * smooth_amount)
        )

    # Additional bilateral passes reduce skin, fabric and foliage noise.
    passes = 1 if smoothness < 35 else (2 if smoothness < 75 else 3)
    filtered = stylized
    for _ in range(passes):
        filtered = cv2.bilateralFilter(
            filtered, d=7,
            sigmaColor=35 + int(75 * smooth_amount),
            sigmaSpace=35 + int(75 * smooth_amount)
        )

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold creates fine contours across uneven lighting.
    block = int(51 - detail * 0.30)
    block = max(19, min(51, block))
    if block % 2 == 0:
        block += 1

    # More detail means a slightly more sensitive threshold.
    c_value = int(5 + (100 - detail) * 0.10)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, c_value
    )

    # Canny adds clear structural boundaries that adaptive threshold can miss.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    low = int(35 + (100 - detail) * 0.45)
    high = int(low * 2.7)
    edges = cv2.Canny(blur, low, high)

    # Make the structural edges readable without making every line heavy.
    edge_kernel_size = 1 if line_strength < 35 else 2
    edge_kernel = np.ones((edge_kernel_size, edge_kernel_size), np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    # Adaptive outlines + structural edges.
    combined = cv2.bitwise_and(adaptive, cv2.bitwise_not(edges))

    # Set line thickness with a restrained morphology operation.
    if line_strength >= 68:
        combined = cv2.erode(
            combined, np.ones((2, 2), np.uint8), iterations=1
        )
    elif line_strength <= 25:
        combined = cv2.dilate(
            combined, np.ones((2, 2), np.uint8), iterations=1
        )

    # Remove isolated dots and close tiny breaks in outlines.
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
    )
    if remove_specks:
        min_area = 3 + int((100 - detail) * 0.10)
        combined = remove_small_components(combined, min_area)

    # Keep the page pure white and the line art pure black.
    _, final = cv2.threshold(combined, 200, 255, cv2.THRESH_BINARY)

    # Print-ready dimensions, preserving the original aspect ratio.
    final_h, final_w = final.shape
    if output_width and final_w != output_width:
        target_h = int(final_h * (output_width / final_w))
        final = cv2.resize(
            final, (output_width, target_h), interpolation=cv2.INTER_CUBIC
        )
        _, final = cv2.threshold(final, 200, 255, cv2.THRESH_BINARY)

    return Image.fromarray(final, mode="L")


def safe_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (stem or "coloring_page") + "_coloring_page.png"


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", dpi=(300, 300), optimize=True)
    return buffer.getvalue()


# ------------------------------- User interface ------------------------------- #

st.set_page_config(
    page_title="Photo to Coloring Book Studio",
    page_icon="✏️",
    layout="wide",
)

st.title("Photo to Coloring Book Studio")
st.write(
    "Turn photos into printable black-outline coloring pages. "
    "Upload one photo for a preview, or upload up to 10 photos for batch processing."
)

with st.sidebar:
    st.header("Coloring page settings")
    detail = st.slider(
        "Detail preservation", 0, 100, 65,
        help="Higher values retain more small features. Lower values simplify busy photos."
    )
    line_strength = st.slider(
        "Outline strength", 0, 100, 50,
        help="Higher values produce stronger outlines."
    )
    smoothness = st.slider(
        "Photo smoothing", 0, 100, 60,
        help="Increase this to reduce skin, fabric, foliage and camera texture."
    )
    output_width = st.selectbox(
        "Output width",
        [1800, 2400, 3000, 3600],
        index=2,
        help="The output height is calculated automatically to preserve the photo's proportions."
    )
    remove_specks = st.checkbox("Remove isolated dots", value=True)

uploads = st.file_uploader(
    "Upload photos (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Maximum 10 photos per batch."
)

if uploads and len(uploads) > 10:
    st.error("Maximum limit is 10 photos. Please remove the extra files.")
    st.stop()

if not uploads:
    st.info("Start by uploading 1 photo. You can add up to 9 more for batch processing.")
    st.stop()

# Read and cache the uploaded photos for this session.
photos = []
for file in uploads:
    try:
        photos.append((file.name, load_rgb(file)))
    except Exception as exc:
        st.warning(f"Could not read {file.name}: {exc}")

if not photos:
    st.error("No valid images were found.")
    st.stop()

st.caption(f"{len(photos)} photo(s) uploaded. Maximum allowed: 10.")

tab_preview, tab_batch = st.tabs(["Single photo preview", "Process all photos"])

with tab_preview:
    names = [item[0] for item in photos]
    selected_name = st.selectbox("Choose a photo to preview", names)
    selected_index = names.index(selected_name)
    original = photos[selected_index][1]

    left, right = st.columns(2)
    with left:
        st.subheader("Original photo")
        st.image(original, use_container_width=True)

    if st.button("Generate coloring-page preview", type="primary"):
        try:
            preview = make_coloring_page(
                original, detail, line_strength, smoothness,
                output_width, remove_specks
            )
            st.session_state["preview_image"] = preview
            st.session_state["preview_name"] = selected_name
        except Exception as exc:
            st.error(f"Preview failed: {exc}")

    if (
        "preview_image" in st.session_state
        and st.session_state.get("preview_name") == selected_name
    ):
        with right:
            st.subheader("Coloring-page preview")
            st.image(st.session_state["preview_image"], use_container_width=True)
        st.download_button(
            "Download this preview (PNG)",
            data=image_to_png_bytes(st.session_state["preview_image"]),
            file_name=safe_filename(selected_name),
            mime="image/png",
        )
        st.success(
            "If the preview looks good, open the 'Process all photos' tab "
            "to continue with the full batch."
        )
    else:
        with right:
            st.subheader("Coloring-page preview")
            st.info("Generate a preview to see the result here.")

with tab_batch:
    st.subheader("Batch processing")
    st.write(
        f"Ready to process {len(photos)} photo(s). "
        "The same settings from the sidebar will be applied to every photo."
    )
    confirm = st.checkbox(
        "I have reviewed the preview or I am ready to process all uploaded photos.",
        value=False,
    )

    if st.button(
        f"Process all {len(photos)} photo(s)",
        type="primary",
        disabled=not confirm,
    ):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for index, (name, rgb) in enumerate(photos, start=1):
            status.write(f"Processing {index}/{len(photos)}: {name}")
            try:
                result = make_coloring_page(
                    rgb, detail, line_strength, smoothness,
                    output_width, remove_specks
                )
                results.append((safe_filename(name), image_to_png_bytes(result), result))
            except Exception as exc:
                st.warning(f"Skipped {name}: {exc}")
            progress.progress(index / len(photos))

        st.session_state["batch_results"] = results
        status.write(f"Finished. {len(results)} image(s) processed successfully.")

    results = st.session_state.get("batch_results", [])
    if results:
        st.subheader("Processed coloring pages")
        for filename, png_data, result in results:
            with st.expander(filename, expanded=False):
                st.image(result, use_container_width=True)
                st.download_button(
                    f"Download {filename}",
                    data=png_data,
                    file_name=filename,
                    mime="image/png",
                    key=f"download_{filename}",
                )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, png_data, _ in results:
                archive.writestr(filename, png_data)

        st.download_button(
            "Download all coloring pages (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="coloring_book_pages.zip",
            mime="application/zip",
            type="primary",
        )

st.divider()
st.caption(
    "Tip: Clear, well-lit photos with visible faces and uncluttered backgrounds "
    "usually produce the best results. This is a computer-vision converter, "
    "not a generative illustration model; very complex photos may need manual cleanup."
)
