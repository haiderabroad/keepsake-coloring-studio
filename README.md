# Keepsake Coloring Studio

A local Python web application that turns exactly 10 uploaded photographs into a personalized printable coloring book.

## Features

- Upload up to 10 JPG, JPEG, PNG or WEBP photographs (the workflow requires exactly 10).
- Automatic EXIF orientation correction and image validation.
- Three conversion styles: Balanced, Bold & Simple, and Detailed.
- Controls for outline weight, detail, smoothing and contrast.
- Preview all 10 generated pages before export.
- Download individual PNG pages.
- Personalized cover with title, subtitle, recipient name and selected cover illustration.
- Print-ready PDF with A4 or US Letter page sizing.
- ZIP containing 300-DPI PNG pages and printing notes.
- Local-first processing: images are processed in the Streamlit session and are not intentionally sent to an external image service.

## Requirements

- Python 3.10 or newer recommended.
- 4 GB RAM minimum; 8 GB recommended for large photographs.
- Internet is only needed to install dependencies and load the optional Google font in the interface. The image conversion itself runs locally.

## Install and run

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows (Command Prompt)

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app normally opens at http://localhost:8501.

## Workflow

1. Upload exactly 10 clear photographs.
2. Choose A4 or US Letter and a conversion style in the sidebar.
3. Adjust the outline and detail controls.
4. Click **Convert all 10 photos into coloring pages**.
5. Review all pages in the Design & preview tab.
6. Add the book title, subtitle and recipient name.
7. Choose which coloring page should appear on the cover.
8. Build and download the PDF and/or PNG ZIP.

## Image quality guidance

Best results generally come from:
- sharp, high-resolution photographs;
- bright, even lighting;
- a clear subject against a relatively simple background;
- faces and important objects that are not too small in the frame;
- minimal motion blur and compression artifacts.

The app uses OpenCV edge extraction and thresholding. It is not a semantic illustration model, so it may interpret complex textures, hair, foliage, patterned clothing or busy backgrounds as excessive lines. Review the pages before printing.

## Important production notes

This is a local prototype, not a hosted multi-customer SaaS product. Before exposing it publicly, add:
- authentication and per-user job isolation;
- upload quotas, rate limits and file scanning;
- background job processing and storage lifecycle controls;
- explicit consent and privacy/deletion controls for personal photos;
- CSRF protection and secure deployment configuration;
- monitoring, error reporting and automated tests;
- a licensed or self-hosted font instead of relying on a remote font;
- a proper image-generation/illustration provider if you need semantic, artist-like line art.

Do not promise that every source photograph will produce perfect outlines. Include a review/approval step and allow customers to request adjustments.
