# Keepsake Coloring Studio

A local-first Streamlit app that turns 10 uploaded photographs into printable coloring pages and exports a personalized PDF book plus PNG files.

## Requirements
- Python 3.10 or newer
- pip

## Run on Windows
1. Extract this ZIP file.
2. Open the extracted folder in Command Prompt or PowerShell.
3. Create a virtual environment (recommended):
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the application:
   ```powershell
   streamlit run app.py
   ```
6. Streamlit will open the app in your browser. If it does not, visit `http://localhost:8501`.

## Run on macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Upload exactly 10 JPG, JPEG, PNG, or WEBP images.
- Each image must be at least 250 × 250 pixels and no larger than 20 MB.
- The app uses OpenCV image processing, not generative AI.
- When run locally, photos are processed on your computer. If hosted online, uploaded files pass through the hosting server.
