# Face Recognition System (Flask + ML)

Enroll faces from your webcam and recognize them live in the browser.

## How it works
1. Browser sends a webcam frame to the Flask server.
2. `face_recognition` (dlib) detects faces and converts each into a 128-number embedding.
3. The embedding is compared with enrolled embeddings (stored in SQLite).
   A distance <= `THRESHOLD` in `app.py` (0.5) counts as a match.

## Run it
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 and allow camera access.

## Install tips
- `dlib` compiles from source and needs CMake plus a C++ compiler
  (Windows: Visual Studio Build Tools "Desktop development with C++").
- Easier on any OS with conda: `conda install -c conda-forge dlib`, then
  `pip install face_recognition flask pillow numpy "setuptools<81"`.

## Files
- `app.py` - server, database, detection and matching
- `templates/index.html` - webcam UI and live overlay
- `faces.db` - created on first run

## Ideas to extend
Login by face, attendance logging with timestamps, liveness detection,
averaging several enrollment photos per person.

Faces are biometric data: get consent before enrolling anyone.
