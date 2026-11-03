"""Face Recognition Lab: Flask + face_recognition (dlib) + SQLite."""
import base64, io, json, sqlite3, time

import numpy as np
import face_recognition
from flask import Flask, jsonify, render_template, request
from PIL import Image

DB_PATH = "faces_v2.db"      # same database as the previous version, so enrollments carry over
DEFAULT_THRESHOLD = 0.5      # max distance that still counts as a match

app = Flask(__name__)


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, thumb TEXT);
        CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL, embedding TEXT NOT NULL);
        """)


def decode_image(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def thumbnail(img, loc):
    top, right, bottom, left = loc
    pad = int((bottom - top) * 0.25)
    crop = img[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad]
    buf = io.BytesIO()
    Image.fromarray(crop).resize((120, 120)).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def gallery():
    """One averaged embedding per person (mean of all their samples)."""
    with db() as c:
        rows = c.execute("SELECT p.id, p.name, s.embedding FROM people p "
                         "JOIN samples s ON s.person_id = p.id").fetchall()
    groups = {}
    for r in rows:
        g = groups.setdefault(r["id"], {"name": r["name"], "emb": []})
        g["emb"].append(json.loads(r["embedding"]))
    return ([g["name"] for g in groups.values()],
            [np.mean(g["emb"], axis=0) for g in groups.values()])


@app.route("/")
def index():
    return render_template("index.html", threshold=DEFAULT_THRESHOLD)


@app.route("/api/people")
def people():
    with db() as c:
        rows = c.execute("SELECT p.id, p.name, p.thumb, COUNT(s.id) AS samples FROM people p "
                         "LEFT JOIN samples s ON s.person_id = p.id GROUP BY p.id ORDER BY p.id").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/people/<int:pid>", methods=["DELETE"])
def delete_person(pid):
    with db() as c:
        c.execute("DELETE FROM samples WHERE person_id = ?", (pid,))
        c.execute("DELETE FROM people WHERE id = ?", (pid,))
    return jsonify(ok=True)


@app.route("/api/enroll", methods=["POST"])
def enroll():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    images = data.get("images", [])
    if not name:
        return jsonify(error="Enter a name first."), 400
    embeddings, thumb = [], None
    for url in images:
        img = decode_image(url)
        locs = face_recognition.face_locations(img)
        if len(locs) != 1:
            continue
        embeddings.append(face_recognition.face_encodings(img, locs)[0])
        thumb = thumb or thumbnail(img, locs[0])
    needed = 1 if data.get("source") == "upload" else 3
    if len(embeddings) < needed:
        return jsonify(error=f"Only {len(embeddings)} of {len(images)} images had exactly one clear face. "
                             "Use well-lit images where only this person is visible."), 400
    with db() as c:
        pid = c.execute("INSERT INTO people (name, thumb) VALUES (?, ?)", (name, thumb)).lastrowid
        c.executemany("INSERT INTO samples (person_id, embedding) VALUES (?, ?)",
                      [(pid, json.dumps(e.tolist())) for e in embeddings])
    return jsonify(ok=True, name=name, samples=len(embeddings))


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json(force=True)
    t0 = time.time()
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    img = decode_image(data["image"])
    locs = face_recognition.face_locations(img)
    encs = face_recognition.face_encodings(img, locs)
    marks = face_recognition.face_landmarks(img, locs) if data.get("landmarks") else [None] * len(locs)
    names, means = gallery()

    faces = []
    for (top, right, bottom, left), enc, lm in zip(locs, encs, marks):
        label, dist = "Unknown", None
        if means:
            d = face_recognition.face_distance(means, enc)
            best = int(np.argmin(d))
            dist = float(d[best])
            if dist <= threshold:
                label = names[best]
        similarity = None if dist is None else round(max(0.0, 1 - dist / (2 * threshold)) * 100)
        faces.append({"box": [left, top, right, bottom], "name": label,
                      "distance": None if dist is None else round(dist, 3),
                      "similarity": similarity, "landmarks": lm,
                      "embedding": [round(float(x), 2) for x in enc]})
    return jsonify(faces=faces, width=img.shape[1], height=img.shape[0],
                   ms=round((time.time() - t0) * 1000))


@app.route("/api/insights")
def insights():
    """2-D PCA map of every sample, plus distances between people."""
    with db() as c:
        rows = c.execute("SELECT p.name, s.embedding FROM samples s "
                         "JOIN people p ON p.id = s.person_id").fetchall()
    if not rows:
        return jsonify(points=[], names=[], matrix=[])
    X = np.array([json.loads(r["embedding"]) for r in rows])
    Xc = X - X.mean(axis=0)
    if len(X) > 2:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        xy = Xc @ vt[:2].T
    else:
        xy = np.zeros((len(X), 2))
    names, means = gallery()
    matrix = [[round(float(np.linalg.norm(a - b)), 2) for b in means] for a in means]
    points = [{"name": r["name"], "x": float(p[0]), "y": float(p[1])} for r, p in zip(rows, xy)]
    return jsonify(points=points, names=names, matrix=matrix)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
