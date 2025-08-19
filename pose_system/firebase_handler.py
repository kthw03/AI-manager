# firebase_handler.py
import cv2
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db, storage

DATABASE_URL = "https://hann-7b7be-default-rtdb.firebaseio.com/"
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
STORAGE_BUCKET = "hann-7b7be.appspot.com"

def _ensure_init():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        app = firebase_admin.initialize_app(cred, {
            "databaseURL": DATABASE_URL,
            "storageBucket": STORAGE_BUCKET
        })
        try:
            bucket = storage.bucket()
            print(f"[FB] init ok | project={app.project_id} bucket={bucket.name} db={DATABASE_URL}")
        except Exception as e:
            print(f"[FB] init check failed: {e}")

def upload_posture(device_id: str, label: str, view: str, bbox=None, ts: str | None = None):
    _ensure_init()
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    payload = {
        "label": label,
        "view": view,
        "bbox": {
            "x1": int(bbox[0]), "y1": int(bbox[1]),
            "x2": int(bbox[2]), "y2": int(bbox[3])
        } if bbox else None,
        "ts": ts
    }
    db.reference(f"devices/{device_id}/events").push(payload)
    db.reference(f"devices/{device_id}/posture").set(payload)

def upload_frame(device_id: str, frame_bgr, jpeg_quality: int = 85) -> str:
    _ensure_init()
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    now = datetime.now()
    date_folder = now.strftime("%Y%m%d")
    fname = now.strftime("%H%M%S") + ".jpg"
    path = f"frames/{device_id}/{date_folder}/{fname}"

    bucket = storage.bucket()
    blob = bucket.blob(path)
    blob.upload_from_string(buf.tobytes(), content_type="image/jpeg")

    try:
        blob.make_public()
        url = blob.public_url
    except Exception as e:
        print(f"[FB][WARN] make_public failed: {e}")
        url = blob.generate_signed_url(expiration=3600)

    db.reference(f"devices/{device_id}/last_image").set({
        "url": url,
        "path": path,
        "ts": now.isoformat(timespec="seconds")
    })
    return url
