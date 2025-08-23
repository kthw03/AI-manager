from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

DATABASE_URL = "https://hann-7b7be-default-rtdb.firebaseio.com"
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"

_initialized = False

def _ensure_init() -> bool:
    global _initialized
    if _initialized and firebase_admin._apps:
        return True
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
        _initialized = True
        print(f"[FB] init ok | db={DATABASE_URL}")
        return True
    except Exception as e:
        print(f"[FB][ERROR] init failed: {e}")
        return False

def upload_posture(device_id: str,
                   label: str,
                   view: str,
                   bbox=None,
                   ts: str | None = None,
                   anomalies: dict | None = None,
                   state: str | None = None) -> bool:
    if not _ensure_init():
        return False
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    payload = {
        "label": label,
        "view": view,
        "bbox": ({
            "x1": int(bbox[0]), "y1": int(bbox[1]),
            "x2": int(bbox[2]), "y2": int(bbox[3])
        } if bbox else None),
        "ts": ts,
        "state": state,
        "anomalies": (anomalies or {
            "falling_warning": False,
            "falling_detect": False,
            "patient_escape": False,
            "standing_freeze": False
        })
    }
    try:
        db.reference(f"devices/{device_id}/events").push(payload)
        db.reference(f"devices/{device_id}/posture").set(payload)
        print(f"[FB] RTDB write ok @ {ts}")
        return True
    except Exception as e:
        print(f"[FB][ERROR] RTDB write failed: {e} | db={DATABASE_URL}")
        return False
