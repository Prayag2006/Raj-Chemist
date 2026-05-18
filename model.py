import os
import cv2
import numpy as np
import pickle
import datetime
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

# Initialize OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---- Utility: extract face crop -> robust LBP spatial histogram vector ----
def crop_face_and_embed(bgr_image, bbox):
    (x, y, w, h) = bbox
    if w <= 0 or h <= 0:
        return None
    face = bgr_image[y:y+h, x:x+w]
    if face.size == 0:
        return None
    
    # 1. Convert to grayscale and resize to 64x64 for spatial grid analysis
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
    
    # 2. Apply Histogram Equalization to eliminate lighting/shadow variations
    face = cv2.equalizeHist(face)
    
    # 3. Compute Local Binary Pattern (LBP)
    h_f, w_f = face.shape
    lbp = np.zeros((h_f-2, w_f-2), dtype=np.uint8)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]
    center = face[1:h_f-1, 1:w_f-1].astype(np.int32)
    for index, (dy, dx) in enumerate(neighbors):
        neighbor = face[1+dy:h_f-1+dy, 1+dx:w_f-1+dx].astype(np.int32)
        lbp += ((neighbor >= center) * (1 << index)).astype(np.uint8)
        
    # Resize LBP back to 64x64 for clean 8x8 grids
    lbp_resized = cv2.resize(lbp, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    # 4. Extract LBP Histogram from 8x8 grids (16 bins per cell)
    grid_size = 8
    features = []
    for r in range(0, 64, grid_size):
        for c in range(0, 64, grid_size):
            cell = lbp_resized[r:r+grid_size, c:c+grid_size]
            hist, _ = np.histogram(cell, bins=16, range=(0, 256))
            features.append(hist)
            
    # 5. Concatenate and normalize histogram to ensure illumination independence
    emb = np.concatenate(features).astype(np.float32)
    norm = np.sum(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def extract_embedding_for_image(stream_or_bytes):
    # accepts a file-like stream (werkzeug FileStorage.stream)
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    
    # take largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    emb = crop_face_and_embed(img, largest_face)
    return emb

# ---- Load model helpers ----
_cached_clf = None

def load_model_if_exists(db=None):
    global _cached_clf
    if _cached_clf is not None:
        return _cached_clf
        
    # Try loading from local file
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _cached_clf = pickle.load(f)
            return _cached_clf
        except Exception:
            pass
            
    # Try loading from MongoDB
    if db is not None:
        try:
            model_doc = db.models.find_one({"_id": "latest_model"})
            if model_doc and "model_bytes" in model_doc:
                clf = pickle.loads(model_doc["model_bytes"])
                _cached_clf = clf
                # Also save to local disk for faster subsequent loads
                try:
                    with open(MODEL_PATH, "wb") as f:
                        f.write(model_doc["model_bytes"])
                except Exception:
                    pass
                return _cached_clf
        except Exception:
            pass
            
    return None

def predict_with_model(clf, emb):
    # returns label and confidence (max probability)
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf

# ---- Training function used in background ----
def train_model_background(dataset_dir, db=None, progress_callback=None):
    """
    dataset_dir/
        employee_id/
            img1.jpg
            img2.jpg
    """
    global _cached_clf

    # 1. Restore dataset files from MongoDB if available
    if db is not None:
        if progress_callback:
            progress_callback(5, "Restoring dataset from database...")
        try:
            cursor = db.face_images.find({})
            restored_count = 0
            for doc in cursor:
                eid = doc["employee_id"]
                fname = doc["filename"]
                img_data = doc["image_data"]
                
                emp_folder = os.path.join(dataset_dir, str(eid))
                os.makedirs(emp_folder, exist_ok=True)
                
                file_path = os.path.join(emp_folder, fname)
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(img_data)
                restored_count += 1
            if progress_callback and restored_count > 0:
                progress_callback(10, f"Restored {restored_count} images from database.")
        except Exception as e:
            if progress_callback:
                progress_callback(10, f"Restore warning: {str(e)}")

    X = []
    y = []
    employee_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_employees = max(1, len(employee_dirs))
    processed = 0

    for eid in employee_dirs:
        folder = os.path.join(dataset_dir, eid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                continue
            
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            emb = crop_face_and_embed(img, largest_face)
            if emb is None:
                continue
            X.append(emb)
            y.append(int(eid))
        processed += 1
        if progress_callback:
            pct = int((processed/total_employees)*80)
            progress_callback(pct, f"Processed {processed}/{total_employees} employees")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    # convert
    X = np.stack(X)
    y = np.array(y)

    # fit RandomForest
    if progress_callback:
        progress_callback(85, "Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X, y)

    # Save to local file
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
    except Exception:
        pass

    # Save to MongoDB
    if db is not None:
        if progress_callback:
            progress_callback(95, "Uploading trained model to database...")
        try:
            db.models.replace_one(
                {"_id": "latest_model"},
                {
                    "_id": "latest_model",
                    "model_bytes": pickle.dumps(clf),
                    "updated_at": datetime.datetime.utcnow().isoformat()
                },
                upsert=True
            )
        except Exception as e:
            if progress_callback:
                progress_callback(95, f"Warning: Could not save model to DB: {str(e)}")

    # Update cache
    _cached_clf = clf

    if progress_callback:
        progress_callback(100, "Training complete")
