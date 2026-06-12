import os
import cv2
import numpy as np
import pickle
import datetime
from sklearn.ensemble import RandomForestClassifier

import os as _os
# Vercel has a read-only filesystem - use /tmp for writable files
if _os.environ.get("VERCEL"):
    MODEL_PATH = "/tmp/model.pkl"
else:
    MODEL_PATH = "model.pkl"

# Initialize OpenCV Haar Cascades for face detection (frontal and profile/side-view fallbacks)
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def detect_faces_robust(gray, min_size_val):
    """Try many cascade + parameter combos to maximize detection rate."""
    configs = [
        (face_cascade_default, 1.3, 4),
        (face_cascade_default, 1.2, 3),
        (face_cascade_default, 1.15, 3),
        (face_cascade_default, 1.1,  3),
        (face_cascade_alt,     1.3, 4),
        (face_cascade_alt,     1.2, 3),
        (face_cascade_alt,     1.15, 3),
        (face_cascade_profile, 1.3, 3),
        (face_cascade_profile, 1.2, 3),
        (face_cascade_default, 1.3, 2),   # more lenient
        (face_cascade_alt,     1.3, 2),
    ]
    for cascade, sf, mn in configs:
        faces = cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn,
                                         minSize=(min_size_val, min_size_val))
        if len(faces) > 0:
            return faces
    return []


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
    h, w = img.shape[:2]
    # Try multiple detection widths to maximize success
    for target_w in [240, 180, 320, 160]:
        scale = float(target_w) / w if w > 0 else 1.0
        detect_img = cv2.resize(img, (target_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        min_size = max(5, int(30 * scale))
        faces = detect_faces_robust(gray, min_size)
        if len(faces) > 0:
            largest = max(faces, key=lambda r: r[2]*r[3])
            (x, y, fw, fh) = largest
            bbox = (int(x/scale), int(y/scale), int(fw/scale), int(fh/scale))
            emb = crop_face_and_embed(img, bbox)
            if emb is not None:
                return emb
    return None

# ---- Load model helpers ----
_cached_clf = None
_cached_model_updated_at = None
_last_db_check_time = 0.0

def load_model_if_exists(db=None):
    global _cached_clf, _cached_model_updated_at, _last_db_check_time
    import time
    
    # If DB is not available, check local cache or file
    if db is None:
        if _cached_clf is not None:
            return _cached_clf
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    _cached_clf = pickle.load(f)
                return _cached_clf
            except Exception:
                pass
        return None
        
    current_time = time.time()
    if _cached_clf is not None and (current_time - _last_db_check_time < 10.0):
        return _cached_clf
        
    try:
        _last_db_check_time = current_time
        # Check database for latest_model metadata first (to check updated_at)
        model_meta = db.models.find_one({"_id": "latest_model"}, {"updated_at": 1})
        if model_meta:
            db_updated_at = model_meta.get("updated_at")
            # If we already have the latest version cached, return it directly
            if _cached_clf is not None and db_updated_at == _cached_model_updated_at:
                return _cached_clf
            
            # Fetch the actual bytes and load the model
            model_doc = db.models.find_one({"_id": "latest_model"})
            if model_doc and "model_bytes" in model_doc:
                clf = pickle.loads(model_doc["model_bytes"])
                _cached_clf = clf
                _cached_model_updated_at = db_updated_at
                
                # Also save to local disk for faster subsequent loads
                try:
                    with open(MODEL_PATH, "wb") as f:
                        f.write(model_doc["model_bytes"])
                except Exception:
                    pass
                return _cached_clf
        else:
            # No model in DB, check local file
            if _cached_clf is not None:
                return _cached_clf
            if os.path.exists(MODEL_PATH):
                try:
                    with open(MODEL_PATH, "rb") as f:
                        _cached_clf = pickle.load(f)
                    return _cached_clf
                except Exception:
                    pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error checking/loading model from DB: {e}")
        if _cached_clf is not None:
            return _cached_clf
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    _cached_clf = pickle.load(f)
                return _cached_clf
            except Exception:
                pass
            
    return None

def predict_with_model(clf, emb):
    # Use Random Forest first to predict the identity robustly
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba([emb])[0]
        idx = np.argmax(proba)
        rf_label = clf.classes_[idx]
        rf_conf = proba[idx]
        
        # Calculate Cosine Similarity to the predicted class centroid
        if hasattr(clf, 'centroids_') and clf.centroids_ and rf_label in clf.centroids_:
            centroid = clf.centroids_[rf_label]
            dot = np.dot(emb, centroid)
            norm_a = np.linalg.norm(emb)
            norm_b = np.linalg.norm(centroid)
            cossim = float(dot / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
            
            # If Random Forest is confident, return the identity and the cosine similarity
            if rf_conf >= 0.80:
                return rf_label, cossim
            else:
                # If RF is not confident, return similarity 0.0 to reject it
                return rf_label, 0.0
                
    # Fallback to pure Centroid similarity if predict_proba is not available
    if hasattr(clf, 'centroids_') and clf.centroids_:
        best_label = None
        best_conf = -1.0
        for label, centroid in clf.centroids_.items():
            dot = np.dot(emb, centroid)
            norm_a = np.linalg.norm(emb)
            norm_b = np.linalg.norm(centroid)
            sim = float(dot / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
            if sim > best_conf:
                best_conf = sim
                best_label = label
        return best_label, best_conf
        
    return None, 0.0

# ---- Training function used in background ----
def train_model_background(dataset_dir, db=None, progress_callback=None):
    global _cached_clf

    try:
        X = []
        y = []

        if db is not None:
            if progress_callback:
                progress_callback(5, "Fetching face embeddings from database...")
            
            # Fetch all face images metadata and embeddings from MongoDB
            cursor = db.face_images.find({}, {"employee_id": 1, "filename": 1, "embedding": 1})
            docs = list(cursor)
            
            total_docs = len(docs)
            processed_docs = 0
            
            for doc in docs:
                eid = doc.get("employee_id")
                emb_list = doc.get("embedding")
                
                if not eid:
                    continue
                
                if emb_list is not None:
                    # Pre-computed embedding exists! Use it directly.
                    X.append(np.array(emb_list, dtype=np.float32))
                    y.append(int(eid))
                else:
                    # Pre-computed embedding is missing. Fetch full document with image_data
                    full_doc = db.face_images.find_one({"_id": doc["_id"]}, {"image_data": 1, "filename": 1})
                    if full_doc and full_doc.get("image_data"):
                        import io
                        # Extract embedding from image bytes
                        img_bytes = full_doc["image_data"]
                        emb = extract_embedding_for_image(io.BytesIO(img_bytes))
                        if emb is not None:
                            # Save back to MongoDB so we don't have to compute it next time
                            try:
                                db.face_images.update_one(
                                    {"_id": doc["_id"]},
                                    {"$set": {"embedding": emb.tolist()}}
                                )
                            except Exception:
                                pass
                            X.append(emb)
                            y.append(int(eid))
                
                processed_docs += 1
                if progress_callback and total_docs > 0:
                    pct = int((processed_docs / total_docs) * 80)
                    progress_callback(pct, f"Processed {processed_docs}/{total_docs} face embeddings")
        
        else:
            # Fallback for offline/local mode when db is not provided
            if progress_callback:
                progress_callback(5, "Scanning local dataset directory...")
            
            if not os.path.exists(dataset_dir):
                if progress_callback:
                    progress_callback(0, "No local training data found")
                return

            employee_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()]
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
                    
                    # Speed up face detection by downscaling the detection image
                    h, w = img.shape[:2]
                    scale = 1.0
                    if w > 240:
                        scale = 240.0 / w
                        detect_img = cv2.resize(img, (240, int(h * scale)), interpolation=cv2.INTER_AREA)
                    else:
                        detect_img = img
                        
                    gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
                    min_size_val = int(30 * scale)
                    faces = detect_faces_robust(gray, min_size_val)
                    if len(faces) == 0:
                        continue
                    
                    # Map the largest face coordinates back to original scale
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    (x, fy, fw, fh) = largest_face
                    bbox = (int(x / scale), int(fy / scale), int(fw / scale), int(fh / scale))
                    
                    emb = crop_face_and_embed(img, bbox)
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
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        clf.fit(X, y)

        # Calculate and store centroids for each class to compute absolute similarity later
        centroids = {}
        for c in np.unique(y):
            centroids[int(c)] = np.mean(X[y == c], axis=0)
        clf.centroids_ = centroids

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

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"CRITICAL ERROR in train_model_background:\n{tb}")
        if progress_callback:
            progress_callback(0, f"Error during training: {str(e)}")
