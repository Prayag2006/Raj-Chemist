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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(min_size_val, min_size_val))
    if len(faces) == 0:
        return None
    
    # take largest face and map coordinates back
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    (x, y, fw, fh) = largest_face
    bbox = (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
    
    emb = crop_face_and_embed(img, bbox)
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
    # returns label and confidence (max probability or cosine similarity)
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    
    # If centroids are attached, compute Cosine Similarity to the class centroid as confidence
    if hasattr(clf, 'centroids_') and label in clf.centroids_:
        centroid = clf.centroids_[label]
        dot = np.dot(emb, centroid)
        norm_a = np.linalg.norm(emb)
        norm_b = np.linalg.norm(centroid)
        if norm_a > 0 and norm_b > 0:
            conf = float(dot / (norm_a * norm_b))
        else:
            conf = 0.0
    else:
        # Fallback to Random Forest vote probability if legacy model
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

    try:
        # 1. Restore dataset files from MongoDB if available (optimized batch-restore)
        if db is not None:
            if progress_callback:
                progress_callback(5, "Checking database for missing dataset files...")
            try:
                # Gather all existing local files to prevent duplicate downloads
                existing_files = set()
                if os.path.exists(dataset_dir):
                    for root, dirs, files in os.walk(dataset_dir):
                        for f in files:
                            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                                rel_dir = os.path.basename(root)
                                existing_files.add((rel_dir, f))
                
                # Fetch only metadata (no heavy binary image data) to check what is in the DB
                metadata_cursor = db.face_images.find({}, {"employee_id": 1, "filename": 1})
                missing_queries = []
                
                for doc in metadata_cursor:
                    eid = doc.get("employee_id")
                    fname = doc.get("filename")
                    if not eid or not fname:
                        continue
                    if (str(eid), fname) not in existing_files:
                        missing_queries.append({"employee_id": int(eid), "filename": fname})
                
                restored_count = 0
                if len(missing_queries) > 0:
                    if progress_callback:
                        progress_callback(8, f"Downloading {len(missing_queries)} missing images...")
                    
                    # Fetch missing documents with their binary image data in chunks of 100
                    chunk_size = 100
                    for i in range(0, len(missing_queries), chunk_size):
                        chunk = missing_queries[i : i + chunk_size]
                        data_cursor = db.face_images.find({"$or": chunk})
                        for doc in data_cursor:
                            eid = doc.get("employee_id")
                            fname = doc.get("filename")
                            img_data = doc.get("image_data")
                            if not eid or not fname or not img_data:
                                continue
                            
                            emp_folder = os.path.join(dataset_dir, str(eid))
                            os.makedirs(emp_folder, exist_ok=True)
                            
                            file_path = os.path.join(emp_folder, fname)
                            with open(file_path, "wb") as f:
                                f.write(img_data)
                            restored_count += 1
                
                if progress_callback:
                    if restored_count > 0:
                        progress_callback(10, f"Restored {restored_count} new images. Database synced.")
                    else:
                        progress_callback(10, "All images already cached locally. Database sync skipped.")
            except Exception as e:
                if progress_callback:
                    progress_callback(10, f"Sync warning: {str(e)}")

        X = []
        y = []
        
        # Filter: only look at numeric directories to prevent ValueError: int(eid) on __pycache__ or other folders
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
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(min_size_val, min_size_val))
                if len(faces) == 0:
                    continue
                
                # Map the largest face coordinates back to original scale
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, fw, fh) = largest_face
                bbox = (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
                
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
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
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
