import os
import io
import queue
import threading
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file, abort, Response
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

# --- MONGODB IMPORTS ---
from pymongo import MongoClient, ReturnDocument
from bson import ObjectId
import certifi

# ---------- Real-time SSE Clients & Broadcast ----------
sse_clients = []

def broadcast_sse(event_type, data):
    for q in list(sse_clients):
        try:
            q.put_nowait({"event": event_type, "data": data})
        except Exception:
            pass

# ---------- Timezone & Datetime Parsing Helper ----------
def parse_dt(ts_str):
    if not ts_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # Assume UTC for legacy naive strings in DB
            return dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        return None


def get_month_range_query(month_str):
    try:
        parts = month_str.split("-")
        year = int(parts[0])
        month = int(parts[1])
        start = f"{year:04d}-{month:02d}-01T00:00:00"
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        end = f"{next_year:04d}-{next_month:02d}-01T00:00:00"
        return {"$gte": start, "$lt": end}
    except Exception:
        return {"$regex": f"^{month_str}"}


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(APP_DIR, "dataset")
try:
    os.makedirs(DATASET_DIR, exist_ok=True)
except Exception:
    pass

if os.environ.get("VERCEL"):
    TRAIN_STATUS_FILE = "/tmp/train_status.json"
else:
    TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- MongoDB Setup ----------
# Reads from Environment Variable for Cloud hosting (like Atlas), or defaults to local.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
# Added short timeouts so serverless execution doesn't hang indefinitely if DB is down
if "localhost" in MONGO_URI:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
else:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, tlsCAFile=certifi.where())
db = client['attendance_system'] # Database Name

# Create database indexes for maximum query performance
try:
    db.employees.create_index("id", unique=True)
except Exception as e:
    print(f"Employees id index warning: {e}")

try:
    db.attendance.create_index([("employee_id", 1), ("timestamp", 1)])
except Exception as e:
    print(f"Attendance compound index warning: {e}")

try:
    db.attendance.create_index("timestamp")
except Exception as e:
    print(f"Attendance timestamp index warning: {e}")

try:
    db.face_images.create_index("employee_id")
except Exception as e:
    print(f"Face images index warning: {e}")

# Try to restore the model.pkl from MongoDB asynchronously in the background so it doesn't block server startup
def restore_model_async():
    try:
        model_doc = db.models.find_one({"_id": "latest_model"})
        if model_doc and "model_bytes" in model_doc:
            with open(MODEL_PATH, "wb") as f:
                f.write(model_doc["model_bytes"])
            print("Successfully restored trained model from MongoDB in background.")
    except Exception as e:
        print(f"Could not restore model from MongoDB in background: {e}")

threading.Thread(target=restore_model_async, daemon=True).start()

# ---------- Render Keepalive: Prevent Cold Starts ----------
def _keepalive_ping():
    """Pings this server every 10 minutes to prevent Render free-tier sleep."""
    import time
    import urllib.request
    # Wait 60 seconds after startup before starting pings
    time.sleep(60)
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not render_url:
        # Not on Render, no keepalive needed
        return
    ping_url = render_url.rstrip("/") + "/ping"
    while True:
        try:
            urllib.request.urlopen(ping_url, timeout=10)
            print(f"[Keepalive] Pinged {ping_url} successfully.")
        except Exception as e:
            print(f"[Keepalive] Ping failed: {e}")
        time.sleep(600)  # Ping every 10 minutes

threading.Thread(target=_keepalive_ping, daemon=True).start()

# Helper for Auto-Incrementing IDs just like SQLite
def get_next_sequence(counter_name):
    ret = db.counters.find_one_and_update(
        {"_id": counter_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return int(ret["seq"])

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    try:
        with open(TRAIN_STATUS_FILE, "w") as f:
            json.dump(status_dict, f)
    except Exception:
        pass

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

# Disabled writing to disk on startup to prevent crashes on read-only Vercel filesystem
# write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Network settings & verification helpers ----------
def get_network_config():
    default_config = {
        "_id": "network_config",
        "wifi_restriction_enabled": False,
        "allowed_subnets": "192.168.1, 192.168.0", # default example local subnets
        "bypass_localhost": True
    }
    try:
        config = db.settings.find_one({"_id": "network_config"})
        if not config:
            config = default_config
            try:
                db.settings.insert_one(config)
            except Exception:
                pass
        return config
    except Exception as e:
        app.logger.error("DB Error in get_network_config: %s", e)
        return default_config

def check_wifi_access(client_ip):
    config = get_network_config()
    if not config.get("wifi_restriction_enabled"):
        return True, "Wi-Fi Restriction Disabled"

    if not client_ip:
        return False, "Could not determine client IP address."

    # Parse first IP if comma-separated proxy chain
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Strip IPv6-mapped IPv4 prefix if present (e.g., ::ffff:192.168.1.50 -> 192.168.1.50)
    if client_ip.startswith("::ffff:"):
        client_ip = client_ip[7:]

    # Always allow localhost if enabled in config
    if config.get("bypass_localhost") and client_ip in ["127.0.0.1", "::1"]:
        return True, "Access from Localhost Allowed"

    allowed_list = [s.strip() for s in config.get("allowed_subnets", "").split(",") if s.strip()]
    
    # Match against prefixes (e.g. 192.168.1 covers 192.168.1.x)
    for allowed in allowed_list:
        if client_ip.startswith(allowed):
            return True, f"Device connected to allowed network: {allowed}"
            
    return False, f"Access Blocked: Device ({client_ip}) not connected to Office Wi-Fi."


# ---------- Context Processors ----------
@app.context_processor
def inject_sidebar_data():
    default_sb = {
        "sb_name": "Loading...",
        "sb_role": "Loading...",
        "sb_total_hours": "0.00",
        "sb_regular_hours": "0.00",
        "sb_rate": "$0.00",
        "sb_salary": "$0.00",
        "sb_dept": "Loading...",
        "sb_eid": 0
    }
    try:
        eid_cookie = request.cookies.get("sidebar_eid")
        if eid_cookie:
            return {
                "sb_name": "Loading...",
                "sb_role": "Loading...",
                "sb_total_hours": "0.00",
                "sb_regular_hours": "0.00",
                "sb_rate": "$0.00",
                "sb_salary": "$0.00",
                "sb_dept": "Loading...",
                "sb_eid": int(eid_cookie)
            }
        return default_sb
    except Exception:
        return default_sb

# ---------- Routes ----------
import traceback

@app.errorhandler(500)
def handle_500(e):
    tb = traceback.format_exc()
    app.logger.error("500 ERROR DETECTED: %s", tb)
    return f"<h1>500 INTERNAL SERVER ERROR DEBUGINFO</h1><pre>{tb}</pre>", 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/stream")
def stream():
    def event_stream():
        q = queue.Queue(maxsize=50)
        sse_clients.append(q)
        try:
            # Send initial connection event
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    event = q.get(timeout=20)  # Heartbeat timeout
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                except queue.Empty:
                    # Keep connection alive
                    yield "event: heartbeat\ndata: {}\n\n"
        except GeneratorExit:
            pass
        finally:
            if q in sse_clients:
                sse_clients.remove(q)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/mock_attendance")
def mock_attendance():
    # Try to find a real employee first
    emp = db.employees.find_one({})
    if emp:
        numeric_eid = emp["id"]
        name = emp["name"]
    else:
        numeric_eid = 99
        name = "Prayag"
        
    import random
    status_options = ["Check In", "Late Check In", "Check Out"]
    status = random.choice(status_options)
    
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    attendance_id = get_next_sequence("attendance_id_seq")
    
    db.attendance.insert_one({
        "id": attendance_id,
        "employee_id": numeric_eid,
        "name": name,
        "timestamp": ts,
        "status": status
    })
    
    broadcast_sse("attendance_marked", {
        "id": int(attendance_id),
        "employee_id": int(numeric_eid),
        "name": name,
        "timestamp": ts,
        "status": status
    })
    
    return jsonify({
        "mocked": True,
        "employee_id": numeric_eid,
        "name": name,
        "timestamp": ts,
        "status": status
    })

# Dashboard stats API (last 30 days counts)
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    from datetime import date, timedelta
    
    # Safe fallback for days
    days = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
    
    try:
        # Fetch all logs
        logs = list(db.attendance.find({}, {"timestamp": 1, "_id": 0}))
        df = pd.DataFrame(logs)
        
        if df.empty:
            return jsonify({"dates": days, "counts": [0]*30})
        
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
        counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
        dates = [d.strftime("%d-%b") for d in last_30]
        return jsonify({"dates": dates, "counts": counts})
    except Exception as e:
        app.logger.error("DB Error in attendance_stats: %s", e)
        return jsonify({"dates": days, "counts": [0]*30})

# -------- Add employee --------
@app.route("/add_employee", methods=["GET", "POST"])
def add_employee():
    if request.method == "GET":
        return render_template("add_employee.html")
    
    try:
        data = request.form
        name = data.get("name", "").strip()
        emp_id = data.get("emp_id", "").strip()
        dept = data.get("dept", "").strip()
        desig = data.get("desig", "").strip()
        joining = data.get("joining", "").strip()
        shift_start = data.get("shift_start", "09:00").strip()
        if not name:
            return jsonify({"error": "name required"}), 400

        hourly_rate = data.get("hourly_rate", "0").strip()
        try:
            hourly_rate = float(hourly_rate)
        except:
            hourly_rate = 0.0

        # Generate numerical ID using counter sequence
        numeric_id = get_next_sequence("employee_id_seq")
        now = datetime.datetime.now().isoformat()
        
        db.employees.insert_one({
            "id": numeric_id,
            "name": name,
            "emp_id": emp_id,
            "department": dept,
            "designation": desig,
            "joining_date": joining,
            "shift_start": shift_start,
            "hourly_rate": hourly_rate,
            "created_at": now
        })
        
        # Keep existing folder structure logic (str(id))
        os.makedirs(os.path.join(DATASET_DIR, str(numeric_id)), exist_ok=True)
        return jsonify({"employee_id": numeric_id})
    except Exception as e:
        app.logger.error("Error in add_employee: %s", e)
        return jsonify({"error": str(e)}), 500

# -------- Upload face images --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    employee_id = request.form.get("employee_id")
    if not employee_id:
        return jsonify({"error": "employee_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, str(employee_id))
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            # Read image bytes
            img_bytes = f.read()
            # Reset pointer for saving locally
            f.seek(0)
            
            # Save to disk
            fname = f"{datetime.datetime.now().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            
            # Save to MongoDB for persistent backup
            db.face_images.insert_one({
                "employee_id": int(employee_id),
                "filename": fname,
                "image_data": img_bytes,
                "created_at": datetime.datetime.utcnow().isoformat()
            })
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

# -------- Train model --------
@app.route("/train_model", methods=["GET"])
def train_model_route():
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status": "already_running"}), 202
    
    def status_callback(p, m):
        payload = {
            "running": (p < 100),
            "progress": p,
            "message": m
        }
        write_train_status(payload)
        broadcast_sse("train_status", payload)
        
    start_payload = {"running": True, "progress": 0, "message": "Starting training"}
    write_train_status(start_payload)
    broadcast_sse("train_status", start_payload)
    
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, db, status_callback))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# -------- Clear all attendance logs --------
@app.route("/clear_attendance", methods=["GET"])
def clear_attendance():
    try:
        res = db.attendance.delete_many({})
        return jsonify({
            "status": "success",
            "message": f"Successfully deleted {res.deleted_count} attendance records."
        }), 200
    except Exception as e:
        app.logger.error("Error clearing attendance: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# -------- Clear all employee and face records --------
@app.route("/clear_employees", methods=["GET"])
def clear_employees():
    try:
        emp_res = db.employees.delete_many({})
        face_res = db.face_images.delete_many({})
        db.counters.delete_many({})
        
        # Clean local dataset directory
        import shutil
        if os.path.exists(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
            os.makedirs(DATASET_DIR, exist_ok=True)
            
        return jsonify({
            "status": "success",
            "message": f"Successfully deleted {emp_res.deleted_count} employees and {face_res.deleted_count} face images."
        }), 200
    except Exception as e:
        app.logger.error("Error clearing employees: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    client_ip = request.remote_addr
    # Handle X-Forwarded-For if behind a proxy
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0]
    
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Normalize IP prefix
    if client_ip and client_ip.startswith("::ffff:"):
        client_ip = client_ip[7:]
        
    allowed, message = check_wifi_access(client_ip)
    return render_template("mark_attendance.html", wifi_allowed=allowed, wifi_message=message, client_ip=client_ip)

# -------- Recognize face endpoint --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    client_ip = request.remote_addr
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0]
        
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Normalize IP prefix
    if client_ip and client_ip.startswith("::ffff:"):
        client_ip = client_ip[7:]
        
    allowed, message = check_wifi_access(client_ip)
    if not allowed:
        return jsonify({"recognized": False, "error": message}), 403

    if "image" not in request.files:
        return jsonify({"recognized": False, "error": "no image"}), 400
    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error": "no face detected"}), 200
        
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists(db)
        if clf is None:
            return jsonify({"recognized": False, "error": "model not trained"}), 200
        
        pred_label, conf = predict_with_model(clf, emb)
        if conf < 0.65:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
        
        # Get accurate employee int ID
        numeric_eid = int(pred_label)
        employee = db.employees.find_one({"id": numeric_eid})
        
        # FIX: If record deleted/legacy residue, don't log as Unknown!
        if not employee:
            return jsonify({"recognized": False, "error": "legacy record detected"}), 200
            
        name = employee["name"]

        # -- COOLDOWN & ALTERNATE LOGIC --
        last_rec = db.attendance.find_one(
            {"employee_id": numeric_eid},
            sort=[("timestamp", -1)]
        )
        
        IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_local = now_dt.astimezone(IST)
        next_status = "Check In"
        
        if last_rec:
            last_ts_str = last_rec.get("timestamp")
            last_status = last_rec.get("status", "Check In")
            try:
                last_dt = parse_dt(last_ts_str)
                if last_dt:
                    delta = (now_dt - last_dt).total_seconds()
                    if delta < 60:
                        return jsonify({"recognized": True, "employee_id": numeric_eid, "name": name, "confidence": float(conf), "status": "Debounced (Wait 1 min)"}), 200
                    next_status = "Check Out" if last_status in ["Check In", "Late Check In"] else "Check In"
            except:
                pass

        # Calculate if late (Only apply to the FIRST attendance of the day in IST)
        if next_status == "Check In":
            start_of_today_ist = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_today_ist = start_of_today_ist + datetime.timedelta(days=1)
            
            # Fetch recent logs to robustly scan for today's attendance regardless of offsets in DB
            recent_logs = list(db.attendance.find({"employee_id": numeric_eid}).sort("timestamp", -1).limit(10))
            
            has_attendance_today = False
            for log in recent_logs:
                log_dt = parse_dt(log.get("timestamp"))
                if log_dt:
                    log_dt_local = log_dt.astimezone(IST)
                    if start_of_today_ist <= log_dt_local < end_of_today_ist:
                        has_attendance_today = True
                        break
            
            if not has_attendance_today: # First time attending today
                shift_start_str = employee.get("shift_start", "09:00")
                try:
                    h, m = map(int, shift_start_str.split(':'))
                    expected_time = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
                    grace_time = expected_time + datetime.timedelta(minutes=15)
                    if now_local > grace_time:
                        next_status = "Late Check In"
                except:
                    pass

        ts = now_dt.isoformat()
        attendance_id = get_next_sequence("attendance_id_seq")
        db.attendance.insert_one({
            "id": attendance_id,
            "employee_id": numeric_eid,
            "name": name,
            "timestamp": ts,
            "status": next_status
        })
        
        # Broadcast real-time attendance marked event
        broadcast_sse("attendance_marked", {
            "id": int(attendance_id),
            "employee_id": int(numeric_eid),
            "name": name,
            "timestamp": ts,
            "status": next_status
        })
        
        return jsonify({"recognized": True, "employee_id": numeric_eid, "name": name, "confidence": float(conf), "status": next_status}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

# -------- Attendance records & filters --------
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")
    query = {}
    formatted_records = []
    
    try:
        IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_local = now_dt.astimezone(IST)

        cursor = db.attendance.find(query).sort("timestamp", -1).limit(5000)
        
        for r in cursor:
            ts_str = r.get("timestamp")
            dt = parse_dt(ts_str)
            if dt:
                dt_local = dt.astimezone(IST)
                
                # Apply timezone-accurate period filtering
                if period == "daily":
                    if dt_local.date() != now_local.date():
                        continue
                elif period in ["weekly", "monthly"]:
                    days = 7 if period == "weekly" else 30
                    if (now_dt - dt).days >= days:
                        continue
                
                date_str = dt_local.strftime("%Y-%m-%d")
                time_str = dt_local.strftime("%I:%M:%S %p")
                if time_str.startswith("0"):
                    time_str = time_str[1:]

                formatted_records.append([
                    r.get("id"),
                    r.get("employee_id"),
                    r.get("name"),
                    date_str,
                    r.get("status", "Check In"),
                    time_str
                ])
    except Exception as e:
        app.logger.error("DB Error in attendance_record: %s", e)
        
    return render_template("attendance_record.html", records=formatted_records, period=period)

# -------- CSV download --------
@app.route("/download_csv", methods=["GET"])
def download_csv():
    cursor = db.attendance.find({}).sort("timestamp", -1)
    output = io.StringIO()
    output.write("id,employee_id,name,timestamp,status\n")
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    for r in cursor:
        ts_str = r.get("timestamp")
        dt = parse_dt(ts_str)
        if dt:
            dt_local = dt.astimezone(IST)
            ts_formatted = dt_local.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_formatted = ts_str
        output.write(f'{r.get("id")},{r.get("employee_id")},{r.get("name")},{ts_formatted},{r.get("status", "N/A")}\n')
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="employee_attendance.csv", mimetype="text/csv")

@app.route("/users")
def users_dashboard_page():
    employees = []
    try:
        cursor = db.employees.find({}).sort("id", -1)
        for emp in cursor:
            # Remove Mongo internal ObjectId from raw passing to prevent serialization crashes
            emp.pop('_id', None) 
            employees.append(emp)
    except Exception as e:
        app.logger.error("DB Error in users_dashboard_page: %s", e)
    return render_template("users.html", employees=employees)

# -------- Employees Update API --------
@app.route("/employees/update", methods=["POST"])
def update_employee():
    eid = request.form.get("id")
    if not eid:
        return jsonify({"error": "id required"}), 400
    
    hourly_rate_str = request.form.get("hourly_rate", "0").strip()
    try:
        hourly_rate = float(hourly_rate_str)
    except:
        hourly_rate = 0.0

    update_data = {
        "name": request.form.get("name", "").strip(),
        "emp_id": request.form.get("emp_id", "").strip(),
        "department": request.form.get("department", "").strip(),
        "designation": request.form.get("designation", "").strip(),
        "joining_date": request.form.get("joining_date", ""),
        "shift_start": request.form.get("shift_start", "09:00").strip(),
        "hourly_rate": hourly_rate
    }
    
    db.employees.update_one({"id": int(eid)}, {"$set": update_data})
    # also sync the attendance records so names reflect update
    db.attendance.update_many({"employee_id": int(eid)}, {"$set": {"name": update_data["name"]}})
    
    return jsonify({"updated": True})

# -------- Employees API for listing/editing --------
@app.route("/employees", methods=["GET"])
def employees_list():
    cursor = db.employees.find({}).sort("id", -1)
    data = []
    for r in cursor:
        data.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "emp_id": r.get("emp_id"),
            "department": r.get("department"),
            "designation": r.get("designation"),
            "joining_date": r.get("joining_date"),
            "shift_start": r.get("shift_start", "09:00"),
            "hourly_rate": r.get("hourly_rate", 0.0),
            "created_at": r.get("created_at")
        })
    return jsonify({"employees": data})

@app.route("/employees/<int:eid>", methods=["DELETE"])
def delete_employee(eid):
    db.employees.delete_one({"id": int(eid)})
    db.attendance.delete_many({"employee_id": int(eid)})
    db.face_images.delete_many({"employee_id": int(eid)})
    
    folder = os.path.join(DATASET_DIR, str(eid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})

# -------- Payroll & Salary Calculation --------
@app.route("/payroll", methods=["GET"])
def payroll_page():
    # Optional month filter, defaults to current month
    month_str = request.args.get("month")
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    if not month_str:
        month_str = datetime.datetime.now(IST).strftime("%Y-%m")
    
    payroll_data = []
    
    try:
        # We want attendance records matching this month prefix in ISO format
        query = {"timestamp": get_month_range_query(month_str)}
        attendance_records = list(db.attendance.find(query).sort([("employee_id", 1), ("timestamp", 1)]))
        
        employees = list(db.employees.find({}))
        emp_map = {emp["id"]: emp for emp in employees}
        
        # Calculate hours per employee
        # Group by employee
        emp_attendance = {}
        for r in attendance_records:
            eid = r["employee_id"]
            if eid not in emp_attendance:
                emp_attendance[eid] = []
            emp_attendance[eid].append(r)
            
        for emp in employees:
            eid = emp["id"]
            records = emp_attendance.get(eid, [])
            total_seconds = 0
            last_checkin_time = None
            
            for r in records:
                status = r.get("status", "")
                ts = parse_dt(r["timestamp"])
                if ts:
                    if status in ["Check In", "Late Check In"]:
                        last_checkin_time = ts
                    elif status == "Check Out":
                        if last_checkin_time:
                            # check if same day in IST
                            if last_checkin_time.astimezone(IST).date() == ts.astimezone(IST).date():
                                delta = (ts - last_checkin_time).total_seconds()
                                if delta > 0:
                                    total_seconds += delta
                            last_checkin_time = None
                        
            total_hours = total_seconds / 3600.0
            rate = emp.get("hourly_rate", 0.0)
            salary = total_hours * rate
            
            payroll_data.append({
                "id": eid,
                "name": emp["name"],
                "emp_id": emp["emp_id"],
                "hourly_rate": rate,
                "total_hours": round(total_hours, 2),
                "salary": round(salary, 2)
            })
    except Exception as e:
        app.logger.error("DB Error in payroll_page: %s", e)
        
    return render_template("payroll.html", payroll_data=payroll_data, month_str=month_str)

@app.route("/download_payroll_csv", methods=["GET"])
def download_payroll_csv():
    month_str = request.args.get("month")
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    if not month_str:
        month_str = datetime.datetime.now(IST).strftime("%Y-%m")
    
    query = {"timestamp": get_month_range_query(month_str)}
    attendance_records = list(db.attendance.find(query).sort([("employee_id", 1), ("timestamp", 1)]))
    
    employees = list(db.employees.find({}))
    
    emp_attendance = {}
    for r in attendance_records:
        eid = r["employee_id"]
        if eid not in emp_attendance:
            emp_attendance[eid] = []
        emp_attendance[eid].append(r)
        
    output = io.StringIO()
    output.write("Employee ID,Name,Department,Designation,Hourly Rate,Total Hours,Calculated Salary\n")
    
    for emp in employees:
        eid = emp["id"]
        records = emp_attendance.get(eid, [])
        total_seconds = 0
        last_checkin_time = None
        
        for r in records:
            status = r.get("status", "")
            ts = parse_dt(r["timestamp"])
            if ts:
                if status in ["Check In", "Late Check In"]:
                    last_checkin_time = ts
                elif status == "Check Out":
                    if last_checkin_time:
                        if last_checkin_time.astimezone(IST).date() == ts.astimezone(IST).date():
                            delta = (ts - last_checkin_time).total_seconds()
                            if delta > 0:
                                total_seconds += delta
                        last_checkin_time = None
                    
        total_hours = total_seconds / 3600.0
        rate = emp.get("hourly_rate", 0.0)
        salary = total_hours * rate
        
        output.write(f'{emp["emp_id"]},{emp["name"]},{emp.get("department", "")},{emp.get("designation", "")},{rate:.2f},{total_hours:.2f},{salary:.2f}\n')
        
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"monthly_report_{month_str}.csv", mimetype="text/csv")

# -------- Employee Performance & Analytics --------
@app.route("/performance", methods=["GET"])
def performance_page():
    month_str = request.args.get("month")
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    if not month_str:
        month_str = datetime.datetime.now(IST).strftime("%Y-%m")
    
    performance_data = []
    
    try:
        # Get all records for selected month
        query = {"timestamp": get_month_range_query(month_str)}
        attendance_records = list(db.attendance.find(query).sort("timestamp", 1))
        
        employees = list(db.employees.find({}))
        
        emp_attendance = {}
        for r in attendance_records:
            eid = r["employee_id"]
            if eid not in emp_attendance:
                emp_attendance[eid] = []
            emp_attendance[eid].append(r)
            
        for emp in employees:
            eid = emp["id"]
            records = emp_attendance.get(eid, [])
            
            total_seconds = 0
            last_checkin_time = None
            
            days_present = set()
            days_late = set()
            
            for r in records:
                status = r.get("status", "")
                ts = parse_dt(r["timestamp"])
                if ts:
                    day_str = ts.astimezone(IST).date().isoformat()
                    
                    if status in ["Check In", "Late Check In"]:
                        days_present.add(day_str)
                        if status == "Late Check In":
                            days_late.add(day_str)
                        last_checkin_time = ts
                    elif status == "Check Out":
                        if last_checkin_time:
                            if last_checkin_time.astimezone(IST).date() == ts.astimezone(IST).date():
                                delta = (ts - last_checkin_time).total_seconds()
                                if delta > 0:
                                    total_seconds += delta
                            last_checkin_time = None
                    
            total_hours = total_seconds / 3600.0
            num_days_present = len(days_present)
            num_days_late = len(days_late)
            num_days_on_time = max(0, num_days_present - num_days_late)
            
            punctuality_rate = (num_days_on_time / num_days_present * 100) if num_days_present > 0 else 0.0
            avg_hours = (total_hours / num_days_present) if num_days_present > 0 else 0.0
            
            # Score weight: 60% Punctuality, 40% Work Hours (Target 8 hrs/day)
            punct_score = punctuality_rate
            hours_score = min(100.0, (avg_hours / 8.0) * 100) if num_days_present > 0 else 0.0
            final_score = (0.6 * punct_score) + (0.4 * hours_score)
            
            if num_days_present == 0:
                grade = "N/A"
            elif final_score >= 90:
                grade = "A+"
            elif final_score >= 80:
                grade = "A"
            elif final_score >= 70:
                grade = "B"
            elif final_score >= 60:
                grade = "C"
            else:
                grade = "D"
                
            performance_data.append({
                "id": eid,
                "name": emp.get("name", "Unknown"),
                "emp_id": emp.get("emp_id", "-"),
                "designation": emp.get("designation", "-"),
                "total_days": num_days_present,
                "on_time_days": num_days_on_time,
                "late_days": num_days_late,
                "punctuality_rate": round(punctuality_rate, 1),
                "total_hours": round(total_hours, 1),
                "avg_hours": round(avg_hours, 1),
                "score": round(final_score, 1),
                "grade": grade
            })
            
        performance_data.sort(key=lambda x: x["score"], reverse=True)
    except Exception as e:
        app.logger.error("DB Error in performance_page: %s", e)
        
    return render_template("performance.html", data=performance_data, month_str=month_str)

@app.route("/api/employee_image/<int:eid>")
def employee_image(eid):
    folder = os.path.join(DATASET_DIR, str(eid))
    if os.path.exists(folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            return send_file(os.path.join(folder, images[0]))
            
    # Try MongoDB if local file not found (due to container sleep / reset)
    try:
        img_doc = db.face_images.find_one({"employee_id": int(eid)})
        if img_doc and "image_data" in img_doc:
            return send_file(
                io.BytesIO(img_doc["image_data"]),
                mimetype="image/jpeg",
                as_attachment=False,
                download_name=img_doc.get("filename", "face.jpg")
            )
    except Exception as e:
        app.logger.error("DB Error in employee_image: %s", e)
        
    abort(404)

@app.route("/api/sidebar_employee/<direction>/<int:current_eid>")
def api_sidebar_employee(direction, current_eid):
    employees = list(db.employees.find({}, {"id": 1}).sort("id", 1))
    if not employees:
        return jsonify({"error": "No employees"}), 404
        
    eids = [e["id"] for e in employees]
    try:
        idx = eids.index(current_eid)
    except ValueError:
        idx = 0
        
    if direction == "next":
        idx = (idx + 1) % len(eids)
    elif direction == "prev":
        idx = (idx - 1) % len(eids)
        
    target_eid = eids[idx]
    
    emp = db.employees.find_one({"id": target_eid})
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    month_str = datetime.datetime.now(IST).strftime("%Y-%m")
    query = {"employee_id": target_eid, "timestamp": get_month_range_query(month_str)}
    records = list(db.attendance.find(query).sort("timestamp", 1))
    
    total_seconds = 0
    last_checkin_time = None
    for r in records:
        status = r.get("status", "")
        ts = parse_dt(r["timestamp"])
        if ts:
            if status in ["Check In", "Late Check In"]:
                last_checkin_time = ts
            elif status == "Check Out":
                if last_checkin_time:
                    if last_checkin_time.astimezone(IST).date() == ts.astimezone(IST).date():
                        total_seconds += (ts - last_checkin_time).total_seconds()
                        last_checkin_time = None
                
    total_hours = total_seconds / 3600.0
    rate = emp.get("hourly_rate", 0.0)
    salary = total_hours * rate
    
    return jsonify({
        "name": emp.get("name", "Unknown"),
        "role": emp.get("designation", "Employee") or "Employee",
        "dept": emp.get("department", "N/A") or "N/A",
        "total_hours": f"{total_hours:.2f}",
        "regular_hours": f"{total_hours:.2f} hrs",
        "rate": f"${rate:.2f}",
        "salary": f"${salary:.2f}",
        "eid": target_eid
    })

# -------- System Settings --------
@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    message = None
    if request.method == "POST":
        wifi_enabled = request.form.get("wifi_restriction_enabled") == "on"
        allowed_subnets = request.form.get("allowed_subnets", "").strip()
        bypass_localhost = request.form.get("bypass_localhost") == "on"
        
        try:
            db.settings.update_one(
                {"_id": "network_config"},
                {
                    "$set": {
                        "wifi_restriction_enabled": wifi_enabled,
                        "allowed_subnets": allowed_subnets,
                        "bypass_localhost": bypass_localhost
                    }
                },
                upsert=True
            )
            message = "Settings updated successfully!"
        except Exception as e:
            app.logger.error("DB Error updating settings: %s", e)
            message = "Error: Could not connect to database to save settings."
        
    config = get_network_config()
    client_ip = request.remote_addr
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0]
        
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Normalize IP prefix
    if client_ip and client_ip.startswith("::ffff:"):
        client_ip = client_ip[7:]
        
    return render_template("settings.html", config=config, message=message, current_ip=client_ip)

if __name__ == "__main__":
    # Warmup ping to Mongo
    try:
        client.admin.command('ping')
        print("Connected successfully to MongoDB")
    except Exception as e:
        print(f"COULD NOT CONNECT TO MONGODB: {e}")
    
    app.run(host="0.0.0.0", debug=True)