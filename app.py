import os
import io
import threading
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file, abort
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

# --- MONGODB IMPORTS ---
from pymongo import MongoClient, ReturnDocument
from bson import ObjectId
import certifi

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
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, tlsCAFile=certifi.where())
db = client['attendance_system'] # Database Name

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
        "sb_name": "No Data",
        "sb_role": "-",
        "sb_total_hours": "0.00",
        "sb_regular_hours": "0.00",
        "sb_rate": "$0.00",
        "sb_salary": "$0.00",
        "sb_dept": "-",
        "sb_eid": 0
    }
    try:
        eid_cookie = request.cookies.get("sidebar_eid")
        emp = None
        
        if eid_cookie:
            try:
                emp = db.employees.find_one({"id": int(eid_cookie)})
            except:
                emp = None
                
        if not emp:
            latest_record = db.attendance.find_one({}, sort=[("timestamp", -1)])
            if latest_record:
                emp = db.employees.find_one({"id": latest_record["employee_id"]})
                
        if not emp:
            emp = db.employees.find_one({})
            
        if not emp:
            return default_sb
            
        eid = emp["id"]
        month_str = datetime.datetime.now().strftime("%Y-%m")
        query = {"employee_id": eid, "timestamp": {"$regex": f"^{month_str}"}}
        records = list(db.attendance.find(query).sort("timestamp", 1))
        
        total_seconds = 0
        last_checkin_time = None
        for r in records:
            status = r.get("status", "")
            ts = datetime.datetime.fromisoformat(r["timestamp"])
            if status in ["Check In", "Late Check In"]:
                last_checkin_time = ts
            elif status == "Check Out":
                if last_checkin_time and last_checkin_time.date() == ts.date():
                    total_seconds += (ts - last_checkin_time).total_seconds()
                    last_checkin_time = None
                    
        total_hours = total_seconds / 3600.0
        rate = emp.get("hourly_rate", 0.0)
        salary = total_hours * rate
        
        return {
            "sb_name": emp.get("name", "Unknown"),
            "sb_role": emp.get("designation", "Employee") or "Employee",
            "sb_dept": emp.get("department", "N/A") or "N/A",
            "sb_total_hours": f"{total_hours:.2f}",
            "sb_regular_hours": f"{total_hours:.2f}",
            "sb_rate": f"${rate:.2f}",
            "sb_salary": f"${salary:.2f}",
            "sb_eid": eid
        }
    except Exception as e:
        app.logger.error("DB Error in inject_sidebar_data: %s", e)
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
            fname = f"{datetime.datetime.now().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
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
        write_train_status({
            "running": (p < 100),
            "progress": p,
            "message": m
        })
        
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, status_callback))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    client_ip = request.remote_addr
    # Handle X-Forwarded-For if behind a proxy
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0]
    
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
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error": "model not trained"}), 200
        
        pred_label, conf = predict_with_model(clf, emb)
        if conf < 0.3:
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
        
        now_dt = datetime.datetime.now()
        next_status = "Check In"
        
        if last_rec:
            last_ts_str = last_rec.get("timestamp")
            last_status = last_rec.get("status", "Check In")
            try:
                last_dt = datetime.datetime.fromisoformat(last_ts_str)
                delta = (now_dt - last_dt).total_seconds()
                if delta < 60:
                    return jsonify({"recognized": True, "employee_id": numeric_eid, "name": name, "confidence": float(conf), "status": "Debounced (Wait 1 min)"}), 200
                next_status = "Check Out" if last_status in ["Check In", "Late Check In"] else "Check In"
            except:
                pass

        # Calculate if late (Only apply to the FIRST attendance of the day)
        if next_status == "Check In":
            today_str = now_dt.date().isoformat()
            has_attendance_today = db.attendance.find_one({
                "employee_id": numeric_eid,
                "timestamp": {"$regex": f"^{today_str}"}
            })
            
            if not has_attendance_today: # First time attending today
                shift_start_str = employee.get("shift_start", "09:00")
                try:
                    h, m = map(int, shift_start_str.split(':'))
                    expected_time = now_dt.replace(hour=h, minute=m, second=0, microsecond=0)
                    grace_time = expected_time + datetime.timedelta(minutes=15)
                    if now_dt > grace_time:
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
        if period == "daily":
            today = datetime.date.today().isoformat()
            query["timestamp"] = {"$regex": f"^{today}"}
        elif period in ["weekly", "monthly"]:
            days = 7 if period == "weekly" else 30
            start_dt = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
            query["timestamp"] = {"$gte": start_dt}

        cursor = db.attendance.find(query).sort("timestamp", -1).limit(5000)
        
        for r in cursor:
            formatted_records.append([
                r.get("id"),
                r.get("employee_id"),
                r.get("name"),
                r.get("timestamp"),
                r.get("status", "Check In")
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
    for r in cursor:
        output.write(f'{r.get("id")},{r.get("employee_id")},{r.get("name")},{r.get("timestamp")},{r.get("status", "N/A")}\n')
    
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
    if not month_str:
        month_str = datetime.datetime.now().strftime("%Y-%m")
    
    payroll_data = []
    
    try:
        # We want attendance records matching this month prefix in ISO format
        query = {"timestamp": {"$regex": f"^{month_str}"}}
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
                ts = datetime.datetime.fromisoformat(r["timestamp"])
                if status in ["Check In", "Late Check In"]:
                    last_checkin_time = ts
                elif status == "Check Out":
                    if last_checkin_time:
                        # check if same day
                        if last_checkin_time.date() == ts.date():
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
    if not month_str:
        month_str = datetime.datetime.now().strftime("%Y-%m")
    
    query = {"timestamp": {"$regex": f"^{month_str}"}}
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
            ts = datetime.datetime.fromisoformat(r["timestamp"])
            if status in ["Check In", "Late Check In"]:
                last_checkin_time = ts
            elif status == "Check Out":
                if last_checkin_time:
                    if last_checkin_time.date() == ts.date():
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
    if not month_str:
        month_str = datetime.datetime.now().strftime("%Y-%m")
    
    performance_data = []
    
    try:
        # Get all records for selected month
        query = {"timestamp": {"$regex": f"^{month_str}"}}
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
                ts = datetime.datetime.fromisoformat(r["timestamp"])
                day_str = ts.date().isoformat()
                
                if status in ["Check In", "Late Check In"]:
                    days_present.add(day_str)
                    if status == "Late Check In":
                        days_late.add(day_str)
                    last_checkin_time = ts
                elif status == "Check Out":
                    if last_checkin_time and last_checkin_time.date() == ts.date():
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
    month_str = datetime.datetime.now().strftime("%Y-%m")
    query = {"employee_id": target_eid, "timestamp": {"$regex": f"^{month_str}"}}
    records = list(db.attendance.find(query).sort("timestamp", 1))
    
    total_seconds = 0
    last_checkin_time = None
    for r in records:
        status = r.get("status", "")
        ts = datetime.datetime.fromisoformat(r["timestamp"])
        if status in ["Check In", "Late Check In"]:
            last_checkin_time = ts
        elif status == "Check Out":
            if last_checkin_time and last_checkin_time.date() == ts.date():
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