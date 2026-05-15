from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client['attendance_system']
res = db.attendance.delete_many({"name": "Unknown"})
print(f"CLEANED UP {res.deleted_count} UNKNOWN RECORDS.")
