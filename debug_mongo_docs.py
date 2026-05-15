from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client['attendance_system']
docs = list(db.employees.find({}).limit(10))
print("--- DOCUMENTS IN EMPLOYEES ---")
for d in docs:
    print(d)
