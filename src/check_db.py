import sqlite3
import os

db_path = os.path.join("data", "spip.db")

if not os.path.exists(db_path):
    print("Database file data/spip.db does not exist yet. Run main_inference.py first.")
    exit()

conn = sqlite3.connect(db_path)
cur = conn.cursor()

print("--- Latest 10 Occupancy Records ---")
for row in cur.execute("SELECT * FROM occupancy_events ORDER BY id DESC LIMIT 10"):
    print(row)

conn.close()