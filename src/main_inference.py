import cv2
import json
import time
import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from ultralytics import YOLO
from shapely.geometry import Polygon

VIDEO_PATH = os.path.join("data", "test_lot.mp4")
SLOTS_JSON = os.path.join("data", "slot_coordinates.json")
COVERAGE_THRESHOLD = 0.40
DB_URL = "postgresql://admin:securepassword@localhost:5432/spip_db"

def init_db(slot_coords):
    """Connects to TimescaleDB, builds tables, and syncs mapped slots."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True  # Automatically commit inserts
    cur = conn.cursor()
    
    # 1. Create the slots reference table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS slots (
            slot_id INT PRIMARY KEY,
            polygon_coordinates JSONB NOT NULL
        );
    """)
    
    # 2. Create the time-series events table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS occupancy_events (
            timestamp TIMESTAMPTZ NOT NULL,
            slot_id INT REFERENCES slots(slot_id),
            is_occupied BOOLEAN NOT NULL
        );
    """)
    
    # 3. Convert into a TimescaleDB hypertable (optimized for time queries)
    cur.execute("SELECT create_hypertable('occupancy_events', 'timestamp', if_not_exists => TRUE);")
        
    # 4. Sync the JSON mapped slots into the database
    for i, coords in enumerate(slot_coords):
        cur.execute(
            "INSERT INTO slots (slot_id, polygon_coordinates) VALUES (%s, %s) ON CONFLICT (slot_id) DO NOTHING",
            (i, json.dumps(coords))
        )
        
    return conn

def main():
    if not os.path.exists(SLOTS_JSON):
        print(f"Error: {SLOTS_JSON} not found. Run map_slots.py first.")
        return

    with open(SLOTS_JSON, "r") as f:
        slot_coords = json.load(f)
    slot_polygons = [Polygon(coords) for coords in slot_coords]

    # Initialize Database connection
    try:
        print("Connecting to TimescaleDB...")
        db_conn = init_db(slot_coords)
        db_cursor = db_conn.cursor()
        print("Database connected successfully!")
    except Exception as e:
        print(f"Database connection failed: {e}. Is Docker running?")
        return

    model = YOLO("yolo11n.pt") 
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    prev_time = 0
    last_db_push = 0
    PUSH_INTERVAL = 2.0  # Write to the DB every 2 seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO11 Inference + ByteTrack
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", 
            classes=[2, 3, 5, 7], verbose=False
        )
        
        vehicle_boxes = []
        for r in results:
            if r.boxes.id is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    vehicle_boxes.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

        slot_states = [] 
        occupied_count = 0

        # Coverage Math
        for i, slot_poly in enumerate(slot_polygons):
            is_occupied = False
            pts = np.array(slot_coords[i], np.int32)
            
            for vehicle_poly in vehicle_boxes:
                if slot_poly.intersects(vehicle_poly):
                    coverage = slot_poly.intersection(vehicle_poly).area / slot_poly.area
                    if coverage > COVERAGE_THRESHOLD:
                        is_occupied = True
                        break
            
            slot_states.append((i, is_occupied))
            
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            if is_occupied:
                occupied_count += 1
                
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, str(i), (pts[0][0], pts[0][1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ---------------------------------------------------------
        # DATABASE INGESTION LOGIC
        # ---------------------------------------------------------
        current_time = time.time()
        if current_time - last_db_push >= PUSH_INTERVAL:
            now = datetime.now(timezone.utc)
            # Create a list of tuples formatted for PostgreSQL
            db_values = [(now, state[0], state[1]) for state in slot_states]
            
            try:
                # execute_values is a fast bulk-insert method
                execute_values(db_cursor, 
                    "INSERT INTO occupancy_events (timestamp, slot_id, is_occupied) VALUES %s", 
                    db_values
                )
                print(f"[{now.strftime('%H:%M:%S')}] Pushed {len(db_values)} slot states to DB.")
                last_db_push = current_time
            except Exception as e:
                print(f"DB Write Error: {e}")
        # ---------------------------------------------------------

        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"Occupied: {occupied_count}/{len(slot_polygons)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("SPIP CV + Database Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if db_conn:
        db_conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()