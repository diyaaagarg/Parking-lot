# import cv2
# import json
# import time
# import os
# import sqlite3
# from datetime import datetime, timezone
# import numpy as np
# from ultralytics import YOLO
# from shapely.geometry import Polygon

# VIDEO_PATH = os.path.join("data", "clip.mp4")
# SLOTS_JSON = os.path.join("data", "slot_coordinates.json")
# DB_PATH = os.path.join("data", "spip.db")
# COVERAGE_THRESHOLD = 0.40  # 40% overlap = occupied

# def init_db(slot_coords):
#     """Initializes local SQLite database and tables."""
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
    
#     # 1. Create slots table
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS slots (
#             slot_id INTEGER PRIMARY KEY,
#             polygon_coordinates TEXT NOT NULL
#         );
#     """)
    
#     # 2. Create time-series occupancy table
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS occupancy_events (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             slot_id INTEGER,
#             is_occupied INTEGER,
#             FOREIGN KEY (slot_id) REFERENCES slots(slot_id)
#         );
#     """)
    
#     # 3. Populate slot metadata
#     for i, coords in enumerate(slot_coords):
#         cur.execute(
#             "INSERT OR IGNORE INTO slots (slot_id, polygon_coordinates) VALUES (?, ?)",
#             (i, json.dumps(coords))
#         )
        
#     conn.commit()
#     return conn

# def main():
#     if not os.path.exists(SLOTS_JSON):
#         print(f"Error: {SLOTS_JSON} not found. Run map_slots.py first.")
#         return

#     if not os.path.exists(VIDEO_PATH):
#         print(f"Error: Video file not found at '{VIDEO_PATH}'.")
#         return

#     with open(SLOTS_JSON, "r") as f:
#         slot_coords = json.load(f)
#     slot_polygons = [Polygon(coords) for coords in slot_coords]

#     # Initialize local SQLite database
#     print(f"Connecting to local SQLite database at {DB_PATH}...")
#     db_conn = init_db(slot_coords)
#     db_cursor = db_conn.cursor()
#     print("Database connected successfully!")

#     # Check if custom weights exist, otherwise fall back to yolo11n.pt
#     custom_weights = os.path.join("weights", "best.pt")
#     model_path = custom_weights if os.path.exists(custom_weights) else "yolo11n.pt"
#     print(f"Loading YOLO model from: {model_path}")
#     model = YOLO(model_path)

#     cap = cv2.VideoCapture(VIDEO_PATH)
#     prev_time = 0
#     last_db_push = 0
#     PUSH_INTERVAL = 2.0  # Write to DB every 2 seconds

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run YOLO11 tracking
#         results = model.predict(
#             frame, 
#             conf=0.15,
#             verbose=False
#         )

#         vehicle_boxes = []
#         for r in results:
#             # if r.boxes.id is not None:
#                 for box in r.boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].tolist()
#                     # Add these two debug lines to visually see the YOLO boxes
#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
#                     vehicle_boxes.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
                    
#         slot_states = []
#         occupied_count = 0

#         # Slot coverage logic
#         for i, slot_poly in enumerate(slot_polygons):
#             is_occupied = False
#             pts = np.array(slot_coords[i], np.int32)

#             for vehicle_poly in vehicle_boxes:
#                 if slot_poly.intersects(vehicle_poly):
#                     coverage = slot_poly.intersection(vehicle_poly).area / slot_poly.area
#                     if coverage > COVERAGE_THRESHOLD:
#                         is_occupied = True
#                         break

#             slot_states.append((i, 1 if is_occupied else 0))

#             color = (0, 0, 255) if is_occupied else (0, 255, 0)
#             if is_occupied:
#                 occupied_count += 1

#             cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
#             cv2.putText(frame, str(i), (pts[0][0], pts[0][1]-8), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

#         # Ingest state into local SQLite DB
#         current_time = time.time()
#         if current_time - last_db_push >= PUSH_INTERVAL:
#             now_iso = datetime.now(timezone.utc).isoformat()
#             db_values = [(now_iso, state[0], state[1]) for state in slot_states]
            
#             try:
#                 db_cursor.executemany(
#                     "INSERT INTO occupancy_events (timestamp, slot_id, is_occupied) VALUES (?, ?, ?)",
#                     db_values
#                 )
#                 db_conn.commit()
#                 print(f"[{now_iso}] Pushed {len(db_values)} slot states to local SQLite.")
#                 last_db_push = current_time
#             except Exception as e:
#                 print(f"DB Write Error: {e}")

#         # FPS overlay
#         fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
#         prev_time = current_time

#         cv2.putText(frame, f"Occupied: {occupied_count}/{len(slot_polygons)}", (20, 40), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#         cv2.putText(frame, f"FPS: {int(fps)}", (20, 75), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#         cv2.imshow("SPIP CV (No Docker / Local SQLite)", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if db_conn:
#         db_conn.close()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
from shapely.geometry import Polygon, Point
import cv2
import json
import time
import os
import sqlite3
from datetime import datetime, timezone
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

VIDEO_PATH = os.path.join("data", "clip.mp4") # Update this if your video is named differently
SLOTS_JSON = os.path.join("data", "slot_coordinates.json")
DB_PATH = os.path.join("data", "spip.db")
COVERAGE_THRESHOLD = 0.15

# Global variables for the drawing UI
drawing_slots = []
current_slot = []

def mouse_click(event, x, y, flags, param):
    global current_slot, drawing_slots
    if event == cv2.EVENT_LBUTTONDOWN:
        current_slot.append((x, y))
        if len(current_slot) == 4:
            drawing_slots.append(current_slot)
            current_slot = []

def init_db(slot_coords):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS slots (
            slot_id INTEGER PRIMARY KEY,
            polygon_coordinates TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS occupancy_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            slot_id INTEGER,
            is_occupied INTEGER,
            FOREIGN KEY (slot_id) REFERENCES slots(slot_id)
        );
    """)
    for i, coords in enumerate(slot_coords):
        cur.execute("INSERT OR IGNORE INTO slots (slot_id, polygon_coordinates) VALUES (?, ?)", (i, json.dumps(coords)))
    conn.commit()
    return conn

def main():
    global drawing_slots, current_slot

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video.")
        return

    # ---------------------------------------------------------
    # 1. DRAWING MODE
    # ---------------------------------------------------------
    window_name = "SPIP - Draw Slots (Press 's' to start, 'c' to clear)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    # Load existing slots if they exist so you don't have to start from scratch
    if os.path.exists(SLOTS_JSON):
        with open(SLOTS_JSON, "r") as f:
            drawing_slots = json.load(f)

    print("--- DRAWING MODE ---")
    print("1. Click 4 corners for each parking slot.")
    print("2. Press 'c' to clear all slots and start over.")
    print("3. Press 's' to save and start AI inference.")

    while True:
        temp_frame = first_frame.copy()
        for i, slot in enumerate(drawing_slots):
            pts = np.array(slot, np.int32)
            cv2.polylines(temp_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(temp_frame, str(i), (slot[0][0], slot[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for pt in current_slot:
            cv2.circle(temp_frame, pt, radius=4, color=(0, 0, 255), thickness=-1)
            
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):   # Save and Start
            break
        elif key == ord('c'): # Clear slots
            drawing_slots = []
            current_slot = []
            
    # Save the updated slots to JSON
    with open(SLOTS_JSON, "w") as f:
        json.dump(drawing_slots, f)
        
    cv2.destroyWindow(window_name)
    slot_polygons = [Polygon(coords) for coords in drawing_slots]

    # ---------------------------------------------------------
    # 2. INFERENCE & DATABASE MODE
    # ---------------------------------------------------------
    print("Initializing Database...")
    db_conn = init_db(drawing_slots)
    db_cursor = db_conn.cursor()

    model_path = "yolo11s.pt"
    print(f"Forcing baseline YOLO model: {model_path}")
    model = YOLO(model_path)

    prev_time = 0
    last_db_push = 0
    PUSH_INTERVAL = 2.0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Using predict() for stationary vehicles
        # Using the COCO model, so we must filter for cars (2), motorcycles (3), buses (5), and trucks (7)
        results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)

        # vehicle_boxes = []
        # for r in results:
        #     for box in r.boxes:
        #         x1, y1, x2, y2 = box.xyxy[0].tolist()
        #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        #         vehicle_boxes.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

        # slot_states = []
        # occupied_count = 0

        # for i, slot_poly in enumerate(slot_polygons):
        #     is_occupied = False
        #     pts = np.array(drawing_slots[i], np.int32)

        #     # --- FIX: Automatically resolve self-intersecting 'bowtie' polygons ---
        #     if not slot_poly.is_valid:
        #         slot_poly = slot_poly.buffer(0)

        #     for vehicle_poly in vehicle_boxes:
        #         # Ensure vehicle boxes are also valid
        #         if not vehicle_poly.is_valid:
        #             vehicle_poly = vehicle_poly.buffer(0)

        #         if slot_poly.intersects(vehicle_poly):
        #             coverage = slot_poly.intersection(vehicle_poly).area / slot_poly.area
        #             if coverage > COVERAGE_THRESHOLD:
        #                 is_occupied = True
        #                 break

        #     slot_states.append((i, 1 if is_occupied else 0))
        #     color = (0, 0, 255) if is_occupied else (0, 255, 0)
        #     if is_occupied: occupied_count += 1

        #     cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        #     cv2.putText(frame, str(i), (pts[0][0], pts[0][1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # 1. Raise confidence to 25% to eliminate ghost detections and flickering
        results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)

        vehicle_centers = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate the exact center of the car
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Draw a blue dot at the center of the car instead of a big box
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                
                vehicle_centers.append(Point(cx, cy))

        slot_states = []
        occupied_count = 0

        for i, slot_poly in enumerate(slot_polygons):
            is_occupied = False
            pts = np.array(drawing_slots[i], np.int32)

            # Auto-repair self-intersecting slots
            if not slot_poly.is_valid:
                slot_poly = slot_poly.buffer(0)

            # 2. Check if the car's center dot is inside the parking slot
            for center_pt in vehicle_centers:
                # buffer(10) adds a 10-pixel grace area around your drawn slot
                if slot_poly.buffer(10).contains(center_pt):
                    is_occupied = True
                    break

            slot_states.append((i, 1 if is_occupied else 0))
            
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            if is_occupied: 
                occupied_count += 1

            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, str(i), (pts[0][0], pts[0][1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        current_time = time.time()
        if current_time - last_db_push >= PUSH_INTERVAL:
            now_iso = datetime.now(timezone.utc).isoformat()
            db_values = [(now_iso, state[0], state[1]) for state in slot_states]
            try:
                db_cursor.executemany("INSERT INTO occupancy_events (timestamp, slot_id, is_occupied) VALUES (?, ?, ?)", db_values)
                db_conn.commit()
                last_db_push = current_time
            except Exception as e:
                print(f"DB Write Error: {e}")

        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"Occupied: {occupied_count}/{len(slot_polygons)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("SPIP Live Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    db_conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()