import os
import json
import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from src.config import YOLO_MODEL_PATH, FALLBACK_MODEL_PATH

class ParkingSlotDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25, coverage_threshold: float = 0.15):
        if model_path is None:
            model_path = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else FALLBACK_MODEL_PATH

        print(f"[CV Engine] Loading YOLO Model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.coverage_threshold = coverage_threshold

    def detect_frame_occupancy(self, frame: np.ndarray, slot_polygons_coords: list) -> tuple:
        """
        Process a single image frame against a list of slot polygon coordinates.
        Returns:
            (slot_states: list of (slot_index, is_occupied: int 0/1),
             annotated_frame: np.ndarray)
        """
        # Run YOLO vehicle detection
        results = self.model.predict(frame, conf=self.conf_threshold, imgsz=640, verbose=False)

        vehicle_centers = []
        vehicle_polygons = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                vehicle_centers.append(Point(cx, cy))
                vehicle_polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

                # Visual overlay dot on vehicle center
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        slot_states = []
        occupied_count = 0

        for i, coords in enumerate(slot_polygons_coords):
            slot_poly = Polygon(coords)
            if not slot_poly.is_valid:
                slot_poly = slot_poly.buffer(0)

            is_occupied = False

            # Center-point in polygon algorithm with grace buffer
            for center_pt in vehicle_centers:
                if slot_poly.buffer(10).contains(center_pt):
                    is_occupied = True
                    break

            # Fallback IoU coverage algorithm
            if not is_occupied:
                for v_poly in vehicle_polygons:
                    if not v_poly.is_valid:
                        v_poly = v_poly.buffer(0)
                    if slot_poly.intersects(v_poly):
                        coverage = slot_poly.intersection(v_poly).area / slot_poly.area
                        if coverage >= self.coverage_threshold:
                            is_occupied = True
                            break

            is_occ_int = 1 if is_occupied else 0
            slot_states.append((i, is_occ_int))
            if is_occupied:
                occupied_count += 1

            # Render slot boundaries on frame
            pts = np.array(coords, np.int32)
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, f"S{i+1}", (pts[0][0], pts[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Overlay total count
        cv2.putText(frame, f"Occupied: {occupied_count}/{len(slot_polygons_coords)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return slot_states, frame
