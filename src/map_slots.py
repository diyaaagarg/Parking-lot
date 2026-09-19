import cv2
import json
import numpy as np
import os

VIDEO_PATH = os.path.join("data", "test_lot.mp4")
OUTPUT_JSON = os.path.join("data", "slot_coordinates.json")

slots = []
current_slot = []

def mouse_click(event, x, y, flags, param):
    global current_slot, slots
    if event == cv2.EVENT_LBUTTONDOWN:
        current_slot.append((x, y))
        if len(current_slot) == 4:
            slots.append(current_slot)
            current_slot = []

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Place your test video at '{VIDEO_PATH}'.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    window_name = "Map Slots - Click 4 Corners per Slot | Press 'q' to Save"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    while True:
        img_copy = frame.copy()
        for i, slot in enumerate(slots):
            pts = np.array(slot, np.int32)
            cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img_copy, f"Slot {i}", (slot[0][0], slot[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for pt in current_slot:
            cv2.circle(img_copy, pt, radius=4, color=(0, 0, 255), thickness=-1)

        cv2.imshow(window_name, img_copy)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    with open(OUTPUT_JSON, "w") as f:
        json.dump(slots, f)
    cv2.destroyAllWindows()
    print(f"Saved {len(slots)} slots to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()