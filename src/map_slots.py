# import cv2
# import json
# import numpy as np
# import os

# VIDEO_PATH = os.path.join("data", "clip.mp4")
# OUTPUT_JSON = os.path.join("data", "slot_coordinates.json")

# slots = []
# current_slot = []

# def mouse_click(event, x, y, flags, param):
#     global current_slot, slots
#     if event == cv2.EVENT_LBUTTONDOWN:
#         current_slot.append((x, y))
#         if len(current_slot) == 4:
#             slots.append(current_slot)
#             current_slot = []

# def main():
#     if not os.path.exists(VIDEO_PATH):
#         print(f"Error: Place your test video at '{VIDEO_PATH}'.")
#         return

#     cap = cv2.VideoCapture(VIDEO_PATH)
#     ret, frame = cap.read()
#     cap.release()

#     window_name = "Map Slots - Click 4 Corners per Slot | Press 'q' to Save"
#     cv2.namedWindow(window_name)
#     cv2.setMouseCallback(window_name, mouse_click)

#     while True:
#         img_copy = frame.copy()
#         for i, slot in enumerate(slots):
#             pts = np.array(slot, np.int32)
#             cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
#             cv2.putText(img_copy, f"Slot {i}", (slot[0][0], slot[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         for pt in current_slot:
#             cv2.circle(img_copy, pt, radius=4, color=(0, 0, 255), thickness=-1)

#         cv2.imshow(window_name, img_copy)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     with open(OUTPUT_JSON, "w") as f:
#         json.dump(slots, f)
#     cv2.destroyAllWindows()
#     print(f"Saved {len(slots)} slots to {OUTPUT_JSON}")

# if __name__ == "__main__":
#     main()

import cv2
import json
import numpy as np
import os
 
VIDEO_PATH = os.path.join("data", "clip.mp4")
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
    global slots, current_slot
 
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Place your test video at '{VIDEO_PATH}'.")
        return
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
 
    if not ret or frame is None:
        print(f"Error: Could not read a frame from '{VIDEO_PATH}'. Check the file is a valid video.")
        return
 
    # Load existing slots so re-running doesn't wipe previous work
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            slots = json.load(f)
        print(f"Loaded {len(slots)} existing slots from {OUTPUT_JSON}")
 
    window_name = "Map Slots - click 4 corners | 'z'=undo point | 'c'=clear all | 's'=save+quit | 'q'=quit no save"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)
 
    print("--- MAP SLOTS ---")
    print("1. Click 4 corners per slot, in order.")
    print("2. Press 'z' to undo the last clicked point (before a slot is completed).")
    print("3. Press 'c' to clear ALL slots and start over.")
    print("4. Press 's' to save all slots and exit.")
    print("5. Press 'q' to quit WITHOUT saving.")
 
    while True:
        img_copy = frame.copy()
 
        for i, slot in enumerate(slots):
            pts = np.array(slot, np.int32)
            cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img_copy, f"Slot {i}", (slot[0][0], slot[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
        for pt in current_slot:
            cv2.circle(img_copy, pt, radius=4, color=(0, 0, 255), thickness=-1)
 
        cv2.putText(img_copy, f"Slots marked: {len(slots)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord('s'):
            with open(OUTPUT_JSON, "w") as f:
                json.dump(slots, f)
            print(f"Saved {len(slots)} slots to {OUTPUT_JSON}")
            break
 
        elif key == ord('q'):
            print("Exiting without saving.")
            break
 
        elif key == ord('c'):
            slots = []
            current_slot = []
            print("Cleared all slots.")
 
        elif key == ord('z'):
            if current_slot:
                current_slot.pop()
                print("Undid last point in current slot.")
            elif slots:
                removed = slots.pop()
                print(f"Removed last completed slot: {removed}")
 
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()