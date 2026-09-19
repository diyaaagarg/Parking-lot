import os
from ultralytics import YOLO

def main():
    # Dynamically calculate the path to data.yaml so it works perfectly in VS Code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(current_dir, "..", "data", "PKLot-1", "data.yaml"))

    if not os.path.exists(yaml_path):
        print(f"Error: Could not find {yaml_path}.")
        print("Did you run download_data.py first?")
        return

    print(f"Loading YOLO11 Nano and starting training with {yaml_path}...")

    # 1. Load the official YOLO11 Nano pre-trained weights
    model = YOLO("yolo11n.pt") 

    # 2. Fine-tune the model on the parking dataset
    results = model.train(
        data=yaml_path,
        epochs=20,            # 20 epochs is a fast, solid baseline for this project
        imgsz=640,            # Standard input size for YOLO
        batch=16,             # Batch size (lower this to 8 if you get an Out of Memory error)
        name="spip_pklot",    # The folder name where your new weights will be saved
        device="auto"         # Ultralytics will automatically use your GPU if available, else CPU
    )

    print("\nTraining Complete!")
    print("Your new custom weights are saved in: runs/detect/spip_pklot/weights/best.pt")

if __name__ == "__main__":
    main()