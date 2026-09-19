from ultralytics import YOLO

def evaluate_baseline():
    model = YOLO("yolo11n.pt") 
    # Update path to a valid YOLO dataset YAML (like PKLot)
    dataset_yaml = "data/pklot_data.yaml" 
    
    print("Starting evaluation...")
    try:
        metrics = model.val(data=dataset_yaml, split='test', classes=[2, 3, 5, 7]) 
        print(f"mAP50-95: {metrics.box.map}")
        print(f"mAP50: {metrics.box.map50}")
        print(f"Precision: {metrics.box.p[0]}")
        print(f"Recall: {metrics.box.r[0]}")
    except Exception as e:
        print("Evaluation requires a YOLO-formatted dataset.yaml. Details:", e)

if __name__ == "__main__":
    evaluate_baseline()