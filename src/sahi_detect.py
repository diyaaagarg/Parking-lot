from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
 
 
def build_sahi_model(model_path: str, conf_threshold: float = 0.25, device: str = "cpu"):
    """
    Build a SAHI-wrapped YOLO model once, outside your main loop
    (this is relatively slow to initialize, so don't call it per frame).
 
    device: "cpu" or "cuda:0" if you have a GPU available.
    """
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device,
    )
 
 
def get_sliced_detections(sahi_model, frame, slice_height=512, slice_width=512,
                           overlap_ratio=0.2, vehicle_class_names=None):
    """
    Run sliced inference on a single frame and return a list of detections:
        [{"bbox": (x1, y1, x2, y2), "class_name": str, "confidence": float}, ...]
 
    Tune slice_height/slice_width smaller (e.g. 384) if cars are still missed,
    or larger (e.g. 640) if inference becomes too slow.
    """
    if vehicle_class_names is None:
        vehicle_class_names = {"car", "truck", "bus", "motorcycle", "vehicle"}
 
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0,
    )
 
    detections = []
    for pred in result.object_prediction_list:
        cls_name = pred.category.name.lower()
        if cls_name not in vehicle_class_names:
            continue
 
        bbox = pred.bbox  # has minx, miny, maxx, maxy
        detections.append({
            "bbox": (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy),
            "class_name": cls_name,
            "confidence": pred.score.value,
        })
 
    return detections
 