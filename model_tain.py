import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Initialize YOLOv11 and point to the preprocessed directory
model = YOLO("yolo11n.pt")
processed_dir = Path("visdrone_processed/sequence_intersection")

# Sort the preprocessed frames chronologically
frames = sorted(processed_dir.glob("*.jpg"))
telemetry_data = []

for frame_path in frames:
    frame_id = int(frame_path.stem.split('_')[1])
    frame = cv2.imread(str(frame_path))
    
    # Run ByteTrack on the Apple Silicon MPS backend
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", device="mps", verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            # Isolate traffic by filtering for cars (2), vans (3), buses (4), and trucks (5)
            if cls in [2, 3, 4, 5]: 
                telemetry_data.append({
                    "frame_id": frame_id,
                    "vehicle_id": track_id,
                    "class_id": cls,
                    "x_center": box[0],
                    "y_center": box[1]
                })

df = pd.DataFrame(telemetry_data)
df.to_parquet("sequence_intersection_telemetry.parquet", index=False)
print(f"Extraction complete. {len(df)} tracking points saved to Parquet.")