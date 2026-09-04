import cv2
from pathlib import Path
import os

raw_dir = Path("/Users/bibekdulal/Desktop/PYSPARK_MOT/visdrone-mot-data/data")
processed_dir = Path("visdrone_processed/sequence_intersection")
os.makedirs(processed_dir, exist_ok=True)

# Matches every frame belonging to video 4, sorted in chronological order
frames = sorted(raw_dir.glob("*-4.jpg"), key=lambda x: int(x.stem.split('-')[0]))

print(f"Found {len(frames)} frames for this sequence.")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for idx, frame_path in enumerate(frames):
    # Downsample: take every 2nd frame (12 FPS equivalent)
    if idx % 2 != 0:
        continue
        
    img = cv2.imread(str(frame_path))
    if img is None:
        continue
    
    # Resize to standard YOLO dimensions
    img_resized = cv2.resize(img, (1280, 736))
    
    # CLAHE contrast enhancement on the Lightness channel
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    output_path = processed_dir / f"frame_{idx:04d}.jpg"
    cv2.imwrite(str(output_path), img_enhanced)

print(f"Preprocessing complete. Frames saved to {processed_dir}")