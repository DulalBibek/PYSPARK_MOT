import cv2
from pathlib import Path
import os

raw_dir = Path("/Users/bibekdulal/Desktop/PYSPARK_MOT/visdrone-mot-data/data")
processed_dir = Path("visdrone_processed/sequence_intersection")
os.makedirs(processed_dir, exist_ok=True)

# Filter for all frames ending in '-5.jpg' (Sequence 5) and sort by the leading frame number
frames = sorted(raw_dir.glob("*-5.jpg"), key=lambda x: int(x.stem.split('-')[0]))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for idx, frame_path in enumerate(frames):
    # Temporal Downsampling: Process every 2nd frame
    if idx % 2 != 0:
        continue
        
    img = cv2.imread(str(frame_path))
    
    # Resize to a YOLO-friendly multiple of 32
    target_width, target_height = 1280, 736
    img_resized = cv2.resize(img, (target_width, target_height))
    
    # Apply CLAHE contrast enhancement
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    output_path = processed_dir / f"frame_{idx:04d}.jpg"
    cv2.imwrite(str(output_path), img_enhanced)

print(f"Preprocessing complete. Frames saved to {processed_dir}")