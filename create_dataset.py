import os
import pickle
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = './data'

# Create hand landmarker with the new MediaPipe API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       min_hand_detection_confidence=0.3)
detector = vision.HandLandmarker.create_from_options(options)

data = []
labels = []

print("Processing images...")

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path) or dir_.startswith('.'):
        continue
    
    print(f"Processing class {dir_}...")
    processed_count = 0
    
    for img_path in os.listdir(dir_path):
        if not img_path.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"  Skipped {img_path} (could not read)")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)
        
        results = detector.detect(mp_image)
        
        if results.hand_landmarks:
            # Use only the first detected hand
            hand_landmarks = results.hand_landmarks[0]
            
            for landmark in hand_landmarks:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            processed_count += 1
        else:
            print(f"  Skipped {img_path} (no hand detected)")
    
    print(f"  Processed {processed_count} images from class {dir_}")

print(f"\nTotal samples collected: {len(data)}")

if len(data) > 0:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"✅ Dataset created successfully with {len(data)} samples!")
    print(f"Saved to: data.pickle")
else:
    print("❌ No data collected. Please check your images.")
