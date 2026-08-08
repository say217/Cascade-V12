import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Load your trained object detection model checkpoint
model_path = '/kaggle/working/runs/detect/flood_detection_delta/yolov8_detection_run/weights/best.pt'
model = YOLO(model_path)

# 2. Specify the path to your test/validation image
sample_image_path = '/kaggle/input/datasets/sayaksamanta/flood-detection/flood detect.v1i.yolov8/train/images/024a082e-1016_jpg.rf.7db0fc1e88dc5097386e89d206f03df6.jpg' 

# 3. Perform prediction
results = model.predict(source=sample_image_path, imgsz=640, conf=0.2, verbose=False)

# 4. Load the original image manually
img = cv2.imread(sample_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 5. Extract detection data and draw thin boxes with embedded bar indicators
boxes = results[0].boxes
for box in boxes:
    # Get coordinates, confidence, and class id
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    
    # Draw thin, white bounding box
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
    
    # Text string structure
    label_text = f"flood {conf*100:.1f}% "
    
    # Calculate baseline text requirements to orient the indicator bar
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    
    # Define text coordinate positions dynamically above the box boundary
    text_x = x1 + 2
    text_y = y1 - 6 if y1 - 6 > 15 else y1 + text_h + 4
    
    # Optional text backing layout for clarity in highly chaotic backgrounds
    cv2.rectangle(img_rgb, (x1, text_y - text_h - 2), (x2, text_y + baseline), (28, 28, 28), -1)
    
    # Render string element
    cv2.putText(img_rgb, label_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    
    # Progress bar sizing configurations dynamically scaling beside the text layout
    bar_x_start = text_x + text_w + 2
    bar_max_width = max(30, (x2 - bar_x_start) - 4)  # Constraints width within bounding boundary
    bar_width = int(bar_max_width * conf)
    bar_height = 5
    
    # Draw background track for progress bar
    cv2.rectangle(img_rgb, (bar_x_start, text_y - text_h + 1), 
                  (bar_x_start + bar_max_width, text_y - text_h + 1 + bar_height), (85, 85, 85), -1)
    
    # Draw fill value track (Solid stark white line representing total model confidence matching box)
    cv2.rectangle(img_rgb, (bar_x_start, text_y - text_h + 1), 
                  (bar_x_start + bar_width, text_y - text_h + 1 + bar_height), (255, 255, 255), -1)

# 6. Render the final sleek image layout inside your notebook
plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()