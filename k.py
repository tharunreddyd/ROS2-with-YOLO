import cv2
from ultralytics import YOLO

# Define paths
model_path = "runs/detect/train/weights/best.pt"  
input_video_path = "/home/roopi66/mars_rocks/synthetic_video.mp4"
output_video_path = "/home/roopi66/mars_rocks/detection_results.mp4"

# Step 1: Load YOLO model
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Step 2: Open input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Unable to open input video.")
    exit()

# Step 3: Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

# Step 4: Initialize video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
print(f"Processing video. Output will be saved to: {output_video_path}")

# Step 5: Process video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Finished processing {frame_count} frames.")
        break

    frame_count += 1
    try:
        # Convert frame to RGB (required by YOLO model)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model.predict(rgb_frame)

        # Draw bounding boxes on the frame
        for pred in results[0].boxes:
            x1, y1, x2, y2 = map(int, pred.xyxy[0].tolist())
            conf = pred.conf[0].item()
            cls = int(pred.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")

# Step 6: Release resources
cap.release()
out.release()

print(f"Video processing completed. Output saved to: {output_video_path}")
