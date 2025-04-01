import cv2
import torch
import numpy as np
from tqdm import tqdm  # For progress bar

print("Loading your trained model...")
try:
    model = torch.jit.load('runs/detect/train/weights/best.pt')
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Input and output video paths
video_path = '/home/roopi66/mars_rocks/synthetic_video.mp4'
output_path = 'rocks_detected.mp4'  # You can change this name
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create video writer with MP4V codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Video will be saved as: {output_path}")
print(f"Frame size: {frame_width}x{frame_height}")
print(f"FPS: {fps}")

def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img

def draw_detections(img, output, conf_threshold=0.25):
    img_copy = img.copy()
    height, width = img_copy.shape[:2]
    
    try:
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        
        if isinstance(output, (list, tuple)):
            output = output[0]
            
        if output is not None and len(output.shape) >= 2:
            for det in output:
                if len(det) >= 5:
                    conf = float(det[4])
                    
                    if conf > conf_threshold:
                        x1 = int(det[0] * width / 640)
                        y1 = int(det[1] * height / 640)
                        x2 = int(det[2] * width / 640)
                        y2 = int(det[3] * height / 640)
                        
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_copy, f'Rock {conf:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error in draw_detections: {e}")
    
    return img_copy

print("Processing video...")

try:
    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        input_tensor = preprocess_image(frame)
        with torch.no_grad():
            predictions = model(input_tensor)
        annotated_frame = draw_detections(frame, predictions)
        
        # Save frame
        out.write(annotated_frame)
        
        # Display frame (optional - you can comment this out to process faster)
        cv2.imshow('Rock Detection', annotated_frame)
        
        # Update progress bar
        pbar.update(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessing interrupted by user")
            break

except Exception as e:
    print(f"\nError during processing: {e}")

finally:
    # Clean up
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing completed. Video saved as: {output_path}")
    print("Note: If the saved video doesn't play, you may need to convert it using:")
    print(f"ffmpeg -i {output_path} -vcodec libx264 converted_{output_path}")
