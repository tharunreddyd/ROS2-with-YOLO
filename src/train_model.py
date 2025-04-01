from ultralytics import YOLO
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a new YOLO model
model = YOLO('yolov8n.pt')  # Using the nano model to start

# Configure training
results = model.train(
    data=os.path.join(current_dir, 'data.yaml'),  # Path to your data.yaml
    epochs=50,          # Number of training epochs
    imgsz=640,         # Image size
    batch=8,           # Batch size (start small, increase if you have good GPU)
    patience=20,       # Early stopping patience
    name='mars_rocks', # Name for this training run
    device='0',        # Use GPU if available, 'cpu' if not
    save=True          # Save the trained model
)
