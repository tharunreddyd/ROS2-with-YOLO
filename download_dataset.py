from roboflow import Roboflow
import os

# Create a specific directory for the dataset
dataset_dir = os.path.expanduser("~/mars_rocks_dataset")
os.makedirs(dataset_dir, exist_ok=True)

# Initialize Roboflow
rf = Roboflow(api_key="T6LNA9woVJklhz1BXtmf")
project = rf.workspace().project("mars-rocks-detection")

# Download dataset with explicit path
dataset = project.version(4).download("yolov8", location=dataset_dir)

print(f"Dataset downloaded to: {dataset_dir}")
