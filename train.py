from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load an official model
model.train(
    data="D:/ids project/dataset/data.yaml",  # Path to your dataset configuration
    epochs=35,                 # Number of epochs to train
    batch=16,                  # Batch size
    imgsz=480,                 # Image size
    workers=4                  # Number of data loader workers
)