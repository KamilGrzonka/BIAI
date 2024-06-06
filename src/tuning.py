from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Tuning the model
model.tune(data="data/animals.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)