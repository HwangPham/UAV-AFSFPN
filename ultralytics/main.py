from ultralytics import YOLO
# training
model = YOLO("ultralytics/backbone/yolov8.yaml")
results = model.train(data="../dataset", epochs=3, imgsz=640)