from ultralytics import YOLO

model = YOLO("yolov12n.pt")

DATA = "data.yaml"

results = model.train(
    data=DATA,
    imgsz=640,
    epochs=100,
    patience=35,
    batch=16,
    device=0,
    workers=2,

    lr0=0.001,
    lrf=0.001,

    rect=True,
    mosaic=1.0,
    close_mosaic=15,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    fliplr=0.0,
    flipud=0.0
)

print("Training erfolgreich beendet.")
