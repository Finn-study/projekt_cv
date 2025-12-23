from ultralytics import YOLO
import cv2, time, json, os

# Klassen-Liste
CLASSES = ["stop","give_way","no_entry","priority_road","keep_right","go_straight"]

# Pfade & Parameter
MODEL_PATH = "models/best.pt"
IMGSZ = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Modell laden
model = YOLO(MODEL_PATH)

# Kamera öffnen
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

log = []

def draw_boxes(frame, result):
    """Zeichnet gefilterte Boxen und Labels ins Frame."""
    for b in result.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        name = result.names[cls_id]

        # Box-Koordinaten
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        # Rechteck & Label zeichnen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# Main Loop
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    # Inferenz
    results = model(frame, imgsz=IMGSZ, conf=0.25, iou=0.45, verbose=False)
    r = results[0]

    # Overlay zeichnen
    vis = draw_boxes(frame.copy(), r)

    # FPS berechnen
    fps = 1.0 / max(1e-6, time.time() - t0)
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Log speichern
    log.append({
        "ts": time.time(),
        "fps": fps,
        "n_det": len(r.boxes)})

    # Fenster anzeigen
    cv2.imshow("Traffic Signs (Realtime)", vis)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Metrics speichern
os.makedirs("reports", exist_ok=True)
with open("reports/realtime_metrics.json", "w") as f:
    json.dump(log, f, indent=2)
print("Metrics wurden gespeichert.")
