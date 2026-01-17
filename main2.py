import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")  # nano version = fast

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = box.conf[0]
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Different color for people
            if label == "person":
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0)  # green

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Disaster Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
