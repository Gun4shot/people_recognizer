import cv2
from ultralytics import YOLO

model = YOLO("rtdetr-l.pt")

cap = cv2.VideoCapture("test2.mp4")

person_ids = set()
bottle_ids = set()

PERSON_CONF = 0.5
BOTTLE_CONF = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.4,              # base filter
        tracker="botsort.yaml"
    )

    for result in results:
        if result.boxes.id is None:
            continue

        for box, track_id in zip(result.boxes, result.boxes.id):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])
            track_id = int(track_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person" and confidence >= PERSON_CONF:
                person_ids.add(track_id)
                color = (0, 0, 255)

            elif label == "bottle" and confidence >= BOTTLE_CONF:
                bottle_ids.add(track_id)
                color = (255, 0, 0)

            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ID:{track_id} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.putText(frame, f"Unique Persons: {len(person_ids)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    cv2.putText(frame, f"Unique Bottles: {len(bottle_ids)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)

    cv2.imshow("Filtered Disaster Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Unique Counts")
print("Persons:", len(person_ids))
print("Bottles:", len(bottle_ids))
