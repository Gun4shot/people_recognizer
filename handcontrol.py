import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.7, maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    cv2.imshow("Hand Test", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
