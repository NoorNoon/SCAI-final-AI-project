import cvzone
from ultralytics import YOLO
import cv2
import math



cap = cv2.VideoCapture(0)
model = YOLO("../yolov8_Project/yolo-weights/yolov8n.pt")
#results = model("imagesForYolo/people.jpg",show=True)


classNames = ["person", "chair","sofa","couch","cell phone","parking meter","fire"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h) )
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), scale = 0.7, thickness = 1)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

