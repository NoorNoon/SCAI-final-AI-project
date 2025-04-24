import cv2
import numpy as np
from ultralytics import YOLO

# حملي نموذج YOLO للكشف عن الأشخاص
model = YOLO("yolov8n.pt")  # هذا نموذج جاهز

# حملي الصورة
image_path = "imagesForYolo/fallen2.jpeg"  # غيري الاسم حسب اسم صورتك
image = cv2.imread(image_path)

# شغلي النموذج على الصورة
results = model(image)

fall_detected = False

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # إذا كان الشخص (YOLO class 0 هو "person")
            x1, y1, x2, y2 = box.xyxy[0]
            width = x2 - x1
            height = y2 - y1

            aspect_ratio = width / height

            # إذا الجسم أعرض من طوله بكثير، احتمال يكون طايح
            if aspect_ratio > 1.3:
                fall_detected = True
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(image, "Fall Detected!", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, "Standing", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()