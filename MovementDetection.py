import cv2
import numpy as np
from ultralytics import YOLO

# تحميل نموذج YOLOv8
model = YOLO("yolov8n.pt")  # استخدم نسخة أقوى إذا كنت بحاجة لدقة أعلى

# فتح الفيديو
cap = cv2.VideoCapture("imagesForYolo/13425236_3840_2160_60fps.mp4")   # استبدل بمسار الفيديو

# متغيرات لحفظ عدد الأشخاص في كل إطار
previous_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # تشغيل YOLOv8 على الإطار الحالي
    results = model(frame)

    # استخراج عدد الأشخاص فقط ورسم المربعات حولهم
    people_count = 0
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # صنف الأشخاص هو 0 في YOLOv8
            people_count += 1
            # رسم مستطيل حول الشخص
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # استخراج إحداثيات الصندوق
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # رسم المربع باللون الأخضر

    # تحليل التدفق البشري
    if people_count > previous_count + 5:  # زيادة كبيرة = تدفق عالي
        flow_status = " very crowded"
    elif people_count > previous_count:
        flow_status = " increase in flow"
    elif people_count < previous_count - 5:  # انخفاض كبير = خروج جماعي
        flow_status = " sharpe decrease in flow"
    elif people_count < previous_count:
        flow_status = " decrease in flow"
    else:
        flow_status = "stable flow"

    # تحديث العدد السابق
    previous_count = people_count

    # عرض النتائج على الشاشة
    cv2.putText(frame, f" no of people: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f" flow state: {flow_status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # عرض الفيديو مع التحليل
    cv2.imshow("People Flow Detection", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
