import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# تحميل YOLOv8
model = YOLO("crowd-counting-umiax/1")

# فتح الكاميرا
cap = cv2.VideoCapture(r"C:\Users\zsaff\PycharmProjects\PythonProject2\crowd counting.v1i.yolov8\train\images\0001_jpg.rf.890a5f9943244f9472a6e1fd32859ea6.jpg")

# تخزين عدد الأشخاص في كل منطقة
zone_counts = defaultdict(list)
frame_counter = 0

# تحديد المناطق يدويًا (مثال)
zones = {
    "A": [(100, 100), (300, 300)],  # مربع منطقة A
    "B": [(400, 100), (600, 300)],  # مربع منطقة B
    "C": [(700, 100), (900, 300)],  # مربع منطقة C
}

def point_in_zone(point, zone):
    """تحقق إذا كانت النقطة داخل المنطقة"""
    (x1, y1), (x2, y2) = zone
    return x1 < point[0] < x2 and y1 < point[1] < y2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    persons = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # فئة "person" في YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append(((x1 + x2) // 2, (y1 + y2) // 2))  # مركز الشخص

    # تحديث عدد الأشخاص في كل منطقة
    counts = {zone: 0 for zone in zones}
    for person in persons:
        for zone_name, zone_coords in zones.items():
            if point_in_zone(person, zone_coords):
                counts[zone_name] += 1

    # حفظ البيانات في القاموس
    timestamp = datetime.now().strftime("%H:%M:%S")
    zone_counts["time"].append(timestamp)
    for zone_name in zones:
        zone_counts[zone_name].append(counts[zone_name])

    # عرض النتائج
    for zone_name, zone_coords in zones.items():
        cv2.rectangle(frame, zone_coords[0], zone_coords[1], (0, 255, 0), 2)
        cv2.putText(frame, f"{zone_name}: {counts[zone_name]}", zone_coords[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# حفظ البيانات في ملف CSV
df = pd.DataFrame(zone_counts)
df.to_csv("traffic_data.csv", index=False)
print("data hase been saved traffic_data.csv")