import cv2
from ultralytics import YOLO
# تحميل الموديل
model = YOLO("yolov8n.pt")  # تأكد أنه يتعرف على السيارات
# فتح الكاميرا أو فيديو
cap = cv2.VideoCapture(0)  # استبدل بـ "parking.mp4" لو عندك ملف
# إعداد الشبكة (عدد الصفوف × الأعمدة لتقسيم الصورة)
GRID_ROWS = 2
GRID_COLS = 4
while True:
   ret, frame = cap.read()
   if not ret:
       break
   height, width = frame.shape[:2]
   results = model(frame)[0]
   cars = []
   for box in results.boxes:
       cls = int(box.cls[0])
       label = model.names[cls]
       x1, y1, x2, y2 = map(int, box.xyxy[0])
       if "car" in label.lower():
           cars.append(((x1 + x2) // 2, (y1 + y2) // 2))  # نحفظ مركز كل سيارة
   # تقسيم الصورة لشبكة مواقف
   cell_w = width // GRID_COLS
   cell_h = height // GRID_ROWS
   for row in range(GRID_ROWS):
       for col in range(GRID_COLS):
           x1 = col * cell_w
           y1 = row * cell_h
           x2 = x1 + cell_w
           y2 = y1 + cell_h
           cell_center = ((x1 + x2) // 2, (y1 + y2) // 2)
           # نشوف إذا فيه سيارة داخل هذه الخانة
           occupied = False
           for car_center in cars:
               cx, cy = car_center
               if x1 < cx < x2 and y1 < cy < y2:
                   occupied = True
                   break
           color = (0, 0, 255) if occupied else (0, 255, 0)
           label = "Occupied" if occupied else "Empty"
           cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
           cv2.putText(frame, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
   cv2.imshow("Parking Spot Detection (Auto Grid)", frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()