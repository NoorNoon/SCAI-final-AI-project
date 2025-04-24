import cv2
import os
from ultralytics import YOLO

# تحميل الموديل
model = YOLO("yolov8n.pt")  # تأكد إن الموديل موجود بنفس المجلد أو عدل المسار

# مسار المجلد اللي فيه الصور والفيديوهات
folder_path = "imagesForYolo"

# دعم صيغ الصور والفيديوهات
image_exts = ['.jpg', '.jpeg', '.png']
video_exts = ['.mp4', '.avi', '.mov']

# لف على كل الملفات
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    ext = os.path.splitext(filename)[1].lower()

    # إذا الملف صورة
    if ext in image_exts:
        img = cv2.imread(filepath)
        results = model(img)[0]

        for box in results.boxes:
            cls = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{cls} ({conf:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Image Detection", img)
        cv2.waitKey(0)

    # إذا الملف فيديو
    elif ext in video_exts:
        cap = cv2.VideoCapture(filepath)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            count = 0

            for box in results.boxes:
                cls = results.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{cls} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                count += 1

            cv2.putText(frame, f"Total: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Video Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

cv2.destroyAllWindows()
