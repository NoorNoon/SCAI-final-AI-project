from ultralytics import YOLO
import cv2

# تحميل النموذج المدرب
model = YOLO("yolov8n.pt")

# زيادة الحساسية
model.overrides['conf'] = 0.25  # الافتراضي عادة 0.25 - جربي تخفضينه أكثر إذا تبين حساسية أكثر

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تكبير الصورة للكشف عن التفاصيل الصغيرة
    resized_frame = cv2.resize(frame, None, fx=1.5, fy=1.5)

    results = model(resized_frame)
    annotated_frame = results[0].plot()

    # التأكد من وجود نار
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if hasattr(box, 'cls') and box.cls is not None:
                    cls_id = int(box.cls[0])
                    class_name = model.names.get(cls_id, "unknown")
                    if class_name == 'fire':
                        print("🔥 Warning: Small or color-varied fire detected!")

    cv2.imshow("Fire Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
