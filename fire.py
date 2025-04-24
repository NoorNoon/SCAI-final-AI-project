from ultralytics import YOLO
import cv2
import time

# تحميل النموذج المدرب مسبقًا للكشف عن الحريق
model = YOLO("../yolo-Weights/yolov8n.pt")  # درب عليه مسبقاً على اللهب، الدخان، التوهج إلخ

# فتح الكاميرا أو فيديو
cap = cv2.VideoCapture()  # أو ضع مسار الفيديو

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تحليل الصورة
    results = model(frame)

    # عرض النتائج
    annotated_frame = results[0].plot()

    # التحقق من وجود حريق بناءً على التصنيفات
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            if cls in ['flame', 'smoke', 'glow', 'ash', 'black_spot']:
                print(f"Warning! Detected: {cls}")

    cv2.imshow("Fire Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()