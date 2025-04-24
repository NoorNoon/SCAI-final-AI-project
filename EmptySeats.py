import cv2

from pyzbar.pyzbar import decode

from ultralytics import YOLO

# تحميل نموذج YOLO (تقدر تستخدم "yolov8n.pt" أو مخصص)

model = YOLO("yolov8n.pt")  # تأكد إن الموديل يعرف الكراسي والأشخاص

# فتح الكاميرا

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:

        break

    # تحليل الكاميرا باستخدام YOLO

    results = model(frame)[0]

    chairs = []

    persons = []

    # استخراج الكائنات المكتشفة

    for box in results.boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if "chair" in label.lower() or "seat" in label.lower():

            chairs.append(((x1, y1, x2, y2)))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, "Chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif "person" in label.lower():

            persons.append(((x1, y1, x2, y2)))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # مطابقة الأشخاص مع الكراسي لتحديد المشغولة

    for chair in chairs:

        cx = (chair[0] + chair[2]) // 2

        cy = (chair[1] + chair[3]) // 2

        occupied = False

        for person in persons:

            if person[0] < cx < person[2] and person[1] < cy < person[3]:

                occupied = True

                break

        color = (0, 0, 255) if occupied else (0, 255, 0)

        status = "Occupied" if occupied else "Empty"

        cv2.putText(frame, status, (chair[0], chair[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # قراءة الباركود إن وجد

    barcodes = decode(frame)

    for barcode in barcodes:

        x, y, w, h = barcode.rect

        barcode_data = barcode.data.decode("utf-8")

        barcode_type = barcode.type

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        text = f"{barcode_data} ({barcode_type})"

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Seat Detection + Barcode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows() 