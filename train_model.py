from ultralytics import YOLO

model = YOLO("../yolo-Weights/yolov8n.pt")

# تدريب النموذج
model.train(data=r"C:/Users/zsaff/PycharmProjects/PythonProject2/Human-Positions-7/data.yaml", epochs=50, batch=16)

# التنبؤ
model.predict(source="C:/Users/zsaff/Desktop/ProjectRelatedData/valid/images", show=True)

# حفظ النموذج
model.save("trained_model.pt")
