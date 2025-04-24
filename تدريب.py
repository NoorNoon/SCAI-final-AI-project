from ultralytics import YOLO

# تحميل النموذج
model = YOLO('yolov8n.pt')  # نستخدم نسخة صغيرة وخفيفة

# تدريب النموذج
model.train(
    data=r"C:\Users\zsaff\PycharmProjects\PythonProject2\Human-Positions-7\data.yaml",
    epochs=25,
    imgsz=416,          # تصغير الصور
    batch=4,            # دفعة صغيرة لتخفيف الضغط
    val=True,           # نسمح بالتحقق، لكن بعد ما نخفف الصور
)
