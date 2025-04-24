import cv2
from ultralytics import YOLO
model = YOLO(r'runs/detect/train38/weights/best.pt')
results = model(r"C:\Users\zsaff\PycharmProjects\PythonProject2\imagesForYolo\ppe-2.mp4", show=True)
cv2.waitKey(0)
