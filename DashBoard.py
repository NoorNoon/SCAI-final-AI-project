import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Dashboard الملعب الذكي")

st.title("Dashboard ⚽")

# تحميل الموديل
model = YOLO(r'runs/detect/train37/weights/best.pt')  # حط اسم موديلك هنا

# رفع صورة أو فيديو
uploaded_file = st.file_uploader("ارفع صورة من الكاميرا أو مشهد من الجمهور", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة الأصلية")

    # نحول الصورة إلى np array
    frame = np.array(image)

    # تحليل الصورة
    results = model(frame)[0]  # نأخذ أول نتيجة

    # رسم البوكسات
    annotated_frame = results.plot()

    # نعرض الصورة بعد التحليل
    st.image(annotated_frame, caption="نتائج التحليل")

    # استخراج البيانات الإحصائية
    classes = results.names
    counts = {}

    for c in results.boxes.cls.cpu().numpy():
        label = classes[int(c)]
        counts[label] = counts.get(label, 0) + 1

    st.subheader("تفاصيل التحليل:")
    for label, count in counts.items():
        st.write(f"{label}: {count}")

    # تحليل المشاعر لو عندك تسميات مثل happy, neutral, sad
    if any(emotion in counts for emotion in ["happy", "neutral", "sad"]):
        st.subheader("تحليل المشاعر:")
        st.bar_chart({k: v for k, v in counts.items() if k in ["happy", "neutral", "sad"]})
