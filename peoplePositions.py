from roboflow import Roboflow

# مفتاح الـ API
rf = Roboflow(api_key="GYKIWryeGmDS01m4iqvM")

# تحميل المشروع ونسخته
project = rf.workspace("t2c").project("human-positions")
version = project.version(7)

# تحميل نسخة YOLOv8
dataset_yolo = version.download("yolov8")

# تحميل نسخة CreateML (لو حاب تستخدمها أيضاً)
dataset_createml = version.download("createml")

