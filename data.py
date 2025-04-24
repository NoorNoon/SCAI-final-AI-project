from roboflow import Roboflow

rf = Roboflow(api_key="GYKIWryeGmDS01m4iqvM")
project = rf.workspace("zainabsaeed").project("deteksiparkirkosong-nw1td")
version = project.version(1)
dataset = version.download("yolov8")
