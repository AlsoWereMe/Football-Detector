import os
from ..config import Config
from ultralytics import YOLO
from roboflow import Roboflow

# def download_dataset():
#     """下载Roboflow数据集"""
#     rf = Roboflow(api_key=Config.ROBOFLOW_API_KEY)
#     project = rf.workspace(Config.ROBOFLOW_WORKSPACE).project(Config.ROBOFLOW_PROJECT)
#     dataset = project.version(Config.ROBOFLOW_VERSION).download("yolov5")
#     return dataset.location


def train_model():
    """训练YOLO模型"""

    # 使用YOLOv5模型进行训练
    model = YOLO(Config.BASE_MODEL)

    results = model.train(
        data=Config.DATASET_PATH,
        epochs=Config.EPOCHS,
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH_SIZE,
        name=Config.EXPERIMENT_NAME,
        project=Config.OUTPUT_MODEL_DIR,
    )
    return results
