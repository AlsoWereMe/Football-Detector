class Config:
    # YOLOv5训练配置
    DATASET_PATH = "data/football-players-detection-9/data.yaml"
    BASE_MODEL = "yolov5x.pt"
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 16
    EXPERIMENT_NAME = "football_detector"

    # 目录
    STUB_DIR = "data/stubs"
    INPUT_VIDEO_DIR = "data/raw/videos"
    OUTPUT_VIDEO_DIR = "data/output/videos"
    OUTPUT_MODEL_DIR = "models"
    OUTPUT_PREDICT_DIR = "data/output/predictions"
    OUTPUT_HEATMAPS_DIR = "data/output/heatmaps"

    # 路径
    MODEL_PATH = "models/best_yolo.pt"
    
    # Roboflow配置
    ROBOFLOW_API_KEY = "Dzyc1PNrMrH36GR1E0w1"
    ROBOFLOW_WORKSPACE = "roboflow-jvuqo"
    ROBOFLOW_PROJECT = "football-players-detection-3zvbc"
    ROBOFLOW_VERSION = 9
