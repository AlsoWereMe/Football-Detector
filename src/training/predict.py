from ultralytics import YOLO
import os
from config import Config
import time
import json


def predict():
    # 初始化配置
    input_video = os.path.join(Config.INPUT_VIDEO_DIR, "08fd33_0.mp4")
    output_dir = Config.OUTPUT_PREDICT_DIR
    model_path = Config.MODEL_PATH

    # 加载模型
    print(f"加载模型: {model_path}")
    start_load = time.time()
    model = YOLO(model_path)
    print(f"模型加载完成，耗时: {time.time() - start_load:.2f}秒")

    # 执行预测
    results = model.predict(
        source=input_video,
        save=True,
        save_txt=True,  # 保存检测结果的文本文件
        save_crop=True,  # 保存裁剪的检测对象
        project=output_dir,
        name="predictions",
        show_labels=True,
        show_conf=True,
        stream=False,
    )

    # 处理并保存结果
    if results:
        # 保存结果统计信息
        result_stats = {
            "video": input_video,
            "frame_count": len(results),
            "detection_count": sum(len(result.boxes) for result in results if result.boxes),
            "class_distribution": {},
        }

        # 类别分布
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    result_stats["class_distribution"].setdefault(cls_id, 0)
                    result_stats["class_distribution"][cls_id] += 1

        # 保存统计信息
        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(result_stats, f, indent=4)

        print(f"结果统计已保存至: {stats_path}")

        # 打印简要信息
        print("\n预测结果摘要:")
        print(f"处理帧数: {len(results)}")
        print(f"总检测对象: {result_stats['detection_count']}")
        print("类别分布:")
        for cls_id, count in result_stats["class_distribution"].items():
            class_name = model.names[cls_id]
            print(f"  {class_name} ({cls_id}): {count}个")
    else:
        print("未检测到任何结果")


if __name__ == "__main__":
    predict()
