# Football-Detector
- 能够对足球比赛视频中的运动员进行识别，并且显示球员热力图、距离等指标
- 使用streamlit搭建前端，yolov5进行目标检测，opencv进行视频处理

## 如何运行本项目
- 输入`pip install -r requirements.txt`安装所有依赖
- 在项目根目录下，启动终端运行命令`python train.py`进行训练
- 训练好的模型文件`XXXX.pt`将存放在`models/`下，将其改名为`best_yolo.py`
- 在项目根目录下，启动终端运行命令`streamlit run run.py`，即可进入前端界面 

## 参考资料
- [football_analysis](https://github.com/abdullahtarek/football_analysis)
