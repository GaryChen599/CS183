# Text Detection using EAST

本项目使用预训练的 EAST 模型检测自然场景图片中的文字区域，并以绿色矩形框标出。

## 环境要求
- Python 3.7+
- OpenCV (opencv-python)
- imutils
- numpy

安装依赖：
pip install opencv-python imutils numpy

## 使用方法
1. 下载 EAST 模型文件 `frozen_east_text_detection.pb` 放入 `model/` 目录。
   下载链接：https://github.com/oyyd/frozen_east_text_detection.pb
2. 将待检测图片放入 `images/` 目录。
3. 修改 `config.py` 中的 `IMAGE_PATH` 为你的图片路径。
4. 运行：
python main.py
5. 结果图片保存在 `output/` 目录。

## 配置说明
在 `config.py` 中可调整：
- `MIN_CONFIDENCE`: 置信度阈值（默认0.5），降低可检测到更多文字但可能误检。
- `RESIZE_WIDTH / RESIZE_HEIGHT`: 缩放尺寸，必须是32的倍数。