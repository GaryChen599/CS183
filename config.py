# config.py
# Configuration file for EAST text detection project
# 配置文件：集中管理所有参数，修改这里即可改变程序行为

# ---- Paths / 路径配置 ----
MODEL_PATH = "model/frozen_east_text_detection.pb"   # Path to pre-trained model 预训练模型路径
IMAGE_PATH = "images/test2.jpg"                       # Input image to detect 待检测图片路径
OUTPUT_DIR = "output"                                # Directory to save result images 结果输出目录

# ---- Detection parameters / 检测参数 ----
MIN_CONFIDENCE = 0.5     # Confidence threshold (0~1) 置信度阈值，越低框越多
RESIZE_WIDTH = 320       # Resized width, must be multiple of 32 缩放宽度（必须为32倍数）
RESIZE_HEIGHT = 320      # Resized height, must be multiple of 32 缩放高度（必须为32倍数）

# ---- Display & save / 显示与保存 ----
SHOW_RESULT = True       # Whether to pop up result window 是否弹窗显示结果
SAVE_RESULT = True       # Whether to save the output image 是否保存结果图片