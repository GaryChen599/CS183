# config.py
# Configuration file for EAST text detection + EasyOCR recognition
# 配置文件：EAST文本检测 + EasyOCR识别，集中管理所有参数

# ---- Detector selection / 检测器选择 ----
DETECTOR = "EAST"            # "EAST" or "PADDLE"  选择使用的检测器
USE_PADDLE = True if DETECTOR == "PADDLE" else False   # 自动判断

# ---- Paths / 路径配置 ----
MODEL_PATH = "model/frozen_east_text_detection.pb"   # EAST model path / EAST模型路径
IMAGE_PATH = "images/test1.jpg"                       # Default input image / 默认测试图片
OUTPUT_DIR = "output"                                 # Output directory / 输出目录

# ---- EAST detection parameters / EAST 检测参数 ----
MIN_CONFIDENCE = 0.3     # Confidence threshold (lower = more boxes) / 置信度阈值，调低以检出更多文字
RESIZE_WIDTH = 640       # Must be multiple of 32 / 必须为32的倍数
RESIZE_HEIGHT = 640

# ---- Display & save / 显示与保存 ----
SHOW_RESULT = True       # Pop-up window / 弹窗显示
SAVE_RESULT = True       # Save result image / 保存结果图片

# ---- OCR settings / OCR 识别设置 ----
ENABLE_OCR = True
OCR_LANGUAGES = ['ch_sim', 'en']   # Languages: ['en'] or ['ch_sim','en'] / 识别语言
OCR_GPU = False                      # Use GPU? / 是否使用GPU
SAVE_TEXT_RESULT = True              # Save .txt file / 保存识别文本