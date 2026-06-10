"""
file name/文件名: config.py
Responsible team member/负责团队成员: Huang Jiancheng
Description/描述:Configuration file for EAST text detection + EasyOCR recognition/用于EAST文本检测 + EasyOCR识别的配置文件

"""

# ---- Detector selection / 检测器选择 ----
DETECTOR = "EAST"            # "EAST" or "PADDLE"  选择使用的检测器
USE_PADDLE = True if DETECTOR == "PADDLE" else False   # 自动判断

# ---- Paths / 路径配置 ----
MODEL_PATH = "model/frozen_east_text_detection.pb"   # EAST model path / EAST模型路径
IMAGE_PATH = "images/test2.jpg"                       # Default input image / 默认测试图片
OUTPUT_DIR = "output"                                 # Output directory / 输出目录

# ---- EAST detection parameters / EAST 检测参数 ----
MIN_CONFIDENCE = 0.3     # Confidence threshold (lower = more boxes) / 置信度阈值，调低以检出更多文字
RESIZE_WIDTH = 640       # Must be multiple of 32 / 必须为32的倍数
RESIZE_HEIGHT = 640

SHOW_RESULT = True       # Pop-up window / 弹窗显示
SAVE_RESULT = True       # Save result image / 保存结果图片

# ---- OCR settings / OCR 识别设置 ----
ENABLE_OCR = True
OCR_LANGUAGES = ['ch_sim', 'en']   # Languages: ['en'] or ['ch_sim','en'] / 识别语言
OCR_GPU = False                      # Use GPU? / 是否使用GPU
SAVE_TEXT_RESULT = True              # Save .txt file / 保存识别文本# ---- Display & save / 显示与保存 ----

"""
Process demonstration:
1. Load image from IMAGE_PATH
         ↓
2. Resize image to RESIZE_WIDTH × RESIZE_HEIGHT (must be multiples of 32)
         ↓
3. Run EAST text detection using MODEL_PATH
         ↓
4. Filter detected boxes by MIN_CONFIDENCE
   (Lower threshold → more boxes, but may include false positives)
         ↓
5. For each detected text region:
   - Crop the region from the original image
   - Pass it to EasyOCR for recognition
   - Use OCR_LANGUAGES to determine which languages to recognize
   - Use OCR_GPU to decide CPU or GPU processing
         ↓
6. Output results:
   - If SHOW_RESULT is True → Pop-up window with bounding boxes
   - If SAVE_RESULT is True → Save image with drawn boxes to OUTPUT_DIR
   - If SAVE_TEXT_RESULT is True → Save recognized text as .txt file
         ↓
7. (Optional) If ENABLE_OCR is False → Only detect boxes, skip text recognition 
    ↓ 
"""