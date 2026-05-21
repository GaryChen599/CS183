# Text Detection and Recognition using EAST + EasyOCR

本项目使用预训练的 EAST 模型检测自然场景图片中的文字区域，并结合 EasyOCR 对检测到的文字进行识别。最终在原图上标出文本框和识别结果，同时将识别文字保存为文本文件。

This project uses a pre-trained EAST model to detect text regions in natural scene images, and integrates EasyOCR to recognise the text inside each detected box. The results are drawn on the image and saved to a `.txt` file.

---

## 环境要求 / Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- imutils
- numpy
- easyocr

安装依赖 / Install dependencies:

```bash
pip install opencv-python imutils numpy easyocr

text_detection_east/
├── model/
│   └── frozen_east_text_detection.pb   # EAST 预训练模型
├── images/                             # 存放待检测图片
├── output/                             # 输出结果（图片 + 文本）
├── config.py                           # 配置文件
├── detector.py                         # EAST 文本检测器
├── ocr_recognizer.py                   # EasyOCR 识别器（Stage 3 新增）
├── main.py                             # 主入口
└── README.md

使用方法 / How to Run
下载 EAST 模型文件 frozen_east_text_detection.pb，放入 model/ 目录。
下载链接：https://github.com/oyyd/frozen_east_text_detection.pb

将待检测图片放入 images/ 目录。

修改 config.py 中的 IMAGE_PATH 为你的图片路径。

运行主程序：

bash
python main.py
结果图片保存在 output/ 目录，识别文字保存在 output/xxx_text.txt。

配置说明 / Configuration (config.py)
参数	说明	默认值
MODEL_PATH	EAST 模型文件路径	model/frozen_east_text_detection.pb
IMAGE_PATH	待检测图片路径	images/test1.jpg
OUTPUT_DIR	输出目录	output
MIN_CONFIDENCE	文本检测置信度阈值	0.5
RESIZE_WIDTH	网络输入宽度（32 的倍数）	320
RESIZE_HEIGHT	网络输入高度（32 的倍数）	320
SHOW_RESULT	是否弹窗显示结果	True
SAVE_RESULT	是否保存结果图片	True
ENABLE_OCR	是否执行文字识别	True
OCR_LANGUAGES	识别语言列表	['ch_sim', 'en']
OCR_GPU	是否使用 GPU（需 CUDA）	False
SAVE_TEXT_RESULT	是否保存识别文字到 txt 文件	True

OCR Results for images/test1.jpg
==================================================
Box 1: (26, 82, 77, 93) -> Today's Activity (conf: 1.00)
Box 2: (204, 200, 243, 212) -> 5.01km (conf: 1.00)
...
Total recognized: 15/21

