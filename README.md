# Text Detection and Recognition with EAST + EasyOCR
# 基于EAST与EasyOCR的文本检测与识别

This project detects text regions in natural scene images using the EAST model, then recognizes the text content using EasyOCR.  
A preprocessing module (skew correction + CLAHE) improves accuracy on tilted or low-contrast text.  
Both CLI and web interfaces are provided; the web server is built solely on Python's standard library.

本项目使用EAST模型检测自然场景中的文字区域，再用EasyOCR识别文字内容。  
通过预处理模块（旋转校正+CLAHE增强）提高倾斜或低对比度文字的识别率。  
提供命令行和Web两种使用方式，Web服务器完全基于Python标准库，无需额外依赖。

## Project Structure / 项目结构
├── model/ # EAST pre-trained model / 预训练模型
├── images/ # Test images / 测试图片
├── output/ # Output files / 输出结果
├── config.py # Configuration / 配置文件
├── detector.py # EAST text detector / EAST检测器
├── preprocessor.py # Preprocessing (skew+CLAHE) / 预处理
├── ocr_recognizer.py # EasyOCR recognizer / 识别器
├── main.py # CLI entry / 命令行入口
├── app_server.py # Web server (zero-dependency) / Web服务器
└── README.md


## Quick Start / 快速开始
1. Download the EAST model `frozen_east_text_detection.pb` and place it in `model/`.  
   Link: [https://github.com/oyyd/frozen_east_text_detection.pb](https://github.com/oyyd/frozen_east_text_detection.pb)  
   下载EAST模型文件放入 `model/` 目录。
2. Install dependencies: `pip install -r requirements.txt`
3. **Command line:** `python main.py` – results will be saved in `output/`.
   **Web:** `python app_server.py` – open `http://localhost:8000`.

## Configuration / 配置说明 (config.py)
- `MIN_CONFIDENCE`: 0.3 (lower for more boxes)
- `RESIZE_WIDTH/HEIGHT`: 640 (improves small text detection)
- `OCR_LANGUAGES`: ['ch_sim','en'] for Chinese+English
- `SAVE_TEXT_RESULT`: True to save recognized text as .txt

## Key Improvements / 主要改进
1. **Coordinate mapping fix** – boxes now correctly map back to original image size.
2. **Preprocessing** – skew correction + CLAHE for tilted/faded text.
3. **Web interface** – zero-dependency HTTP server, supports drag-and-drop upload and result display.
4. **Modular design** – detector interface allows easy swapping to other models (e.g., PaddleOCR).

## FAQ / 常见问题
- First run of EasyOCR will download language models; keep network connected.
- `[No text]` indicates no recognizable text in that region (normal for icons/background).

## Acknowledgements / 致谢
- EAST: https://github.com/argman/EAST
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- 