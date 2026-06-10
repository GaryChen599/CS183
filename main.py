"""
File name/文件：main.py
Responsible team member/负责团队成员：Lan Shuchang
Description/描述:Orchestrates the whole image text pipeline: load image → EAST detects regions → OCR recognizes text → draw/save/show results./串联整个图像文字处理流程：加载图像 → EAST 检测文本区域 → OCR 识别文字 → 绘制/保存/显示结果

"""
# main.py
# Command line entry / 命令行入口
import os, cv2# 导入系统文件操作模块 / Import system file operation module    # 导入OpenCV计算机视觉库，用于图像处理和显示 / Import OpenCV library for image processing and display
from config import *
from detector import EASTTextDetector
from ocr_recognizer import OCRRecognizer

def draw_text_on_image(img, box, text):#在图像上绘制识别出的文字 / Draw recognized text on image"""
    (x1, y1, x2, y2) = box
    y_text = y1 - 10 if y1 > 20 else y2 + 20
    cv2.putText(img, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def main():# 创建输出目录（如果需要） / Create output dir if needed
    if SAVE_RESULT and not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    detector = EASTTextDetector(MODEL_PATH, MIN_CONFIDENCE, RESIZE_WIDTH, RESIZE_HEIGHT)# 初始化检测器并执行文本检测 / Init detector and detect text boxes
    result_img, boxes = detector.detect(IMAGE_PATH)
    print(f"[INFO] Found {len(boxes)} boxes")
    recognized = []
    if ENABLE_OCR and len(boxes) > 0:# 如果启用OCR且检测到文本框，进行文字识别 / If OCR enabled and boxes found, recognize text
        recognizer = OCRRecognizer(languages=OCR_LANGUAGES, gpu=OCR_GPU)
        original_img = cv2.imread(IMAGE_PATH)
        recognized = recognizer.recognize_boxes(original_img, boxes)
        for (box, text, _) in recognized: # 在结果图上绘制识别出的文本 / Draw recognized text on result image
            draw_text_on_image(result_img, box, text if text else "?")
        print(f"[INFO] Recognized {len(recognized)} boxes")
        if SAVE_TEXT_RESULT: # 保存文本结果到文件（如果需要） / Save text results to file if needed
            base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
            with open(os.path.join(OUTPUT_DIR, f"{base}_text.txt"), 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for {IMAGE_PATH}\n{'='*50}\n")
                count = 0
                for i, (box, text, conf) in enumerate(recognized, 1):
                    if text: f.write(f"Box {i}: {box} -> {text}\n"); count += 1
                    else: f.write(f"Box {i}: {box} -> [No text]\n")
                f.write(f"\nTotal recognized: {count}/{len(recognized)}\n")
    if SHOW_RESULT:# 显示结果窗口 / Show result window
        cv2.imshow("Result", result_img); cv2.waitKey(0); cv2.destroyAllWindows()
    if SAVE_RESULT: # 保存带检测框的结果图像 / Save result image with bo
        out_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(IMAGE_PATH)}")
        cv2.imwrite(out_path, result_img)

if __name__ == "__main__": main()
