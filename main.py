import os
import cv2
from config import *
from detector import EASTTextDetector
from ocr_recognizer import OCRRecognizer   # 新增导入
import easyocr, cv2

reader = easyocr.Reader(['ch_sim', 'en'])
img = cv2.imread('images/test1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = reader.readtext(img_rgb, detail=0)
print("全图识别结果:", result)


def draw_text_on_image(img, box, text, color=(255, 0, 0)):
    """
    Draw recognized text above the bounding box.
    在矩形框上方绘制识别文字。
    """
    (x1, y1, x2, y2) = box
    # Put text slightly above the box / 文字放在框上方一点
    y_text = y1 - 10 if y1 > 20 else y2 + 20
    cv2.putText(img, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def main():
    # Create output directory if saving / 如果需要保存，则创建输出文件夹
    if SAVE_RESULT and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize detector / 初始化检测器
    detector = EASTTextDetector(
        model_path=MODEL_PATH,
        conf_thresh=MIN_CONFIDENCE,
        resize_width=RESIZE_WIDTH,
        resize_height=RESIZE_HEIGHT
    )

    # Run detection / 执行检测
    print(f"[INFO] Detecting text in: {IMAGE_PATH} / 正在检测图片: {IMAGE_PATH}")
    result_img, boxes = detector.detect(IMAGE_PATH)
    print(f"[INFO] Found {len(boxes)} text region(s) / 检测到 {len(boxes)} 个文本区域")

    # ---- Stage 3: OCR Recognition / 第三阶段：OCR 识别 ----
    recognized = []
    if ENABLE_OCR and len(boxes) > 0:          # <-- 修改了这一行
        recognizer = OCRRecognizer(languages=OCR_LANGUAGES, gpu=OCR_GPU)
        original_img = cv2.imread(IMAGE_PATH)
        recognized = recognizer.recognize_boxes(original_img, boxes)
        print("[DEBUG] recognized sample:", recognized[:3])

        # Draw text labels on result image / 在结果图上绘制文字标注
        for (box, text, conf) in recognized:
            label = text if text.strip() else "?"
            draw_text_on_image(result_img, box, label)

        print(f"[INFO] OCR finished. Recognized {len(recognized)} boxes / OCR 完成，已识别 {len(recognized)} 个框")

        # Save recognized text to file / 保存识别结果到文件
        if SAVE_TEXT_RESULT:
            base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
            txt_path = os.path.join(OUTPUT_DIR, f"{base}_text.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for {IMAGE_PATH}\n")
                f.write("=" * 50 + "\n")
                count = 0
                for i, (box, text, conf) in enumerate(recognized, 1):
                    if text:
                        f.write(f"Box {i}: {box} -> {text} (conf: {conf:.2f})\n")
                        count += 1
                    else:
                        f.write(f"Box {i}: {box} -> [No text]\n")
                f.write(f"\nTotal recognized: {count}/{len(recognized)}\n")
            print(f"[INFO] OCR text saved to {txt_path}")

    # Show result window / 显示结果
    if SHOW_RESULT:
        cv2.imshow("Text Detection + OCR Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save output image / 保存输出图像
    if SAVE_RESULT:
        base_name = os.path.basename(IMAGE_PATH)
        out_path = os.path.join(OUTPUT_DIR, f"detected_{base_name}")
        cv2.imwrite(out_path, result_img)
        print(f"[INFO] Result saved to: {out_path} / 结果已保存至: {out_path}")

if __name__ == "__main__":
    main()