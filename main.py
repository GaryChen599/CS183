# main.py
# Command line entry / 命令行入口
import os, cv2
from config import *
from detector import EASTTextDetector
from ocr_recognizer import OCRRecognizer

def draw_text_on_image(img, box, text):
    (x1, y1, x2, y2) = box
    y_text = y1 - 10 if y1 > 20 else y2 + 20
    cv2.putText(img, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def main():
    if SAVE_RESULT and not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    detector = EASTTextDetector(MODEL_PATH, MIN_CONFIDENCE, RESIZE_WIDTH, RESIZE_HEIGHT)
    result_img, boxes = detector.detect(IMAGE_PATH)
    print(f"[INFO] Found {len(boxes)} boxes")
    recognized = []
    if ENABLE_OCR and len(boxes) > 0:
        recognizer = OCRRecognizer(languages=OCR_LANGUAGES, gpu=OCR_GPU)
        original_img = cv2.imread(IMAGE_PATH)
        recognized = recognizer.recognize_boxes(original_img, boxes)
        for (box, text, _) in recognized:
            draw_text_on_image(result_img, box, text if text else "?")
        print(f"[INFO] Recognized {len(recognized)} boxes")
        if SAVE_TEXT_RESULT:
            base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
            with open(os.path.join(OUTPUT_DIR, f"{base}_text.txt"), 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for {IMAGE_PATH}\n{'='*50}\n")
                count = 0
                for i, (box, text, conf) in enumerate(recognized, 1):
                    if text: f.write(f"Box {i}: {box} -> {text}\n"); count += 1
                    else: f.write(f"Box {i}: {box} -> [No text]\n")
                f.write(f"\nTotal recognized: {count}/{len(recognized)}\n")
    if SHOW_RESULT:
        cv2.imshow("Result", result_img); cv2.waitKey(0); cv2.destroyAllWindows()
    if SAVE_RESULT:
        out_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(IMAGE_PATH)}")
        cv2.imwrite(out_path, result_img)

if __name__ == "__main__": main()