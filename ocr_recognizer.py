# ocr_recognizer.py
import easyocr
import cv2

class OCRRecognizer:
    def __init__(self, languages=['en'], gpu=False):
        print(f"[INFO] Loading OCR model for {languages} / 正在加载 OCR 模型...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def recognize_boxes(self, image, boxes):
        results = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 只做放大：高度不足 64 像素就等比放大到 64
            h, w = roi.shape[:2]
            if h < 64:
                scale = 64.0 / h
                new_w = max(int(w * scale), 1)
                roi = cv2.resize(roi, (new_w, 64), interpolation=cv2.INTER_CUBIC)

            # 转 RGB 送给 EasyOCR
            if len(roi.shape) == 3:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

            ocr_result = self.reader.readtext(roi_rgb, detail=0)
            text = ' '.join(ocr_result).strip()

            results.append(((x1, y1, x2, y2), text, 1.0 if text else 0.0))

        return results