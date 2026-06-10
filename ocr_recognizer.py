"""
File Name/文件名: ocr_recognizer.py
Responsible Team Member/负责团队成员:HuYang(胡杨)
Description/描述: Performs OCR text recognition, including model loading,image preprocessing, and recognition result processing./执行OCR文本识别，包含模型加载、图像预处理以及识别结果处理。
"""
import easyocr, cv2
from preprocessor import ImagePreprocessor

class OCRRecognizer:
    def __init__(self, languages=['en'], gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.preprocessor = ImagePreprocessor()

    def recognize_boxes(self, image, boxes):
        results = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1: continue
            roi = image[y1:y2, x1:x2]
            if roi.size == 0: continue
            proc_roi = self.preprocessor.process(roi)
            h, w = proc_roi.shape[:2]
            if h < 64:
                scale = 64.0 / h
                new_w = max(int(w*scale), 1)
                proc_roi = cv2.resize(proc_roi, (new_w, 64), interpolation=cv2.INTER_CUBIC)
            proc_rgb = cv2.cvtColor(proc_roi, cv2.COLOR_BGR2RGB)
            ocr_result = self.reader.readtext(proc_rgb, detail=0)
            text = ' '.join(ocr_result).strip()
            results.append(((x1, y1, x2, y2), text, 1.0 if text else 0.0))
        return results