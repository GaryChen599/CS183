# preprocessor.py
# Image preprocessing: skew correction + CLAHE enhancement / 图像预处理：旋转校正 + CLAHE增强
# In charge: Gary/Chen Jialue
import cv2, numpy as np

class ImagePreprocessor:
    @staticmethod
    def process(roi):
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5: return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angle = 0
        if contours:
            all_pts = np.vstack(contours)
            rect = cv2.minAreaRect(all_pts)
            angle = rect[2]
            if angle < -45: angle += 90
            elif angle > 45: angle -= 90
        if abs(angle) > 1:
            h, w = gray.shape
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos, sin = abs(M[0,0]), abs(M[0,1])
            new_w = int(h*sin + w*cos); new_h = int(h*cos + w*sin)
            M[0,2] += new_w/2 - center[0]; M[1,2] += new_h/2 - center[1]
            gray = cv2.warpAffine(gray, M, (new_w, new_h), borderValue=255)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
