# detector.py
# EAST text detector class / EAST文本检测器类
import cv2, numpy as np, time
from imutils.object_detection import non_max_suppression

class EASTTextDetector:
    def __init__(self, model_path, conf_thresh=0.5, resize_width=320, resize_height=320):
        self.conf_thresh = conf_thresh
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.net = cv2.dnn.readNet(model_path)
        self.layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    def detect(self, image_path):
        original_img = cv2.imread(image_path)
        (H, W) = original_img.shape[:2]
        display_img = original_img.copy()
        rW = W / float(self.resize_width)
        rH = H / float(self.resize_height)
        resized = cv2.resize(original_img, (self.resize_width, self.resize_height))
        (newH, newW) = resized.shape[:2]
        blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        scores, geometry = self.net.forward(self.layer_names)
        print(f"[INFO] Detection took {time.time()-start:.3f} sec")
        rects, confidences = self._decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        mapped_boxes = []
        for (sX, sY, eX, eY) in boxes:
            sX_m = int(sX * rW); sY_m = int(sY * rH)
            eX_m = int(eX * rW); eY_m = int(eY * rH)
            cv2.rectangle(display_img, (sX_m, sY_m), (eX_m, eY_m), (0,255,0), 2)
            mapped_boxes.append((sX_m, sY_m, eX_m, eY_m))
        return display_img, mapped_boxes

    def _decode_predictions(self, scores, geometry):
        (numRows, numCols) = scores.shape[2:4]
        rects, confidences = [], []
        for y in range(numRows):
            scores_row = scores[0,0,y]
            geo_top = geometry[0,0,y]; geo_right = geometry[0,1,y]
            geo_bottom = geometry[0,2,y]; geo_left = geometry[0,3,y]; geo_angle = geometry[0,4,y]
            for x in range(numCols):
                if scores_row[x] < self.conf_thresh: continue
                offsetX, offsetY = x*4.0, y*4.0
                cos, sin = np.cos(geo_angle[x]), np.sin(geo_angle[x])
                h = geo_top[x] + geo_bottom[x]; w = geo_right[x] + geo_left[x]
                endX = int(offsetX + cos*geo_right[x] + sin*geo_bottom[x])
                endY = int(offsetY - sin*geo_right[x] + cos*geo_bottom[x])
                startX = int(endX - w); startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scores_row[x])
        return rects, confidences