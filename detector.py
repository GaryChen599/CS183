# detector.py
# EAST text detector class
# EAST文本检测器类，封装了模型加载、推理、解码和画框

import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression

class EASTTextDetector:
    """
    A simple EAST text detector.
    一个简单的EAST文本检测器，可以检测图片中的文字区域。
    """

    def __init__(self, model_path, conf_thresh=0.5, resize_width=320, resize_height=320):
        """
        Initialize detector, load EAST model
        初始化检测器，加载EAST预训练模型
        Parameters / 参数:
        model_path : str         path to .pb model file  模型文件路径
        conf_thresh: float       confidence threshold    置信度阈值
        resize_width: int        target image width      缩放宽度
        resize_height: int       target image height     缩放高度
        """
        self.conf_thresh = conf_thresh
        self.resize_width = resize_width
        self.resize_height = resize_height

        print("[INFO] Loading EAST model... / 加载EAST模型...")
        self.net = cv2.dnn.readNet(model_path)

        # Output layer names in the EAST network
        # EAST网络的两个输出层名称
        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",  # Score map / 分数图（文本概率）
            "feature_fusion/concat_3"  # Geometry map / 几何图（框的形状信息）
        ]

    def detect(self, image_path):
        """
        Run text detection on given image
        对指定图片执行文本检测，返回带框的图像和框列表
        Parameters / 参数:
        image_path : str    path to input image  输入图片路径
        Returns / 返回:
        display_img : ndarray    image with boxes drawn   画好框的结果图像
        boxes : list             list of rectangles       检测到的框坐标列表
        """
        # 1. Load image / 读取图片
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path} / 无法读取图片: {image_path}")

        (H, W) = original_img.shape[:2]
        display_img = original_img.copy()

        # 2. Compute scaling ratios / 计算缩放比例（用于坐标映射）
        rW = W / float(self.resize_width)
        rH = H / float(self.resize_height)

        # 3. Resize image to fit network / 缩放到模型要求的尺寸
        resized = cv2.resize(original_img, (self.resize_width, self.resize_height))
        (newH, newW) = resized.shape[:2]

        # 4. Build blob and forward pass / 构建blob并进行前向推理
        #    Mean subtraction uses ImageNet RGB means  均值减法使用了ImageNet的RGB均值
        blob = cv2.dnn.blobFromImage(
            resized, 1.0, (newW, newH),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        start = time.time()
        scores, geometry = self.net.forward(self.layer_names)
        end = time.time()
        print(f"[INFO] Detection took {end - start:.3f} sec / 检测耗时 {end - start:.3f} 秒")

        # 5. Decode feature maps into candidate boxes / 将特征图解码为候选矩形框
        rects, confidences = self._decode_predictions(scores, geometry)

        # 6. Non‑Maximum Suppression (remove duplicates) / 非极大值抑制，去掉重叠框
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # 7. Map coordinates back and draw rectangles / 坐标还原到原图并画框
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # Draw green rectangle / 画绿色矩形框
            cv2.rectangle(display_img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return display_img, boxes

    def _decode_predictions(self, scores, geometry):
        """
        Decode score map and geometry map into bounding boxes.
        解码分数图和几何图，输出矩形框坐标和对应的置信度。
        This part is a bit math-heavy; you only need to understand that
        it extracts text box locations from the network outputs.
        这部分数学计算较多，只需了解它从网络输出中提取出文本框位置即可。
        """
        (numRows, numCols) = scores.shape[2:4]  # Feature map size  特征图尺寸
        rects = []  # Bounding box list  候选框列表
        confidences = []  # Confidence list  置信度列表

        # Loop over each pixel in the feature map / 遍历特征图上每一个像素
        for y in range(numRows):
            # Extract data for current row / 取出当前行的数据
            scores_row = scores[0, 0, y]  # probabilities 概率
            geo_top = geometry[0, 0, y]  # distance to top 到上边距离
            geo_right = geometry[0, 1, y]  # distance to right 到右边距离
            geo_bottom = geometry[0, 2, y]  # distance to bottom 到下边距离
            geo_left = geometry[0, 3, y]  # distance to left 到左边距离
            geo_angle = geometry[0, 4, y]  # rotation angle 旋转角度

            for x in range(numCols):
                # Skip low-confidence pixels / 忽略低置信度像素
                if scores_row[x] < self.conf_thresh:
                    continue

                # Since feature map is 1/4 of input size, multiply by 4
                # 特征图尺寸是输入的四分之一，坐标需乘以4映射回缩放后图像
                offsetX = x * 4.0
                offsetY = y * 4.0

                # Get sin and cos of the angle / 获取角度的正余弦值
                angle = geo_angle[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # Box height and width / 计算矩形的高和宽
                h = geo_top[x] + geo_bottom[x]
                w = geo_right[x] + geo_left[x]

                # Compute rotated corner coordinates / 计算经旋转补偿后的右下角坐标
                endX = int(offsetX + cos * geo_right[x] + sin * geo_bottom[x])
                endY = int(offsetY - sin * geo_right[x] + cos * geo_bottom[x])
                # Top-left corner / 左上角坐标
                startX = int(endX - w)
                startY = int(endY - h)

                # Save this box and its confidence / 保存该框及其置信度
                rects.append((startX, startY, endX, endY))
                confidences.append(scores_row[x])

        return rects, confidences