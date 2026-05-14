# main.py
# ============================================================
# Entry point of the project / 项目入口
# Reads config, runs detection, displays/saves result
# 读取配置文件 → 初始化检测器 → 执行检测 → 显示/保存结果
# ============================================================

import os
import cv2
from config import *                  # Import all configuration / 导入所有配置参数
from detector import EASTTextDetector # Import detector class / 导入检测器类

def main():
    """
    Main function: sets up everything and runs text detection.
    主函数：完成初始化、检测、结果输出。
    """
    # Create output directory if saving / 如果需要保存，则创建输出文件夹
    if SAVE_RESULT and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize detector with config values / 用配置文件中的参数初始化检测器
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

    # Show result in a window / 弹窗显示结果
    if SHOW_RESULT:
        cv2.imshow("Text Detection Result", result_img)
        cv2.waitKey(0)            # Wait for key press / 按任意键关闭
        cv2.destroyAllWindows()

    # Save result to file / 保存结果图片
    if SAVE_RESULT:
        base_name = os.path.basename(IMAGE_PATH)
        out_path = os.path.join(OUTPUT_DIR, f"detected_{base_name}")
        cv2.imwrite(out_path, result_img)
        print(f"[INFO] Result saved to: {out_path} / 结果已保存至: {out_path}")

if __name__ == "__main__":
    main()