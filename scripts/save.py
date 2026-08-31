import cv2
import numpy as np

# 读取图片，缩放到 YOLOv8 标准输入 640x640 灰度图
img = cv2.imread("test_images/people_classroom_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (640, 640))

# 存为裸数据（总字节数：640*640 = 409600）
resized.astype(np.uint8).tofile("input.raw")
print("Generated input.raw (640x640, 409600 bytes)")