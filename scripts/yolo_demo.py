from ultralytics import YOLO

# 1. 加载预训练模型（第一次运行时会自动下载权重文件，约 6MB）
model = YOLO("yolov8n.pt")

# 2. 用一张图片做推理（官方会去网上下载一张测试图）
results = model("https://ultralytics.com/images/bus.jpg")

# 3. 显示检测结果（会在本地保存一张带框的图片）
results[0].show()  # 弹窗显示
results[0].save(filename="result_bus.jpg")  # 保存到本地