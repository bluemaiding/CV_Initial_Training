import cv2
import time
from pathlib import Path
from ultralytics import YOLO
from preprocess_pipeline import ImagePreprocessor

# 1. 加载 YOLO
model = YOLO("yolov8n.pt")

# 2. 预处理配置（当前仍为 128，你可以随时改成 320 或 640）
pre = ImagePreprocessor(target_size=(640,640))

# 3. 扫描测试图片
image_dir = Path("./test_images")
output_dir = Path("./yolo_results")
output_dir.mkdir(exist_ok=True)

img_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
print(f"📸 找到 {len(img_files)} 张图片，开始批量检测...\n")

for img_path in img_files:
    print(f"--- 处理: {img_path.name} ---")

    # 预处理
    result = pre.process_single_image(str(img_path))
    if result is None:
        print("   ❌ 读取失败，跳过\n")
        continue

    gray = result["enhanced"]
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # YOLO 推理（verbose=False 抑制内部日志）
    results = model(gray_3ch, verbose=False)

    # 保存带框结果图
    save_path = output_dir / f"yolo_{img_path.name}"
    results[0].save(filename=str(save_path))

    # 提取所有检测框（不管置信度多少）
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        # 获取边界框坐标（xywh 归一化格式）
        xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height] 归一化
        detections.append((label, conf, xywh))

    if detections:
        # 按置信度从高到低排序
        detections.sort(key=lambda x: x[1], reverse=True)
        print(f"   ✅ 总共检出 {len(detections)} 个目标:")
        for label, conf, xywh in detections:
            # 打印置信度保留两位小数，坐标保留三位
            print(
                f"      {label}: {conf:.2f}  (框: x={xywh[0]:.3f}, y={xywh[1]:.3f}, w={xywh[2]:.3f}, h={xywh[3]:.3f})")
    else:
        print("   ⚠️ 未检测到任何目标")

    print("")  # 空行分隔

print(f"🎉 批量检测完成！所有带框结果已保存至 {output_dir} 文件夹")