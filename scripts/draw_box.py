import numpy as np
import cv2

IMG_SIZE = 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45

CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
           "traffic light","fire hydrant","stop sign","parking lot","bench","bird","cat",
           "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
           "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
           "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
           "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
           "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
           "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
           "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
           "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

# 读取增强图
img_raw = np.fromfile("enhanced.raw", dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

# 读取输出张量
tensor = np.fromfile("output_tensor.raw", dtype=np.float32).reshape(1, 84, 8400)
preds = tensor[0].T  # (8400, 84)

# ====== 【修正点1】提取中心点+宽高，转成 x1,y1,x2,y2 ======
cx = preds[:, 0]
cy = preds[:, 1]
w = preds[:, 2]
h = preds[:, 3]

x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2

boxes = np.stack([x1, y1, x2, y2], axis=1)  # (8400, 4)

# ====== 【修正点2】坐标已经是原始像素值，不需要再乘以 IMG_SIZE ======
# boxes[:, 0] *= IMG_SIZE
# boxes[:, 1] *= IMG_SIZE
# boxes[:, 2] *= IMG_SIZE
# boxes[:, 3] *= IMG_SIZE

# 提取置信度和类别
scores = preds[:, 4:]  # (8400, 80)
class_ids = np.argmax(scores, axis=1)
confidences = np.max(scores, axis=1)

# 过滤低分框
mask = confidences > CONF_THRESHOLD
boxes = boxes[mask]
confidences = confidences[mask]
class_ids = class_ids[mask]

print(f"Detected {len(boxes)} boxes with conf > {CONF_THRESHOLD}")

if len(boxes) == 0:
    print("No objects detected.")
    cv2.imwrite("final_result.jpg", img_bgr)
    exit()

# 确保坐标在图像范围内
boxes[:, 0] = np.clip(boxes[:, 0], 0, IMG_SIZE)
boxes[:, 1] = np.clip(boxes[:, 1], 0, IMG_SIZE)
boxes[:, 2] = np.clip(boxes[:, 2], 0, IMG_SIZE)
boxes[:, 3] = np.clip(boxes[:, 3], 0, IMG_SIZE)

# NMS
def nms(boxes, scores, iou_thresh):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        x1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        inter = w * h
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area1 + area2 - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_thresh]
    return keep

keep = nms(boxes, confidences, NMS_THRESHOLD)
final_boxes = boxes[keep].astype(np.int32)
final_conf = confidences[keep]
final_cls = class_ids[keep]

print(f"After NMS: {len(final_boxes)} boxes remain.")

# 画框
for i, box in enumerate(final_boxes):
    x1, y1, x2, y2 = box
    label = f"{CLASSES[final_cls[i]]} {final_conf[i]:.2f}"
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("final_result.jpg", img_bgr)
print("✅ Saved to final_result.jpg")