# CLAHE + YOLOv8n 嵌入式预处理推理流水线

> 学习项目：纯 C 语言实现 CLAHE 增强 → ONNX Runtime 推理 YOLOv8n → Python 后处理画框。

## 流水线

```
test_images/*.jpg ──(save.py)──> input.raw ──(pipeline.exe)──> enhanced.raw + output_tensor.raw ──(draw_box.py)──> final_result.jpg
```

- **`src/pipeline.cpp`**：读取 `input.raw` → CLAHE 增强 → 转为 `1×3×640×640` float 张量 → ONNX Runtime 推理 → 保存增强图 `enhanced.raw` 和原始输出张量 `output_tensor.raw`。
- **`scripts/draw_box.py`**：读取 `enhanced.raw` + `output_tensor.raw`，做置信度过滤 + NMS，在图上画框，输出 `final_result.jpg`。

## 快速开始

```bash
# 1. 生成 input.raw（640x640 灰度图裸数据，409600 字节）
python scripts/save.py

# 2. 编译（首次需要，模型路径已指向 models/yolov8n.onnx）
g++ -O2 -std=c++17 src/pipeline.cpp -I<onnxruntime>/include -L<onnxruntime>/lib -lonnxruntime -o pipeline.exe
# 确保 onnxruntime.dll 在 exe 同目录（当前就在项目根目录）

# 3. CLAHE 增强 + ONNX 推理
./pipeline.exe

# 4. 后处理 + NMS + 画框
python scripts/draw_box.py
```

> 注意：所有命令在项目根目录（`cv/`）下运行，中间产物 `input.raw` / `enhanced.raw` / `output_tensor.raw` / `final_result.jpg` 都生成在根目录。

## 文件说明

```
cv/
├── src/                      # C/C++ 核心源码
│   ├── clahe.c               # 纯 C 语言 CLAHE 实现
│   ├── clahe.h               # 函数声明
│   └── pipeline.cpp          # C++ ONNX Runtime 推理主程序
├── scripts/                  # Python 脚本
│   ├── save.py               # 生成 input.raw
│   ├── draw_box.py           # 后处理 + NMS + 画框
│   ├── preprocess_pipeline.py  # 早期 OpenCV 预处理库（可独立使用）
│   ├── yolo.py               # 早期 ultralytics 批量检测
│   └── yolo_demo.py          # ultralytics 官方 demo
├── models/
│   └── yolov8n.onnx          # YOLOv8n 模型
├── cpp/                      # 独立 VS/CMake 工程（OpenCV DNN 版推理，与 pipeline.cpp 是两套实现）
├── test_images/              # 测试图片
├── onnxruntime.dll           # ONNX Runtime 动态库（运行时依赖）
├── yolov8n.pt                # PyTorch 权重（旧 Python 链路使用）
├── _trash/                   # 归档的临时产物（确认后手动删除，不参与 git）
└── README.md
```

## 依赖

- **ONNX Runtime 1.28.1**（`onnxruntime.dll` + 编译用 `include/`、`lib/`）
- **Python 3.x**（`numpy`、`opencv-python`）
- **g++ / MinGW**（编译 `src/pipeline.cpp`）

## 早期 Python 链路

`preprocess_pipeline.py` + `yolo.py` + `yolov8n.pt` 是早期探索阶段的 Python 实现（OpenCV CLAHE 预处理 + ultralytics 检测），已保留作参考。用法：

```bash
python scripts/preprocess_pipeline.py --input ./test_images --output ./output --size 640
python scripts/yolo.py
```
