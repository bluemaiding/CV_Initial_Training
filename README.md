

```markdown
# 📸 嵌入式 AI 图像预处理工具库 (OpenCV Python)

> 面向嵌入式 NPU（如 RK3588 / Jetson）的轻量级图像预处理流水线  
> 输出格式直接对齐 YOLO / MobileNet 等主流模型输入规范

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Ubuntu-lightgrey.svg)]()

---

## 🎯 项目背景

本项目为 **嵌入式计算机视觉** 场景设计，旨在构建一个 **低延迟、可量化、跨平台** 的图像预处理前端。  

核心设计目标：
- 输入：任意尺寸的 JPEG / PNG 图像
- 输出：**128×128 灰度图**，同时提供 **归一化 (0~1)** 和 **INT8 量化** 两种格式
- 适配场景：无人机避障、工业质检、微光成像等资源受限的嵌入式系统

---

## 🔧 核心功能

| 模块 | 说明 |
| :--- | :--- |
| **自适应缩放** | 保持宽高比缩放到目标尺寸（支持 `--size` 命令行配置） |
| **BGR → 灰度** | 三通道降维为单通道，减少 2/3 算力消耗 |
| **CLAHE 增强** | 限制对比度自适应直方图均衡化，暗光场景专用 |
| **归一化** | 像素值映射到 `[0, 1]` 浮点数，适配 GPU / CPU 推理 |
| **INT8 量化** | 输出 `uint8` 整数，适配 NPU 零拷贝输入 |

---

## 📊 性能基准 (Windows 11, Intel i5, 128×128)

| 图片样本 | 预处理耗时 | INT8 输出范围 |
| :--- | :--- | :--- |
| fish_across_glass_4.jpg | 24.89 ms | [9, 255] |
| parrot_book_2.jpg | 70.54 ms | [7, 255] |
| people_classroom_3.jpg | 13.33 ms | [3, 255] |
| person_guitar_1.jpg | 18.52 ms | [6, 255] |
| scene_on_bridge_5.jpg | 18.79 ms | [4, 255] |
| **平均 (含离群值)** | **29.21 ms/张** | - |
| **平均 (常规图)** | **18.88 ms/张** | - |

> 💡 注：`parrot_book_2.jpg` 为高复杂度 JPEG，解码开销较大，在实际嵌入式场景中建议上游限制图像分辨率或使用 MJPEG 直出流以规避此类抖动。

---

## 🐧 跨平台验证

| 平台 | 状态 | 平均耗时 |
| :--- | :--- | :--- |
| Windows 11 (宿主机) | ✅ 通过 | 29.21 ms/张 |
| Ubuntu 24.04 (虚拟机) | ✅ 通过 | 待补充 |

> Ubuntu 端数据将在虚拟机恢复后补充，已确认代码零改动直接运行。


```

### 2. 安装依赖
```bash
pip install opencv-python numpy
```

### 3. 运行预处理
```bash
# 默认尺寸 224×224
python preprocess_pipeline.py --input ./test_images --output ./output

# 指定尺寸 128×128 (推荐嵌入式场景)
python preprocess_pipeline.py --input ./test_images --output ./output --size 128
```

### 4. 查看结果
- `output/` 目录下生成：
  - `enhanced_xxx.jpg` — 增强后灰度图
  - `compare_xxx.jpg` — 四宫格对比图（原图 / 灰度 / 增强 / AI 回显）
  - `enhanced_ai_xxx.npy` — 归一化后的 NumPy 矩阵（可直接喂给模型）

---

## 📁 项目结构

```
.
├── preprocess_pipeline.py    # 主程序
├── test_images/              # 测试图片
├── output/                   # 输出目录
├── README.md
```

---

## 🧠 技术细节

- **CLAHE 参数**：`clipLimit=2.0, tileGridSize=(8,8)`，在暗光增强与噪点抑制之间取得平衡。
- **INT8 量化**：`(enhanced_ai * 255).astype(np.uint8)`，确保输出数据可直接 `memcpy` 至 NPU 输入缓冲区。
- **性能监控**：使用 `cv2.getTickCount()` 测量单帧耗时，误差 < 1ms。

---

## 🔭 后续计划

- [x] Windows 端核心功能验证
- [x] Ubuntu 端跨平台验证 (虚拟机)
- [ ] 接入 YOLO 目标检测模型，形成完整推理链路
- [ ] 在真实嵌入式开发板 (Jetson / RK3588) 上进行实测
- [ ] C++ 版本重构，进一步降低预处理延迟



