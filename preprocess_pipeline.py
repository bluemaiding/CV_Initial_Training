import cv2
import numpy as np
import os
import time
from pathlib import Path


class ImagePreprocessor:
    """面向嵌入式AI的图像预处理库 (适配YOLO/CNN输入)"""

    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize  # 🧠 是否归一化到0~1

    def process_single_image(self, img_path):
        """流水线: 缩放 -> 灰度 -> CLAHE增强 -> (可选)归一化"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        # 1. 缩放 (保持宽高比，补黑边letterbox的简化版)
        h, w = img.shape[:2]
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # 2. 转灰度
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. 增强 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)  # 此时是 uint8 类型 (0~255)

        # 🧠 4. 【AI核心】归一化到 0~1 浮点数 (神经网络输入标准)
        if self.normalize:
            # astype转float32, 除以255.0
            enhanced_ai = enhanced.astype(np.float32) / 255.0
            # 🧠 INT8量化版本 (嵌入式部署专用) —— 放在这里！
            enhanced_int8 = (enhanced_ai * 255).astype(np.uint8)
        else:
            enhanced_ai = enhanced
            enhanced_int8 = enhanced  # 不归一化时，INT8就等于原图

        # 5. 返回结果 (保留原始uint8用于存图，保留归一化版本用于AI)
        return {
            "original": img,
            "resized": resized,
            "gray": gray,  # uint8
            "enhanced": enhanced,  # uint8 (用来保存对比图)
            "enhanced_ai": enhanced_ai,  # 🧠 float32 (0~1) 准备喂给模型
            "enhanced_int8": enhanced_int8
        }

    def batch_process(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        img_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        if not img_files:
            print(f"❌ 在 {input_dir} 里没找到图片！")
            return

        total_time = 0
        count = 0

        for f in img_files[:20]:  # 先处理20张
            start = cv2.getTickCount()
            result = self.process_single_image(str(f))
            if not result:
                continue
            elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            total_time += elapsed
            count += 1

            # --- 保存增强后的图 (uint8) ---
            cv2.imwrite(str(output_path / f"enhanced_{f.name}"), result["enhanced"])

            # 🧠 额外保存一份归一化的.npy文件 (用于验证数值)
            np.save(str(output_path / f"enhanced_ai_{f.stem}.npy"), result["enhanced_ai"])

            # --- 生成四宫格对比图 (唬人专用) ---
            h, w = result["gray"].shape
            base = cv2.resize(result["original"], (w, h))
            gray_3ch = cv2.cvtColor(result["gray"], cv2.COLOR_GRAY2BGR)
            enhanced_3ch = cv2.cvtColor(result["enhanced"], cv2.COLOR_GRAY2BGR)

            # 为了展示AI效果，把归一化的小数映射回0~255用于显示
            if self.normalize:
                ai_display = (result["enhanced_ai"] * 255).astype(np.uint8)
                ai_3ch = cv2.cvtColor(ai_display, cv2.COLOR_GRAY2BGR)
            else:
                ai_3ch = enhanced_3ch

            top_row = np.hstack((base, gray_3ch))
            bottom_row = np.hstack((enhanced_3ch, ai_3ch))
            compare_img = np.vstack((top_row, bottom_row))
            cv2.imwrite(str(output_path / f"compare_{f.name}"), compare_img)

            # 🧠 打印AI数值信息 (证明归一化成功)
            ai_min = result["enhanced_ai"].min()
            ai_max = result["enhanced_ai"].max()
            print(f"✅ {f.name} | 耗时: {elapsed * 1000:.2f}ms | AI输入范围: [{ai_min:.3f}, {ai_max:.3f}]")

        if count > 0:
            # 🧠 打印AI数值信息 + INT8范围
            ai_min = result["enhanced_ai"].min()
            ai_max = result["enhanced_ai"].max()
            int8_min = result["enhanced_int8"].min()
            int8_max = result["enhanced_int8"].max()
            print(
                f"✅ {f.name} | 耗时: {elapsed * 1000:.2f}ms | AI: [{ai_min:.3f}, {ai_max:.3f}] | INT8: [{int8_min}, {int8_max}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="嵌入式AI图像预处理工具")
    parser.add_argument("--input", default="./test_images", help="输入图片文件夹")
    parser.add_argument("--output", default="./output", help="输出文件夹")
    parser.add_argument("--no_norm", action="store_true", help="加上此参数则不进行归一化")
    parser.add_argument("--size", default=224, type=int, help="目标尺寸（宽高相等）")
    args = parser.parse_args()

    # 默认开启归一化，加了 --no_norm 才关闭
    pre = ImagePreprocessor(target_size=(args.size, args.size), normalize=not args.no_norm)
    pre.batch_process(args.input, args.output)