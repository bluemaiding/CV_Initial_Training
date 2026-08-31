#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

#include "clahe.h"

const int INPUT_SIZE = 640;
const int TOTAL_PIXELS = INPUT_SIZE * INPUT_SIZE;

// 读取 RAW 文件
unsigned char* read_raw(const std::string& path, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return nullptr;
    unsigned char* data = new unsigned char[size];
    file.read((char*)data, size);
    return data;
}

// 灰度图转 1x3x640x640 float 张量 (0~1 归一化)
std::vector<float> gray_to_tensor(const unsigned char* gray_data) {
    std::vector<float> tensor(1 * 3 * INPUT_SIZE * INPUT_SIZE);
    float* ptr = tensor.data();
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        float val = (float)gray_data[i] / 255.0f;
        ptr[0 * TOTAL_PIXELS + i] = val; // R
        ptr[1 * TOTAL_PIXELS + i] = val; // G
        ptr[2 * TOTAL_PIXELS + i] = val; // B
    }
    return tensor;
}

int main() {
    // 1. 读取 input.raw
    unsigned char* raw_input = read_raw("input.raw", TOTAL_PIXELS);
    if (!raw_input) {
        std::cerr << "Failed to read input.raw" << std::endl;
        return -1;
    }

    // 2. CLAHE 增强 (tile_size = 80, 因为 640/8 = 80)
    unsigned char* enhanced = clahe_enhance(raw_input, INPUT_SIZE, INPUT_SIZE, 80);
    if (!enhanced) {
        std::cerr << "CLAHE failed!" << std::endl;
        delete[] raw_input;
        return -1;
    }
    std::cout << "CLAHE enhancement done." << std::endl;

    // 3. 转为 ONNX Tensor
    auto input_tensor_data = gray_to_tensor(enhanced);

    // 4. ONNX Runtime 推理
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    // 新版本 API：不再用 AddCpuProvider，默认就是 CPU
    // 如果需要显式设置 CPU，可以用下面这行（可选）：
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Cpu(session_options, 0));

    // 加载模型（用宽字符路径）
    std::wstring model_path = L"models/yolov8n.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);

    // 准备输入
    std::vector<int64_t> input_shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_data.data(), input_tensor_data.size(),
        input_shape.data(), input_shape.size()
    );

    // 执行推理（新版本推荐写法）
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    std::vector<Ort::Value> input_values;
    input_values.push_back(std::move(input_tensor));

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, input_values.data(), input_values.size(),
                                      output_names, 1);

    // 5. 打印输出形状（证明跑通了）
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Inference success! Output shape: ";
    for (auto d : shape) std::cout << d << " ";
    std::cout << std::endl;

    // 释放内存
    delete[] raw_input;
        // ... 推理代码 ...

    // 5. 保存增强后的灰度图（供 Python 画框用）
    FILE* fout_img = fopen("enhanced.raw", "wb");
    fwrite(enhanced, 1, TOTAL_PIXELS, fout_img);
    fclose(fout_img);
    std::cout << "Enhanced image saved to enhanced.raw" << std::endl;

    // 6. 保存原始输出张量（供 Python 后处理用）
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t num_elements = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    FILE* fout_tensor = fopen("output_tensor.raw", "wb");
    fwrite(output_data, sizeof(float), num_elements, fout_tensor);
    fclose(fout_tensor);
    std::cout << "Output tensor saved to output_tensor.raw (size: " << num_elements << " floats)" << std::endl;
    free(enhanced);
    return 0;
}