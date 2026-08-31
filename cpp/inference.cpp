#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
using namespace dnn;

int main() {
    string modelPath = "C:/Users/bluem/Desktop/cv/yolov8n.onnx";
    string imagePath = "C:/Users/bluem/Desktop/cv/test_images/people_classroom_3.jpg";

    Net net = readNetFromONNX(modelPath);
    Mat img = imread(imagePath);
    if (img.empty()) { cout << "Image not found" << endl; return -1; }

    // 预处理
    Mat blob = blobFromImage(img, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    net.setInput(blob);
    Mat output = net.forward();  // shape: [1, 84, 8400]

    // 解析输出
    int rows = output.size[2];      // 8400
    int cols = output.size[1];      // 84
    // 将 output 重塑为 (8400, 84) 以便逐行处理
    Mat det_output = output.reshape(1, rows); // 8400 x 84
    transpose(det_output, det_output);        // 84 x 8400  (这一步可选，但便于索引)

    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    vector<Rect> boxes;
    vector<float> confs;
    vector<int> classIds;

    vector<string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // 现在 det_output 是 84 x 8400，每一列是一个检测框
    for (int i = 0; i < rows; i++) {
        // 获取第 i 个框的数据（84个值）
        float* data = det_output.ptr<float>(0, i);
        float x_center = data[0];
        float y_center = data[1];
        float w = data[2];
        float h = data[3];
        // 置信度：从索引4到83，取最大值（已经 sigmoid，无需再次）
        float maxConf = 0.0f;
        int maxId = -1;
        for (int c = 4; c < 84; c++) {
            if (data[c] > maxConf) {
                maxConf = data[c];
                maxId = c - 4;
            }
        }
        if (maxConf < confThreshold || maxId == -1) continue;

        // 转换为像素坐标（原图尺寸）
        int x1 = (x_center - w / 2) * img.cols;
        int y1 = (y_center - h / 2) * img.rows;
        int x2 = (x_center + w / 2) * img.cols;
        int y2 = (y_center + h / 2) * img.rows;
        // 裁剪
        x1 = max(0, min(x1, img.cols));
        y1 = max(0, min(y1, img.rows));
        x2 = max(0, min(x2, img.cols));
        y2 = max(0, min(y2, img.rows));
        if (x2 <= x1 || y2 <= y1) continue;
        boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
        confs.push_back(maxConf);
        classIds.push_back(maxId);
    }

    vector<int> indices;
    NMSBoxes(boxes, confs, confThreshold, nmsThreshold, indices);

    cout << "Detected " << indices.size() << " objects:" << endl;
    for (int idx : indices) {
        Rect box = boxes[idx];
        string label = class_names[classIds[idx]] + " " + to_string(confs[idx]).substr(0, 4);
        cout << "   " << label << " (x=" << box.x << ", y=" << box.y << ", w=" << box.width << ", h=" << box.height << ")" << endl;
        rectangle(img, box, Scalar(0, 255, 0), 2);
        putText(img, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }

    imwrite("cpp_result_opencv.jpg", img);
    cout << "Result saved to cpp_result_opencv.jpg" << endl;
    return 0;
}