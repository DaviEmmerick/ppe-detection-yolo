#include "inference.hpp"
#include <chrono>

YoloDetector::YoloDetector(const std::string& modelPath) : env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference") {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(8);
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    std::cout << "--- V2: C++ Engine Ativa ---" << std::endl;
}

std::vector<Detection> YoloDetector::detect(cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, inputSize, cv::Scalar(0, 0, 0), true, false);

    std::vector<int64_t> inputDims = {1, 3, 640, 640};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)blob.data, blob.total(), inputDims.data(), inputDims.size()
    );

    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1
    );

    float* allData = outputTensors[0].GetTensorMutableData<float>();
    auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int rows = shape[1]; 
    int dimensions = shape[2];

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    cv::Mat output(rows, dimensions, CV_32F, allData);
    cv::Mat detData = output.t(); 

    for (int i = 0; i < detData.rows; ++i) {
        float* data = detData.ptr<float>(i);
        float confidence = data[4]; 

        if (confidence >= confThreshold) {
            float* classesScores = data + 5;
            cv::Mat scores(1, 5, CV_32F, classesScores); 
            cv::Point classIdPoint;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

            if (maxClassScore > 0.25) {
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * (frame.cols / 640.0));
                int top = int((y - 0.5 * h) * (frame.rows / 640.0));
                int width = int(w * (frame.cols / 640.0));
                int height = int(h * (frame.rows / 640.0));

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                classIds.push_back(classIdPoint.x);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> result;
    for (int idx : indices) {
        result.push_back({boxes[idx], confidences[idx], classIds[idx]});
    }
    return result;
}

int main() {
    YoloDetector detector("../best.onnx");
    cv::VideoCapture cap(0);
    
    cv::Mat frame;
    while (cv::waitKey(1) != 27) {
        auto start = std::chrono::high_resolution_clock::now(); 

        cap >> frame;
        if (frame.empty()) break;

        auto detections = detector.detect(frame);

        bool hasHelmet = false;
        bool hasVest = false;

        for (const auto& det : detections) {
            if (det.classId == 0) hasHelmet = true; 
            if (det.classId == 1) hasVest = true;   

            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        if (!hasHelmet) {
            cv::putText(frame, "ALERTA: SEM CAPACETE!", cv::Point(10, 80), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        if (!hasVest) {
            cv::putText(frame, "ALERTA: SEM COLETE!", cv::Point(10, 110), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Cálculo de FPS e Latência
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        float ms = duration.count();
        float fps = 1000.0f / ms;

        std::string info = "FPS: " + std::to_string((int)fps) + " | Latencia: " + std::to_string((int)ms) + "ms";
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

        cv::imshow("PPE Detection V2 - High Performance", frame);
    }
    return 0;
}