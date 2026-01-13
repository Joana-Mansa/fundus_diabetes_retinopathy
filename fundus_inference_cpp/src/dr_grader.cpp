#include "dr_grader.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace fundus {

DRGrader::DRGrader(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "DRGrader"),
      session_(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), Ort::SessionOptions{}) {

    // Get input info
    auto input_count = session_.GetInputCount();
    if (input_count != 1) {
        throw std::runtime_error("Expected 1 input, got " + std::to_string(input_count));
    }

    auto input_name = session_.GetInputNameAllocated(0, allocator_);
    input_names_.push_back(strdup(input_name.get()));

    auto input_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();

    // Get output info
    auto output_name = session_.GetOutputNameAllocated(0, allocator_);
    output_names_.push_back(strdup(output_name.get()));
}

DRGrader::~DRGrader() {
    for (auto* name : input_names_) free(const_cast<char*>(name));
    for (auto* name : output_names_) free(const_cast<char*>(name));
}

cv::Mat DRGrader::crop_circle(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Find the retinal region
    cv::Mat thresh;
    cv::threshold(gray, thresh, 10, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return image;
    }

    // Find largest contour
    auto largest = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    cv::Rect bbox = cv::boundingRect(*largest);

    // Add small padding
    int pad = 5;
    bbox.x = std::max(0, bbox.x - pad);
    bbox.y = std::max(0, bbox.y - pad);
    bbox.width = std::min(image.cols - bbox.x, bbox.width + 2 * pad);
    bbox.height = std::min(image.rows - bbox.y, bbox.height + 2 * pad);

    return image(bbox).clone();
}

cv::Mat DRGrader::apply_ben_graham(const cv::Mat& image, int sigma) {
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(0, 0), sigma);

    cv::Mat processed;
    cv::addWeighted(image, 4, blurred, -4, 128, processed);

    return processed;
}

cv::Mat DRGrader::preprocess(const cv::Mat& image) {
    // 1. Crop circle (remove black borders)
    cv::Mat cropped = crop_circle(image);

    // 2. Resize to model input size
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(input_size_, input_size_));

    // 3. Apply Ben Graham preprocessing
    cv::Mat processed = apply_ben_graham(resized);

    // 4. Convert to float and normalize
    cv::Mat float_img;
    processed.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // 5. Apply ImageNet normalization
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    // OpenCV uses BGR, but model expects RGB
    channels[0] = (channels[0] - mean_[2]) / std_[2]; // B -> R
    channels[1] = (channels[1] - mean_[1]) / std_[1]; // G -> G
    channels[2] = (channels[2] - mean_[0]) / std_[0]; // R -> B

    cv::Mat normalized;
    cv::merge(channels, normalized);

    return normalized;
}

float DRGrader::run_inference(const cv::Mat& preprocessed) {
    // Convert to CHW format (channels first)
    std::vector<float> input_tensor(3 * input_size_ * input_size_);

    std::vector<cv::Mat> channels(3);
    cv::split(preprocessed, channels);

    // Reorder BGR -> RGB and flatten to CHW
    for (int c = 0; c < 3; ++c) {
        int src_channel = 2 - c; // BGR to RGB
        std::memcpy(input_tensor.data() + c * input_size_ * input_size_,
                    channels[src_channel].data,
                    input_size_ * input_size_ * sizeof(float));
    }

    // Create input tensor
    std::vector<int64_t> shape = {1, 3, input_size_, input_size_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        shape.data(), shape.size());

    // Run inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor_ort, 1,
        output_names_.data(), 1);

    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return output_data[0];
}

DRGrade DRGrader::score_to_grade(float score) {
    if (score < thresholds_[0]) return DRGrade::NoDR;
    if (score < thresholds_[1]) return DRGrade::Mild;
    if (score < thresholds_[2]) return DRGrade::Moderate;
    if (score < thresholds_[3]) return DRGrade::Severe;
    return DRGrade::Proliferative;
}

std::string DRGrader::grade_to_string(DRGrade grade) {
    switch (grade) {
        case DRGrade::NoDR: return "No DR";
        case DRGrade::Mild: return "Mild";
        case DRGrade::Moderate: return "Moderate";
        case DRGrade::Severe: return "Severe";
        case DRGrade::Proliferative: return "Proliferative";
        default: return "Unknown";
    }
}

GradingResult DRGrader::grade(const cv::Mat& image) {
    cv::Mat preprocessed = preprocess(image);
    float score = run_inference(preprocessed);
    DRGrade grade = score_to_grade(score);

    return GradingResult{
        .raw_score = score,
        .grade = grade,
        .grade_name = grade_to_string(grade),
        .confidence = 1.0f // Could implement proper confidence later
    };
}

GradingResult DRGrader::grade_from_file(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return grade(image);
}

std::vector<GradingResult> DRGrader::grade_batch(const std::vector<cv::Mat>& images) {
    std::vector<GradingResult> results;
    results.reserve(images.size());

    for (const auto& image : images) {
        results.push_back(grade(image));
    }

    return results;
}

} // namespace fundus