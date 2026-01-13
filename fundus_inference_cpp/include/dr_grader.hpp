#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace fundus {

enum class DRGrade {
    NoDR = 0,
    Mild = 1,
    Moderate = 2,
    Severe = 3,
    Proliferative = 4
};

struct GradingResult {
    float raw_score;
    DRGrade grade;
    std::string grade_name;
    float confidence;
};

class DRGrader {
public:
    explicit DRGrader(const std::string& model_path);
    ~DRGrader();

    // Prevent copying
    DRGrader(const DRGrader&) = delete;
    DRGrader& operator=(const DRGrader&) = delete;

    // Main inference function
    GradingResult grade(const cv::Mat& image);
    GradingResult grade_from_file(const std::string& image_path);

    // Batch processing
    std::vector<GradingResult> grade_batch(const std::vector<cv::Mat>& images);

private:
    // Preprocessing
    cv::Mat preprocess(const cv::Mat& image);
    cv::Mat crop_circle(const cv::Mat& image);
    cv::Mat apply_ben_graham(const cv::Mat& image, int sigma = 30);

    // Inference
    float run_inference(const cv::Mat& preprocessed);

    // Postprocessing
    DRGrade score_to_grade(float score);
    std::string grade_to_string(DRGrade grade);

    // ONNX Runtime members
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Model info
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;

    // Optimized thresholds from training
    static constexpr std::array<float, 4> thresholds_ = {0.55f, 1.51f, 2.50f, 3.17f};

    // ImageNet normalization constants
    static constexpr std::array<float, 3> mean_ = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> std_ = {0.229f, 0.224f, 0.225f};

    static constexpr int input_size_ = 512;
};

} // namespace fundus