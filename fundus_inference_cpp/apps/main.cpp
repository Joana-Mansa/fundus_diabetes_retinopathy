#include "dr_grader.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void print_usage(const char* program) {
    std::cout << "Diabetic Retinopathy Grader\n"
              << "Usage: " << program << " <model_path> <image_path>\n"
              << "\nExample:\n"
              << "  " << program << " models/dr_model.onnx test_images/fundus.jpg\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // Validate paths
    if (!fs::exists(model_path)) {
        std::cerr << "Error: Model not found: " << model_path << "\n";
        return 1;
    }
    if (!fs::exists(image_path)) {
        std::cerr << "Error: Image not found: " << image_path << "\n";
        return 1;
    }

    try {
        std::cout << "Loading model: " << model_path << "\n";
        fundus::DRGrader grader(model_path);

        std::cout << "Processing image: " << image_path << "\n";
        auto result = grader.grade_from_file(image_path);

        std::cout << "\n=== Results ===\n"
                  << "Raw score: " << result.raw_score << "\n"
                  << "Grade: " << static_cast<int>(result.grade)
                  << " (" << result.grade_name << ")\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}