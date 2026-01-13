
cmake_minimum_required(VERSION 3.20)
project(fundus_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# Header-only library
add_library(dr_grader STATIC
    src/dr_grader.cpp
)

target_include_directories(dr_grader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(dr_grader PUBLIC
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

# Main executable
add_executable(fundus_inference apps/main.cpp)
target_link_libraries(fundus_inference PRIVATE dr_grader)

# Copy model to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION ${CMAKE_CURRENT_BINARY_DIR})