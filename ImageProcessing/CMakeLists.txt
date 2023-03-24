cmake_minimum_required(VERSION 3.0)

project(ImageProcessing LANGUAGES C)

add_executable(ImageProcessing RawImageProcessing.c)

target_compile_options(ImageProcessing PRIVATE -Wall -Wextra)

target_compile_features(ImageProcessing PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(ImageProcessing PRIVATE OpenCL::OpenCL)