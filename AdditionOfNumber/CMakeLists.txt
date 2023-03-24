cmake_minimum_required(VERSION 3.0)

project(AdditionOfNumber LANGUAGES C)

add_executable(AdditionOfNumber add.c)

target_compile_options(AdditionOfNumber PRIVATE -Wall -Wextra)

target_compile_features(AdditionOfNumber PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(AdditionOfNumber PRIVATE OpenCL::OpenCL)