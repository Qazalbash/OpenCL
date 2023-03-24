cmake_minimum_required(VERSION 3.0)

project(MatrixMultiplication LANGUAGES C)

add_executable(MatrixMultiplication mult.c)

target_compile_options(MatrixMultiplication PRIVATE -Wall -Wextra)

target_compile_features(MatrixMultiplication PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(MatrixMultiplication PRIVATE OpenCL::OpenCL)