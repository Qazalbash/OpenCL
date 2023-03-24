cmake_minimum_required(VERSION 3.0)

project(MatrixMultiplicationTiled LANGUAGES C)

add_executable(MatrixMultiplicationTiled MatrixMultiplicationTiled.c)

target_compile_options(MatrixMultiplicationTiled PRIVATE -Wall -Wextra)

target_compile_features(MatrixMultiplicationTiled PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(MatrixMultiplicationTiled PRIVATE OpenCL::OpenCL)