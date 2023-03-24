cmake_minimum_required(VERSION 3.0)

project(VectorAddition LANGUAGES C)

add_executable(VectorAddition add.c)

target_compile_options(VectorAddition PRIVATE -Wall -Wextra)

target_compile_features(VectorAddition PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(VectorAddition PRIVATE OpenCL::OpenCL)