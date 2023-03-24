cmake_minimum_required(VERSION 3.0)

project(MemoryBitmap LANGUAGES C)

add_executable(MemoryBitmap WaveformsMemoryBitmap.c)

target_compile_options(MemoryBitmap PRIVATE -Wall -Wextra)

target_compile_features(MemoryBitmap PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(MemoryBitmap PRIVATE OpenCL::OpenCL)